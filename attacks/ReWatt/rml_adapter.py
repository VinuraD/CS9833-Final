import copy
import os
import sys
from typing import Optional, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

REWATT_DIR = os.path.dirname(__file__)
if REWATT_DIR not in sys.path:
    sys.path.append(REWATT_DIR)

from edge_policy_nets import RandomAgent  # noqa: E402


def _predict_label(model, data: Data, device) -> int:
    try:
        return int(model.predict(data, None, device))
    except TypeError:
        return int(model.predict(data, device))


def _data_to_nx(data: Data) -> Tuple[nx.Graph, torch.Tensor]:
    graph = nx.Graph()
    num_nodes = data.num_nodes
    graph.add_nodes_from(range(num_nodes))
    edge_index = data.edge_index
    if edge_index.numel() > 0:
        edges = edge_index.t().cpu().numpy().tolist()
        graph.add_edges_from(edges)
    if data.x is not None:
        node_features = data.x.detach().cpu()
    else:
        node_features = torch.ones((num_nodes, 1), dtype=torch.float32)
    for idx in range(num_nodes):
        graph.nodes[idx]['feat'] = node_features[idx].numpy()
    if getattr(data, 'y', None) is not None:
        graph.graph['label'] = int(data.y[0])
    return graph, node_features


def _graph_to_data(graph: nx.Graph, node_features: torch.Tensor, y: Optional[torch.Tensor]) -> Data:
    edges = list(graph.edges())
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    x = node_features.clone()
    data = Data(x=x, edge_index=edge_index)
    if y is not None:
        data.y = y
    return data


class PyGModelEnv:
    def __init__(self, model, device, include_self: bool = True, use_all: bool = False, targeted: bool = False):
        self.model = model
        self.device = device
        self.include_self = include_self
        self.use_all = use_all
        self.targeted = targeted
        self.node_features: Optional[torch.Tensor] = None
        self.graph: Optional[nx.Graph] = None
        self.label: Optional[int] = None
        self.target_label: Optional[int] = None
        self.negative_reward: float = -0.5
        self.done: int = 0

    def reset(
        self,
        graph: nx.Graph,
        node_features: torch.Tensor,
        negative_reward: float,
        base_label: Optional[int],
        target_label: Optional[int],
    ):
        self.graph = copy.deepcopy(graph)
        self.node_features = node_features
        self.negative_reward = negative_reward
        self.target_label = target_label
        self.done = 0
        self.label = base_label if base_label is not None else self.get_predicted_label()
        return self.graph, self.label

    def _as_data(self) -> Data:
        assert self.graph is not None
        data = _graph_to_data(self.graph, self.node_features, None)
        return data

    def get_predicted_label(self) -> int:
        data = self._as_data()
        return _predict_label(self.model, data, self.device)

    def evaluate_reward(self) -> Tuple[int, float]:
        current_label = self.get_predicted_label()
        success = current_label == self.target_label if self.targeted else current_label != self.label
        if success:
            self.done = 1
            return current_label, 1.0
        return current_label, self.negative_reward

    def step(self, action):
        if self.graph is None:
            raise RuntimeError('Environment not initialised; call reset first.')
        u, v, w = action
        if self.include_self or u != v:
            if self.graph.has_edge(u, v):
                self.graph.remove_edge(u, v)
        if self.include_self or u != w:
            self.graph.add_edge(u, w)
        current_label, reward = self.evaluate_reward()
        return [self.graph, reward, self.done, current_label]

    def constraint_action_space(self, selected_nodes):
        if len(selected_nodes) == 1:
            candidate_nodes = set(self.graph.neighbors(selected_nodes[0]))
            if self.include_self:
                candidate_nodes.add(selected_nodes[0])
        elif len(selected_nodes) == 2:
            target_node = selected_nodes[0]
            if self.use_all:
                one_hop = set(self.graph.neighbors(target_node))
                all_nodes = set(self.graph.nodes)
                candidate_nodes = all_nodes - one_hop
                candidate_nodes.add(target_node)
            else:
                one_hop_neighbors = set(self.graph.neighbors(target_node))
                two_hop_neighbors = set()
                for neigh in one_hop_neighbors:
                    two_hop_neighbors.update(self.graph.neighbors(neigh))
                candidate_nodes = two_hop_neighbors - one_hop_neighbors
                if self.include_self:
                    candidate_nodes.add(target_node)
        else:
            candidate_nodes = set()
        return list(candidate_nodes)


class ReWattAttacker:
    def __init__(
        self,
        model,
        device,
        action_percent: float = 0.03,
        max_actions: Optional[int] = None,
        negative_reward: float = -0.5,
        include_self: bool = True,
        use_all: bool = False,
        edge_based: bool = False,
    ):
        self.model = model
        self.device = device
        self.action_percent = action_percent
        self.max_actions = max_actions
        self.negative_reward = negative_reward
        self.include_self = include_self
        self.use_all = use_all
        self.edge_based = edge_based

    def _compute_budget(self, graph: nx.Graph) -> int:
        if self.max_actions is not None:
            return max(1, int(self.max_actions))
        edges = max(1, graph.number_of_edges())
        return max(1, int(np.ceil(edges * self.action_percent)))

    def attack(
        self,
        data: Data,
        targeted: bool = False,
        target_label: Optional[int] = None,
        initial_pred: Optional[int] = None,
    ):
        graph, node_features = _data_to_nx(data)
        if initial_pred is None:
            initial_pred = _predict_label(self.model, data, self.device)
        env = PyGModelEnv(
            model=self.model,
            device=self.device,
            include_self=self.include_self,
            use_all=self.use_all,
            targeted=targeted,
        )
        agent = RandomAgent(env, int(self.edge_based))
        num_possible_actions = self._compute_budget(graph)
        state, _ = env.reset(graph, node_features, self.negative_reward, int(initial_pred), target_label)
        success = False
        steps = 0
        for _ in range(num_possible_actions):
            action = agent.select_action(state)
            if action is None:
                break
            steps += 1
            state, reward, done, _ = env.step(action)
            if done or reward > 0:
                success = True
                break

        if success:
            adv_data = _graph_to_data(state, node_features, getattr(data, 'y', None))
        else:
            adv_data = None
        return adv_data, success, steps
