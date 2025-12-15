import copy
import math
from typing import Callable, Optional, Union

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_networkx

from attacks.BB_attack.Sign_OPT import new_graph


def _edge_index_from_networkx(graph_nx: nx.Graph, device: torch.device) -> torch.Tensor:
    """
    Convert a NetworkX graph into a bidirectional PyG edge_index tensor.
    """
    edges = list(graph_nx.edges())
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    edge_tensor = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    reverse_edges = edge_tensor[[1, 0], :]
    return torch.cat([edge_tensor, reverse_edges], dim=1)


def _graph_distance(a: Data, b: Data) -> float:
    """
    Count undirected edge flips needed to transform graph b into graph a.
    """
    adj_a = nx.adjacency_matrix(to_networkx(a, to_undirected=True)).toarray()
    adj_b = nx.adjacency_matrix(to_networkx(b, to_undirected=True)).toarray()
    max_nodes = max(adj_a.shape[0], adj_b.shape[0])
    def pad(mat):
        if mat.shape[0] == max_nodes:
            return mat
        pad_width = max_nodes - mat.shape[0]
        return np.pad(mat, ((0, pad_width), (0, pad_width)), mode='constant')
    adj_a = pad(adj_a)
    adj_b = pad(adj_b)
    return float(np.sum(np.abs(adj_a - adj_b)) / 2)


def _ensure_single_graph(graph: Union[Data, Batch]) -> Data:
    if isinstance(graph, Batch):
        if graph.num_graphs < 1:
            raise ValueError("Received an empty batch for random counterfactual generation.")
        return graph.to_data_list()[0]
    return graph


class RandExplainer:
    """
    Counterfactual generator that samples random edge flips, mirroring the RAND attack.
    """

    def __init__(self,
                 device: Optional[Union[str, torch.device]] = None,
                 trials: int = 500,
                 budget: float = 0.1,
                 distance_fn: Optional[Callable[[Data, Data], float]] = None) -> None:
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.trials = max(1, int(trials))
        self.budget = float(budget)
        self.distance_fn = distance_fn or _graph_distance

    def _edge_budget(self, graph: Data) -> int:
        num_nodes = int(graph.num_nodes)
        space_edges = max(1, num_nodes * (num_nodes - 1) // 2)
        if self.budget <= 1:
            return max(1, int(math.ceil(self.budget * space_edges)))
        return max(1, int(min(int(self.budget), space_edges)))

    def explain(self, graph: Union[Data, Batch], model) -> Optional[Data]:
        base_graph = _ensure_single_graph(graph).to(self.device)
        if not hasattr(base_graph, "targets"):
            raise ValueError("RandExplainer requires target labels stored in graph.targets.")
        target_label = int(base_graph.targets.view(-1)[0].item())
        orig_pred = int(model.predict(base_graph, device=self.device).item())
        if orig_pred == target_label:
            return None

        base_nx = to_networkx(base_graph.cpu(), to_undirected=True)
        max_budget = self._edge_budget(base_graph)
        best_cf = None
        best_dist = float("inf")

        for _ in range(self.trials):
            theta = torch.rand((base_graph.num_nodes, base_graph.num_nodes)).cpu()
            perturbed_nx = new_graph(base_nx, theta)
            candidate = copy.deepcopy(base_graph)
            candidate.edge_index = _edge_index_from_networkx(perturbed_nx, self.device)
            candidate.num_nodes = base_graph.num_nodes
            pred = int(model.predict(candidate, device=self.device).item())
            if pred == target_label:
                dist = self.distance_fn(candidate.cpu(), base_graph.cpu())
                if dist <= max_budget and dist < best_dist:
                    best_dist = dist
                    best_cf = candidate

        return best_cf
