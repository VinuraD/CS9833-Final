import copy
from typing import Callable

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from attacks.BB_attack.Sign_OPT import new_graph


def _edge_index_from_networkx(graph_nx: nx.Graph, device: torch.device) -> torch.Tensor:
    edges = list(graph_nx.edges())
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    edge_tensor = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    reverse_edges = edge_tensor[[1, 0], :]
    edge_index = torch.cat([edge_tensor, reverse_edges], dim=1)
    return edge_index


def random_attack_sample(graph: Data,
                         model,
                         device: torch.device,
                         trials: int,
                         max_budget: int,
                         distance_fn: Callable[[Data, Data], float]):
    """
    Perform a simple random edge perturbation attack by sampling random adjacency masks.
    Returns (adv_graph, adv_label, success_flag, perturbation_distance).
    """
    model.eval()
    graph = graph.to(device)
    y0 = graph.y[0]
    base_graph_nx = to_networkx(graph.cpu(), to_undirected=True)
    best_adv = None
    best_dist = float('inf')

    for _ in range(max(1, trials)):
        num_nodes = graph.num_nodes
        theta = torch.rand((num_nodes, num_nodes)).cpu()
        perturbed_nx = new_graph(base_graph_nx, theta)
        adv_graph = copy.deepcopy(graph)
        adv_graph.edge_index = _edge_index_from_networkx(perturbed_nx, device)
        adv_graph.num_nodes = graph.num_nodes
        pred = model.predict(adv_graph, None, device)
        if pred != y0:
            dis = distance_fn(adv_graph.cpu(), graph.cpu())
            if dis <= max_budget and dis < best_dist:
                best_dist = dis
                best_adv = copy.deepcopy(adv_graph)

    if best_adv is None:
        return graph, y0, False, -1
    adv_label = model.predict(best_adv, None, device)
    return best_adv, adv_label, True, best_dist
