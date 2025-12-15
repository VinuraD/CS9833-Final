from __future__ import annotations

from typing import Optional

import torch
import torch.optim as optim
from torch_geometric.data import Data


def get_optimizer(cfg, model):
    name = cfg.optimizer.name.lower()
    lr = cfg.optimizer.lr
    if name == "sgd":
        momentum = cfg.optimizer.n_momentum
        if momentum == 0.0:
            return optim.SGD(model.parameters(), lr=lr)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    if name == "adadelta":
        return optim.Adadelta(model.parameters(), lr=lr)
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    raise ValueError("Unsupported optimizer {} for COMBINEX".format(cfg.optimizer.name))


def discretize_to_nearest_integer(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(tensor)


def build_counterfactual_graph_gc(x: torch.Tensor,
                                  edge_index: torch.Tensor,
                                  graph: Data,
                                  oracle,
                                  output_actual: torch.Tensor,
                                  device: Optional[torch.device] = None) -> Data:
    device = device or x.device
    cf = Data(
        x=x.detach().cpu(),
        edge_index=edge_index.detach().cpu(),
        y=torch.argmax(output_actual.detach().cpu(), dim=1)
    )
    if hasattr(graph, "targets"):
        cf.targets = graph.targets.detach().cpu()
    if hasattr(graph, "batch"):
        with torch.no_grad():
            # embedding = oracle.get_embedding_repr(x, edge_index, graph.batch)
            embedding = oracle.get_embedding_repr(x, edge_index, graph.batch)
        cf.x_projection = embedding.detach().cpu()
    return cf
