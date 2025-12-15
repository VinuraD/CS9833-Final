from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
from torch_geometric.data import Data

from .perturber import Perturber
from .utils import discretize_to_nearest_integer


class GraphPerturber(Perturber):
    def __init__(self, cfg, model: nn.Module, graph: Data, datainfo, device: torch.device) -> None:
        super().__init__(cfg=cfg, model=model)
        self.device = device
        self.graph_sample = graph
        self.num_classes = datainfo.num_classes
        self.num_nodes = graph.x.shape[0]
        self.num_features = datainfo.num_features
        self.min_range = datainfo.min_range
        self.max_range = datainfo.max_range
        self.discrete_features_mask = datainfo.discrete_mask
        self.continuous_features_mask = 1 - datainfo.discrete_mask

        self.P_x = Parameter(torch.zeros(self.num_nodes, self.num_features, device=self.device))
        self.EP_x = Parameter(torch.ones(graph.edge_index.shape[1], device=self.device))

        self.edge_index = graph.edge_index
        self.x = graph.x

    @staticmethod
    def discretize_tensor(tensor: torch.Tensor) -> torch.Tensor:
        return torch.where(tensor <= 0.5, torch.zeros_like(tensor), torch.ones_like(tensor))

    def _feature_clamp(self, value: Tensor) -> Tensor:
        min_vals = self.min_range.view(1, -1)
        max_vals = self.max_range.view(1, -1)
        value = torch.max(value, min_vals)
        value = torch.min(value, max_vals)
        return value

    def forward(self, V_x: Tensor, batch: Tensor) -> Tensor:
        tanh_discrete_features = torch.tanh(self.P_x)
        perturbation_discrete_rescaling = self.min_range + (self.max_range - self.min_range) * tanh_discrete_features
        perturbed_discrete = perturbation_discrete_rescaling + V_x
        discrete_perturbation = self.discrete_features_mask * self._feature_clamp(perturbed_discrete)
        continuous_perturbation = self.continuous_features_mask * self._feature_clamp(self.P_x + V_x)
        perturbed_features = discrete_perturbation + continuous_perturbation

        edge_weights = torch.sigmoid(self.EP_x)
        return self.model(perturbed_features, self.graph_sample.edge_index, batch, edge_weights=edge_weights)

    def forward_prediction(self, V_x: Tensor, batch: Tensor):
        discrete_perturbation = self.discrete_features_mask * discretize_to_nearest_integer(
            self.min_range + (self.max_range - self.min_range) * torch.tanh(self.P_x) + V_x
        )
        discrete_perturbation = self._feature_clamp(discrete_perturbation)
        continuous_perturbation = self.continuous_features_mask * self._feature_clamp(self.P_x + V_x)
        V_pert = discrete_perturbation + continuous_perturbation
        EP_x_discrete = self.discretize_tensor(torch.sigmoid(self.EP_x))
        out = self.model(V_pert, self.edge_index, batch, edge_weights=EP_x_discrete)
        return out, V_pert, self.EP_x

    def edge_loss(self, graph: Data) -> Tuple[Tensor, Tensor]:
        cf_edge_weights = torch.sigmoid(self.EP_x)
        sparsity_loss = torch.sum(torch.abs(cf_edge_weights - 1))
        cf_edge_weights_discrete = self.discretize_tensor(cf_edge_weights)
        edge_mask = cf_edge_weights_discrete == 1
        cf_edge_index = graph.edge_index[:, edge_mask]
        return sparsity_loss, cf_edge_index

    def node_loss(self, graph: Data) -> Tuple[Tensor, Tensor]:
        loss_discrete = F.l1_loss(
            graph.x * self.discrete_features_mask,
            torch.clamp(self.discrete_features_mask * torch.tanh(self.P_x) + graph.x, 0, 1),
        )
        loss_continuous = F.mse_loss(
            graph.x * self.continuous_features_mask,
            self.continuous_features_mask * (self.P_x + graph.x),
        )
        return loss_discrete + loss_continuous, self.edge_index
