from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data


class GraphModelAdapter(nn.Module):
    """
    Wraps the project-specific graph classifiers so they expose the COMBINEX signature.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @staticmethod
    def _build_data(x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Data:
        data = Data(x=x, edge_index=edge_index)
        data.batch = batch
        return data

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, edge_weights=None):
        data = self._build_data(x, edge_index, batch)
        if edge_weights is not None:
            return self.model(data, edge_weight=edge_weights)
        return self.model(data)

    def get_embedding_repr(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        data = self._build_data(x, edge_index, batch)
        if hasattr(self.model, "get_embedding_repr"):
            return self.model.get_embedding_repr(data)
        raise AttributeError("Underlying model does not implement get_embedding_repr required by COMBINEX.")
