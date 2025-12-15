from __future__ import annotations

from typing import Iterable

import torch
from torch_geometric.data import Data


class GraphDataInfo:
    """
    Minimal dataset descriptor required by the COMBINEX graph-level explainer.
    Computes feature statistics once so the explainer can clamp/discretize perturbations.
    """

    def __init__(self, dataset: Iterable[Data], num_classes: int, device: torch.device) -> None:
        features = []
        for graph in dataset:
            if graph.x is None:
                raise ValueError("COMBINEX requires node features; got a graph without 'x'.")
            features.append(graph.x.detach().cpu().float())

        if not features:
            raise ValueError("The provided dataset is empty; cannot build COMBINEX data info.")

        stacked = torch.cat(features, dim=0)
        self.num_features = stacked.shape[1]
        self.num_classes = num_classes
        self.device = device

        self.min_range = stacked.min(dim=0).values.to(device)
        self.max_range = stacked.max(dim=0).values.to(device)
        self.discrete_mask = self._infer_discrete_mask(stacked).to(device)

    @staticmethod
    def _infer_discrete_mask(features: torch.Tensor) -> torch.Tensor:
        """
        Heuristically detect discrete/binary features so the explainer can treat them carefully.
        A feature is considered discrete if it only takes a small integer range (<=10 unique values).
        """
        rounded = torch.round(features)
        integer_like = (features - rounded).abs() < 1e-6
        integer_like = integer_like.all(dim=0)

        min_vals = features.min(dim=0).values
        max_vals = features.max(dim=0).values
        bounded = (min_vals >= 0.0) & (max_vals <= 1.0)

        unique_counts = []
        for idx in range(features.shape[1]):
            unique_counts.append(int(torch.unique(features[:, idx]).numel()))
        unique_counts = torch.tensor(unique_counts, dtype=torch.long)
        few_unique = unique_counts <= 10

        mask = integer_like & (bounded | few_unique)
        return mask.float()
