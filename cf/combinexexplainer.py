from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch_geometric.data import Data

from .combinex_utils import (
    CombinedExplainer,
    GraphDataInfo,
    GraphModelAdapter,
    build_default_config,
)


class CombinexExplainer:
    """
    Thin wrapper that mirrors the API of the other counterfactual generators.
    """

    def __init__(self,
                 dataset: Iterable[Data],
                 num_classes: int,
                 model,
                 device: Optional[torch.device] = None,
                 config_overrides: Optional[dict] = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        cfg_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.cfg = build_default_config(device=cfg_device, overrides=config_overrides)
        self.datainfo = GraphDataInfo(dataset=dataset, num_classes=num_classes, device=self.device)
        self.oracle = GraphModelAdapter(model).to(self.device)
        self.explainer = CombinedExplainer(cfg=self.cfg, datainfo=self.datainfo)

    def explain(self, graph: Data, _unused=None):
        return self.explainer.explain(graph=graph, oracle=self.oracle)
