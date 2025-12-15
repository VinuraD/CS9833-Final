from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import numpy as np


class Explainer(ABC):
    def __init__(self, cfg, datainfo) -> None:
        self.cfg = cfg
        self.datainfo = datainfo
        self.device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
        self.verbose = getattr(cfg, "verbose", False)

    @abstractmethod
    def explain(self, graph, oracle, **kwargs):
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    def set_reproducibility(self) -> None:
        seed = getattr(self.cfg.general, "seed", 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
        np.random.seed(seed)
