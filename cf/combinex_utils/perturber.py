from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class Perturber(nn.Module, ABC):
    def __init__(self, cfg, model: nn.Module) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.deactivate_model()
        self.set_reproducibility()

    def deactivate_model(self) -> None:
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def set_reproducibility(self) -> None:
        seed = getattr(self.cfg.general, "seed", 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward_prediction(self, *args, **kwargs):
        raise NotImplementedError
