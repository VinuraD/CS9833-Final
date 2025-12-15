from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 0.1
    n_momentum: float = 0.0
    beta: float = 0.5
    gamma: float = 1.0
    num_epochs: int = 700


@dataclass
class SchedulerConfig:
    policy: str = "linear"
    initial_alpha: float = 1.0
    step_size: float = 0.05
    decay_rate: float = 0.1


@dataclass
class GeneralConfig:
    seed: int = 42


@dataclass
class CombinexConfig:
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    general: GeneralConfig = field(default_factory=GeneralConfig)
    timeout: float = 500.0
    clip_grad_norm: float = 2.0
    verbose: bool = False
    device: str = field(default_factory=_default_device)


def _apply_overrides(obj: Any, overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if isinstance(value, dict) and not isinstance(current, (str, int, float, bool)):
            _apply_overrides(current, value)
        else:
            setattr(obj, key, value)


def build_default_config(device: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> CombinexConfig:
    cfg = CombinexConfig()
    if device is not None:
        cfg.device = device
    else:
        cfg.device = _default_device()
    if overrides:
        _apply_overrides(cfg, overrides)
    return cfg
