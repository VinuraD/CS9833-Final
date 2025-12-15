from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import Any

import torch


def ensure_pyg_explain_available() -> None:
    """Install a lightweight torch_geometric.explain shim if PyG lacks it."""

    try:
        import torch_geometric.explain  # type: ignore  # noqa:F401
        return
    except ModuleNotFoundError:
        pass

    import sys

    if 'torch_geometric' not in sys.modules:
        # Create a minimal torch_geometric namespace so submodules register cleanly.
        torch_geometric_mod = ModuleType('torch_geometric')
        sys.modules['torch_geometric'] = torch_geometric_mod
    else:
        torch_geometric_mod = sys.modules['torch_geometric']

    explain_mod = ModuleType('torch_geometric.explain')
    config_mod = ModuleType('torch_geometric.explain.config')
    algorithm_mod = ModuleType('torch_geometric.explain.algorithm')

    class MaskType(Enum):
        object = 'object'
        common_attributes = 'common_attributes'
        attributes = 'attributes'

    class ModelMode(Enum):
        classification = 'classification'
        regression = 'regression'

    class ModelTaskLevel(Enum):
        node = 'node'
        edge = 'edge'
        graph = 'graph'

    class ModelReturnType(Enum):
        raw = 'raw'
        probs = 'probs'
        log_probs = 'log_probs'

    config_mod.MaskType = MaskType
    config_mod.ModelMode = ModelMode
    config_mod.ModelTaskLevel = ModelTaskLevel
    config_mod.ModelReturnType = ModelReturnType

    @dataclass
    class ExplainerConfig:
        explanation_type: str | None = None
        node_mask_type: str | None = None
        edge_mask_type: str | None = None

    @dataclass
    class ModelConfig:
        mode: ModelMode
        task_level: ModelTaskLevel
        return_type: ModelReturnType

    class Explanation(dict):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)

        def __getattr__(self, item: str) -> Any:
            try:
                return self[item]
            except KeyError as exc:
                raise AttributeError(item) from exc

        def __setattr__(self, key: str, value: Any) -> None:
            self[key] = value

    class ExplainerAlgorithm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model_config: ModelConfig | None = None

        @staticmethod
        def _num_hops(model) -> int:
            return int(getattr(model, 'num_layers', 1))

        @staticmethod
        def _flow(model) -> str:
            return getattr(model, 'flow', 'source_to_target')

        def supports(self) -> bool:
            return True

    class Explainer:
        def __init__(self,
                     model,
                     algorithm,
                     *,
                     model_config: ModelConfig,
                     **_unused):
            self.model = model
            self.algorithm = algorithm
            self.model_config = model_config
            if isinstance(self.algorithm, ExplainerAlgorithm):
                self.algorithm.model_config = model_config

        def __call__(self, **kwargs):
            return self.algorithm.forward(model=self.model, **kwargs)

    explain_mod.Explainer = Explainer
    explain_mod.ExplainerConfig = ExplainerConfig
    explain_mod.ModelConfig = ModelConfig
    explain_mod.Explanation = Explanation

    algorithm_mod.ExplainerAlgorithm = ExplainerAlgorithm

    sys.modules['torch_geometric.explain'] = explain_mod
    sys.modules['torch_geometric.explain.config'] = config_mod
    sys.modules['torch_geometric.explain.algorithm'] = algorithm_mod
    torch_geometric_mod.explain = explain_mod
