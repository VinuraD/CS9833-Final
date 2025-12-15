from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from .pyg_explain_compat import ensure_pyg_explain_available
from .pyg_utils_compat import ensure_coalesce_available, ensure_sort_edge_index_compat

ensure_pyg_explain_available()
ensure_coalesce_available()
ensure_sort_edge_index_compat()

from torch_geometric.explain import Explainer, ModelConfig
from torch_geometric.explain.config import ModelMode, ModelTaskLevel, ModelReturnType


class _ExplainerModelAdapter(nn.Module):
    """Adapts project models to the signature expected by PyG's Explainer."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, batch=None, edge_weight=None, **kwargs):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if batch.numel() < x.size(0):
            pad_size = x.size(0) - batch.numel()
            pad_value = batch[-1].item() if batch.numel() > 0 else 0
            pad_tensor = torch.full((pad_size,), int(pad_value), dtype=torch.long, device=x.device)
            batch = torch.cat([batch, pad_tensor], dim=0)
        data = Data(x=x, edge_index=edge_index)
        data.batch = batch
        return self.model(data, edge_weight=edge_weight)


class C2ExplainerWrapper:
    """Thin wrapper around the upstream C2Explainer implementation."""

    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 repo_root: Optional[str] = None,
                 epochs: int = 1200,
                 lr: float = 0.05,
                 print_loss: bool = False,
                 silent_mode: bool = False,
                 ar_mode: bool = False,
                 fpm: bool = False,
                 subgraph_mode: bool = False,
                 wo_st: bool = False,
                 at_loss: bool = False,
                 ent_loss: bool = False,
                 undirected: bool = True) -> None:
        self.device = device
        self.model = model
        self.repo_root = self._resolve_repo(repo_root)
        self._import_c2()
        self._c2_cls = self._build_c2_explainer()
        self._c2_kwargs = dict(
            epochs=epochs,
            lr=lr,
            print_loss=print_loss,
            silent_mode=silent_mode,
            AR_mode=ar_mode,
            FPM=fpm,
            subgraph_mode=subgraph_mode,
            wo_ST=wo_st,
            AT_loss=at_loss,
            ENT_loss=ent_loss,
            undirected=undirected,
        )
        self._adapter = _ExplainerModelAdapter(self.model).to(device)

    @staticmethod
    def _ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 1:
            return tensor.unsqueeze(0)
        return tensor

    def _build_c2_explainer(self):
        from cf_explainer import C2Explainer as ExternalC2Explainer  # type: ignore

        outer = self

        class PatchedC2Explainer(ExternalC2Explainer):
            def _cf_loss(self, y_hat: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                y_hat_mod = outer._ensure_2d(y_hat)
                target_idx = torch.tensor([int(self.desired_y)], device=y_hat_mod.device)
                return outer._compute_loss_from_return_type(y_hat_mod, target_idx, getattr(self, 'model_config', None))

            def _at_cf_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                y_hat_mod = outer._ensure_2d(y_hat)
                target_idx = torch.tensor([int(y)], device=y_hat_mod.device)
                loss = outer._compute_loss_from_return_type(y_hat_mod, target_idx, getattr(self, 'model_config', None))
                pred_same = getattr(self, 'pred_same', True)
                return -loss * (1 if pred_same else 0)

        return PatchedC2Explainer

    @staticmethod
    def _compute_loss_from_return_type(logits: torch.Tensor,
                                       target_idx: torch.Tensor,
                                       model_config: Optional[ModelConfig]) -> torch.Tensor:
        return_type = getattr(model_config, 'return_type', ModelReturnType.raw)
        if return_type == ModelReturnType.raw:
            return F.cross_entropy(logits, target_idx)
        if return_type == ModelReturnType.probs:
            probs = torch.clamp(logits, min=1e-12)
            return F.nll_loss(probs.log(), target_idx)
        if return_type == ModelReturnType.log_probs:
            return F.nll_loss(logits, target_idx)
        return F.cross_entropy(logits, target_idx)

    @staticmethod
    def _resolve_repo(repo_root: Optional[str]) -> Path:
        if repo_root:
            candidate = Path(repo_root).expanduser().resolve()
        else:
            candidate = Path(__file__).resolve().parents[2] / 'Git' / 'C2Explainer'
        if not candidate.exists():
            raise FileNotFoundError(f'Could not locate C2Explainer repository at {candidate}')
        return candidate

    def _import_c2(self) -> None:
        src_dir = self.repo_root / 'src'
        if not src_dir.exists():
            raise FileNotFoundError(f'C2Explainer src directory not found at {src_dir}')
        src_path = str(src_dir)
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

    def _build_explainer(self) -> Explainer:
        algorithm = self._c2_cls(**self._c2_kwargs)
        model_config = ModelConfig(
            mode=ModelMode.classification,
            task_level=ModelTaskLevel.graph,
            return_type=ModelReturnType.raw,
        )
        return Explainer(
            model=self._adapter,
            algorithm=algorithm,
            explanation_type='phenomenon',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=model_config,
        )

    def explain(self, graph, _unused=None):
        if graph is None:
            return None
        batch_vec = getattr(graph, 'batch', torch.zeros(graph.x.size(0), dtype=torch.long, device=graph.x.device))
        target_tensor = getattr(graph, 'targets', getattr(graph, 'y', None))
        if target_tensor is None:
            raise ValueError('Graph must contain targets or labels for C2Explainer.')
        if target_tensor.dim() == 0:
            target_tensor = target_tensor.view(1)
        else:
            target_tensor = target_tensor.view(-1)
        target_tensor = target_tensor[:1].to(graph.x.device).long()

        explainer = self._build_explainer()
        explanation = explainer(
            x=graph.x,
            edge_index=graph.edge_index,
            batch=batch_vec,
            target=target_tensor,
        )
        cf_edge_index = getattr(explanation, 'cf', None)
        if cf_edge_index is None or cf_edge_index.numel() == 0:
            return None
        return self._build_counterfactual_graph(graph, cf_edge_index)

    def _build_counterfactual_graph(self, original, cf_edge_index):
        cf_edge_index = cf_edge_index.detach().cpu()
        x = original.x.detach().cpu()
        cf_x = self._augment_features(x, cf_edge_index)
        cf_graph = Data(x=cf_x, edge_index=cf_edge_index)
        if hasattr(original, 'y'):
            cf_graph.y = original.y.detach().cpu()
        if hasattr(original, 'targets'):
            cf_graph.targets = original.targets.detach().cpu()
        if hasattr(cf_graph, 'batch'):
            del cf_graph.batch
        return cf_graph

    @staticmethod
    def _augment_features(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return x.clone()
        max_node = int(edge_index.max().item())
        required_nodes = max_node + 1
        if required_nodes <= x.size(0):
            return x.clone()
        deficit = required_nodes - x.size(0)
        repeats = []
        remaining = deficit
        while remaining > 0:
            take = min(remaining, x.size(0))
            repeats.append(x[:take])
            remaining -= take
        extra = torch.cat(repeats, dim=0)
        return torch.cat([x, extra], dim=0)
