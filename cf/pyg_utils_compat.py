from __future__ import annotations

import inspect
import sys
from typing import Optional, Tuple

import torch


def ensure_coalesce_available() -> None:
    try:
        import torch_geometric.utils as pyg_utils
    except ModuleNotFoundError:
        return

    if hasattr(pyg_utils, 'coalesce'):
        return

    def _coalesce(edge_index: torch.Tensor,
                  edge_attr: Optional[torch.Tensor] = None,
                  m: Optional[int] = None,
                  n: Optional[int] = None,
                  reduce: str = 'add') -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if edge_index.numel() == 0:
            return edge_index, edge_attr

        cols = edge_index.t()
        unique_cols, inverse = torch.unique(cols, dim=0, return_inverse=True)

        new_attr = None
        if edge_attr is not None:
            if reduce != 'add':
                raise NotImplementedError("Fallback coalesce only supports 'add' reduction")
            if edge_attr.dim() == 0:
                edge_attr_flat = edge_attr.view(1, 1)
            else:
                edge_attr_flat = edge_attr.view(edge_attr.size(0), -1)
            new_attr = torch.zeros((unique_cols.size(0), edge_attr_flat.size(1)),
                                   dtype=edge_attr.dtype,
                                   device=edge_attr.device)
            scatter_index = inverse.view(-1, 1).expand_as(edge_attr_flat)
            new_attr.scatter_add_(0, scatter_index, edge_attr_flat)
            if edge_attr.dim() <= 1:
                new_attr = new_attr.view(unique_cols.size(0))
            else:
                new_attr = new_attr.view((unique_cols.size(0),) + tuple(edge_attr.shape[1:]))

        return unique_cols.t().contiguous(), new_attr

    pyg_utils.coalesce = _coalesce
    sys.modules['torch_geometric.utils'].coalesce = _coalesce


def ensure_sort_edge_index_compat() -> None:
    try:
        import torch_geometric.utils as pyg_utils
    except ModuleNotFoundError:
        return

    sort_edge_index = getattr(pyg_utils, 'sort_edge_index', None)
    if sort_edge_index is None:
        return

    try:
        test = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        result = sort_edge_index(test.clone())
    except Exception:
        return

    if not isinstance(result, tuple):
        return

    supports_return_perm = 'return_perm' in inspect.signature(sort_edge_index).parameters

    def compat(edge_index, *args, **kwargs):
        return_perm = supports_return_perm and kwargs.get('return_perm', False)
        out = sort_edge_index(edge_index, *args, **kwargs)
        if isinstance(out, tuple):
            edge_idx, perm = out
            if return_perm:
                return edge_idx, perm
            return edge_idx
        return out

    pyg_utils.sort_edge_index = compat
    sys.modules['torch_geometric.utils'].sort_edge_index = compat
