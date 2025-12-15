import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from typing import Iterable, Sequence, Tuple, Union

from .models import GIN

GraphInput = Union[Data, Batch, Sequence[Data], Iterable[Data]]


def _to_batch(graph_data: GraphInput, device: torch.device) -> Batch:
    if isinstance(graph_data, Batch):
        batch = graph_data
    elif isinstance(graph_data, Data):
        batch = Batch.from_data_list([graph_data])
    else:
        batch = Batch.from_data_list(list(graph_data))
    return batch.to(device)


class ConformalPredictor(torch.nn.Module):
    """
    Split-conformal predictor that wraps a base graph classifier and exposes
    prediction uncertainty via conformal p-values.
    """

    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("_calibration_scores", torch.empty(0))

    def forward(self, data: Batch) -> torch.Tensor:
        return self.base_model(data)

    @torch.no_grad()
    def fit_calibration(
        self,
        loader: DataLoader,
        device: torch.device = None,
    ) -> None:
        if device is None:
            device = next(self.base_model.parameters()).device

        scores = []
        self.eval()
        self.base_model.eval()
        for batch in loader:
            batch = batch.to(device)
            logits = self.base_model(batch)
            probs = torch.softmax(logits, dim=-1)
            true_probs = probs.gather(1, batch.y.view(-1, 1)).squeeze(1)
            scores.append(1.0 - true_probs)

        if not scores:
            raise ValueError("Calibration loader must yield at least one batch.")

        self._calibration_scores = torch.cat(scores).detach().cpu()

    @torch.no_grad()
    def predict(self, data: GraphInput, device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = next(self.base_model.parameters()).device
        batch = _to_batch(data, device)
        logits = self.base_model(batch)
        return logits.argmax(dim=-1).cpu()

    @torch.no_grad()
    def get_uncertainty(
        self,
        data: GraphInput,
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._calibration_scores.numel() == 0:
            raise RuntimeError("Call fit_calibration before requesting uncertainty.")

        if device is None:
            device = next(self.base_model.parameters()).device

        batch = _to_batch(data, device)
        logits = self.base_model(batch)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        alpha_hat = 1.0 - probs.gather(1, preds.view(-1, 1)).squeeze(1)
        p_values = self._conformal_p_value(alpha_hat)
        return preds.cpu(), p_values

    def _conformal_p_value(self, alpha_hat: torch.Tensor) -> torch.Tensor:
        calibration = self._calibration_scores.view(1, -1)
        comp = (calibration >= alpha_hat.view(-1, 1).cpu()).sum(dim=1)
        n = calibration.shape[1]
        return ((comp + 1) / (n + 1)).float()


def build_gin_conformal_predictor(
    num_layers: int,
    num_mlp_layers: int,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
) -> ConformalPredictor:
    gin = GIN(num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, dropout)
    return ConformalPredictor(gin)
