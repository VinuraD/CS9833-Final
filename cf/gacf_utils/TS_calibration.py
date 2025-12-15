"""
Lightweight temperature scaling utilities for calibrating GNN logits.
Credits: https://github.com/gpleiss/temperature_scaling/
"""

from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


class TemperatureScaledModel(nn.Module):
    """
    Wrap a classifier with a learnable temperature parameter for post-hoc calibration.

    The wrapped model is left untouched; only the temperature scalar is optimized.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))
        # Freeze the underlying model parameters during calibration.
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, batch):
        logits = self._get_logits(batch)
        return self.temp_scale(logits)

    def _get_logits(self, batch, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Obtain logits from the wrapped model. Falls back to model(batch) when
        the wrapped model does not expose a dedicated get_logits method.
        """
        device = device or next(self.parameters()).device
        if hasattr(self.model, "get_logits"):
            logits = self.model.get_logits(batch)
        else:
            logits = self.model(batch.to(device) if hasattr(batch, "to") else batch)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        return logits

    def temp_scale(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.clamp(self.temperature, min=1e-6)
        temperature = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    @staticmethod
    def _split_batch(batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Accepts batches in the form (data, labels) or a single object with a `.y` attribute.
        """
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            data, labels = batch
            return data, labels
        if hasattr(batch, "y"):
            return batch, batch.y
        raise ValueError("Could not infer labels from the provided batch.")

    def set_temperature(self,
                        val_loader: Iterable,
                        device: Union[str, torch.device] = "cpu",
                        verbose: bool = False):
        """
        Fit the temperature parameter using a validation loader.
        """
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = _ECELoss()
        logits_list = []
        labels_list = []

        self.to(device)
        self.model.eval()
        with torch.no_grad():
            for batch in val_loader:
                data, labels = self._split_batch(batch)
                data = data.to(device) if hasattr(data, "to") else data
                labels = labels.to(device)
                logits = self._get_logits(data, device=device)
                logits_list.append(logits)
                labels_list.append(labels.view(-1))
        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=200)

        def eval_closure():
            optimizer.zero_grad()
            loss = nll_criterion(self.temp_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_closure)

        after_temperature_nll = nll_criterion(self.temp_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temp_scale(logits), labels).item()

        if verbose:
            print(f"Optimal temperature: {self.temperature.item():.3f}")
            print(f"Before temperature - NLL: {before_temperature_nll:.3f}, ECE: {before_temperature_ece:.3f}")
            print(f"After temperature  - NLL: {after_temperature_nll:.3f}, ECE: {after_temperature_ece:.3f}")

        return self

    def predict_proba(self, batch, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
        """
        Return calibrated probabilities for a single graph or a batch of graphs.
        """
        self.eval()
        with torch.no_grad():
            logits = self._get_logits(batch, device=device)
            scaled = self.temp_scale(logits)
            return F.softmax(scaled, dim=-1)

    def predict_entropy(self, batch, device: Union[str, torch.device] = "cpu") -> float:
        """
        Predict entropy for the given batch.
        """
        probs = self.predict_proba(batch, device=device)
        return float(_entropy_from_probs(probs))


class _ECELoss(nn.Module):
    def __init__(self, n_bins: int = 15):
        super().__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


def _entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy for a batch of probability vectors.
    """
    probs = probs.clamp(min=1e-10)
    return -torch.sum(probs * probs.log(), dim=-1).mean()


# Backwards compatibility alias.
model_with_TS = TemperatureScaledModel
