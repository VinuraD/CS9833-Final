from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .config import CombinexConfig
from .explainer import Explainer
from .graph_perturber import GraphPerturber
from .utils import build_counterfactual_graph_gc, get_optimizer


class CombinedExplainer(Explainer):
    """
    Simplified COMBINEX graph-level explainer tailored for this project.
    """

    def __init__(self, cfg: CombinexConfig, datainfo) -> None:
        super().__init__(cfg=cfg, datainfo=datainfo)
        self.best_loss: float = np.inf
        self.graph_perturber: Optional[GraphPerturber] = None
        self.set_reproducibility()

    def explain(self, graph: Data, oracle):
        graph = graph.to(self.device)
        self.best_loss = np.inf
        self.graph_perturber = GraphPerturber(
            cfg=self.cfg,
            model=oracle,
            graph=graph,
            datainfo=self.datainfo,
            device=self.device,
        ).to(self.device)
        self.optimizer = get_optimizer(self.cfg, self.graph_perturber)
        best_cf_example = None

        start = time.time()
        for epoch in range(self.cfg.optimizer.num_epochs):
            new_sample = self.train(graph, oracle, epoch)

            if time.time() - start > self.cfg.timeout:
                break

            if new_sample is not None:
                best_cf_example = new_sample

        return best_cf_example

    def train(self, graph: Data, oracle, epoch: int):
        self.optimizer.zero_grad()

        differentiable_output = self.graph_perturber.forward(graph.x, graph.batch)
        model_out, V_pert, cf_edge_weights = self.graph_perturber.forward_prediction(graph.x, graph.batch)

        y_pred_actual = torch.argmax(model_out, dim=1)
        y_pred_diff = torch.argmax(differentiable_output, dim=1)
        target = graph.targets
        if target.dim() > 1:
            target = target.view(-1)
        target = target.long().to(graph.x.device)
        if target.numel() > 1:
            target = target[:1]

        edge_loss, cf_edges = self.graph_perturber.edge_loss(graph)
        node_loss, _ = self.graph_perturber.node_loss(graph)
        alpha = self.get_alpha(epoch, edge_loss, node_loss)

        mismatch = ((y_pred_actual != target) | (y_pred_diff != target)).float()
        loss_pred = F.cross_entropy(differentiable_output, target.view(1))
        loss = mismatch * loss_pred + (1 - alpha) * edge_loss + alpha * node_loss
        loss.backward()
        self.optimizer.step()

        counterfactual = None
        matches_target = torch.equal(y_pred_actual.view(-1), target.view(-1))

        if matches_target and loss.item() < self.best_loss:
            counterfactual = build_counterfactual_graph_gc(
                x=V_pert,
                edge_index=cf_edges,
                graph=graph,
                oracle=oracle,
                output_actual=model_out,
                device=self.device,
            )
            self.best_loss = loss.item()

        return counterfactual

    @property
    def name(self):
        return "CombinedExplainer"

    def get_alpha(self, epoch: int, edge_loss, node_loss) -> float:
        policy = self.cfg.scheduler.policy
        if policy == "linear":
            return max(0.0, 1.0 - epoch / self.cfg.optimizer.num_epochs)
        if policy == "exponential":
            return max(0.0, np.exp(-epoch / max(self.cfg.scheduler.decay_rate, 1e-6)))
        if policy == "sinusoidal":
            return max(0.0, 0.5 * (1 + np.cos(np.pi * epoch / self.cfg.optimizer.num_epochs)))
        if policy == "dynamic":
            return 0.0 if edge_loss > node_loss else 1.0
        return self.cfg.scheduler.initial_alpha
