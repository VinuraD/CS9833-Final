"""
Level-1 genetic algorithm for graph counterfactuals via edge addition/removal.

Pipeline:
    1) Calibrate a GNN with temperature scaling.
    2) Remove isolated nodes in the input graph.
    3) Run community detection and operate on the resulting supergraph.
    4) Use a basic genetic algorithm over superedges to propose edge edits that
       increase predictive entropy and push the model toward a target class.

Level-2 refinement is intentionally left unimplemented for now.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.utils import subgraph

from .TS_calibration import TemperatureScaledModel
from .community_detection import community_supergraph


@dataclass
class Gene:
    pair: Tuple[int, int]
    action_type: str  # "remove" or "add"
    weight: float = 1.0


def _entropy_from_probs(probs: Tensor) -> float:
    probs = probs.clamp(min=1e-10)
    return float(-(probs * probs.log()).sum())


def _ensure_batch(graph: Union[Data, Batch]) -> Batch:
    if isinstance(graph, Batch):
        return graph
    if hasattr(graph, "batch"):
        return graph if isinstance(graph, Batch) else Batch.from_data_list([graph])
    return Batch.from_data_list([graph])


def remove_isolated_nodes(graph: Union[Data, Batch]) -> Data:
    """
    Remove nodes with zero degree from a PyG graph.
    """
    if isinstance(graph, Batch):
        # Use the first graph in the batch for simplicity.
        graph = graph.to_data_list()[0]

    num_nodes = int(graph.num_nodes) if getattr(graph, "num_nodes", None) is not None else int(graph.x.size(0))
    edge_index = graph.edge_index
    device = edge_index.device

    deg = torch.zeros(num_nodes, device=device, dtype=torch.long)
    deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=device, dtype=torch.long))
    deg.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=device, dtype=torch.long))
    keep_mask = deg > 0

    if bool(keep_mask.all()):
        return graph

    filtered_edge_index, filtered_edge_attr = subgraph(
        keep_mask,
        edge_index,
        edge_attr=getattr(graph, "edge_attr", None),
        relabel_nodes=True,
    )

    new_graph = Data(
        x=graph.x[keep_mask],
        edge_index=filtered_edge_index,
    )
    if filtered_edge_attr is not None:
        new_graph.edge_attr = filtered_edge_attr
    for attr_name in ("y", "edge_weight"):
        if hasattr(graph, attr_name):
            setattr(new_graph, attr_name, getattr(graph, attr_name))
    return new_graph


def calibrate_model(model: torch.nn.Module,
                    val_loader=None,
                    device: Union[str, torch.device] = "cpu",
                    verbose: bool = False) -> TemperatureScaledModel:
    """
    Wrap the model with temperature scaling. If a validation loader is provided,
    the temperature parameter is fitted before returning.
    """
    calibrated = TemperatureScaledModel(model).to(device)
    if val_loader is not None:
        calibrated.set_temperature(val_loader, device=device, verbose=verbose)
    return calibrated


def _build_candidate_genes(supergraph, superedge_weights: Dict[Tuple[int, int], float], max_new_edges: int = 20) -> List[Gene]:
    """
    Construct gene list from the community supergraph. Existing superedges
    become "remove" genes; a limited set of missing pairs become "add" genes.
    """
    isolated = {v.index for v in supergraph.vs if supergraph.degree(v.index) == 0}
    candidates: List[Gene] = []

    for (u, v), weight in superedge_weights.items():
        if u in isolated or v in isolated:
            continue
        candidates.append(Gene(pair=(u, v), action_type="remove", weight=weight))

    existing = {tuple(sorted(edge.tuple)) for edge in supergraph.es}
    non_isolated_nodes = [v.index for v in supergraph.vs if v.index not in isolated]
    avg_weight = sum(superedge_weights.values()) / max(len(superedge_weights), 1)
    added = 0
    for i, u in enumerate(non_isolated_nodes):
        if added >= max_new_edges:
            break
        for v in non_isolated_nodes[i + 1:]:
            if tuple(sorted((u, v))) in existing:
                continue
            candidates.append(Gene(pair=(u, v), action_type="add", weight=avg_weight or 1.0))
            added += 1
            if added >= max_new_edges:
                break
    return candidates


def _apply_actions(
    graph: Data,
    genes: List[Gene],
    actions: List[int],
    community_nodes: Dict[int, List[int]],
    community_edge_map: Dict[Tuple[int, int], List[int]],
) -> Data:
    """
    Apply gene actions to the graph to produce a perturbed counterfactual graph.
    """
    edge_index = graph.edge_index
    edge_attr = getattr(graph, "edge_attr", None)
    edge_weight = getattr(graph, "edge_weight", None)

    keep_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
    for gene, action in zip(genes, actions):
        if gene.action_type == "remove" and action == -1:
            for e_idx in community_edge_map.get(gene.pair, []):
                keep_mask[e_idx] = False

    new_edge_index = edge_index[:, keep_mask]
    new_edge_attr = edge_attr[keep_mask] if edge_attr is not None else None
    new_edge_weight = edge_weight[keep_mask] if edge_weight is not None else None

    # Prepare a set of existing undirected edges to avoid duplicate additions.
    existing_pairs = {tuple(sorted(pair)) for pair in new_edge_index.t().tolist()}
    added_edges: List[List[int]] = []
    added_attr: List[Tensor] = []
    added_weight: List[Tensor] = []

    for gene, action in zip(genes, actions):
        if gene.action_type != "add" or action != 1:
            continue
        u_nodes = community_nodes.get(gene.pair[0], [])
        v_nodes = community_nodes.get(gene.pair[1], [])
        if not u_nodes or not v_nodes:
            continue
        src = u_nodes[0]
        dst = v_nodes[0] if v_nodes[0] != src else (v_nodes[1] if len(v_nodes) > 1 else None)
        if dst is None:
            continue
        if tuple(sorted((src, dst))) in existing_pairs:
            continue
        added_edges.extend([[src, dst], [dst, src]])
        existing_pairs.add(tuple(sorted((src, dst))))
        if new_edge_attr is not None:
            fill_val = torch.tensor(gene.weight, device=new_edge_attr.device, dtype=new_edge_attr.dtype)
            added_attr.extend([fill_val, fill_val])
        if new_edge_weight is not None:
            fill_w = torch.tensor(gene.weight, device=new_edge_weight.device, dtype=new_edge_weight.dtype)
            added_weight.extend([fill_w, fill_w])

    if added_edges:
        added_tensor = torch.tensor(added_edges, dtype=new_edge_index.dtype, device=new_edge_index.device).t()
        new_edge_index = torch.cat([new_edge_index, added_tensor], dim=1)
        if new_edge_attr is not None and added_attr:
            new_edge_attr = torch.cat([new_edge_attr, torch.stack(added_attr)], dim=0)
        if new_edge_weight is not None and added_weight:
            new_edge_weight = torch.cat([new_edge_weight, torch.stack(added_weight)], dim=0)

    cf_graph = Data(x=graph.x, edge_index=new_edge_index, y=getattr(graph, "y", None))
    if new_edge_attr is not None:
        cf_graph.edge_attr = new_edge_attr
    if new_edge_weight is not None:
        cf_graph.edge_weight = new_edge_weight
    return cf_graph


def _mutate(chromosome: List[int], genes: List[Gene], mutation_rate: float) -> List[int]:
    mutated = chromosome.copy()
    for idx, gene in enumerate(genes):
        if random.random() < mutation_rate:
            if gene.action_type == "remove":
                mutated[idx] = -1 if mutated[idx] == 0 else 0
            else:
                mutated[idx] = 1 if mutated[idx] == 0 else 0
    return mutated


def _crossover(a: List[int], b: List[int], crossover_rate: float) -> List[int]:
    assert len(a) == len(b)
    child = a.copy()
    for i in range(len(a)):
        if random.random() < crossover_rate:
            child[i] = b[i]
    return child


def genetic_counterfactual_level_one(
    graph: Data,
    model: torch.nn.Module,
    val_loader=None,
    target_label: Optional[int] = None,
    population_size: int = 20,
    generations: int = 30,
    mutation_rate: float = 0.25,
    crossover_rate: float = 0.5,
    max_new_edges: int = 20,
    entropy_weight: float = 1.0,
    delta_weight: float = 1.0,
    label_bonus: float = 0.5,
    device: Union[str, torch.device] = "cpu",
    seed: Optional[int] = None,
) -> Dict[str, Union[Data, List[int], float, Tensor]]:
    """
    Run the level-1 genetic search to find an initial edge-perturbation direction.

    Returns a dictionary containing the best counterfactual graph found, the
    gene actions that produced it, and diagnostics (fitness, entropy, deltas).
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    graph = remove_isolated_nodes(graph)
    calibrated = calibrate_model(model, val_loader=val_loader, device=device, verbose=False)
    supergraph, membership, community_nodes, superedge_weights, community_edge_map = community_supergraph(graph)
    genes = _build_candidate_genes(supergraph, superedge_weights, max_new_edges=max_new_edges)
    if not genes:
        raise RuntimeError("No candidate genes available after community pruning.")

    base_batch = _ensure_batch(graph).to(device)
    with torch.no_grad():
        base_probs = calibrated.predict_proba(base_batch, device=device)[0]
    base_label = int(base_probs.argmax().item())

    population: List[List[int]] = [[0 for _ in genes] for _ in range(population_size)]
    best_result = {"fitness": float("-inf")}

    for _ in range(generations):
        scored = []
        for chromosome in population:
            cf_graph = _apply_actions(graph, genes, chromosome, community_nodes, community_edge_map)
            cf_batch = _ensure_batch(cf_graph).to(device)
            with torch.no_grad():
                probs = calibrated.predict_proba(cf_batch, device=device)[0]
            entropy = _entropy_from_probs(probs)
            delta = torch.norm(probs - base_probs, p=2).item()
            pred_label = int(probs.argmax().item())
            flip_reward = label_bonus if (target_label is not None and pred_label == target_label) or (target_label is None and pred_label != base_label) else 0.0
            fitness = entropy_weight * entropy + delta_weight * delta + flip_reward
            scored.append((fitness, chromosome, cf_graph, entropy, delta, probs))

        scored.sort(key=lambda tup: tup[0], reverse=True)
        top_fitness, top_actions, top_graph, top_entropy, top_delta, top_probs = scored[0]
        if top_fitness > best_result.get("fitness", float("-inf")):
            best_result = {
                "fitness": top_fitness,
                "actions": top_actions,
                "counterfactual": top_graph,
                "entropy": top_entropy,
                "delta": top_delta,
                "probs": top_probs,
            }

        # Elitist selection with uniform crossover/mutation.
        parents = [entry[1] for entry in scored[: max(2, population_size // 2)]]
        new_population = [top_actions]
        while len(new_population) < population_size:
            p1, p2 = random.sample(parents, 2)
            child = _crossover(p1, p2, crossover_rate)
            child = _mutate(child, genes, mutation_rate)
            new_population.append(child)
        population = new_population

    best_result["base_probs"] = base_probs
    best_result["base_label"] = base_label
    best_result["genes"] = genes
    return best_result
