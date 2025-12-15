#!/usr/bin/env python3
"""
Run Louvain community detection on MalNetTiny graphs and plot the result.
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import igraph as ig
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


# Ensure we can import dataset utilities from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dataset_utils import load_malnet_tiny  # type: ignore  # noqa: E402


def _graph_num_nodes(data_obj) -> int:
    if getattr(data_obj, 'num_nodes', None) is not None:
        return int(data_obj.num_nodes)
    if getattr(data_obj, 'x', None) is not None:
        return int(data_obj.x.size(0))
    raise ValueError('Graph object is missing node count information.')


def _resolve_label_name(dataset, label_idx: int) -> str:
    """
    Try to map the numeric label back to the original MalNet family name.
    """
    for attr in ('class_mapping', 'label_dict', 'class_dict', 'label_map'):
        mapping = getattr(dataset, attr, None)
        if isinstance(mapping, dict):
            for key, value in mapping.items():
                if isinstance(value, (int, float)) and int(value) == label_idx:
                    return str(key)
                if isinstance(key, (int, float)) and int(key) == label_idx:
                    return str(value)

    for attr in ('class_names', 'classes', 'labels'):
        names: Optional[Sequence] = getattr(dataset, attr, None)
        if names is not None and 0 <= label_idx < len(names):
            return str(names[label_idx])

    return str(label_idx)


def _pyg_to_igraph(data_obj) -> ig.Graph:
    """
    Convert a PyG Data object to an igraph Graph.
    """
    num_nodes = int(data_obj.num_nodes) if data_obj.num_nodes is not None else int(data_obj.x.size(0))
    edges: List[Tuple[int, int]] = [tuple(map(int, pair)) for pair in data_obj.edge_index.t().tolist()]
    g = ig.Graph(n=num_nodes, edges=edges, directed=False)

    edge_weight = getattr(data_obj, 'edge_weight', None)
    if edge_weight is not None:
        weights = edge_weight.view(-1).cpu().tolist()
        g.es['weight'] = weights
    return g


def _aggregate_superedges(graph: ig.Graph, membership: Sequence[int]) -> Dict[Tuple[int, int], float]:
    """
    Build a mapping of inter-community edges to their aggregated weight/count.
    """
    has_weights = 'weight' in graph.es.attributes()
    weights: Iterable[float] = graph.es['weight'] if has_weights else [1.0] * graph.ecount()
    edge_weights: Dict[Tuple[int, int], float] = {}

    for (src, dst), weight in zip(graph.get_edgelist(), weights):
        c_src, c_dst = membership[src], membership[dst]
        if c_src == c_dst:
            continue
        key = (c_src, c_dst) if c_src <= c_dst else (c_dst, c_src)
        edge_weights[key] = edge_weights.get(key, 0.0) + float(weight)
    return edge_weights


def _plot_communities(graph: ig.Graph,
                      membership: Sequence[int],
                      label_name: str,
                      split: str,
                      index: int,
                      output_path: Path) -> None:
    """
    Draw the graph with nodes colored by community membership.
    """
    num_comms = len(set(membership))
    layout = graph.layout("fr")
    coords = np.array(layout.coords, dtype=float)

    cmap = plt.cm.get_cmap('tab20', max(num_comms, 1))
    colors = [cmap(m % cmap.N) for m in membership]

    plt.figure(figsize=(10, 8))
    for src, dst in graph.get_edgelist():
        plt.plot([coords[src, 0], coords[dst, 0]],
                 [coords[src, 1], coords[dst, 1]],
                 color='#bbbbbb',
                 linewidth=0.5,
                 alpha=0.3,
                 zorder=1)

    plt.scatter(coords[:, 0],
                coords[:, 1],
                c=colors,
                s=25,
                edgecolors='k',
                linewidths=0.3,
                zorder=2)

    plt.title(f'MalNetTiny: {label_name} | split={split} | idx={index}\n'
              f'Louvain communities: {num_comms}',
              fontsize=11)
    plt.axis('off')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Louvain community detection on MalNetTiny.')
    parser.add_argument('--split',
                        choices=['train', 'val', 'test', 'all'],
                        default='train',
                        help='Which MalNetTiny split to sample from (default: train).')
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help='Random seed for reproducibility.')
    parser.add_argument('--index',
                        type=int,
                        default=42,
                        help='Optional specific graph index within the chosen split.')
    parser.add_argument('--graph_size',
                        type=int,
                        default=None,
                        help='Pick the graph whose node count is closest to this value.')
    parser.add_argument('--get_extreme',
                        action='store_true',
                        help='Plot both the smallest and largest graphs (by node count) in the split.')
    parser.add_argument('--output',
                        type=Path,
                        default=None,
                        help='Where to save the community plot (default: alongside this script).')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)

    split_candidates = ['train', 'val', 'test'] if args.split == 'all' else [args.split]
    chosen_split = rng.choice(split_candidates)
    dataset = load_malnet_tiny(split=chosen_split)

    if len(dataset) == 0:
        raise ValueError(f'Chosen split "{chosen_split}" is empty.')

    def select_graphs() -> List[Tuple[int, str, int]]:
        if args.get_extreme:
            min_idx = max_idx = 0
            min_nodes = max_nodes = _graph_num_nodes(dataset[0])
            for idx in range(1, len(dataset)):
                n_nodes = _graph_num_nodes(dataset[idx])
                if n_nodes < min_nodes:
                    min_idx, min_nodes = idx, n_nodes
                if n_nodes > max_nodes:
                    max_idx, max_nodes = idx, n_nodes
            if min_idx == max_idx:
                return [(min_idx, 'smallest=largest', min_nodes)]
            return [
                (min_idx, 'smallest', min_nodes),
                (max_idx, 'largest', max_nodes),
            ]

        if args.graph_size is not None:
            target = args.graph_size
            best_idx = 0
            best_nodes = _graph_num_nodes(dataset[0])
            best_diff = abs(best_nodes - target)
            if best_diff == 0:
                return [(best_idx, f'closest_to_{target}', best_nodes)]
            for idx in range(1, len(dataset)):
                n_nodes = _graph_num_nodes(dataset[idx])
                diff = abs(n_nodes - target)
                if diff < best_diff:
                    best_idx, best_nodes, best_diff = idx, n_nodes, diff
                    if best_diff == 0:
                        break
            return [(best_idx, f'closest_to_{target}', best_nodes)]

        graph_index = args.index if args.index is not None else rng.randrange(len(dataset))
        if not (0 <= graph_index < len(dataset)):
            raise IndexError(f'Index {graph_index} out of range for split "{chosen_split}" (len={len(dataset)}).')
        n_nodes = _graph_num_nodes(dataset[graph_index])
        return [(graph_index, 'selected', n_nodes)]

    output_base = args.output or Path(__file__).with_name('community_detection_plot.png')
    selections = select_graphs()
    multiple = len(selections) > 1

    for idx, tag, n_nodes_hint in selections:
        data_obj = dataset[idx]
        label_idx = int(data_obj.y.view(-1)[0].item())
        label_name = _resolve_label_name(dataset, label_idx)

        ig_graph = _pyg_to_igraph(data_obj)
        isolated_nodes = sum(1 for deg in ig_graph.degree() if deg == 0)
        weights = ig_graph.es['weight'] if 'weight' in ig_graph.es.attributes() else None
        communities = ig_graph.community_multilevel(weights=weights)
        membership = communities.membership

        superedge_weights = _aggregate_superedges(ig_graph, membership)
        num_communities = len(set(membership))
        comm_degree = {c: 0 for c in range(num_communities)}
        for c_u, c_v in superedge_weights.keys():
            comm_degree[c_u] += 1
            comm_degree[c_v] += 1
        isolated_comms = sum(1 for deg in comm_degree.values() if deg == 0)

        output_path = output_base
        if multiple:
            output_path = output_base.with_name(
                f'{output_base.stem}_split-{chosen_split}_idx-{idx}_n{ig_graph.vcount()}{output_base.suffix}'
            )
        _plot_communities(ig_graph, membership, label_name, chosen_split, idx, output_path)

        prefix = f'[{tag}] ' if tag else ''
        print(f'{prefix}MalNetTiny graph -> split: {chosen_split}, index: {idx}')
        print(f'Label: {label_name} (class id: {label_idx})')
        print(f'Nodes: {ig_graph.vcount()} | Edges: {ig_graph.ecount()}')
        print(f'Isolated nodes: {isolated_nodes}')
        print(f'Communities detected: {num_communities}')
        print(f'Supernodes (community graph): {num_communities}')
        print(f'Superedges (unique inter-community edges): {len(superedge_weights)}')
        print(f'Isolated communities (no inter-community edges): {isolated_comms}')

        if weights is not None and len(weights) > 0:
            weight_vals = [float(w) for w in weights]
            mean_w = sum(weight_vals) / len(weight_vals)
            print(f'Edge weights present -> min: {min(weight_vals):.3f}, '
                  f'max: {max(weight_vals):.3f}, mean: {mean_w:.3f}')
        else:
            print('Edge weights: unweighted (all edges treated as weight=1)')

        if superedge_weights:
            sorted_edges = sorted(superedge_weights.items(), key=lambda kv: kv[1], reverse=True)
            top_edges = sorted_edges[:min(5, len(sorted_edges))]
            print('Top superedge weights (community_u, community_v => weight):')
            for (c_u, c_v), w in top_edges:
                print(f'  ({c_u}, {c_v}) => {w:.3f}')

        print(f'Community plot saved to: {output_path}')
        if multiple:
            print('-' * 60)


if __name__ == '__main__':
    main()
