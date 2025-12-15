from collections import defaultdict
from typing import Dict, List, Sequence, Tuple, Union

import igraph as ig
import torch
from torch_geometric.data import Batch, Data


def pyg_to_igraph(data_obj: Union[Data, Batch]) -> ig.Graph:
    """
    Convert a PyG Data/Batch object into an igraph Graph.
    """
    if isinstance(data_obj, Batch):
        # Expecting a single graph inside the batch for this use case.
        data_obj = data_obj.to_data_list()[0]

    num_nodes = int(data_obj.num_nodes) if getattr(data_obj, "num_nodes", None) is not None else int(data_obj.x.size(0))
    edges: List[Tuple[int, int]] = [tuple(map(int, pair)) for pair in data_obj.edge_index.t().tolist()]
    g = ig.Graph(n=num_nodes, edges=edges, directed=False)

    edge_weight = getattr(data_obj, "edge_weight", None)
    if edge_weight is None:
        # Fall back to a 1D edge_attr if provided.
        attr = getattr(data_obj, "edge_attr", None)
        if attr is not None and attr.dim() == 1:
            edge_weight = attr
    if edge_weight is not None:
        weights = torch.as_tensor(edge_weight).view(-1).cpu().tolist()
        g.es["weight"] = weights
    return g


def detect_communities(graph: Union[Data, Batch]) -> Tuple[ig.Graph, List[int]]:
    """
    Run Louvain (multilevel) community detection on the provided graph.

    Returns the igraph representation of the graph and a membership list
    mapping node index -> community id.
    """
    ig_graph = pyg_to_igraph(graph)
    weights = ig_graph.es["weight"] if "weight" in ig_graph.es.attributes() else None
    communities = ig_graph.community_multilevel(weights=weights)
    membership: List[int] = [int(m) for m in communities.membership]
    return ig_graph, membership


def aggregate_superedges(graph: ig.Graph, membership: Sequence[int]) -> Dict[Tuple[int, int], float]:
    """
    Aggregate edges between communities and sum their weights (or counts).
    """
    has_weights = "weight" in graph.es.attributes()
    weights = graph.es["weight"] if has_weights else [1.0] * graph.ecount()
    edge_weights: Dict[Tuple[int, int], float] = {}

    for (src, dst), weight in zip(graph.get_edgelist(), weights):
        c_src, c_dst = membership[src], membership[dst]
        if c_src == c_dst:
            continue
        key = (c_src, c_dst) if c_src <= c_dst else (c_dst, c_src)
        edge_weights[key] = edge_weights.get(key, 0.0) + float(weight)
    return edge_weights


def map_edges_to_communities(data_obj: Union[Data, Batch], membership: Sequence[int]) -> Dict[Tuple[int, int], List[int]]:
    """
    Map each pair of communities to the list of edge indices connecting them.
    """
    if isinstance(data_obj, Batch):
        data_obj = data_obj.to_data_list()[0]

    edge_pairs = data_obj.edge_index.t().tolist()
    mapping: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, (u, v) in enumerate(edge_pairs):
        c_u, c_v = membership[u], membership[v]
        if c_u == c_v:
            continue
        key = (c_u, c_v) if c_u <= c_v else (c_v, c_u)
        mapping[key].append(idx)
    return mapping


def community_supergraph(data_obj: Union[Data, Batch]) -> Tuple[ig.Graph, List[int], Dict[int, List[int]], Dict[Tuple[int, int], float], Dict[Tuple[int, int], List[int]]]:
    """
    Build a community-level supergraph for a given PyG graph.

    Returns:
        supergraph: igraph Graph with communities as nodes and weights on superedges.
        membership: list mapping original node -> community id.
        community_nodes: dict mapping community id -> list of member node indices.
        superedge_weights: dict of (comm_u, comm_v) -> aggregated weight/count.
        community_edge_map: dict of (comm_u, comm_v) -> list of edge indices in the original graph.
    """
    ig_graph, membership = detect_communities(data_obj)
    superedge_weights = aggregate_superedges(ig_graph, membership)

    community_nodes: Dict[int, List[int]] = defaultdict(list)
    for node_idx, comm in enumerate(membership):
        community_nodes[int(comm)].append(int(node_idx))

    supergraph = ig.Graph(n=len(set(membership)), directed=False)
    for (u, v), weight in superedge_weights.items():
        supergraph.add_edge(u, v, weight=weight)

    community_edge_map = map_edges_to_communities(data_obj, membership)
    return supergraph, membership, community_nodes, superedge_weights, community_edge_map
