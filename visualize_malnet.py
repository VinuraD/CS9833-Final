import igraph as ig
from dataset_utils import load_malnet_tiny

def plot_graph(graph, data_path):
    ig_graph = ig.Graph.Adjacency((graph.edge_index.numpy().T).tolist())
    path = 
    ig.plot(ig_graph, target=path)