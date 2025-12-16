import os
import argparse
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Batch, DataLoader
from torch_geometric.utils import to_networkx, from_networkx
from models import GIN, GCN, SAG, GUNet
from typing import List
import matplotlib.pyplot as plt
import copy
from scipy.linalg import qr
import networkx as nx
from networkx.classes.function import is_path
import community
from torch_geometric.utils import to_networkx, from_networkx
from dataset_utils import (
    load_tud_dataset,
    load_dataset_splits,
    is_malnet_tiny,
    load_malnet_tiny_filtered_splits,
    MALNET_TINY_MAX_NODES,
)


def idx_correct(model,data,device):
    model.eval()
    y=data.y.item()
    output = model.predict(data,device=device)
    return output.item() == y

def _pad_adj(mat: np.ndarray, target_size: int) -> np.ndarray:
    if mat.shape[0] == target_size:
        return mat
    pad = target_size - mat.shape[0]
    return np.pad(mat, ((0, pad), (0, pad)), mode='constant')


def distance(x_adv, x):
    adj_adv = nx.adjacency_matrix(to_networkx(x_adv, to_undirected=True)).toarray()
    adj_x = nx.adjacency_matrix(to_networkx(x, to_undirected=True)).toarray()
    target_size = max(adj_adv.shape[0], adj_x.shape[0])
    adj_adv = _pad_adj(adj_adv, target_size)
    adj_x = _pad_adj(adj_x, target_size)
    return np.sum(np.abs(adj_adv - adj_x)) / 2

def validity(y_pred,y_target):
    return int(y_pred==y_target)

def _ensure_single_graph(data):
    """
    find_coarse_perturb expects to work on a single graph. If a Batch is passed
    (e.g., directly from a DataLoader), unwrap the first graph to avoid slicing
    mismatches when the edge_index is modified.
    """
    if isinstance(data, Batch):
        if data.num_graphs < 1:
            raise ValueError("Received an empty batch for perturbation search.")
        data = data.to_data_list()[0]
    return data

def new_graph(G, perturb, index1=None, index2=None):
    #generate a perturbed graph
    G_new = copy.deepcopy(G)
    if index2 == None:
        if index1 == None:  #change the edges in the whole graph
            num_nodes = len(G_new)
            for i in range(num_nodes-1):
                for j in range(i+1, num_nodes):
                    if perturb[i, j] > 0.5:  #we will change this link
                        if is_path(G_new, [i, j]):
                            G_new.remove_edge(i, j)
                        else:
                            G_new.add_edge(i, j)
        else:  #change the edges in one cluster
            num_nodes = len(index1)
            for i in range(num_nodes-1):
                for j in range(i+1, num_nodes):
                    if perturb[i, j] > 0.5:  #we will change this link
                        if is_path(G_new, [index1[i], index1[j]]):
                            G_new.remove_edge(index1[i], index1[j])
                        else:
                            G_new.add_edge(index1[i], index1[j])
    else:  #change the edges between clusters
        num_nodes1, num_nodes2 = len(index1), len(index2)
        for i in range(num_nodes1):
            for j in range(num_nodes2):
                if perturb[i, j] > 0.5:
                    if is_path(G_new, [index1[i], index2[j]]):
                        G_new.remove_edge(index1[i], index2[j])
                    else:
                        G_new.add_edge(index1[i], index2[j])
    return G_new 

def find_coarse_perturb(model,x0, y0,device):
        model.eval()
        x0 = _ensure_single_graph(x0).to(device)
        num_query = 0
        num_nodes = x0.num_nodes
        G = to_networkx(x0, to_undirected=True) #tansfer from PyG data to networkx
        partition = community.best_partition(G) #decompose G into clusters
        num_cluster = len(list(set(partition.values())))
        cluster = {}
        for i in range(num_cluster):
            cluster[i] = list(np.where(np.array(list(partition.values())) == i)[0])

        g_theta = float('inf')  #initial g_theta
        g_theta = torch.FloatTensor([g_theta]).to(device)
        F_theta = float('inf')  #initial F_theta
        F_theta = torch.FloatTensor([F_theta]).to(device)
        x_new = copy.deepcopy(x0).to(device)
        flag_inner, flag_outer = 0, 0
        
        #final_theta = torch.zeros((num_nodes, num_nodes)).to(device)
        final_theta = torch.normal(mean=0.5,std=0.1,size=(num_nodes,num_nodes)).to(device)
        final_theta = torch.clamp(final_theta, 0.0, 0.5)
        search_type = -1       
        
        #inner cluster perturbation
        for i in range(num_cluster):
            nodes = cluster[i]
            num_cluster_nodes = len(nodes)
            if num_cluster_nodes > 1:
                for j in range(15*num_cluster_nodes): #search initial directions 
                    theta = torch.normal(mean=torch.rand(1).item(),std=0.5,size=(num_cluster_nodes,num_cluster_nodes)).to(device)
                    theta = torch.triu(theta, diagonal=1).to(device)
                    G_new = new_graph(G, theta, index1=nodes)
                    x_new.edge_index = from_networkx(G_new).to(device).edge_index.long()
                    if model.predict(x_new, device=device) != y0:  #we find a direction
                        F_lbd = distance(x_new, x0)
                        if F_lbd < F_theta:
                            F_theta = F_lbd
                            flag_inner = 1
                            search_type = 0
                            for p in range(num_cluster_nodes-1):
                                for q in range(p+1, num_cluster_nodes):
                                    final_theta[nodes[p], nodes[q]] = theta[p, q]
                                    final_theta[nodes[q], nodes[p]] = theta[p, q]   
                    num_query += 1   
        
        ##perturbations between clusters
        if (num_cluster > 1) and (flag_inner == 0):
            for i in range(num_cluster - 1):
                for j in range(i+1, num_cluster):
                    nodes1, nodes2 = cluster[i], cluster[j]
                    num_cluster_nodes1, num_cluster_nodes2 = len(nodes1), len(nodes2)
                    for k in range(15*(num_cluster_nodes1+num_cluster_nodes2)):
                        theta = torch.normal(mean=torch.rand(1).item(), std=0.5, size=(num_cluster_nodes1,num_cluster_nodes2)).to(device)
                        G_new = new_graph(G, theta, nodes1, nodes2)
                        x_new.edge_index = from_networkx(G_new).to(device).edge_index.long()
                        if model.predict(x_new, device=device) != y0:
                            F_lbd = distance(x_new, x0)
                            if F_lbd < F_theta:  
                                F_theta = F_lbd
                                flag_outer = 1
                                search_type = 1
                                for p in range(num_cluster_nodes1):
                                    for q in range(num_cluster_nodes2):
                                        final_theta[nodes1[p], nodes2[q]] = theta[p, q]     
                                        final_theta[nodes2[q], nodes1[p]] = theta[p, q]     
                        num_query += 1   
        
        #perturbations on the whole graph
        if (flag_inner == 0) and (flag_outer == 0):
            for k in range(15*num_nodes):
                theta = torch.normal(mean=torch.rand(1).item(), std=0.5, size=(num_nodes,num_nodes)).to(device)
                theta = torch.triu(theta, diagonal=1).to(device)
                G_new = new_graph(G, theta)
                x_new.edge_index = from_networkx(G_new).to(device).edge_index.long()
                if model.predict(x_new, device=device) != y0:
                    F_lbd = distance(x_new, x0)
                    if F_lbd < F_theta:
                        search_type = 2
                        F_theta = F_lbd
                        #g_theta = lbd
                        final_theta = theta    
                num_query += 1
        
        if F_theta.item() == float('inf'):  #can not find an initial direction
            return final_theta, F_theta, g_theta, num_query, search_type
        else:  #find initial direction
            final_theta = torch.triu(final_theta, diagonal=1).to(device)
            init_lbd_whole = torch.norm(final_theta)
            final_theta_norm = torch.div(final_theta, init_lbd_whole)
            g_theta_whole, c = fine_grained_binary_search(
                model,
                x0,
                y0,
                final_theta_norm,
                init_lbd_whole,
                torch.FloatTensor([float('inf')]).to(device),
                device=device,
            )
            F_theta_whole = L1_norm(g_theta_whole*final_theta_norm, device)
            return final_theta_norm, F_theta_whole, g_theta_whole, num_query + c, search_type
        
def fine_grained_binary_search(model, x0, y0, theta, initial_lbd,current_best, index1=None, index2=None,device='cuda'):
        #theta:  torch(N,N)
        #initial_lbd: torch([1])
        #current_best: torch([1])
        
        x0 = _ensure_single_graph(x0).to(device)
        x_new = copy.deepcopy(x0)
        G0 = to_networkx(x0, to_undirected=True)
        nquery = 0
        
        if current_best < initial_lbd:
            G_new  = new_graph(G0, torch.clamp(current_best*theta,0.0,1.0), index1, index2)
            x_new.edge_index = from_networkx(G_new).to(device).edge_index.long()
            if model.predict(x_new, device=device) == y0:
                nquery += 1
                return torch.FloatTensor([float('inf')]).to(device), nquery
            lbd = current_best.to(device)
        else:
            lbd = initial_lbd.to(device)
        
        lbd_hi = lbd
        lbd_lo = torch.FloatTensor([0.0]).to(device)

        while lbd_hi-lbd_lo > 1e-2: 
            lbd_mid = (lbd_lo+lbd_hi)/2.0
            nquery += 1
            G_new = new_graph(G0, torch.clamp(lbd_mid * theta, 0.0, 1.0), index1, index2)
            x_new.edge_index = from_networkx(G_new).to(device).edge_index.long()
            if model.predict(x_new, device=device) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        g_theta = lbd_hi
        
        return g_theta, nquery

def L0_norm(theta, device):
    theta[theta==float('inf')]=1
    theta[theta==float('-inf')]=1
    theta = torch.triu(theta, diagonal=1).to(device)  
    theta = torch.where(theta>0.5, torch.FloatTensor([1]).to(device), torch.FloatTensor([0]).to(device))
    return torch.sum(theta)

def L1_norm(theta, device):
    theta[theta==float('inf')]=1
    theta[theta==float('-inf')]=1
    theta = torch.triu(theta-0.5, diagonal=1).to(device)
    theta = torch.clamp(theta, 0.0, 1.0)
    return torch.sum(theta)

def count_edges(x_adv, x):
    adj_adv = nx.adjacency_matrix(to_networkx(x_adv, to_undirected=True)).toarray()
    adj_x = nx.adjacency_matrix(to_networkx(x, to_undirected=True)).toarray()
    target_size = max(adj_adv.shape[0], adj_x.shape[0])
    adj_adv = _pad_adj(adj_adv, target_size)
    adj_x = _pad_adj(adj_x, target_size)
    difference = adj_adv - adj_x
    num_add = sum(sum(difference==1)) / 2
    num_delete = sum(sum(difference==-1)) / 2
    return num_add, num_delete


def compute_binary_metrics(model, loader, device):
    """
    Compute accuracy, TPR, and FPR treating label '1' as the positive class.
    Returns dict with accuracy, tpr, fpr, and confusion counts.
    """
    model.eval()
    tp = fp = tn = fn = 0
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            preds = logits.max(dim=1)[1]
            labels = batch.y.view(-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            for pred, label in zip(preds, labels):
                pred_val = int(pred.item())
                label_val = int(label.item())
                if label_val == 1:
                    if pred_val == 1:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if pred_val == 1:
                        fp += 1
                    else:
                        tn += 1
    accuracy = correct / total if total > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        'accuracy': accuracy,
        'tpr': tpr,
        'fpr': fpr,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }

def _load_dataset(dataset_name: str):
    return load_tud_dataset(dataset_name)


def _build_model(model_name: str, dataset_name: str, dataset, device: torch.device, hidden_dim: int = 64, dropout: float = 0.5):
    input_dim = dataset.num_node_features
    output_dim = dataset.num_classes
    gin_layers = 6 if is_malnet_tiny(dataset_name) else 5
    if model_name.upper() == 'GIN':
        model = GIN(gin_layers, 2, input_dim, hidden_dim, output_dim, dropout).to(device)
    elif model_name.upper() == 'GCN':
        model = GCN(5, input_dim, hidden_dim, output_dim, dropout).to(device)
    elif model_name.upper() == 'SAG':
        model = SAG(5, input_dim, hidden_dim, output_dim, 0.8, dropout).to(device)
    elif model_name.upper() == 'GUNET':
        model = GUNet(input_dim, hidden_dim, output_dim, 0.8, 3, dropout).to(device)
    else:
        raise ValueError(f'Unsupported model name {model_name}')
    return model


def _load_model(model_name: str,
                dataset_name: str,
                dataset,
                device: torch.device,
                hidden_dim: int = 64,
                dropout: float = 0.5,
                weights_path: str = None):
    model = _build_model(model_name, dataset_name, dataset, device, hidden_dim=hidden_dim, dropout=dropout)

    candidates = []
    if weights_path is not None:
        candidates.append(weights_path)
    else:
        candidates.extend([
            f'./trained_model/{dataset_name}_{model_name}.pt',
            f'./trained_model/{dataset_name.upper()}_{model_name}.pt',
            f'./trained_model/{dataset_name}_{model_name.upper()}.pt'
        ])
    weight_path = None
    for path in candidates:
        if path is not None and os.path.exists(path):
            weight_path = path
            break
    if weight_path is None:
        if weights_path is not None:
            raise FileNotFoundError(f'Specified weights_path not found: {weights_path}')
        raise FileNotFoundError(f'Model weights not found for {dataset_name}_{model_name}')
    if weights_path is not None and weight_path != weights_path:
        raise FileNotFoundError(f'Specified weights_path not found: {weights_path}')
    print(f'Loading weights from: {weight_path}')
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _load_split_dataset(dataset_name: str):
    if is_malnet_tiny(dataset_name):
        dataset, splits = load_malnet_tiny_filtered_splits(max_nodes=MALNET_TINY_MAX_NODES, seed=42)
    else:
        dataset = load_tud_dataset(dataset_name)
        splits = load_dataset_splits(dataset_name, len(dataset))
    train_dataset = dataset[splits['train']]
    val_dataset = dataset[splits['val']]
    test_dataset = dataset[splits['test']]
    return train_dataset, val_dataset, test_dataset, dataset


def eval_model_on_attack(model_name: str,
                         dataset_name: str,
                         defense_folder: str,
                         device: str = 'cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    dataset = _load_dataset(dataset_name)
    model = _load_model(model_name, dataset_name, dataset, device)

    defense_folder = defense_folder.lower()
    if defense_folder == 'defense3':
        prefix = f'{dataset_name}_{model_name}_Rand_'
    else:
        prefix = f'{dataset_name}_{model_name}_Our_'

    base_dir = os.path.join('./attacks', defense_folder)
    normal_path = os.path.join(base_dir, prefix + 'test_normal.pt')
    advers_path = os.path.join(base_dir, prefix + 'test_advers.pt')

    if not os.path.exists(normal_path) or not os.path.exists(advers_path):
        raise FileNotFoundError(f'Could not find saved attack data under {base_dir} with prefix {prefix}')

    normal_graphs = torch.load(normal_path)
    advers_graphs = torch.load(advers_path)

    def _compute_accuracy(graphs):
        if len(graphs) == 0:
            return float('nan')
        correct = 0
        for g in graphs:
            g = g.to(device)
            pred = model.predict(g, device=device)
            label = g.y.view(-1)[0]
            if pred.item() == label.item():
                correct += 1
        return correct / len(graphs)

    normal_acc = _compute_accuracy(normal_graphs)
    advers_acc = _compute_accuracy(advers_graphs)
    print(f'Accuracy on {defense_folder} normal set: {normal_acc:.4f}')
    print(f'Accuracy on {defense_folder} adversarial set: {advers_acc:.4f}')
    return {'normal_accuracy': normal_acc, 'adversarial_accuracy': advers_acc}


def evaluate_saved_model(weights_path: str,
                         dataset_name: str,
                         model_name: str,
                         device: str = 'cuda',
                         batch_size: int = 32):
    """
    Evaluate a saved model checkpoint on the dataset test split, reporting accuracy, TPR, and FPR.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    _, _, test_dataset, dataset_for_stats = _load_split_dataset(dataset_name)
    model = _load_model(model_name, dataset_name, dataset_for_stats, device, weights_path=weights_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    metrics = compute_binary_metrics(model, test_loader, device)
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"TPR: {metrics['tpr']:.4f}, FPR: {metrics['fpr']:.4f}")
    print(f"Counts - TP: {metrics['tp']}, FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']}")
    return metrics

def plot_uq(p_vals:List,set_sizes:List=None):
    plt.figure()
    plt.hist(p_vals, bins=5, color='blue', alpha=0.7)
    plt.title('Histogram of p-values for Uncertainty Quantification')
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    plt.savefig('uq_pvalue_histogram.png')

    if set_sizes is not None:   
        plt.figure()
        plt.hist(set_sizes, bins=range(1, max(set_sizes)+2), color='green', alpha=0.7, align='left')
        plt.title('Histogram of Prediction Set Sizes')
        plt.xlabel('Set Size')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.show()
        plt.savefig('uq_setsize_histogram.png')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model accuracy on saved attack datasets.')
    parser.add_argument('--model', type=str, default='GIN', help='Model name (GIN, GCN, SAG, GUNet)')
    parser.add_argument('--dataset', type=str, default='NCI1', help='Dataset name')
    parser.add_argument('--defense', type=str, default='defense1', help='Defense folder to evaluate (defense1/2/3)')
    parser.add_argument('--device', type=str, default='cuda', help='Device for evaluation')
    parser.add_argument('--mode', type=str, choices=['attack', 'model'], default='attack', help='Evaluation mode: attack sets or a saved model on the test split')
    parser.add_argument('--weights', type=str, default=None, help='Optional path to a saved model checkpoint for --mode model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for saved model evaluation')
    args = parser.parse_args()
    if args.mode == 'model' or args.weights is not None:
        evaluate_saved_model(args.weights, args.dataset, args.model, device=args.device, batch_size=args.batch_size)
    else:
        eval_model_on_attack(args.model, args.dataset, args.defense, device=args.device)
