
'''
Credit to: https://github.com/Anonymous-21/CCAS21_GNNattack
'''
import os
import torch
import torch_geometric #torch_geometric == 1.6.1
import community
import numpy as np
import networkx as nx
import argparse
import random
import copy
from torch_geometric.data import Batch, Data
from attacks.BB_attack.Sign_OPT import *
from attacks.PR_attack.pr_att import *
from attacks.R_attack.rand_att import random_attack_sample
from torch_geometric.utils import to_networkx
from models import GIN, SAG, GUNet
from time import time
from dataset_utils import (
    load_tud_dataset,
    load_dataset_splits,
    is_malnet_tiny,
    load_malnet_tiny_filtered_splits,
    MALNET_TINY_MAX_NODES,
)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='attack graph classification models')

parser.add_argument('--attack', type=str, default='BB', help='Type of attack: BB for black-box attack, WB for white-box attack, RAND for random attack (default: BB)')
parser.add_argument('--cf_pct', type=float, default=0, help='Percentage of edges to perturb for cf generation (default: 0.25)')
parser.add_argument('--cf_type', type=str, default='cf_gnn', help='Type of counterfactual generation method (default: cf_gnn)')
#these are parameters for attack model (Black box attack)
parser.add_argument('--effective', type=int, default=1)
parser.add_argument('--max_query', type=int, default=20000)
parser.add_argument('--id', type= int, default=1)
parser.add_argument('--search', type=int, default=1)
parser.add_argument('--rand_trials', type=int, default=500, help='Number of random perturbations to sample for RAND attack')
parser.add_argument('--rand_budget', type=float, default=0.1, help='Maximum perturbation budget for RAND attack (absolute if >1 else ratio of possible edges)')
parser.add_argument('--targeted', action='store_true', help='Whether to perform targeted attack (default: False)')
#these are parameters for white box attack model (PR-attack)    
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--mlp_hidden', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--k', type=int, default=2)
#parameters for BayesOpt attack
parser.add_argument('--bo_batch_size', type=int, default=5, help='Batch size for BayesOpt proposals (default: 5)')
parser.add_argument('--bo_n_init', type=int, default=5, help='Number of initial random proposals for BayesOpt (default: 5)')
parser.add_argument('--bo_budget', type=float, default=0.03, help='Edit budget for BayesOpt attack (float as ratio, >=1 as absolute) (default: 0.03)')
parser.add_argument('--bo_budget_by', type=str, choices=['nnodes_sq', 'nedges'], default='nnodes_sq', help='Budget scaling method for BayesOpt (default: nnodes_sq)')
parser.add_argument('--bo_query_per_perturb', type=int, default=40, help='Queries allowed per edge edit for BayesOpt (default: 40)')
parser.add_argument('--bo_patience', type=int, default=200, help='Early-stop patience for BayesOpt when loss stagnates (default: 200)')
parser.add_argument('--bo_max_h', type=int, default=0, help='Weisfeiler-Lehman height for BO surrogate (default: 0)')
parser.add_argument('--bo_mode', type=str, default='flip', choices=['flip', 'rewire'], help='Edit operation type for BayesOpt attack (default: flip)')
parser.add_argument('--bo_no_greedy', action='store_true', help='Use less greedy per-stage edit allocation for BayesOpt')
parser.add_argument('--bo_acq', type=str, default='mutation', choices=['mutation', 'random'], help='Acquisition optimiser for BayesOpt (default: mutation)')
parser.add_argument('--bo_constrain_n_hop', type=int, default=None, help='Constrain BayesOpt edits within n-hop neighbourhood (default: None)')
parser.add_argument('--bo_preserve_components', action='store_true', help='Preserve connected components during BayesOpt attack')
parser.add_argument('--bo_surrogate', type=str, default='bayeslinregress', choices=['gpwl', 'bayeslinregress', 'null'], help='Surrogate choice for BayesOpt attack (default: bayeslinregress)')
parser.add_argument('--rewatt_action_pct', type=float, default=0.7, help='Fraction of edges to consider for ReWatt rewiring steps (default: 0.03)')
parser.add_argument('--rewatt_max_actions', type=int, default=None, help='Hard cap on rewiring steps for ReWatt attack (default: None)')
parser.add_argument('--rewatt_negative_reward', type=float, default=-0.6, help='Negative reward to use when ReWatt fails to flip prediction (default: -0.5)')
parser.add_argument('--rewatt_include_self', type=int, default=1, help='Allow rewiring back to the source node in ReWatt (1) or force different nodes (0) (default: 1)')
parser.add_argument('--rewatt_use_all', action='store_true', help='Use all nodes as ReWatt candidates instead of two-hop neighbourhood')

#these are general parameters
parser.add_argument('--dataset', type=str, default="NCI1")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32, help='social dataset:64 bio dataset:32')
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--model_path', type=str, default='./trained_model/')
parser.add_argument('--model', type=str, default='GIN')
parser.add_argument('--use_pairs', action='store_true', help='Load models trained on counterfactual/original pairs (cf_lm dir)')
args = parser.parse_args()

set_seed(42)

#from BB attack
def distance(x_adv, x):
    adj_adv = nx.adjacency_matrix(to_networkx(x_adv, to_undirected=True))
    adj_x = nx.adjacency_matrix(to_networkx(x, to_undirected=True))
    return np.sum(np.abs(adj_adv-adj_x)) / 2
    
def count_edges(x_adv, x):
    adj_adv = nx.adjacency_matrix(to_networkx(x_adv, to_undirected=True)).A #todense
    adj_x = nx.adjacency_matrix(to_networkx(x, to_undirected=True)).A #todense
    difference = adj_adv - adj_x
    num_add = sum(sum(difference==1)) / 2
    num_delete = sum(sum(difference==-1)) / 2
    return num_add, num_delete

def check_class(x):
    if x.y[0]==1:
        return True
    else:
        return False

def _bidirectional_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    edges = edge_index.t().cpu()
    reversed_edges = edges[:, [1, 0]]
    all_edges = torch.cat([edges, reversed_edges], dim=0)
    unique_edges = torch.unique(all_edges, dim=0)
    return unique_edges.t()

def pyg_to_dgl_graph(data: Data):
    import dgl
    edge_index = data.edge_index.cpu() if data.edge_index is not None else torch.empty((2, 0), dtype=torch.long)
    bidir_edge_index = _bidirectional_edge_index(edge_index, data.num_nodes)
    graph = dgl.graph((bidir_edge_index[0], bidir_edge_index[1]), num_nodes=data.num_nodes)
    node_attr = data.x
    if node_attr is None:
        node_attr = torch.ones((data.num_nodes, 1), dtype=torch.float32)
    graph.ndata['node_attr'] = node_attr.cpu().float()
    return graph

def dgl_to_pyg_graph(graph, y: torch.Tensor = None) -> Data:
    import dgl
    src, dst = graph.edges()
    edge_index = torch.stack([src, dst], dim=0)
    node_attr = graph.ndata['node_attr'] if 'node_attr' in graph.ndata else graph.ndata.get('feat')
    if node_attr is None:
        node_attr = torch.ones((graph.num_nodes(), 1), dtype=torch.float32)
    data = Data(x=node_attr, edge_index=edge_index)
    if y is not None:
        data.y = y
    return data

class DGLModelWrapper:
    def __init__(self, pyg_model, device):
        self.pyg_model = pyg_model
        self.device = device

    def __call__(self, graphs):
        import dgl
        dgl_graph_types = tuple(
            t for t in (getattr(dgl, 'DGLGraph', None), getattr(dgl, 'DGLHeteroGraph', None)) if t is not None
        )
        if dgl_graph_types and isinstance(graphs, dgl_graph_types):
            try:
                graph_list = dgl.unbatch(graphs)
            except Exception:
                # dgl.unbatch fails on non-batched graphs; treat it as a singleton in that case.
                graph_list = [graphs]
        elif isinstance(graphs, (list, tuple)):
            graph_list = list(graphs)
        else:
            raise TypeError(f'Unsupported graph container type for BO attack: {type(graphs)}')
        pyg_graphs = [dgl_to_pyg_graph(g) for g in graph_list]
        batch = Batch.from_data_list(pyg_graphs).to(self.device)
        self.pyg_model.eval()
        with torch.no_grad():
            logits = self.pyg_model(batch)
        return logits

def compute_bo_budget(graph, budget, budget_by, query_per_perturb, mode):
    budget = float(budget)
    query_per_perturb = max(1, int(query_per_perturb))
    n_nodes = int(graph.num_nodes())
    n_edges = int(graph.num_edges())
    if budget >= 1:
        return int(min(budget, 1)), query_per_perturb
    if mode == 'rewire':
        qpp = query_per_perturb * 2
        if budget_by == 'nnodes_sq':
            edit = 1 + min(int(2e4 / qpp), int(np.round(budget * n_nodes ** 2 // 2)) // 2)
        else:
            edit = 1 + min(int(2e4 / qpp), int(np.round(budget * n_edges // 2 // 2)) // 2)
    else:
        qpp = query_per_perturb
        if budget_by == 'nnodes_sq':
            edit = 1 + min(int(2e4 / qpp), int(np.round(budget * n_nodes ** 2)))
        else:
            edit = 1 + min(int(2e4 / qpp), int(np.round(budget * n_edges // 2)))
    return max(1, int(edit)), qpp
    
if __name__ == '__main__':
    set_seed(42)
    attack=args.attack
    dataset_name = args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    dropout = args.dropout
    model_path = args.model_path
    model_name = args.model
    
    if is_malnet_tiny(dataset_name):
        dataset, splits = load_malnet_tiny_filtered_splits(
            max_nodes=MALNET_TINY_MAX_NODES, seed=42)
        test_index = splits['test']
        test_dataset = dataset[test_index]
    else:
        dataset = load_tud_dataset(dataset_name)
        splits = load_dataset_splits(dataset_name, len(dataset))
        test_index = splits['test']
        test_dataset = dataset[test_index]
    input_dim = dataset.num_node_features
    output_dim = dataset.num_classes
    print('input dim: ', input_dim)
    print('output dim: ', output_dim)

    cf_pct = args.cf_pct
    cf_suffix = '' if cf_pct == 0 else str(cf_pct)
    if args.use_pairs:
        model_path = os.path.join(model_path, 'cf_lm', args.cf_type)
    elif cf_pct != 0:
        model_path = os.path.join(model_path, 'cf', args.cf_type)
    if model_name=='GCN':
        model = GCN(5,input_dim,hidden_dim,output_dim,0.8,dropout).to(device)
        if cf_suffix=='':
            load_path = os.path.join(model_path, '{}_{}.pt'.format(dataset_name, model_name))
        else:
            load_path = os.path.join(model_path, '{}_{}_{}.pt'.format(dataset_name, model_name, cf_suffix))
    elif model_name=='GIN':
        gin_layers = 6 if is_malnet_tiny(dataset_name) else 5
        model = GIN(gin_layers,2,input_dim,hidden_dim,output_dim,dropout).to(device)
        if cf_suffix=='':
            load_path = os.path.join(model_path, '{}_{}.pt'.format(dataset_name, model_name))
        else:
            load_path = os.path.join(model_path, '{}_{}_{}.pt'.format(dataset_name, model_name, cf_suffix))

    model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()

    if attack == 'BB':
        attacker = OPT_attack_sign_SGD(model, device, args.effective)
        num_test = len(test_dataset)
        perturbation = [] #perturbation for each poisoned graph
        perturbation_ratio = [] #perturbation ratio for each poisoned graph

        no_need_count = 0
        num_query = []    
        fail_count = 0
        distortion = []
        attack_time = []

        init_perturbation = [] #perturbation for each poisoned graph
        init_perturbation_ratio = [] #perturbation ratio for each poisoned graph
        init_num_query = []    
        init_distortion = []
        init_attack_time = []
        search_type = []

        detect_test_normal = []
        detect_test_advers = []
        success_count = 0
        num_add_edge, num_delete_edge = [], []
        for i in range(num_test):
            print('begin to attack instance {}'.format(i))
            x0 = test_dataset[i].to(device)
            y0 = x0.y[0]
            y1 = model.predict(x0, None,device)
            num_nodes = x0.num_nodes
            space = num_nodes * (num_nodes - 1) / 2
            success = False
            if y0 == y1:
                time_start = time()
                adv_x0, adv_y0, query, success, dis, init = attacker.attack_untargeted(x0, y0, query_limit=args.max_query)
                time_end = time()
                init_num_query.append(init[2])
                num_query.append(query)
                init_attack_time.append(init[3])
                attack_time.append(time_end-time_start)
                if success:
                    #process results in Stage 1
                    init_perturb, init_dis, init_query, init_time, s_type = init
                    init_ratio = init_perturb / space
                    init_perturbation.append(init_perturb)
                    init_distortion.append(init_dis)
                    search_type.append(s_type)
                    init_perturbation_ratio.append(init_ratio)

                    #process results in Stage 2
                    perturb = distance(adv_x0, x0)
                    perturbation.append(perturb)
                    perturbation_ratio.append(perturb/space)
                    distortion.append(dis)
                    
                    add_edge, delete_edge = count_edges(adv_x0, x0)
                    num_delete_edge.append(delete_edge)
                    num_add_edge.append(add_edge)

                    #test dataset for defense
                    #x0.y = torch.tensor([0])
                    #adv_x0.y = torch.tensor([1])
                    adv_x0.y = x0.y
                    detect_test_advers.append(adv_x0)
                    detect_test_normal.append(x0)
                    success_count += 1
                else:
                    detect_test_advers.append(x0)
                    detect_test_normal.append(x0)
                    init_distortion.append(-1)
                    init_perturbation.append(-1)
                    init_perturbation_ratio.append(-1)
                    search_type.append(-1)

                    perturbation.append(-1)
                    perturbation_ratio.append(-1)
                    distortion.append(-1) 
            else:
                print('instance {} is wrongly classified, No Need to Attack'.format(i))
                no_need_count += 1
                num_query.append(0)
                attack_time.append(0)
                perturbation.append(0)
                perturbation_ratio.append(0)
                distortion.append(0)
                
                init_perturbation.append(0)
                init_distortion.append(0)
                init_num_query.append(0)
                init_attack_time.append(0)
                search_type.append(0)
                init_perturbation_ratio.append(0)

        
        print('{} instances don\'t need to be attacked'.format(no_need_count))
        print("successs ratio:{}".format((success_count / (num_test - no_need_count)*100)))
        '''
        success_ratio = success_count / (num_test - no_need_count)*100
        avg_perturbation = sum(perturbation) / success_count
        print("Sign-Opt: the success rate of black-box attack is {}/{} = {:.4f}".format(success_count,num_test-no_need_count, success_ratio))
        print('Sign-Opt: the average perturbation is {:.4f}'.format(avg_perturbation))
        print('Sign-Opt: the average perturbation ratio is {:.4f}'.format(sum(perturbation_ratio) / success_count*100))
        print('Sign-Opt: the average query count is {:.4f}'.format(sum(num_query)/(num_test-no_need_count)))
        print('Sign-Opt: the average attacking time is {:.4f}'.format(sum(attack_time)/(num_test-no_need_count)))
        print('Sign-Opt: the average distortion is {:.4f}'.format(sum(distortion)/success_count))
        print('dataset: {}'.format(dataset_name))
        '''
        if args.search == 1 and args.effective == 1 and args.id ==1: 
            detect_test_path = './attacks/defense1/'+dataset_name+'_'+model_name+'_Our_'
            torch.save(detect_test_normal, detect_test_path+'test_normal.pt')
            torch.save(detect_test_advers, detect_test_path+'test_advers.pt')
            print('test dataset for defense saved!')


        init_path = './attacks/out1/init_{}_{}_{}_{}_'.format(dataset_name, args.id, args.effective , args.search)
        with open(init_path+'search_type.txt', 'w') as f:
            f.write(str(search_type))
        with open(init_path+'P.txt', 'w') as f:
            f.write(str(init_perturbation))
        with open(init_path+'PR.txt', 'w') as f:
            f.write(str(init_perturbation_ratio))
        with open(init_path+'D.txt', 'w') as f:
            f.write(str(init_distortion))
        with open(init_path+'Q.txt', 'w') as f:
            f.write(str(init_num_query))
        with open(init_path+'T.txt', 'w') as f:
            f.write(str(init_attack_time))  
        
        
        our_path = './attacks/out1/our_{}_{}_{}_{}_'.format(dataset_name, args.id, args.effective , args.search)
        with open(our_path+'Q.txt', 'w') as f:
            f.write(str(num_query))
        with open(our_path+'T.txt', 'w') as f:
            f.write(str(attack_time))
        with open(our_path+'P.txt', 'w') as f:
            f.write(str(perturbation))
        with open(our_path+'PR.txt', 'w') as f:
            f.write(str(perturbation_ratio))
        with open(our_path+'D.txt', 'w') as f:
            f.write(str(distortion))
        with open(our_path+'ADD.txt', 'w') as f:
            f.write(str(num_delete_edge))
        with open(our_path+'DEL.txt', 'w') as f:
            f.write(str(num_add_edge))
                
        print("the numbers of deleted edges are:", num_delete_edge)
        print("the numbers od added edges are:", num_add_edge)
        print("the average number of deleted edges for %s: %d"%(dataset_name, float(sum(num_delete_edge)/len(num_delete_edge))))
        print("the average number of added edges for %s: %d"%(dataset_name, float(sum(num_add_edge)/len(num_add_edge))))
        '''
        out_path = './out/{}_Opt_{}.txt'.format(dataset_name, bound)  
        with open(out_path, 'w') as f:
            f.write('{} instances don\'t need to be attacked\n'.format(no_need_count))
            f.write('Sign-Opt fails to attack {} instance\n'.format(fail_count))
            f.write("Sign-Opt: the success rate of black-box attack is {}/{} = {:.4f}\n".format(success_count,num_test-no_need_count, success_ratio))
            f.write('Sign-Opt: the average perturbation is {:.4f}\n'.format(avg_perturbation))
            f.write('Sign-Opt: the average perturbation ratio is {:.4f}\n'.format(sum(perturbation_ratio) / success_count*100))
            f.write('Sign-Opt: the average query count is {:.4f}\n'.format(sum(num_query)/(num_test-no_need_count)))
            f.write('Sign-Opt: the average attacking time is {:.4f}\n'.format(sum(attack_time)/(num_test-no_need_count)))
            f.write('Sign-Opt: the average distortion is {:.4f}\n'.format(sum(distortion)/success_count))
            f.write('Sign-Opt: detail perturbation are: {}\n'.format(perturbation))
            f.write('Sign-Opt: detail perturbation ratio are: {}\n'.format(perturbation_ratio))
        '''
    elif attack == 'WB':
        num_test = len(test_dataset)
        perturbation = [] #perturbation for each poisoned graph
        perturbation_ratio = [] #perturbation ratio for each poisoned graph

        no_need_count = 0
        num_query = []    
        fail_count = 0
        distortion = []
        attack_time = []

        init_perturbation = [] #perturbation for each poisoned graph
        init_perturbation_ratio = [] #perturbation ratio for each poisoned graph
        init_num_query = []    
        init_distortion = []
        init_attack_time = []
        search_type = []

        detect_test_normal = []
        detect_test_advers = []

        num_add_edge, num_delete_edge = [], []
        success_count = 0
        for i in range(num_test):
            attacker = PR_Attack(model, device, emb_dim=hidden_dim, k=args.k, train_epochs=args.epochs, lr=args.lr)
            print('begin to attack instance {}'.format(i))
            x0 = test_dataset[i].to(device)
            y0 = x0.y[0]
            y1 = model.predict(x0, None, device)
            num_nodes = x0.num_nodes
            space = num_nodes * (num_nodes - 1) / 2
            success=False
            if y0 == y1:
                time_start = time()
                attacker.train(x0)
                adv_x0, adv_x0_index,adv_pred = attacker.projection(x0)
                time_end = time()
                if adv_pred != y0:
                    success=True
                # init_num_query.append(init[2])
                # num_query.append(query)
                # init_attack_time.append(init[3])
                adv_x0 = Data(x=adv_x0, edge_index=adv_x0_index, y=adv_pred)
                attack_time.append(time_end-time_start)
                if success:
                    #process results in Stage 1
                    # init_perturb, init_dis, init_query, init_time, s_type = init
                    # init_ratio = init_perturb / space
                    # init_perturbation.append(init_perturb)
                    # init_distortion.append(init_dis)
                    # search_type.append(s_type)
                    # init_perturbation_ratio.append(init_ratio)

                    #process results in Stage 2
                    print('attack success!')
                    success_count += 1
                    perturb = distance(adv_x0, x0)
                    perturbation.append(perturb)
                    perturbation_ratio.append(perturb/space)
                    distortion.append(1) #to be fixed
                    
                    add_edge, delete_edge = count_edges(adv_x0, x0)
                    num_delete_edge.append(delete_edge)
                    num_add_edge.append(add_edge)

                    #test dataset for defense
                    #x0.y = torch.tensor([0])
                    #adv_x0.y = torch.tensor([1])
                    adv_x0.y = x0.y
                    detect_test_advers.append(adv_x0)
                    detect_test_normal.append(x0)
                else:
                    detect_test_advers.append(x0)
                    detect_test_normal.append(x0)
                    init_distortion.append(-1)
                    init_perturbation.append(-1)
                    init_perturbation_ratio.append(-1)
                    search_type.append(-1)

                    perturbation.append(-1)
                    perturbation_ratio.append(-1)
                    distortion.append(-1) 
            else:
                print('instance {} is wrongly classified, No Need to Attack'.format(i))
                no_need_count += 1
                num_query.append(0)
                attack_time.append(0)

                perturbation.append(0)
                perturbation_ratio.append(0)
                distortion.append(0)
                
                init_perturbation.append(0)
                init_distortion.append(0)
                init_num_query.append(0)
                init_attack_time.append(0)
                search_type.append(0)
                init_perturbation_ratio.append(0)

        
        print('{} instances don\'t need to be attacked'.format(no_need_count))
        print("Success ratio:{}".format(success_count/(num_test - no_need_count)))
        '''
        success_ratio = success_count / (num_test - no_need_count)*100
        avg_perturbation = sum(perturbation) / success_count
        print("Sign-Opt: the success rate of black-box attack is {}/{} = {:.4f}".format(success_count,num_test-no_need_count, success_ratio))
        print('Sign-Opt: the average perturbation is {:.4f}'.format(avg_perturbation))
        print('Sign-Opt: the average perturbation ratio is {:.4f}'.format(sum(perturbation_ratio) / success_count*100))
        print('Sign-Opt: the average query count is {:.4f}'.format(sum(num_query)/(num_test-no_need_count)))
        print('Sign-Opt: the average attacking time is {:.4f}'.format(sum(attack_time)/(num_test-no_need_count)))
        print('Sign-Opt: the average distortion is {:.4f}'.format(sum(distortion)/success_count))
        print('dataset: {}'.format(dataset_name))
        '''
        if args.search == 1 and args.effective == 1 and args.id ==1: 
            detect_test_path = './attacks/defense2/'+dataset_name+'_'+model_name+'_Our_'
            torch.save(detect_test_normal, detect_test_path+'test_normal.pt')
            torch.save(detect_test_advers, detect_test_path+'test_advers.pt')
            print('test dataset for defense saved!')
    
        
        init_path = './attacks/out2/init_{}_{}_{}_{}_'.format(dataset_name, args.id, args.effective , args.search)
        with open(init_path+'search_type.txt', 'w') as f:
            f.write(str(search_type))
        with open(init_path+'P.txt', 'w') as f:
            f.write(str(init_perturbation))
        with open(init_path+'PR.txt', 'w') as f:
            f.write(str(init_perturbation_ratio))
        with open(init_path+'D.txt', 'w') as f:
            f.write(str(init_distortion))
        with open(init_path+'Q.txt', 'w') as f:
            f.write(str(init_num_query))
        with open(init_path+'T.txt', 'w') as f:
            f.write(str(init_attack_time))  


        our_path = './attacks/out2/our_{}_{}_{}_{}_'.format(dataset_name, args.id, args.effective , args.search)
        with open(our_path+'Q.txt', 'w') as f:
            f.write(str(num_query))
        with open(our_path+'T.txt', 'w') as f:
            f.write(str(attack_time))
        with open(our_path+'P.txt', 'w') as f:
            f.write(str(perturbation))
        with open(our_path+'PR.txt', 'w') as f:
            f.write(str(perturbation_ratio))
        with open(our_path+'D.txt', 'w') as f:
            f.write(str(distortion))
        with open(our_path+'ADD.txt', 'w') as f:
            f.write(str(num_delete_edge))
        with open(our_path+'DEL.txt', 'w') as f:
            f.write(str(num_add_edge))
                
        print("the numbers of deleted edges are:", num_delete_edge)
        print("the numbers od added edges are:", num_add_edge)
        print("the average number of deleted edges for %s: %d"%(dataset_name, float(sum(num_delete_edge)/len(num_delete_edge))))
        print("the average number of added edges for %s: %d"%(dataset_name, float(sum(num_add_edge)/len(num_add_edge))))
        '''
        out_path = './out/{}_Opt_{}.txt'.format(dataset_name, bound)  
        with open(out_path, 'w') as f:
            f.write('{} instances don\'t need to be attacked\n'.format(no_need_count))
            f.write('Sign-Opt fails to attack {} instance\n'.format(fail_count))
            f.write("Sign-Opt: the success rate of black-box attack is {}/{} = {:.4f}\n".format(success_count,num_test-no_need_count, success_ratio))
            f.write('Sign-Opt: the average perturbation is {:.4f}\n'.format(avg_perturbation))
            f.write('Sign-Opt: the average perturbation ratio is {:.4f}\n'.format(sum(perturbation_ratio) / success_count*100))
            f.write('Sign-Opt: the average query count is {:.4f}\n'.format(sum(num_query)/(num_test-no_need_count)))
            f.write('Sign-Opt: the average attacking time is {:.4f}\n'.format(sum(attack_time)/(num_test-no_need_count)))
            f.write('Sign-Opt: the average distortion is {:.4f}\n'.format(sum(distortion)/success_count))
            f.write('Sign-Opt: detail perturbation are: {}\n'.format(perturbation))
            f.write('Sign-Opt: detail perturbation ratio are: {}\n'.format(perturbation_ratio))
        '''
    elif attack == 'BO':
        try:
            import dgl  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError('BayesOpt attack requires the dgl package; install dgl to use --attack BO.') from exc
        from attacks.BO_attack.bayesopt_attack import BayesOptAttack
        from attacks.BO_attack.utils import nettack_loss, nettack_loss_gunet

        num_test = len(test_dataset)
        perturbation = []
        perturbation_ratio = []
        attack_time = []
        num_add_edge, num_delete_edge = [], []
        num_query = []
        detect_test_normal = []
        detect_test_advers = []
        success_flags = []
        success_count = 0
        no_need_count = 0

        loss_fn = nettack_loss_gunet if model_name == 'GUNet' else nettack_loss
        wrapped_model = DGLModelWrapper(model, device)

        for i in range(num_test):
            print('begin to attack instance {}'.format(i))
            base_graph = test_dataset[i]
            x0 = copy.deepcopy(base_graph).to(device)
            y0 = x0.y[0]
            y1 = model.predict(x0, None, device)
            num_nodes = base_graph.num_nodes
            space = max(1, num_nodes * (num_nodes - 1) / 2)
            if y0 == y1:
                dgl_graph = pyg_to_dgl_graph(base_graph)
                edit, queries_per_perturb = compute_bo_budget(
                    dgl_graph,
                    args.bo_budget,
                    args.bo_budget_by,
                    args.bo_query_per_perturb,
                    args.bo_mode,
                )
                max_queries = min(edit * queries_per_perturb, args.max_query)
                attacker = BayesOptAttack(
                    wrapped_model,
                    loss_fn,
                    surrogate=args.bo_surrogate,
                    batch_size=args.bo_batch_size,
                    n_init=args.bo_n_init,
                    edit_per_stage=min(5, edit) if args.bo_no_greedy else 1,
                    surrogate_settings={'h': args.bo_max_h, 'extractor_mode': 'continuous'},
                    acq_settings={'acq_optimiser': args.bo_acq, 'random_frac': 0.0},
                    verbose=True,
                    mode=args.bo_mode,
                    terminate_after_n_fail=args.bo_patience,
                    n_hop_constraint=args.bo_constrain_n_hop,
                    preserve_disconnected_components=args.bo_preserve_components,
                )
                label = base_graph.y.to(device).view(1)
                time_start = time()
                try:
                    df, adv_example = attacker.attack(dgl_graph, label, edit, max_queries)
                except Exception as exc:
                    print(f'BayesOpt attack failed on instance {i} with error: {exc}. Skipping this instance.')
                    df, adv_example = None, None
                time_end = time()
                attack_time.append(time_end - time_start)
                if df is not None and not df.empty and 'queries' in df.columns:
                    num_query.append(int(df['queries'].max()))
                else:
                    num_query.append(max_queries)
                if adv_example is not None:
                    success_count += 1
                    adv_data = dgl_to_pyg_graph(adv_example, y=base_graph.y)
                    perturb = distance(adv_data, base_graph)
                    perturbation.append(perturb)
                    perturbation_ratio.append(perturb / space)
                    add_edge, delete_edge = count_edges(adv_data, base_graph)
                    num_add_edge.append(add_edge)
                    num_delete_edge.append(delete_edge)
                    detect_test_advers.append(adv_data)
                    detect_test_normal.append(base_graph)
                    success_flags.append(True)
                else:
                    perturbation.append(-1)
                    perturbation_ratio.append(-1)
                    detect_test_advers.append(base_graph)
                    detect_test_normal.append(base_graph)
                    success_flags.append(False)
            else:
                print('instance {} is wrongly classified, No Need to Attack'.format(i))
                no_need_count += 1
                perturbation.append(0)
                perturbation_ratio.append(0)
                attack_time.append(0)
                num_query.append(0)
                detect_test_advers.append(base_graph)
                detect_test_normal.append(base_graph)
                success_flags.append(False)

        total_attempts = max(1, num_test - no_need_count)
        success_ratio = success_count / total_attempts * 100
        success_perturbs = [p for p, flag in zip(perturbation, success_flags) if flag]
        success_ratios = [r for r, flag in zip(perturbation_ratio, success_flags) if flag]
        success_queries = [q for q, flag in zip(num_query, success_flags) if flag]
        success_times = [t for t, flag in zip(attack_time, success_flags) if flag]
        print(f'BayesOpt attack success rate: {success_count}/{total_attempts} = {success_ratio:.2f}%')
        if success_count > 0:
            print(f'BayesOpt average perturbation: {np.mean(success_perturbs):.2f}')
            print(f'BayesOpt average perturbation ratio: {np.mean(success_ratios) * 100:.2f}%')
            print(f'BayesOpt average queries: {np.mean(success_queries):.2f}')
            print(f'BayesOpt average time: {np.mean(success_times):.4f}s')
            if num_add_edge and num_delete_edge:
                print(f'BayesOpt edges added (mean): {np.mean(num_add_edge):.2f}, edges deleted (mean): {np.mean(num_delete_edge):.2f}')
        defense_dir = os.path.join('./attacks', 'defense_bo')
        os.makedirs(defense_dir, exist_ok=True)
        defense_prefix = os.path.join(defense_dir, f'{dataset_name}_{model_name}_BO_')
        torch.save(detect_test_normal, defense_prefix + 'test_normal.pt')
        torch.save(detect_test_advers, defense_prefix + 'test_advers.pt')

    elif attack == 'REWATT':
        from attacks.ReWatt import ReWattAttacker

        num_test = len(test_dataset)
        perturbation = []
        perturbation_ratio = []
        attack_time = []
        num_add_edge, num_delete_edge = [], []
        success_flags = []
        detect_test_normal = []
        detect_test_advers = []
        success_count = 0
        no_need_count = 0
        step_counts = []

        attacker = ReWattAttacker(
            model=model,
            device=device,
            action_percent=args.rewatt_action_pct,
            max_actions=args.rewatt_max_actions,
            negative_reward=args.rewatt_negative_reward,
            include_self=bool(args.rewatt_include_self),
            use_all=args.rewatt_use_all,
        )

        for i in range(num_test):
            print('begin to attack instance {}'.format(i))
            x0 = test_dataset[i]
            y0 = int(x0.y[0])
            x0_device = x0.to(device)
            y1 = int(model.predict(x0_device, None, device))
            num_nodes = x0.num_nodes
            space = max(1, num_nodes * (num_nodes - 1) / 2)
            if y0 == y1:
                time_start = time()
                adv_data, success, steps = attacker.attack(x0, targeted=False, initial_pred=y1)
                time_end = time()
                attack_time.append(time_end - time_start)
                step_counts.append(steps)
                if success and adv_data is not None:
                    success_count += 1
                    perturb = distance(adv_data, x0)
                    perturbation.append(perturb)
                    perturbation_ratio.append(perturb / space)
                    add_edge, delete_edge = count_edges(adv_data, x0)
                    num_add_edge.append(add_edge)
                    num_delete_edge.append(delete_edge)
                    adv_data.y = x0.y
                    detect_test_advers.append(adv_data)
                    detect_test_normal.append(x0)
                    success_flags.append(True)
                else:
                    perturbation.append(-1)
                    perturbation_ratio.append(-1)
                    detect_test_advers.append(x0)
                    detect_test_normal.append(x0)
                    success_flags.append(False)
            else:
                print('instance {} is wrongly classified, No Need to Attack'.format(i))
                no_need_count += 1
                perturbation.append(0)
                perturbation_ratio.append(0)
                attack_time.append(0)
                step_counts.append(0)
                detect_test_advers.append(x0)
                detect_test_normal.append(x0)
                success_flags.append(False)

        total_attempts = max(1, num_test - no_need_count)
        success_ratio = success_count / total_attempts * 100
        success_perturbs = [p for p, flag in zip(perturbation, success_flags) if flag]
        success_ratios = [r for r, flag in zip(perturbation_ratio, success_flags) if flag]
        success_times = [t for t, flag in zip(attack_time, success_flags) if flag]
        success_steps = [s for s, flag in zip(step_counts, success_flags) if flag]
        print(f'ReWatt attack success rate: {success_count}/{total_attempts} = {success_ratio:.2f}%')
        if success_count > 0:
            print(f'ReWatt average perturbation: {np.mean(success_perturbs):.2f}')
            print(f'ReWatt average perturbation ratio: {np.mean(success_ratios) * 100:.2f}%')
            print(f'ReWatt average time: {np.mean(success_times):.4f}s')
            print(f'ReWatt average steps: {np.mean(success_steps):.2f}')
            if num_add_edge and num_delete_edge:
                print(f'ReWatt edges added (mean): {np.mean(num_add_edge):.2f}, edges deleted (mean): {np.mean(num_delete_edge):.2f}')

        defense_dir = os.path.join('./attacks', 'defense_rewatt')
        os.makedirs(defense_dir, exist_ok=True)
        defense_prefix = os.path.join(defense_dir, f'{dataset_name}_{model_name}_ReWatt_')
        torch.save(detect_test_normal, defense_prefix + 'test_normal.pt')
        torch.save(detect_test_advers, defense_prefix + 'test_advers.pt')

    elif attack == 'RAND':
        num_test = len(test_dataset)
        perturbation = []
        perturbation_ratio = []
        attack_time = []
        num_add_edge, num_delete_edge = [], []
        success_count = 0
        fail_count = 0
        no_need_count = 0
        success_flags = []
        detect_test_normal = []
        detect_test_advers = []

        for i in range(num_test):
            print('begin to attack instance {}'.format(i))
            x0 = test_dataset[i].to(device)
            y0 = x0.y[0]
            y1 = model.predict(x0, None, device)
            num_nodes = x0.num_nodes
            space = max(1, num_nodes * (num_nodes - 1) / 2)
            space_edges = max(1, int(space))
            if y0 == y1:
                if args.rand_budget <= 1:
                    max_budget_edges = max(1, int(np.ceil(args.rand_budget * space_edges)))
                else:
                    max_budget_edges = max(1, int(min(args.rand_budget, space_edges)))
                time_start = time()
                adv_x0, adv_y0, success, perturb = random_attack_sample(
                    x0, model, device, args.rand_trials, max_budget_edges, distance)
                time_end = time()
                attack_time.append(time_end - time_start)
                if success:
                    success_count += 1
                    perturbation.append(perturb)
                    perturbation_ratio.append(perturb / space)
                    add_edge, delete_edge = count_edges(adv_x0, x0)
                    num_add_edge.append(add_edge)
                    num_delete_edge.append(delete_edge)
                    adv_x0.y = x0.y
                    detect_test_advers.append(adv_x0)
                    detect_test_normal.append(x0)
                    success_flags.append(True)
                else:
                    fail_count += 1
                    perturbation.append(-1)
                    perturbation_ratio.append(-1)
                    detect_test_advers.append(x0)
                    detect_test_normal.append(x0)
                    success_flags.append(False)
            else:
                print('instance {} is wrongly classified, No Need to Attack'.format(i))
                no_need_count += 1
                perturbation.append(0)
                perturbation_ratio.append(0)
                attack_time.append(0)
                detect_test_advers.append(x0)
                detect_test_normal.append(x0)
                success_flags.append(False)

        total_attempts = max(1, num_test - no_need_count)
        success_ratio = success_count / total_attempts * 100
        success_perturbs = [p for p, flag in zip(perturbation, success_flags) if flag]
        success_ratios = [r for r, flag in zip(perturbation_ratio, success_flags) if flag]
        success_times = [t for t, flag in zip(attack_time, success_flags) if flag]
        avg_perturb = np.mean(success_perturbs) if success_perturbs else 0
        avg_ratio = np.mean(success_ratios) if success_ratios else 0
        avg_time = np.mean(success_times) if success_times else 0
        print(f'Random attack success rate: {success_count}/{total_attempts} = {success_ratio:.2f}%')
        if success_count > 0:
            print(f'Random attack average perturbation: {avg_perturb:.2f}')
            print(f'Random attack average perturbation ratio: {avg_ratio * 100:.2f}%')
            print(f'Random attack average time: {avg_time:.4f}s')
            print(f'Random attack edges added (mean): {np.mean(num_add_edge):.2f}, edges deleted (mean): {np.mean(num_delete_edge):.2f}')

        defense_dir = os.path.join('./attacks', 'defense3')
        os.makedirs(defense_dir, exist_ok=True)
        defense_prefix = os.path.join(defense_dir, f'{dataset_name}_{model_name}_Rand_')
        torch.save(detect_test_normal, defense_prefix + 'test_normal.pt')
        torch.save(detect_test_advers, defense_prefix + 'test_advers.pt')
    
    
