'''
Credit: This file is adapted from CCAS21_GNNattack (https://github.com/Anonymous-21/CCAS21_GNNattack)
'''
import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (GINConv, GCNConv, SAGPooling, TopKPooling,
                                global_add_pool, global_max_pool as gmp,
                                global_mean_pool as gap)
from torch_geometric.nn.models import GraphUNet
from torch_geometric.nn.pool.topk_pool import filter_adj, topk
from torch.nn import Parameter

class WeightedGINConv(GINConv):
    """
    GINConv that supports edge_weight for PyG 1.7.
    Usage: out = conv(x, edge_index, edge_weight=edge_weight)
    """
    def forward(self, x, edge_index, edge_weight=None, size=None):
        # propagate will pass edge_weight to message()
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        # (1 + eps) * x + sum(messages)
        out = out + (1.0 + float(self.eps)) * x
        return self.nn(out)

    def message(self, x_j, edge_weight=None):
        # x_j: messages from source nodes; scale by edge weights if provided
        if edge_weight is None:
            return x_j
        # ensure shape [num_edges, 1] for broadcasting
        return x_j * edge_weight.view(-1, 1)

class MLP(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = torch.nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(torch.nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(torch.nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(torch.nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class ConformalUncertaintyMixin:
    """
    Conformal predictor mixin that buckets calibration data by graph size,
    stores per-bucket nonconformity scores, and exposes predictions, p-values,
    and prediction-set sizes during inference.
    """

    def __init__(self):
        super().__init__()
        self._conformal_scores = torch.empty(0)
        self._bucket_scores = {}
        self._bins = [(-10**9, 10**9)]
        self._num_classes = None
        self._threshold_cache = {}
        self._bin_mode = "quantile"
        self._num_bins = 5

    def _batchify_graphs(self, data, device):
        if isinstance(data, Batch):
            batch = data
        elif isinstance(data, Data):
            batch = Batch.from_data_list([data])
        else:
            data_list = list(data)
            if not data_list:
                raise ValueError("No graph data provided for uncertainty inference.")
            batch = Batch.from_data_list(data_list)
        return batch.to(device)

    def _graph_sizes_from_dataset(self, dataset):
        sizes = []
        for idx in range(len(dataset)):
            graph = dataset[idx]
            sizes.append(int(getattr(graph, "num_nodes", graph.x.size(0))))
        return sizes

    def _graph_sizes_from_batch(self, batch: Batch):
        if hasattr(batch, "ptr") and batch.ptr is not None:
            ptr = batch.ptr.cpu()
            return (ptr[1:] - ptr[:-1]).tolist()
        batch_index = batch.batch.cpu()
        counts = torch.bincount(batch_index, minlength=batch.num_graphs)
        return counts.tolist()

    def _make_bins(self, sizes, mode=None, num_bins=None):
        if not sizes:
            return [(-10**9, 10**9)]
        mode = mode or self._bin_mode
        num_bins = num_bins or self._num_bins
        sizes_tensor = torch.tensor(sizes, dtype=torch.float32)
        s_min = int(sizes_tensor.min().item())
        s_max = int(sizes_tensor.max().item())
        if mode == "none" or num_bins <= 1 or s_min == s_max:
            return [(-10**9, 10**9)]
        if mode == "fixed":
            edges = torch.linspace(float(s_min), float(s_max), steps=num_bins + 1)
        else:
            quantiles = torch.linspace(0.0, 1.0, steps=num_bins + 1)
            edges = torch.quantile(sizes_tensor, quantiles)
            edges, _ = torch.sort(torch.unique(edges))
            if edges.numel() - 1 < num_bins:
                edges = torch.linspace(float(s_min), float(s_max), steps=num_bins + 1)
        bins = []
        for idx in range(edges.numel() - 1):
            lo = edges[idx].item()
            hi = edges[idx + 1].item()
            bins.append((int(lo) - 1 if idx == 0 else int(lo), int(hi)))
        merged = []
        for lo, hi in bins:
            if not merged or lo > merged[-1][1]:
                merged.append((lo, hi))
            else:
                merged[-1] = (merged[-1][0], hi)
        return merged or [(-10**9, 10**9)]

    def _which_bin(self, size):
        for idx, (lo, hi) in enumerate(self._bins):
            if lo < size <= hi:
                return idx
        return 0

    def _quantile(self, scores: torch.Tensor, q: float) -> float:
        if scores.numel() == 0:
            return 1.0
        sorted_scores, _ = torch.sort(scores)
        rank = math.ceil(q * sorted_scores.numel()) - 1
        rank = max(0, min(rank, sorted_scores.numel() - 1))
        return sorted_scores[rank].item()

    def _build_bin_marginals(self):
        marginals = {b: [] for b in range(len(self._bins))}
        for (b, _), scores in self._bucket_scores.items():
            if scores.numel() > 0:
                marginals[b].append(scores)
        return {
            b: torch.cat(vals) if vals else torch.empty(0)
            for b, vals in marginals.items()
        }

    def _thresholds_for_alpha(self, alpha: float=0.1):
        cache_key = round(float(alpha), 6)
        if cache_key in self._threshold_cache:
            return self._threshold_cache[cache_key]
        target_q = 1.0 - alpha
        bin_marginals = self._build_bin_marginals()
        thresholds = {}
        for b in range(len(self._bins)):
            per_class = []
            for c in range(self._num_classes or 0):
                scores = self._bucket_scores.get((b, c))
                if scores is not None and scores.numel() > 0:
                    q = self._quantile(scores, target_q)
                else:
                    bm = bin_marginals.get(b)
                    q = self._quantile(bm, target_q) if bm is not None else 1.0
                per_class.append(q)
            thresholds[b] = torch.tensor(per_class or [1.0], dtype=torch.float32)
        self._threshold_cache[cache_key] = thresholds
        return thresholds

    def _conformal_p_value(self, bucket_key, score: float) -> float:
        scores = self._bucket_scores.get(bucket_key)
        if scores is None or scores.numel() == 0:
            return 1.0
        ge = (scores >= score).sum().item()
        n = scores.numel()
        return (ge + 1) / (n + 1)

    @torch.no_grad()
    def _calibrate_conformal(self, loader, device):
        if loader is None:
            raise RuntimeError("Validation loader is required for conformal calibration.")
        dataset = loader.dataset
        sizes = self._graph_sizes_from_dataset(dataset)
        if not sizes:
            raise ValueError("Calibration loader yielded no graphs.")
        self._bins = self._make_bins(sizes)
        bucket_scores = {}
        prev_mode = self.training
        self.eval()
        num_classes = None
        for batch in loader:
            batch = batch.to(device)
            logits = self(batch)
            probs = torch.softmax(logits, dim=-1).detach().cpu()
            if num_classes is None:
                num_classes = probs.size(-1)
            labels = batch.y.view(-1).detach().cpu().tolist()
            graph_sizes = self._graph_sizes_from_batch(batch)
            for size, label, prob_vec in zip(graph_sizes, labels, probs):
                bucket = self._which_bin(size)
                score = 1.0 - float(prob_vec[label])
                bucket_scores.setdefault((bucket, label), []).append(score)
        self.train(prev_mode)
        if num_classes is None:
            raise ValueError("Calibration loader yielded no batches.")
        self._num_classes = num_classes
        self._bucket_scores = {
            key: torch.tensor(values, dtype=torch.float32)
            for key, values in bucket_scores.items()
        }
        if self._bucket_scores:
            self._conformal_scores = torch.cat(list(self._bucket_scores.values()))
        else:
            self._conformal_scores = torch.empty(0)
        self._threshold_cache = {}

    @torch.no_grad()
    def get_uncertainty(
        self,
        data,
        val_loader=None,
        device=None,
        force_recalibrate=False,
        alpha: float = 0.1,
    ):
        """
        Returns (prediction, conformal p-value, prediction-set size) for each graph.
        """
        if device is None:
            device = next(self.parameters()).device
        needs_calibration = force_recalibrate or not self._bucket_scores
        if needs_calibration:
            if val_loader is None:
                raise RuntimeError("Validation loader must be provided for calibration.")
            self._calibrate_conformal(val_loader, device)
        if self._num_classes is None:
            raise RuntimeError("Model must be calibrated before inference.")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must lie in (0, 1).")
        thresholds = self._thresholds_for_alpha(alpha)
        batch = self._batchify_graphs(data, device)
        logits = self(batch)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        nonconformity_all = 1.0 - probs
        graph_sizes = self._graph_sizes_from_batch(batch)
        p_values = []
        set_sizes = []
        for size, prob_vec, pred_idx, nonconformity_vec in zip(
            graph_sizes, probs.detach().cpu(), preds.tolist(), nonconformity_all.detach().cpu()
        ):
            bucket = self._which_bin(size)
            threshold_vec = thresholds[bucket].to(prob_vec.device)
            set_sizes.append(int((nonconformity_vec <= threshold_vec).sum().item()))
            score = float(nonconformity_vec[pred_idx].item())
            p_val = self._conformal_p_value((bucket, pred_idx), score)
            p_values.append(p_val)
        return preds.cpu(), torch.tensor(p_values, dtype=torch.float32), torch.tensor(set_sizes, dtype=torch.long)

class GIN(ConformalUncertaintyMixin, torch.nn.Module):
    def __init__(self, num_layers, num_mlp_layers,input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        self.liners_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.liners_prediction.append(torch.nn.Linear(input_dim, output_dim))
            else:
                self.liners_prediction.append(torch.nn.Linear(hidden_dim, output_dim))
        
        self.ginconv = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            self.ginconv.append(WeightedGINConv(self.mlps[layer], train_eps=True))
        
    def forward(self, data,edge_weight=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        hidden_rep = [x]

        for layer in range(self.num_layers - 1):
            x = self.ginconv[layer](x, edge_index,edge_weight)
            x = F.relu(self.batch_norms[layer](x))
            hidden_rep.append(x)
            
        score_over_layer = 0

        for layer, h in enumerate(hidden_rep):
            pooled_h = global_add_pool(h, batch)
            score_over_layer += F.dropout(self.liners_prediction[layer](pooled_h),self.dropout,training=self.training)
        return score_over_layer

    def get_embedding_repr(self, data, edge_weight=None) -> torch.Tensor:
        if isinstance(data, Batch):
            batch_data = data
        elif isinstance(data, Data):
            if hasattr(data, 'batch'):
                batch_data = data
            else:
                batch_data = Batch.from_data_list([data])
        else:
            batch_data = Batch.from_data_list(data)
        x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch

        for layer in self.ginconv:
            x = layer(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return x

    def predict(self, data,edge_weight=None ,device='cuda'):
        #this is the prediction for single graph
        self.eval() 
        if isinstance(data,Batch):
            data = data.to_data_list()[0]
        graph = Batch.from_data_list([data]).to(device)
        output = self(graph,edge_weight)  #logits of graph: [[0.2,0.3,0.5]]
        pred = output.max(1, keepdim = True)[1] #final predicted label: [[1]]
        return pred[0][0]
        
    def predict_vector(self, data,edge_weight=None ,device='cuda'):
        self.eval()
        graph = Batch.from_data_list([data]).to(device)
        output = self(graph,edge_weight)
        vector = output[0]
        return torch.nn.functional.softmax(vector, dim=0)
        #return vector

class GCN(torch.nn.Module):
    def __init__(self, num_layers,input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        self.gcns = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.liners_prediction = torch.nn.ModuleList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.gcns.append(GCNConv(input_dim, hidden_dim))
            else:
                self.gcns.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        for layer in range(self.num_layers):
            if layer == 0:
                self.liners_prediction.append(torch.nn.Linear(input_dim, output_dim))
            else:
                self.liners_prediction.append(torch.nn.Linear(hidden_dim, output_dim))

    def get_embedding_repr(self, data, edge_weight=None) -> torch.Tensor:   
        if isinstance(data, Batch):
            batch_data = data
        elif isinstance(data, Data):
            batch_data = data if hasattr(data, 'batch') else Batch.from_data_list([data])
        else:
            batch_data = Batch.from_data_list(data)
        x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch

        for layer in range(self.num_layers - 1):
            x = self.gcns[layer](x, edge_index,edge_weight)
            x = F.relu(self.batch_norms[layer](x))
            x = F.dropout(x, self.dropout, training=self.training)
        return x
    
    def forward(self, data,edge_weight=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        hidden_rep = [x]
        for layer in range(self.num_layers - 1):
            x = self.gcns[layer](x, edge_index,edge_weight)
            x = F.relu(self.batch_norms[layer](x))
            hidden_rep.append(x)
       

        score_over_layer = 0
        for layer, h in enumerate(hidden_rep):
            pooled_h = global_add_pool(h, batch)
            score_over_layer += F.dropout(self.liners_prediction[layer](pooled_h),self.dropout,training=self.training)
        return score_over_layer

    
    def predict(self, data, device):
        self.eval()
        graph = Batch.from_data_list([data]).to(device)
        output = self(graph)
        pred = output.max(1, keepdim = True)[1]
        return pred[0][0]
    
    def predict_vector(self, data, device):
        self.eval()
        graph = Batch.from_data_list([data]).to(device)
        output = self(graph)
        vector = output[0]
        return torch.nn.functional.softmax(vector, dim=0)
        #return vector

class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x,edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm

class SAG(torch.nn.Module):
    def __init__(self,num_layers, input_dim, hidden_dim, output_dim, pooling_ratio, dropout):
        super(SAG, self).__init__()
        self.num_layers = num_layers
        self.num_features = input_dim
        self.nhid = hidden_dim
        self.num_classes = output_dim
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers-1):
            if layer == 0:
                self.convs.append(GCNConv(self.num_features, self.nhid))
            else:
                self.convs.append(GCNConv(self.nhid, self.nhid))
            self.batch_norms.append(torch.nn.BatchNorm1d(self.nhid))

        self.liners_prediction = torch.nn.ModuleList()
        
        for layer in range(num_layers):
            if layer == 0:
                self.liners_prediction.append(torch.nn.Linear(input_dim, output_dim))
            else:
                self.liners_prediction.append(torch.nn.Linear(hidden_dim, output_dim)) 

        self.sagpool = torch.nn.ModuleList()
        for layer in range(self.num_layers-1):
            self.sagpool.append(SAGPool(self.nhid, ratio=self.pooling_ratio))       

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        hidden_rep = [x]
        batch_rep = [batch]
        for layer in range(self.num_layers-1):
            x = self.convs[layer](x, edge_index)
            x, edge_index, _, batch, _ = self.sagpool[layer](x, edge_index, None, batch)
            x = F.relu(self.batch_norms[layer](x))
            hidden_rep.append(x)
            batch_rep.append(batch)
           
        score_over_layer = 0
        
        for layer, h in enumerate(hidden_rep):
            pooled_h = global_add_pool(h, batch_rep[layer])
            score_over_layer += F.dropout(self.liners_prediction[layer](pooled_h),self.dropout_ratio,training=self.training)
    
        return score_over_layer       
 
    def predict(self, data, device):
        #this is the prediction for single graph
        self.eval() 
        graph = Batch.from_data_list([data]).to(device)
        output = self(graph)  #logits of graph: [[0.2,0.3,0.5]]
        pred = output.max(1, keepdim = True)[1] #final predicted label: [[1]]
        return pred[0][0]


class GUNet(torch.nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim, pooling_ratio, deepth, dropout):
        super(GUNet, self).__init__()
        self.num_features = input_dim
        self.nhid = hidden_dim
        self.num_classes = output_dim
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout
        self.deepth = deepth
        self.num_layers = deepth

        self.liners_prediction = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            if layer == 0:
                self.liners_prediction.append(torch.nn.Linear(input_dim, output_dim))
            else:
                self.liners_prediction.append(torch.nn.Linear(hidden_dim, output_dim)) 

        self.gunpool = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers-1):
            if layer ==0:
                self.gunpool.append(GraphUNet(self.num_features, 32, self.nhid, 2, self.pooling_ratio))      
            else:
                self.gunpool.append(GraphUNet(self.nhid, 32, self.nhid, 2, self.pooling_ratio))
            self.batch_norms.append(torch.nn.BatchNorm1d(self.nhid))

       # self.pool = GraphUNet(self.num_features,32, self.nhid, self.deepth,self.pooling_ratio)

        #self.lin1 = torch.nn.Linear(self.num_features, self.num_classes)
       # self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)
      #  self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        hidden_rep = [x]

        for layer in range(self.num_layers-1):   
            x = self.gunpool[layer](x, edge_index)
            x = F.relu(self.batch_norms[layer](x))
            hidden_rep.append(x)
  
        score_over_layer = 0
        
        for layer, h in enumerate(hidden_rep):
            pooled_h = global_add_pool(h, batch)
            score_over_layer += F.dropout(self.liners_prediction[layer](pooled_h),self.dropout_ratio,training=self.training)
        
        return score_over_layer

    def predict(self, data, device):
        #this is the prediction for single graph
        self.eval() 
        graph = Batch.from_data_list([data]).to(device)
        output = self(graph)  #logits of graph: [[0.2,0.3,0.5]]
        pred = output.max(1, keepdim = True)[1] #final predicted label: [[1]]
        return pred[0][0]
