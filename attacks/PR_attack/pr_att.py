from models import *
from time import time
import numpy as np 
from numpy import linalg as LA
import torch
import scipy.spatial
import copy
from scipy.linalg import qr
import random
import networkx as nx
from networkx.classes.function import is_path
import community
import torch_geometric
from torch_geometric.utils import to_networkx, from_networkx, to_dense_adj, dense_to_sparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import Optional


class score_mlp(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(score_mlp, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 8)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.fc3 = torch.nn.Linear(8, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.sigmoid(x)

class PR_Attack:
    def __init__(self,model,device,emb_dim=64,train_epochs=100,lr=0.01,k=2):
        self.model = model
        self.device = device
        self.k = k #budget, only one addition allowed
        self.mlp = score_mlp(emb_dim*2).to(device)
        self.epochs = train_epochs
        self.lr = lr
        self.projection_k = max(0, k)

    def pred_loss(self,pred,original,kappa=0.01):
        # B = pred.shape[0]
        pred=pred.view(-1, pred.shape[-1])
        mask = torch.ones((1,2),dtype=torch.bool).to(self.device)
        mask[:, original]=False
        max_other=pred.masked_fill(~mask,torch.tensor(float('-inf'))).max(dim=1).values
        margin = pred[:,original]-max_other
        loss=F.relu(margin+kappa).mean()
        return loss

    def score(self,embeddings):
        scores=torch.zeros_like(embeddings)
        # embeddings = torch.cat([emb1,emb2],dim=1)
        score_model=self.mlp
        scores = score_model(embeddings)
        return scores

    def ranking(self,graph_adj,mask):
        identity_mat=torch.eye(graph_adj.shape[0]).to(self.device)
        ones_mat=torch.ones((graph_adj.shape[0],graph_adj.shape[0])).to(self.device)
        inv_adj_mat=(ones_mat-identity_mat)-graph_adj
        if mask is not None:
            delta_adj=torch.mul(inv_adj_mat,mask)
        else:
            print('no mask provided')
        return delta_adj

    def _discretize_edges(self, scores: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
        num_nodes = scores.shape[0]
        perturbations = torch.zeros_like(scores)
        if num_nodes <= 1:
            return perturbations

        upper_idx = torch.triu_indices(num_nodes, num_nodes, offset=1, device=scores.device)
        if upper_idx.shape[1] == 0:
            return perturbations
        upper_scores = scores[upper_idx[0], upper_idx[1]]
        limit = self.projection_k if top_k is None else int(top_k)
        limit = max(0, min(limit, upper_scores.numel()))
        k = limit
        if k == 0:
            return perturbations
        _, top_indices = torch.topk(upper_scores, k)
        rows = upper_idx[0][top_indices]
        cols = upper_idx[1][top_indices]
        perturbations[rows, cols] = 1.0
        perturbations[cols, rows] = 1.0
        return perturbations
    
    def projection(self,graph):
        self.model.eval()
        self.mlp.eval()
        num_nodes = graph.x.shape[0]
        embeddings = self.model.get_embedding_repr(graph)
        mask=torch.zeros((num_nodes,num_nodes),device=self.device)
        upper_mask = torch.triu_indices(num_nodes,num_nodes,1)
        for _, (i,j) in enumerate(zip(*upper_mask)):
            pair=torch.cat([embeddings[i],embeddings[j]],dim=0)
            score=self.score(pair)
            mask[i,j]=score
            mask[j,i]=score
        adj=to_dense_adj(edge_index=graph.edge_index)[0] 
        perturbations = self.ranking(adj,mask)
        discretized_perturbations = self._discretize_edges(perturbations)
        perturbed_graph = adj + discretized_perturbations
        att_edge_index = dense_to_sparse(perturbed_graph)[0]
        # graph.edge_index = att_edge_index
        att_graph= copy.deepcopy(graph)
        att_graph.edge_index = att_edge_index
        pred_new = self.model.predict(att_graph,device=self.device)
        return graph.x, att_edge_index,pred_new
    
    def train(self, test_graphs):
        print('training the attack model')
        self.mlp.train()
        graph=test_graphs
        original_pred = self.model.predict(graph,device=self.device)
        best_overall_loss = float("inf")
        best_overall_k = self.projection_k
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=self.lr)

        for e in range(self.epochs):

            num_nodes = graph.x.shape[0]
            embeddings = self.model.get_embedding_repr(graph)
            mask=torch.zeros((num_nodes,num_nodes),device=self.device)
            upper_mask = torch.triu_indices(num_nodes,num_nodes,1)
            for _, (i,j) in enumerate(zip(*upper_mask)):
                pair=torch.cat([embeddings[i],embeddings[j]],dim=0)
                score=self.score(pair)
                mask[i,j]=score
                mask[j,i]=score
            adj=to_dense_adj(edge_index=graph.edge_index)[0] 
            perturbations = self.ranking(adj,mask)
            best_loss = None
            best_prediction = None
            best_k_epoch = None
            for current_k in range(1, self.k + 1):
                discretized_perturbations = self._discretize_edges(perturbations, top_k=current_k)
                if discretized_perturbations.sum() == 0:
                    continue
                perturbed_graph = adj + discretized_perturbations
                att_edge_index = dense_to_sparse(perturbed_graph)[0]
                att_graph = copy.deepcopy(graph)
                att_graph.edge_index = att_edge_index
                att_prediction = self.model.predict_vector(att_graph,device=self.device)
                loss = self.pred_loss(att_prediction, original_pred)
                if best_loss is None or loss < best_loss:
                    best_loss = loss
                    best_prediction = att_prediction
                    best_k_epoch = current_k
            if best_loss is None:
                continue

            optimizer.zero_grad()                
            best_loss.backward()
            optimizer.step()
            loss_value = best_loss.item()
            if loss_value < best_overall_loss and best_k_epoch is not None:
                best_overall_loss = loss_value
                best_overall_k = best_k_epoch
        if best_overall_loss < float("inf"):
            print('Best training loss {:.4f} achieved with k={}'.format(best_overall_loss, best_overall_k))
            self.projection_k = best_overall_k
        else:
            print('Training did not find a better perturbation; keeping k={}'.format(self.projection_k))
        return self.mlp

   
