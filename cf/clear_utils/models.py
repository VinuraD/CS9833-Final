import random
from numbers import Number
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from torch_geometric.nn import DenseGCNConv, DenseGraphConv, DenseGINConv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GraphCFE(nn.Module):
    def __init__(self, init_params, args):
        super(GraphCFE, self).__init__()
        self.vae_type = init_params['vae_type']  # graphVAE
        self.x_dim = init_params['x_dim']
        self.h_dim = args.dim_h
        self.z_dim = args.dim_z
        self.u_dim = 1 # init_params['u_dim']
        self.dropout = args.dropout
        self.max_num_nodes = init_params['max_num_nodes']
        self.encoder_type = 'gcn'
        self.graph_pool_type = 'mean'
        self.disable_u = args.disable_u

        if self.disable_u:
            self.u_dim = 0
            print('disable u!')
        if self.encoder_type == 'gcn':
            self.graph_model = DenseGCNConv(self.x_dim, self.h_dim)
        elif self.encoder_type == 'graphConv':
            self.graph_model = DenseGraphConv(self.x_dim, self.h_dim)

        # prior
        self.prior_mean = MLP(self.u_dim, self.z_dim, self.h_dim, n_layers=1, activation='none', slope=.1, device=device)
        self.prior_var = nn.Sequential(MLP(self.u_dim, self.z_dim, self.h_dim, n_layers=1, activation='none', slope=.1, device=device), nn.Sigmoid())

        # encoder
        self.encoder_mean = nn.Sequential(nn.Linear(self.h_dim + self.u_dim + 1, self.z_dim), nn.BatchNorm1d(self.z_dim), nn.ReLU())
        self.encoder_var = nn.Sequential(nn.Linear(self.h_dim + self.u_dim + 1, self.z_dim), nn.BatchNorm1d(self.z_dim), nn.ReLU(), nn.Sigmoid())

        # decoder
        self.decoder_x = nn.Sequential(nn.Linear(self.z_dim + 1, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.max_num_nodes*self.x_dim))
        self.decoder_a = nn.Sequential(nn.Linear(self.z_dim + 1, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.max_num_nodes*self.max_num_nodes), nn.Sigmoid())
        self.graph_norm = nn.BatchNorm1d(self.h_dim)

    def encoder(self, features, u, adj, y_cf):
        # Q(Z|X,U,A,Y^CF)
        # input: x, u, A, y^cf
        # output: z
        graph_rep = self.graph_model(features, adj)  # n x num_node x h_dim
        graph_rep = self.graph_pooling(graph_rep, self.graph_pool_type)  # n x h_dim
        #graph_rep = self.graph_norm(graph_rep)

        if self.disable_u:
            z_mu = self.encoder_mean(torch.cat((graph_rep, y_cf), dim=1))
            z_logvar = self.encoder_var(torch.cat((graph_rep, y_cf), dim=1))
        else:
            z_mu = self.encoder_mean(torch.cat((graph_rep, u, y_cf), dim=1))
            z_logvar = self.encoder_var(torch.cat((graph_rep, u, y_cf), dim=1))

        return z_mu, z_logvar

    def get_represent(self, features, u, adj, y_cf):
        u_onehot = u
        # encoder
        z_mu, z_logvar = self.encoder(features, u_onehot, adj, y_cf)

        return z_mu, z_logvar

    def decoder(self, z, y_cf, u):
        if self.disable_u:
            adj_reconst = self.decoder_a(torch.cat((z, y_cf), dim=1)).view(-1, self.max_num_nodes,
                                                                              self.max_num_nodes)
        else:
            adj_reconst = self.decoder_a(torch.cat((z, y_cf), dim=1)).view(-1, self.max_num_nodes, self.max_num_nodes)

        features_reconst = self.decoder_x(torch.cat((z, y_cf), dim=1)).view(-1, self.max_num_nodes, self.x_dim)
        return features_reconst, adj_reconst

    def graph_pooling(self, x, type='mean'):
        if type == 'max':
            out, _ = torch.max(x, dim=1, keepdim=False)
        elif type == 'sum':
            out = torch.sum(x, dim=1, keepdim=False)
        elif type == 'mean':
            out = torch.sum(x, dim=1, keepdim=False)
        return out

    def prior_params(self, u):  # P(Z|U)
        if self.disable_u:
            z_u_mu = torch.zeros((len(u),self.h_dim)).to(device)
            z_u_logvar = torch.ones((len(u),self.h_dim)).to(device)
        else:
            z_u_logvar = self.prior_var(u)
            z_u_mu = self.prior_mean(u)
        return z_u_mu, z_u_logvar

    def reparameterize(self, mu, logvar):
        '''
        compute z = mu + std * epsilon
        '''
        if self.training:
            # compute the standard deviation from logvar
            std = torch.exp(0.5 * logvar)
            # sample epsilon from a normal distribution with mean 0 and
            # variance 1
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def score(self):
        return

    def forward(self, features, u, adj, y_cf):
        u_onehot = u

        z_u_mu, z_u_logvar = self.prior_params(u_onehot)
        # encoder
        z_mu, z_logvar = self.encoder(features, u_onehot, adj, y_cf)
        # reparameterize
        z_sample = self.reparameterize(z_mu, z_logvar)
        # decoder
        features_reconst, adj_reconst = self.decoder(z_sample, y_cf, u_onehot)

        return {'z_mu': z_mu, 'z_logvar': z_logvar, 'adj_permuted': adj, 'features_permuted': features,
                'adj_reconst': adj_reconst, 'features_reconst': features_reconst, 'z_u_mu': z_u_mu, 'z_u_logvar': z_u_logvar}


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none', slope=.1, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h


def _gin_layers_for_dataset(dataset: str, default_layers: int = 5) -> int:
    """
    Use a deeper GIN for MalNetTiny to mirror the main training setup.
    """
    name = (dataset or '').lower()
    if name.startswith('malnet'):
        return 6
    return default_layers


class Graph_pred_model(nn.Module):
    """
    Dense variant of the GIN architecture used for CLEAR prediction models.
    Mirrors the sparse GIN structure but operates on padded dense tensors.
    """

    def __init__(self,
                 x_dim,
                 h_dim,
                 n_out,
                 max_num_nodes,
                 dataset='synthetic',
                 num_layers=None,
                 num_mlp_layers=2,
                 dropout=0.5):
        super(Graph_pred_model, self).__init__()
        self.dataset = dataset
        self.max_num_nodes = max_num_nodes
        self.input_dim = x_dim
        self.hidden_dim = h_dim
        self.num_layers = _gin_layers_for_dataset(dataset, default_layers=num_layers or 5)
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        class DenseGINMLP(nn.Module):
            def __init__(self, layers, in_dim, hid_dim, out_dim):
                super().__init__()
                self.layers = nn.ModuleList()
                if layers == 1:
                    self.layers.append(nn.Linear(in_dim, out_dim))
                else:
                    for idx in range(layers - 1):
                        input_dim = in_dim if idx == 0 else hid_dim
                        self.layers.append(nn.Sequential(
                            nn.Linear(input_dim, hid_dim),
                            nn.LayerNorm(hid_dim),
                            nn.ReLU()
                        ))
                    self.layers.append(nn.Linear(hid_dim, out_dim))

            def forward(self, x):
                h = x
                for layer in self.layers:
                    h = layer(h)
                return h

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = DenseGINMLP(num_mlp_layers, x_dim, h_dim, h_dim)
            else:
                mlp = DenseGINMLP(num_mlp_layers, h_dim, h_dim, h_dim)
            self.convs.append(DenseGINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(h_dim))

        self.lin_predictors = nn.ModuleList()
        for layer in range(self.num_layers):
            if layer == 0:
                self.lin_predictors.append(nn.Linear(x_dim, n_out))
            else:
                self.lin_predictors.append(nn.Linear(h_dim, n_out))

    def _prepare_inputs(self, x, adj):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        if x.size(0) != adj.size(0):
            raise ValueError('Feature and adjacency batch sizes must match.')

        # if self.dataset in {'synthetic', 'community', 'imdb_b'}:
        # x = torch.ones_like(x, device=x.device)
        # elif self.dataset == 'ogbg_molhiv':
        #     x = x.clone()
        #     x[:, :, 2:] = 0.
        #     x[:, :, 0] = 0.
        return x, adj

    @staticmethod
    def _pool(h):
        # Sum pooling over padded nodes imitates global_add_pool for dense tensors.
        return torch.sum(h, dim=1)

    def forward(self, x, adj):
        x, adj = self._prepare_inputs(x, adj)
        batch_size = x.size(0)

        hidden_rep = [x]
        h = x
        for layer in range(self.num_layers - 1):
            h = self.convs[layer](h, adj)
            h = F.relu(h)
            h = h.view(-1, self.hidden_dim)
            h = self.batch_norms[layer](h)
            h = h.view(batch_size, self.max_num_nodes, self.hidden_dim)
            h = F.dropout(h, self.dropout, training=self.training)
            hidden_rep.append(h)

        score = 0
        for layer, h in enumerate(hidden_rep):
            pooled_h = self._pool(h)
            score += F.dropout(self.lin_predictors[layer](pooled_h), self.dropout, training=self.training)

        rep_graph = self._pool(hidden_rep[-1])
        return {'y_pred': score, 'rep_graph': rep_graph}
