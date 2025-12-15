
import argparse
import glob
import os
import pickle
import shutil
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, to_dense_adj

try:
    from .clear_utils import data_preprocessing as dpp  # type: ignore
except ImportError:
    dpp = None

from .clear_utils import data_sampler, models, utils

sys.path.append('../')
from dataset_utils import (
    dataset_root,
    is_malnet_tiny,
    load_dataset_splits,
    load_malnet_tiny_filtered_splits,
    load_tud_dataset,
    MALNET_TINY_MAX_NODES,
)

font_sz = 28

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Graph counterfactual explanation generation')
    parser.add_argument('--batch_size', type=int, default=500, metavar='N',
                        help='input batch size for training (default: 500)')
    parser.add_argument('--num_workers', type=int, default=0, metavar='N')
    parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                        help='number of epochs to train (default: 2000)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--trn_rate', type=float, default=0.6, help='training data ratio')
    parser.add_argument('--tst_rate', type=float, default=0.2, help='test data ratio')

    parser.add_argument('--lamda', type=float, default=200, help='weight for CFE loss')
    parser.add_argument('--kl_weight', type=float, default=1.0, help='weight for KL loss')
    parser.add_argument('--disable_u', type=int, default=0, help='disable u in VAE')
    parser.add_argument('--dim_z', type=int, default=16, metavar='N', help='dimension of z')
    parser.add_argument('--dim_h', type=int, default=16, metavar='N', help='dimension of h')
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--dataset', default='NCI1', help='dataset to use',
                        choices=['NCI1', 'IMDB-BINARY'])
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay')
    
    parser.add_argument('--prediction_model_type', type=str, default='GIN',)

    parser.add_argument('--experiment_type', default='train', choices=['train', 'test', 'baseline'],
                        help='train: train CLEAR model; test: load CLEAR from file; baseline: run a baseline')
    return parser


parser = build_arg_parser()


def get_default_args() -> argparse.Namespace:
    """
    Return a Namespace populated with the default CLEAR arguments.
    """
    return parser.parse_args([])


def configure_environment(args: argparse.Namespace) -> None:
    """
    Configure random seeds for reproducibility.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and bool(getattr(args, 'cuda', True)):
        torch.cuda.manual_seed(args.seed)

# select gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OGGB={} #to be implemented


def _prepare_clear_data_bundle(dataset_name: str) -> dict:
    if is_malnet_tiny(dataset_name):
        dataset, splits = load_malnet_tiny_filtered_splits(
            max_nodes=MALNET_TINY_MAX_NODES, seed=42)
    else:
        dataset = load_tud_dataset(dataset_name)
        splits = load_dataset_splits(dataset_name, len(dataset))
    train_indices = np.array(splits['train'], dtype=np.int64)
    rng = np.random.default_rng(0)
    rng.shuffle(train_indices)
    test_indices = np.array(splits['test'], dtype=np.int64)

    if len(train_indices) < 2:
        raise ValueError('CLEAR training requires at least two training graphs to create a validation split.')

    val_size = max(1, int(0.1 * len(train_indices)))
    val_indices = train_indices[:val_size]
    train_indices_refined = train_indices[val_size:]
    if len(train_indices_refined) == 0:
        train_indices_refined = train_indices
        val_indices = train_indices[:1]

    adj_all: List[np.ndarray] = []
    features_all: List[np.ndarray] = []
    u_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    max_num_nodes = 0

    for data in dataset:
        num_nodes = data.num_nodes
        max_num_nodes = max(max_num_nodes, num_nodes)
        dense_adj = to_dense_adj(data.edge_index, max_num_nodes=num_nodes).squeeze(0)
        dense_adj = torch.maximum(dense_adj, dense_adj.T)
        dense_adj.fill_diagonal_(1.0)
        adj_all.append(dense_adj.cpu().numpy().astype(np.float32))

        if data.x is None:
            features = torch.ones((num_nodes, 1), dtype=torch.float32)
        else:
            features = data.x.to(torch.float32)
        features_all.append(features.cpu().numpy())

        if data.y is None:
            raise ValueError('Graph labels are required for CLEAR training.')
        labels_all.append(data.y.view(-1).cpu().numpy().astype(np.int64))
        u_all.append(np.zeros((1,), dtype=np.float32))

    graph_data = data_sampler.GraphData(adj_all, features_all, u_all, labels_all, max_num_nodes, padded=True)

    data_bundle = {
        'data': graph_data,
        'idx_train_list': [train_indices_refined.copy()],
        'idx_val_list': [val_indices.copy()],
        'idx_test_list': [test_indices.copy()],
    }
    return data_bundle


class CLEARExplainer:
    """
    Lightweight wrapper around the CLEAR counterfactual generator to integrate with the
    generate_cf.py workflow. The class mirrors the API of other explainers in the project,
    exposing an `explain` method that accepts a PyG Batch (containing a single graph) and
    returns a counterfactual `Data` object.
    """

    @classmethod
    def weight_filename(cls,
                        dataset_name: str,
                        exp_id: int = 0,
                        epoch: int = 900,
                        disable_u: bool = False) -> str:
        variant = 'VAE' if disable_u else 'CLEAR'
        return f'weights_graphCFE_{variant}_{dataset_name}_exp{exp_id}_epoch{epoch}.pt'

    @classmethod
    def weight_path(cls,
                    model_root: str,
                    dataset_name: str,
                    exp_id: int = 0,
                    epoch: int = 900,
                    disable_u: bool = False) -> str:
        return os.path.join(model_root, cls.weight_filename(dataset_name=dataset_name,
                                                            exp_id=exp_id,
                                                            epoch=epoch,
                                                            disable_u=disable_u))

    def __init__(self,
                 dataset_name: str,
                 train_dataset,
                 model_root: str = './trained_model/clear',
                 exp_id: int = 0,
                 epoch: int = 900,
                 disable_u: bool = False,
                 edge_threshold: float = 0.5,
                 device: Optional[torch.device] = None) -> None:
        self.dataset_name = dataset_name
        self.model_root = model_root
        self.exp_id = exp_id
        self.epoch = epoch
        self.edge_threshold = edge_threshold
        self.disable_u = disable_u
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Derive dataset-specific metadata using the same preprocessing pipeline as CLEAR training.
        metadata = self._load_clear_metadata(dataset_name, train_dataset)
        self.max_num_nodes = metadata['max_num_nodes']
        self.x_dim = metadata['x_dim']
        self.num_classes = metadata['num_classes']

        # Re-use CLEAR defaults unless explicitly overridden.
        self.args = get_default_args()
        self.args.dataset = dataset_name
        self.args.disable_u = bool(disable_u)
        configure_environment(self.args)

        init_params = {
            'vae_type': 'graphVAE',
            'x_dim': self.x_dim,
            'u_dim': 1 if not disable_u else 0,
            'max_num_nodes': self.max_num_nodes,
        }
        self.model = models.GraphCFE(init_params=init_params, args=self.args).to(self.device)
        self.model.eval()

        self.variant = 'VAE' if disable_u else 'CLEAR'
        self.weights_path = self.weight_path(model_root=model_root,
                                             dataset_name=dataset_name,
                                             exp_id=exp_id,
                                             epoch=epoch,
                                             disable_u=disable_u)
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"CLEAR generator weights not found at {self.weights_path}. "
                "Please train or place the pretrained weights before using the CLEAR explainer."
            )
        state_dict = torch.load(self.weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def _construct_inputs(self, graph: Data):
        num_nodes = graph.num_nodes
        if num_nodes > self.max_num_nodes:
            raise ValueError(
                f"Graph with {num_nodes} nodes exceeds CLEAR max nodes ({self.max_num_nodes}). "
                "Please ensure the CLEAR generator was trained with a sufficient padding size."
            )

        if graph.x is None:
            x = torch.ones((graph.num_nodes, self.x_dim), device=self.device)
        else:
            x = graph.x.to(self.device)
        edge_index = graph.edge_index.to(self.device)

        dense_adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
        dense_adj = torch.maximum(dense_adj, dense_adj.T)
        dense_adj.fill_diagonal_(1.0)

        adj_input = torch.zeros((1, self.max_num_nodes, self.max_num_nodes), device=self.device)
        adj_input[:, :num_nodes, :num_nodes] = dense_adj

        features_input = torch.zeros((1, self.max_num_nodes, self.x_dim), device=self.device)
        features_input[:, :num_nodes] = x

        u_tensor = torch.zeros((1, 1), device=self.device)

        y_cf = self._get_cf_target(graph).view(1, -1).float().to(self.device)
        return features_input, u_tensor, adj_input, y_cf, num_nodes

    @staticmethod
    def _get_cf_target(graph: Data) -> torch.Tensor:
        if hasattr(graph, 'targets'):
            target = graph.targets
        else:
            y = graph.y.view(-1)
            if y.numel() != 1:
                raise ValueError("CLEAR explainer expects a single-graph batch with a scalar label.")
            target = 1 - y
        return target.view(-1)

    def explain(self, batch, oracle) -> Optional[Data]:
        """
        Generate a counterfactual graph using the pretrained CLEAR generator.
        """
        if hasattr(batch, "to_data_list"):
            graph = batch.to_data_list()[0]
        else:
            graph = batch

        graph = graph.to(self.device)
        features_input, u_tensor, adj_input, y_cf, num_nodes = self._construct_inputs(graph)

        with torch.no_grad(): #generate cf
            outputs = self.model(features_input, u_tensor, adj_input, y_cf)

        adj_reconst = outputs['adj_reconst'][0, :num_nodes, :num_nodes]
        features_reconst = outputs['features_reconst'][0, :num_nodes]

        adj_reconst = torch.clamp(adj_reconst, min=0.0, max=1.0)
        adj_binary = torch.bernoulli(adj_reconst)
        adj_binary.fill_diagonal_(0.0)

        edge_index_cf, _ = dense_to_sparse(adj_binary)

        cf_targets = y_cf.view(-1).long()

        cf_graph = Data(
            x=features_reconst.detach().cpu(),
            edge_index=edge_index_cf.detach().cpu(),
            y=cf_targets.cpu()
        )
        cf_graph.targets = cf_targets.cpu()
        cf_graph.num_nodes = num_nodes
        return cf_graph

    @staticmethod
    def _metadata_from_pyg_dataset(train_dataset) -> dict:
        max_num_nodes = max(int(data.num_nodes) for data in train_dataset)
        x_dim = getattr(train_dataset, 'num_node_features', 0)
        if x_dim == 0:
            sample_graph = train_dataset[0]
            x_dim = sample_graph.x.shape[1] if sample_graph.x is not None else 1
        num_classes = getattr(train_dataset, 'num_classes', 0)
        if not num_classes:
            labels = set()
            for graph in train_dataset:
                if graph.y is not None:
                    labels.add(int(graph.y.view(-1)[0]))
            num_classes = max(1, len(labels))
        return {'max_num_nodes': max_num_nodes, 'x_dim': x_dim, 'num_classes': num_classes}

    @staticmethod
    def _load_clear_metadata(dataset_name: str, train_dataset) -> dict:
        try:
            data_bundle = _prepare_clear_data_bundle(dataset_name)
            graph_data = data_bundle['data']
            x_dim = graph_data.feature_all[0].shape[1]
            labels_array = np.array([int(np.array(lbl).reshape(-1)[0]) for lbl in graph_data.labels_all], dtype=np.int64)
            num_classes = len(np.unique(labels_array))
            return {
                'max_num_nodes': graph_data.max_num_nodes,
                'x_dim': x_dim,
                'num_classes': num_classes
            }
        except Exception as exc:
            print(f'[CLEARExplainer] Warning: failed to load CLEAR metadata via preprocessing ({exc}). '
                  'Falling back to statistics computed from the provided training dataset.')
            return CLEARExplainer._metadata_from_pyg_dataset(train_dataset)


def _train_clear_prediction_model(data_bundle: dict,
                                  dataset_name: str,
                                  prediction_dir: str,
                                  device: torch.device,
                                  lr: float = 1e-3,
                                  weight_decay: float = 1e-5,
                                  epochs: int = 200) -> None:
    graph_data = data_bundle['data']
    idx_train = data_bundle['idx_train_list'][0]
    idx_val = data_bundle['idx_val_list'][0]
    idx_test = data_bundle['idx_test_list'][0]

    x_dim = graph_data.feature_all[0].shape[1]
    labels_array = np.array([int(np.array(lbl).reshape(-1)[0]) for lbl in graph_data.labels_all], dtype=np.int64)
    num_class = len(np.unique(labels_array))
    max_num_nodes = graph_data.max_num_nodes

    model = models.Graph_pred_model(x_dim, 64, num_class, max_num_nodes, dataset_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    batch_size = min(256, max(1, len(idx_train)))
    train_loader = utils.select_dataloader(graph_data, idx_train, batch_size=batch_size, num_workers=0)
    val_loader = utils.select_dataloader(graph_data, idx_val, batch_size=batch_size, num_workers=0)
    test_loader = utils.select_dataloader(graph_data, idx_test, batch_size=batch_size, num_workers=0)

    def run_epoch(loader, train: bool):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        if train:
            model.train()
        else:
            model.eval()
        for batch in loader:
            features = torch.as_tensor(batch['features'], dtype=torch.float32, device=device)
            adj = torch.as_tensor(batch['adj'], dtype=torch.float32, device=device)
            labels = torch.as_tensor(batch['labels'], dtype=torch.long, device=device).view(-1)

            if train:
                optimizer.zero_grad()

            logits = model(features, adj)['y_pred']
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        mean_loss = total_loss / max(1, total_samples)
        accuracy = total_correct / max(1, total_samples)
        return mean_loss, accuracy

    best_state = None
    best_accuracy = 0.0

    for epoch in range(epochs):
        run_epoch(train_loader, train=True)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(val_loader, train=False)
        if val_acc >= best_accuracy:
            best_accuracy = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = model.state_dict()

    model.load_state_dict(best_state if isinstance(best_state, dict) else best_state.state_dict())
    with torch.no_grad():
        _, test_acc = run_epoch(test_loader, train=False)

    os.makedirs(prediction_dir, exist_ok=True)
    weight_path = os.path.join(prediction_dir, f'weights_graphPred__{dataset_name}.pt')
    torch.save(best_state, weight_path)
    print(f'Saved CLEAR predictor to {weight_path} (val acc: {best_accuracy:.4f}).')
    print(f'CLEAR predictor test accuracy: {test_acc:.4f}')
    return test_acc


def train_clear_model(dataset_name: str,
                      disable_u: bool = False,
                      exp_id: int = 0,
                      target_epoch: int = 900,
                      model_root: str = './trained_model/clear',
                      device: Optional[torch.device] = None) -> Tuple[str, int]:
    """
    Train the CLEAR generator if weights are missing and copy the checkpoint to the expected location.
    Returns the path to the checkpoint.
    """
    args = get_default_args()
    data_bundle = _prepare_clear_data_bundle(dataset_name)

    prediction_dir = os.path.join(model_root, 'prediction')
    os.makedirs(prediction_dir, exist_ok=True)
    prediction_weights = os.path.join(prediction_dir, f'weights_graphPred__{dataset_name}.pt')
    if not os.path.exists(prediction_weights):
        print(f'Prediction model not found at {prediction_weights}. Training a new one for CLEAR...')
        pred_device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _train_clear_prediction_model(data_bundle, dataset_name, prediction_dir, pred_device)
    
    args.dataset = dataset_name
    args.disable_u = bool(disable_u)
    args.epochs = max(args.epochs, target_epoch)
    args.batch_size = min(args.batch_size, max(1, len(data_bundle['idx_train_list'][0])))
    configure_environment(args)

    module_dir = os.path.dirname(__file__)
    models_save_dir = os.path.abspath(os.path.join(module_dir, '..', 'models_save'))
    os.makedirs(models_save_dir, exist_ok=True)

    run_clear(args, 'train', save_model=True, data_bundle=data_bundle)

    variant = 'VAE' if disable_u else 'CLEAR'
    expected_name = CLEARExplainer.weight_filename(dataset_name=dataset_name,
                                                   exp_id=exp_id,
                                                   epoch=target_epoch,
                                                   disable_u=disable_u)

    candidate_path = os.path.join(models_save_dir, expected_name)

    if not os.path.exists(candidate_path):
        pattern = os.path.join(models_save_dir,
                               f'weights_graphCFE_{variant}_{dataset_name}_exp{exp_id}_epoch*.pt')
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"Expected CLEAR checkpoint matching `{expected_name}` in {models_save_dir}, "
                "but none was found after training. Check training configuration."
            )
        candidate_path = matches[-1]
        epoch_str = os.path.splitext(candidate_path)[0].split('_epoch')[-1]
        try:
            target_epoch = int(epoch_str)
        except ValueError:
            raise ValueError(f"Unable to parse epoch from checkpoint name: {candidate_path}") from None

    os.makedirs(model_root, exist_ok=True)
    destination_path = CLEARExplainer.weight_path(model_root=model_root,
                                                  dataset_name=dataset_name,
                                                  exp_id=exp_id,
                                                  epoch=target_epoch,
                                                  disable_u=disable_u)
    if os.path.abspath(candidate_path) != os.path.abspath(destination_path):
        shutil.copyfile(candidate_path, destination_path)
    return destination_path, target_epoch


def add_list_in_dict(key, dict, elem):
    if key not in dict:
        dict[key] = [elem]
    else:
        dict[key].append(elem)
    return dict

def distance_feature(feat_1, feat_2):
    pdist = nn.PairwiseDistance(p=2)
    output = pdist(feat_1, feat_2) /4
    return output

def distance_graph_prob(adj_1, adj_2_prob):
    dist = F.binary_cross_entropy(adj_2_prob, adj_1)
    return dist

def proximity_feature(feat_1, feat_2, type='cos'):
    if type == 'cos':
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        output = cos(feat_1, feat_2)
        output = torch.mean(output)
    return output


def compute_loss(params):
    model, pred_model, z_mu, z_logvar, adj_permuted, features_permuted, adj_reconst, features_reconst, \
    adj_input, features_input, y_cf, z_u_mu, z_u_logvar, z_mu_cf, z_logvar_cf = params['model'], params['pred_model'], params['z_mu'], \
        params['z_logvar'], params['adj_permuted'], params['features_permuted'], params['adj_reconst'], params['features_reconst'], \
        params['adj_input'], params['features_input'], params['y_cf'], params['z_u_mu'], params['z_u_logvar'], params['z_mu_cf'], params['z_logvar_cf']

    # kl loss
    loss_kl = 0.5 * (((z_u_logvar - z_logvar) + ((z_logvar.exp() + (z_mu - z_u_mu).pow(2)) / z_u_logvar.exp())) - 1)
    loss_kl = torch.mean(loss_kl)

    # similarity loss
    size = len(features_permuted)
    dist_x = torch.mean(distance_feature(features_permuted.view(size, -1), features_reconst.view(size, -1)))
    dist_a = distance_graph_prob(adj_permuted, adj_reconst)

    beta = 15

    loss_sim = beta * dist_x + 10 * dist_a

    # CFE loss
    y_pred = pred_model(features_reconst, adj_reconst)['y_pred']  # n x num_class
    loss_cfe = F.nll_loss(F.log_softmax(y_pred, dim=-1), y_cf.view(-1).long())

    # rep loss
    if z_mu_cf is None:
        loss_kl_cf = 0.0
    else:
        loss_kl_cf = 0.5 * (((z_logvar_cf - z_logvar) + ((z_logvar.exp() + (z_mu - z_mu_cf).pow(2)) / z_logvar_cf.exp())) - 1)
        loss_kl_cf = torch.mean(loss_kl_cf)

    loss = 1. * loss_sim + 1 * loss_kl + 1.0 * loss_cfe

    loss_results = {'loss': loss, 'loss_kl': loss_kl, 'loss_sim': loss_sim, 'loss_cfe': loss_cfe, 'loss_kl_cf':loss_kl_cf}
    return loss_results


def train(params):
    epochs, pred_model, model, optimizer, y_cf_all, train_loader, val_loader, test_loader, exp_i, dataset, metrics, variant = \
        params['epochs'], params['pred_model'], params['model'], params['optimizer'], params['y_cf'],\
        params['train_loader'], params['val_loader'], params['test_loader'], params['exp_i'], params['dataset'], params['metrics'], params['variant']
    save_model = params['save_model'] if 'save_model' in params else True
    models_save_dir = params.get('models_save_dir', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models_save')))
    print("start training!")

    time_begin = time.time()
    best_loss = 100000

    for epoch in range(epochs + 1):
        model.train()

        loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf = 0.0, 0.0, 0.0, 0.0, 0.0
        batch_num = 0
        for batch_idx, data in enumerate(train_loader):
            batch_num += 1

            features = data['features'].float().to(device)
            adj = data['adj'].float().to(device)
            u = data['u'].float().to(device)
            orin_index = data['index']
            y_cf = y_cf_all[orin_index]

            optimizer.zero_grad()

            # forward pass
            model_return = model(features, u, adj, y_cf)

            # z_cf
            z_mu_cf, z_logvar_cf = model.get_represent(model_return['features_reconst'], u, model_return['adj_reconst'], y_cf)

            # compute loss
            loss_params = {'model': model, 'pred_model': pred_model, 'adj_input': adj, 'features_input': features, 'y_cf': y_cf, 'z_mu_cf': z_mu_cf, 'z_logvar_cf':z_logvar_cf}
            loss_params.update(model_return)

            loss_results = compute_loss(loss_params)
            loss_batch, loss_kl_batch, loss_sim_batch, loss_cfe_batch, loss_kl_batch_cf = loss_results['loss'], loss_results['loss_kl'], loss_results['loss_sim'], loss_results['loss_cfe'], loss_results['loss_kl_cf']
            loss += loss_batch
            loss_kl += loss_kl_batch
            loss_sim += loss_sim_batch
            loss_cfe += loss_cfe_batch
            loss_kl_cf += loss_kl_batch_cf

        # backward propagation
        loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf = loss / batch_num, loss_kl/batch_num, loss_sim/batch_num, loss_cfe/batch_num, loss_kl_cf/batch_num

        alpha = 5

        if epoch < 450:
            ((loss_sim + loss_kl + 0* loss_cfe)/ batch_num).backward()
        else:
            ((loss_sim + loss_kl + alpha * loss_cfe)/ batch_num).backward()
        optimizer.step()

        # evaluate
        if epoch % 100 == 0:
            model.eval()
            eval_params_val = {'model': model, 'data_loader': val_loader, 'pred_model': pred_model, 'y_cf': y_cf_all, 'dataset': dataset, 'metrics': metrics}
            eval_params_tst = {'model': model, 'data_loader': test_loader, 'pred_model': pred_model, 'y_cf': y_cf_all, 'dataset': dataset, 'metrics': metrics}
            eval_results_val = test(eval_params_val)
            eval_results_tst = test(eval_params_tst)
            val_loss, val_loss_kl, val_loss_sim, val_loss_cfe = eval_results_val['loss'], eval_results_val['loss_kl'], eval_results_val['loss_sim'], eval_results_val['loss_cfe']

            metrics_results_val = ""
            metrics_results_tst = ""
            for k in metrics:
                metrics_results_val += f"{k}_val: {eval_results_val[k]:.4f} | "
                metrics_results_tst += f"{k}_tst: {eval_results_tst[k]:.4f} | "

            print(f"[Train] Epoch {epoch}: train_loss: {(loss):.4f} |" +
                  metrics_results_val + metrics_results_tst +
                  f"time: {(time.time() - time_begin):.4f} |")

            # save
            if save_model:
                if epoch % 300 == 0 and epoch > 450:
                    CFE_model_path = os.path.join(models_save_dir, f'weights_graphCFE_{variant}_{dataset}_exp{exp_i}_epoch{epoch}.pt')
                    torch.save(model.state_dict(), CFE_model_path)
                    print('saved CFE model in: ', CFE_model_path)

        # if epoch % 2000 == 0 and args.dataset == 'imdb_m':
        #     x_range = [0.0, 6]
        #     y_range = [0.0, 5]
        #
        #     if epoch == 0:
        #         title = args.dataset + 'origin'
        #         features_permuted = model_return['features_permuted']
        #         adj_permuted = model_return['adj_permuted']
        #
        #         size = len(adj_permuted)
        #         num_nodes = adj_permuted.shape[-1]
        #         ave_x0 = torch.mean(features_permuted[:, :, 0], dim=-1)
        #         ave_degree = (torch.sum(adj_permuted.reshape(size, -1), dim=-1) - num_nodes) / (2 * num_nodes)  # size
        #         plot.draw_scatter(ave_degree.detach().cpu().numpy(), ave_x0.detach().cpu().numpy(),
        #                           c=u.detach().cpu().numpy(),
        #                           x_label='degree',
        #                           y_label='x0',  # title='Community original',
        #                           alpha=0.5,
        #                           x_range=x_range,
        #                           y_range=y_range,
        #                           save_file='../exp_results/' + title + '.pdf'
        #                           )
        #         continue
        #
        #     features_reconst = model_return['features_reconst']
        #     ave_x0_cf = torch.mean(features_reconst[:, :, 0], dim=-1)
        #     adj_reconst = model_return['adj_reconst']
        #     adj_reconst_binary = torch.bernoulli(adj_reconst)
        #
        #     ave_degree_cf_prob = (torch.sum(adj_reconst.reshape(size, -1), dim=-1) - num_nodes) / (2 * num_nodes)
        #     ave_degree_cf = (torch.sum(adj_reconst_binary.reshape(size, -1), dim=-1) - num_nodes) / (2 * num_nodes)
        #
        #     method = 'VAE' if args.disable_u else 'CLEAR'
        #     title = args.dataset + ' CFE' + ', ' + method + ' epoch' + str(epoch)
        #     plot.draw_scatter(ave_degree_cf.detach().cpu().numpy(), ave_x0_cf.detach().cpu().numpy(),
        #                       c=u.detach().cpu().numpy(),
        #                       x_label='degree',
        #                       y_label='x0', title=None, alpha=0.5,
        #                       x_range=x_range,
        #                       y_range=y_range,
        #                       save_file='../exp_results/' + title + '.pdf'
        #                       )
        #     title = args.dataset + ' CFE' + ', prob, ' + method + ' epoch' + str(epoch)
        #     plot.draw_scatter(ave_degree_cf_prob.detach().cpu().numpy(), ave_x0_cf.detach().cpu().numpy(),
        #                       c=u.detach().cpu().numpy(),
        #                       x_label='degree',
        #                       y_label='x0',
        #                       title=None,
        #                       alpha=0.5,
        #                       x_range=x_range,
        #                       y_range=y_range,
        #                       save_file='../exp_results/' + title + '.pdf')
        #
        # if epoch % 3000 == 0 and args.dataset == 'community':
        #     x_range = [0.0, 4.5]
        #     y_range = [0.0, 4.5]
        #
        #     if epoch == 0:
        #         title = 'Community_original'
        #         n0 = 10
        #         n1 = 10
        #         adj_permuted = model_return['adj_permuted']
        #         size = len(adj_permuted)
        #         ave_degree_0 = (torch.sum(adj_permuted[:, :n0, :n0].reshape(size, -1), dim=-1) - n0) / (2 * n0)  # size
        #         ave_degree_1 = (torch.sum(adj_permuted[:, n0:, n0:].reshape(size, -1), dim=-1) - n1) / (2 * n1)  # size
        #         plot.draw_scatter(ave_degree_0.detach().cpu().numpy(), ave_degree_1.detach().cpu().numpy(),
        #                           c=u.detach().cpu().numpy(),
        #                           x_label='degree of community 1',
        #                           y_label='degree of community 2',
        #                           alpha=0.5,
        #                           x_range=x_range,
        #                           y_range=y_range,
        #                           save_file='../exp_results/' + title + '.pdf'
        #                           )
        #         continue

    return

def test(params):
    model, data_loader, pred_model, y_cf_all, dataset, metrics = params['model'], params['data_loader'], params['pred_model'], params['y_cf'], params['dataset'], params['metrics']
    model.eval()
    pred_model.eval()

    eval_results_all = {k: 0.0 for k in metrics}
    size_all = 0
    loss, loss_kl, loss_sim, loss_cfe = 0.0, 0.0, 0.0, 0.0
    batch_num = 0
    for batch_idx, data in enumerate(data_loader):
        batch_num += 1
        batch_size = len(data['labels'])
        size_all += batch_size

        features = data['features'].float().to(device)
        adj = data['adj'].float().to(device)
        u = data['u'].float().to(device)
        labels = data['labels'].float().to(device)
        orin_index = data['index']
        y_cf = y_cf_all[orin_index]

        model_return = model(features, u, adj, y_cf)
        adj_reconst, features_reconst = model_return['adj_reconst'], model_return['features_reconst']

        adj_reconst_binary = torch.bernoulli(adj_reconst)
        y_cf_pred = pred_model(features_reconst, adj_reconst_binary)['y_pred']
        y_pred = pred_model(features, adj)['y_pred']

        # z_cf
        z_mu_cf, z_logvar_cf = None, None

        # compute loss
        loss_params = {'model': model, 'pred_model': pred_model, 'adj_input': adj, 'features_input': features, 'y_cf': y_cf, 'z_mu_cf': z_mu_cf, 'z_logvar_cf':z_logvar_cf}
        loss_params.update(model_return)

        loss_results = compute_loss(loss_params)
        loss_batch, loss_kl_batch, loss_sim_batch, loss_cfe_batch = loss_results['loss'], loss_results['loss_kl'], \
                                                                    loss_results['loss_sim'], loss_results['loss_cfe']
        loss += loss_batch
        loss_kl += loss_kl_batch
        loss_sim += loss_sim_batch
        loss_cfe += loss_cfe_batch

        # evaluate metrics
        eval_params = model_return.copy()
        eval_params.update({'y_cf': y_cf, 'metrics': metrics, 'y_cf_pred': y_cf_pred, 'dataset': dataset, 'adj_input': adj, 'features_input': features, 'labels':labels, 'u': u, 'y_pred':y_pred})

        eval_results = evaluate(eval_params)
        for k in metrics:
            eval_results_all[k] += (batch_size * eval_results[k])

    for k in metrics:
        eval_results_all[k] /= size_all

    loss, loss_kl, loss_sim, loss_cfe = loss / batch_num, loss_kl / batch_num, loss_sim / batch_num, loss_cfe / batch_num
    eval_results_all['loss'], eval_results_all['loss_kl'], eval_results_all['loss_sim'], eval_results_all['loss_cfe'] = loss, loss_kl, loss_sim, loss_cfe

    return eval_results_all

def evaluate(params):
    adj_permuted, features_permuted, adj_reconst_prob, features_reconst, metrics, dataset, y_cf, y_cf_pred, labels, u, y_pred = \
        params['adj_permuted'], params['features_permuted'], params['adj_reconst'], \
        params['features_reconst'], params['metrics'], params['dataset'], params['y_cf'], params['y_cf_pred'], params['labels'], params['u'], params['y_pred']

    adj_reconst = torch.bernoulli(adj_reconst_prob)
    eval_results = {}
    if 'causality' in metrics:
        score_causal = evaluate_causality(dataset, adj_permuted, features_permuted, adj_reconst, features_reconst,  y_cf, labels, u)
        eval_results['causality'] = score_causal
    if 'proximity' in metrics or 'proximity_x' in metrics or 'proximity_a' in metrics:
        score_proximity, dist_x, dist_a = evaluate_proximity(dataset, adj_permuted, features_permuted, adj_reconst_prob, adj_reconst, features_reconst)
        eval_results['proximity'] = score_proximity
        eval_results['proximity_x'] = dist_x
        eval_results['proximity_a'] = dist_a
    if 'validity' in metrics:
        score_valid = evaluate_validity(y_cf, y_cf_pred)
        eval_results['validity'] = score_valid
    if 'correct' in metrics:
        score_correct = evaluate_correct(dataset, adj_permuted, features_permuted, adj_reconst, features_reconst, y_cf, labels, y_cf_pred, y_pred)
        eval_results['correct'] = score_correct

    return eval_results

def evaluate_validity(y_cf, y_cf_pred):
    y_cf_pred_binary = F.softmax(y_cf_pred, dim=-1)
    y_cf_pred_binary = y_cf_pred_binary.argmax(dim=1).view(-1,1)
    y_eq = torch.where(y_cf == y_cf_pred_binary, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))
    score_valid = torch.mean(y_eq)
    return score_valid

def evaluate_causality(dataset, adj_permuted, features_permuted, adj_reconst, features_reconst, y_cf, labels, u):
    score_causal = 0.0
    if dataset == 'synthetic' or dataset == 'imdb_m':
        size = len(features_permuted)
        max_num_nodes = adj_reconst.shape[-1]

        # Constraint
        ave_degree = (torch.sum(adj_permuted.view(size, -1), dim=-1) - max_num_nodes) / (2 * max_num_nodes) # size
        ave_degree_cf = (torch.sum(adj_reconst.view(size, -1), dim=-1) - max_num_nodes) / (2 * max_num_nodes)
        ave_x0 = torch.mean(features_permuted[:, :, 0], dim=-1)  # size
        ave_x0_cf = torch.mean(features_reconst[:, :, 0], dim=-1)  # size

        count_good = torch.where(
            (((ave_degree > ave_degree_cf) & (ave_x0 > ave_x0_cf)) |
             ((ave_degree == ave_degree_cf) & (ave_x0 == ave_x0_cf)) |
             ((ave_degree < ave_degree_cf) & (ave_x0 < ave_x0_cf))), torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))

        score_causal = torch.mean(count_good)

    elif dataset == 'ogbg_molhiv':
        ave_x0 = torch.mean(features_permuted[:, :, 0], dim=-1)  # size
        ave_x0_cf = torch.mean(features_reconst[:, :, 0], dim=-1)  # size
        ave_x1 = torch.mean(features_permuted[:, :, 1], dim=-1)  # size
        ave_x1_cf = torch.mean(features_reconst[:, :, 1], dim=-1)  # size

        count_good = torch.where(
            (((ave_x0 > ave_x0_cf) & (ave_x1 > ave_x1_cf)) |
             ((ave_x0 == ave_x0_cf) & (ave_x1 == ave_x1_cf)) |
             ((ave_x0 < ave_x0_cf) & (ave_x1 < ave_x1_cf))), torch.tensor(1.0).to(device),
            torch.tensor(0.0).to(device))
        score_causal = torch.mean(count_good)

    elif dataset == 'community':
        size = len(features_permuted)
        max_num_nodes = adj_reconst.shape[-1]

        # Constraint
        n0 = int(max_num_nodes/2)
        n1 = max_num_nodes - n0

        ave_degree_0 = (torch.sum(adj_permuted[:, :n0, :n0].reshape(size, -1), dim=-1) - n0) / (2 * n0)  # size
        ave_degree_cf_0 = (torch.sum(adj_reconst[:, :n0, :n0].reshape(size, -1), dim=-1) - n0) / (2 * n0)
        ave_degree_1 = (torch.sum(adj_permuted[:, n0:, n0:].reshape(size, -1), dim=-1) - n1) / (2 * n1)  # size
        ave_degree_cf_1 = (torch.sum(adj_reconst[:, n0:, n0:].reshape(size, -1), dim=-1) - n1) / (2 * n1)

        max_dg = ave_degree_1.max().tile(len(ave_degree_1))
        min_dg = ave_degree_1.min().tile(len(ave_degree_1))

        count_good = torch.where(
            (((ave_degree_0 > ave_degree_cf_0) & (((ave_degree_1 < max_dg) & (ave_degree_1 < ave_degree_cf_1)) | (ave_degree_1 == max_dg))) |
             ((ave_degree_0 == ave_degree_cf_0) & (ave_degree_1 == ave_degree_cf_1)) |
             ((ave_degree_0 < ave_degree_cf_0) & (((ave_degree_1 > min_dg) & (ave_degree_1 > ave_degree_cf_1)) | (ave_degree_1 == min_dg)))), torch.tensor(1.0).to(device),
            torch.tensor(0.0).to(device))

        score_causal = torch.mean(count_good)

    return score_causal

def evaluate_proximity(dataset, adj_permuted, features_permuted, adj_reconst_prob, adj_reconst, features_reconst):
    size = len(features_permuted)
    dist_x = torch.mean(distance_feature(features_permuted.view(size, -1), features_reconst.view(size, -1)))
    dist_a = distance_graph_prob(adj_permuted, adj_reconst_prob)
    score = dist_x + dist_a

    proximity_x = proximity_feature(features_permuted, features_reconst, 'cos')

    acc_a = (adj_permuted == adj_reconst).float().mean()
    return score, proximity_x, acc_a

def evaluate_correct(dataset, adj_permuted, features_permuted, adj_reconst, features_reconst, y_cf, labels, y_cf_pred, y_pred):
    y_cf_pred_binary = F.softmax(y_cf_pred, dim=-1)
    y_cf_pred_binary = y_cf_pred_binary.argmax(dim=1).view(-1, 1)
    y_pred_binary = F.softmax(y_pred, dim=-1)
    y_pred_binary = y_pred_binary.argmax(dim=1).view(-1, 1)

    score = -1.0
    if dataset == 'synthetic' or dataset == 'imdb_m':
        size = len(features_permuted)
        max_num_nodes = adj_reconst.shape[-1]
        ave_degree = (torch.sum(adj_permuted.view(size, -1), dim=-1) - max_num_nodes) / (2 * max_num_nodes)  # size
        ave_degree_cf = (torch.sum(adj_reconst.view(size, -1), dim=-1) - max_num_nodes) / (2 * max_num_nodes)

        count_good = torch.where(
            (((ave_degree > ave_degree_cf) & (labels.view(-1) > y_cf.view(-1))) |
            ((ave_degree < ave_degree_cf) & (labels.view(-1) < y_cf.view(-1)))), torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))

        score = torch.sum(count_good)
        all = (labels.view(-1) != y_cf.view(-1)).sum()
        if all.item() == 0:
            return score / (all+1)
        score = score / all
    elif dataset == 'ogbg_molhiv':
        ave_x1 = torch.mean(features_permuted[:, :, 1], dim=-1)  # size
        ave_x1_cf = torch.mean(features_reconst[:, :, 1], dim=-1)  # size

        count_good = torch.where(
            (((ave_x1 > ave_x1_cf) & (y_pred_binary.view(-1) > y_cf_pred_binary.view(-1))) |
             ((ave_x1 < ave_x1_cf) & (y_pred_binary.view(-1) < y_cf_pred_binary.view(-1)))),
            torch.tensor(1.0).to(device),
            torch.tensor(0.0).to(device))

        score = torch.sum(count_good)
        all = (y_pred_binary.view(-1) != y_cf_pred_binary.view(-1)).sum()
        if all.item() == 0:
            return score / (all + 1)
        score = score / all

    elif dataset == 'community':
        size = len(features_permuted)
        max_num_nodes = adj_reconst.shape[-1]

        n0 = int(max_num_nodes / 2)
        n1 = max_num_nodes - n0
        ave_degree_0 = (torch.sum(adj_permuted[:, :n0, :n0].reshape(size, -1), dim=-1) - n0) / (2 * n0)  # size
        ave_degree_cf_0 = (torch.sum(adj_reconst[:, :n0, :n0].reshape(size, -1), dim=-1) - n0) / (2 * n0)

        count_good = torch.where(
            (((ave_degree_0 > ave_degree_cf_0) & (y_pred_binary.view(-1) > y_cf_pred_binary.view(-1))) |
             ((ave_degree_0 < ave_degree_cf_0) & (y_pred_binary.view(-1) < y_cf_pred_binary.view(-1)))), torch.tensor(1.0).to(device),
            torch.tensor(0.0).to(device))
        # score = torch.mean(count_good)
        score = torch.sum(count_good)
        all = (y_pred_binary.view(-1) != y_cf_pred_binary.view(-1)).sum()
        if all.item() == 0:
            return score / (all + 1)
        score = score / all

    return score

def perturb_graph(adj, type='random', num_rounds=1):
    num_node = adj.shape[0]
    num_entry = num_node * num_node
    adj_cf = adj.clone()
    if type == 'random':
        # randomly add/remove edges for T rounds
        for rd in range(num_rounds):
            [row, col] = np.random.choice(num_node, size=2, replace=False)
            adj_cf[row, col] = 1 - adj[row, col]
            adj_cf[col, row] = adj_cf[row, col]

    elif type == 'IST':
        # randomly add edge
        for rd in range(num_rounds):
            idx_select = (adj_cf == 0).nonzero()  # 0
            if len(idx_select) <= 0:
                continue
            ii = np.random.choice(len(idx_select), size=1, replace=False)
            idx = idx_select[ii].view(-1)
            row, col = idx[0], idx[1]
            adj_cf[row, col] = 1
            adj_cf[col, row] = 1

    elif type == 'RM':
        # randomly remove edge
        for rd in range(num_rounds):
            idx_select = adj_cf.nonzero()  # 1
            if len(idx_select) <= 0:
                continue
            ii = np.random.choice(len(idx_select), size=1, replace=False)
            idx = idx_select[ii].view(-1)
            row, col = idx[0], idx[1]
            adj_cf[row, col] = 0
            adj_cf[col, row] = 0

    return adj_cf



def run_clear(args, exp_type, save_model=False, data_bundle: Optional[dict] = None):
    data_path_root = os.path.join(dataset_root(), args.dataset)
    model_path = './trained_model/clear/'
    os.makedirs(model_path, exist_ok=True)
    prediction_dir = os.path.join(model_path, 'prediction')
    os.makedirs(prediction_dir, exist_ok=True)
    models_save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models_save'))
    os.makedirs(models_save_dir, exist_ok=True)
    assert exp_type == 'train' or exp_type == 'test' or exp_type == 'test_small'
    small_test = 20

    # load data
    if data_bundle is not None:
        data_load = data_bundle
    else:
        if dpp is None:
            raise ImportError('data_preprocessing module is required when data_bundle is not provided.')
        data_load = dpp.load_data(data_path_root, args.dataset)
    idx_train_list, idx_val_list, idx_test_list = data_load['idx_train_list'], data_load['idx_val_list'], data_load[
        'idx_test_list']
    data = data_load['data']
    x_dim = data[0]["features"].shape[1]
    u_unique = np.unique(np.array(data.u_all))
    u_dim = len(u_unique)

    n = len(data)
    max_num_nodes = data.max_num_nodes
    labels_array = np.array([int(np.array(lbl).reshape(-1)[0]) for lbl in data.labels_all], dtype=np.int64)
    unique_class = np.unique(labels_array)
    num_class = len(unique_class)
    print('n ', n, 'x_dim: ', x_dim, ' max_num_nodes: ', max_num_nodes, ' num_class: ', num_class)

    results_all_exp = {}
    exp_num = len(idx_train_list)
    init_params = {'vae_type': 'graphVAE', 'x_dim': x_dim, 'u_dim': u_dim,
                   'max_num_nodes': max_num_nodes}  # parameters for initialize GraphCFE model

    # load model
    prediction_path = os.path.join(prediction_dir, f'weights_graphPred__{args.dataset}.pt')
    if not os.path.exists(prediction_path):
        print(f'Prediction model missing at {prediction_path}. Training one now...')
        _train_clear_prediction_model(data_load, args.dataset, prediction_dir, device)
    pred_model_type = args.prediction_model_type
    input_dim = x_dim
    hidden_dim = 64
    output_dim = num_class
    dropout = 0.5
    if pred_model_type == 'GIN':
        pred_model = models.Graph_pred_model(input_dim, hidden_dim, output_dim, max_num_nodes, args.dataset).to(device)
    else:
        raise ValueError(f'Unsupported prediction model type for CLEAR: {pred_model_type}')
    pred_model.load_state_dict(torch.load(prediction_path, map_location=device))
    pred_model.eval()

    if num_class > 1:
        y_cf_targets = (labels_array + 1) % num_class
    else:
        y_cf_targets = labels_array
    y_cf = torch.FloatTensor(y_cf_targets).unsqueeze(1).to(device)

    metrics = ['causality', 'validity', 'proximity_x', 'proximity_a']
    time_spent_all = []

    for exp_i in range(0, exp_num):
        print('============================= Start experiment ', str(exp_i),
              ' =============================================')
        idx_train = idx_train_list[exp_i]
        idx_val = idx_val_list[exp_i]
        idx_test = idx_test_list[exp_i]

        if args.disable_u:
            model = models.GraphCFE(init_params=init_params, args=args)
        else:
            model = models.GraphCFE(init_params=init_params, args=args)


        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # data loader
        train_loader = utils.select_dataloader(data, idx_train, batch_size=args.batch_size,
                                               num_workers=args.num_workers)
        val_loader = utils.select_dataloader(data, idx_val, batch_size=args.batch_size, num_workers=args.num_workers)
        test_loader = utils.select_dataloader(data, idx_test, batch_size=args.batch_size, num_workers=args.num_workers)

        
        model = model.to(device)

        variant = 'VAE' if args.disable_u else 'CLEAR'
        if exp_type == 'train':
            # train
            train_params = {'epochs': args.epochs, 'model': model, 'pred_model': pred_model, 'optimizer': optimizer,
                            'y_cf': y_cf,
                            'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader,
                            'exp_i': exp_i,
                            'dataset': args.dataset, 'metrics': metrics, 'save_model': save_model, 'variant': variant,
                            'models_save_dir': models_save_dir}
            train(train_params)
        else:
            # test
            CFE_model_path = os.path.join(models_save_dir, f'weights_graphCFE_{variant}_{args.dataset}_exp{exp_i}_epoch900.pt')
            model.load_state_dict(torch.load(CFE_model_path))
            print('CFE generator loaded from: ' + CFE_model_path)
            if exp_type == 'test_small':
                subset = idx_test[:small_test] if isinstance(idx_test, np.ndarray) else idx_test[:small_test]
                test_loader = utils.select_dataloader(data, subset, batch_size=args.batch_size, num_workers=args.num_workers)

        test_params = {'model': model, 'dataset': args.dataset, 'data_loader': test_loader, 'pred_model': pred_model,
                       'metrics': metrics, 'y_cf': y_cf}

        time_begin = time.time()

        eval_results = test(test_params)

        time_end = time.time()
        time_spent = time_end - time_begin
        time_spent = time_spent / small_test
        time_spent_all.append(time_spent)

        for k in metrics:
            results_all_exp = add_list_in_dict(k, results_all_exp, eval_results[k]) #.detach().cpu().numpy())

        print('=========================== Exp ', str(exp_i), ' Results ==================================')
        for k in eval_results:
            if isinstance(eval_results[k], list):
                print(k, ": ", eval_results[k])
            else:
                print(k, f": {eval_results[k]:.4f}")
        print('time: ', time_spent)

    print('============================= Overall Results =============================================')
    record_exp_result = {}  # save in file
    for k in results_all_exp:
        values = []
        for item in results_all_exp[k]:
            if torch.is_tensor(item):
                values.append(item.detach().cpu().numpy())
            else:
                values.append(item)
        values = np.array(values)
        print(k, f": mean: {np.mean(values):.4f} | std: {np.std(values):.4f}")
        record_exp_result[k] = {'mean': np.mean(values), 'std': np.std(values)}

    time_spent_all = np.array(time_spent_all)
    record_exp_result['time'] = {'mean': np.mean(time_spent_all), 'std': np.std(time_spent_all)}

    save_result = False
    print("====save in file ====")
    print(record_exp_result)
    if save_result:
        exp_save_path = '../exp_results/'
        if args.disable_u:
            exp_save_path = exp_save_path + 'CVAE' + '.pickle'
        else:
            exp_save_path = exp_save_path + 'CLEAR' + '.pickle'
        with open(exp_save_path, 'wb') as handle:
            pickle.dump(record_exp_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('saved data: ', exp_save_path)
    return


if __name__ == '__main__':
    args = parser.parse_args()
    configure_environment(args)
    experiment_type = args.experiment_type
    print('running experiment: ', experiment_type)

    if experiment_type == 'train':
        run_clear(args, 'train')
    elif experiment_type == 'test':
        run_clear(args, 'test_small')
    
