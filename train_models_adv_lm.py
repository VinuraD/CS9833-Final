import argparse
import os
import torch
from torch_geometric.data import DataLoader,Data
from torch_geometric.data import Batch, InMemoryDataset
from models import GIN, SAG, GUNet
from train_models import main, set_seed
from dataset_utils import (
    load_tud_dataset,
    load_dataset_splits,
    is_malnet_tiny,
    load_malnet_tiny_filtered_splits,
    MALNET_TINY_MAX_NODES,
)
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='NCI1', help='Dataset to use (default: NCI1)')
parser.add_argument('--model',type=str, default='GIN', help='Model to attack (default: GIN)')
parser.add_argument('--cf_type',
                    type=str,
                    default='cf_gnn',
                    choices=['cf_gnn', 'clear', 'combinex', 'c2', 'gcf', 'rand'],
                    help='Type of counterfactual generation method (default: cf_gnn; supports rand)')
parser.add_argument('--cf_pct', type=float, default=0.25, help='Percentage of edges to perturb for cf generation (default: 0.25)')
parser.add_argument('--batch_size', type=int, default=32, help='social dataset:64 bio dataset:32')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--get_uncertainty', action='store_true', help='Whether to compute uncertainty of generated counterfactuals')
parser.add_argument('--targeted', action='store_true', help='Use targeted CF pairs where CF label is 0 and original is 1')
parser.add_argument('--base_conf', action='store_true', help='Filter targeted CFs to those from high-uncertainty base samples')
parser.add_argument('--flip_label',action='store_true',help='for experimental purposes, flips the true label')


args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(42)


def fetch_train_cf(dataset_name,cf_type,model_name,cf_pct,targeted=False):
    generated_cf_path=f'./cf/generated/{cf_type}/pairs/cf_{model_name}_{dataset_name}_{cf_pct}.pt'
    if is_malnet_tiny(dataset_name):
        dataset, splits = load_malnet_tiny_filtered_splits(
            max_nodes=MALNET_TINY_MAX_NODES, seed=42)
    else:
        dataset = load_tud_dataset(dataset_name)
        splits = load_dataset_splits(dataset_name, len(dataset))

    train_data = dataset[splits['train']]
    test_data = dataset[splits['test']]
    val_data = dataset[splits['val']]

    cf_data = torch.load(generated_cf_path)
    if targeted:
        balanced_cf, balanced_orig = get_targeted_cf(cf_data)
    else:
        balanced_cf, balanced_orig = get_balanced_cf(cf_data)

    return train_data,test_data,val_data,balanced_cf,balanced_orig

def _unpack_cf_pairs(cf_data):
    if not isinstance(cf_data, dict) or 'cf' not in cf_data or 'orig' not in cf_data:
        raise ValueError("Expected counterfactual pairs with 'cf' and 'orig' entries.")

    cf_batch = cf_data['cf']
    orig_batch = cf_data['orig']

    if cf_batch is None or orig_batch is None:
        raise ValueError("Counterfactual pairs missing data for 'cf' or 'orig'.")

    if hasattr(cf_batch, 'to'):
        cf_batch = cf_batch.to('cpu')
    if hasattr(orig_batch, 'to'):
        orig_batch = orig_batch.to('cpu')

    def _to_data_list(item):
        if isinstance(item, Batch):
            return Batch.to_data_list(item)
        if isinstance(item, list):
            return item
        if isinstance(item, Data):
            return [item]
        raise TypeError("Unsupported data container for counterfactual pairs.")

    cf_list = _to_data_list(cf_batch)
    orig_list = _to_data_list(orig_batch)

    if len(cf_list) != len(orig_list):
        raise ValueError(f"Mismatched pair counts: {len(cf_list)} counterfactuals vs {len(orig_list)} originals.")
    if len(cf_list) == 0:
        raise ValueError("No counterfactual pairs available to balance.")

    return cf_list, orig_list

def _get_label(data_item):
    if not hasattr(data_item, 'y') or data_item.y is None:
        raise ValueError("Data item missing label 'y' required for balancing.")
    label = data_item.y
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label)
    return int(label.view(-1)[0].item())

def _infer_num_node_features(*datasets):
    """Best-effort inference of input feature dimensionality from provided data containers."""
    for ds in datasets:
        if ds is None:
            continue
        if hasattr(ds, 'num_node_features') and getattr(ds, 'num_node_features') is not None:
            try:
                return int(ds.num_node_features)
            except Exception:
                pass
        first_item = None
        if isinstance(ds, Batch):
            ds_list = Batch.to_data_list(ds)
            first_item = ds_list[0] if len(ds_list) > 0 else None
        elif isinstance(ds, Data):
            first_item = ds
        else:
            try:
                ds_iter = list(ds)
                first_item = ds_iter[0] if len(ds_iter) > 0 else None
            except Exception:
                first_item = None
        if first_item is not None and hasattr(first_item, 'num_node_features'):
            return int(first_item.num_node_features)
    return None

def _infer_num_classes(*datasets):
    """Infer number of classes by scanning labels across provided datasets."""
    for ds in datasets:
        if ds is None:
            continue
        if hasattr(ds, 'num_classes') and getattr(ds, 'num_classes') is not None:
            try:
                return int(ds.num_classes)
            except Exception:
                pass
    labels = set()
    for ds in datasets:
        if ds is None:
            continue
        if isinstance(ds, Batch):
            ds_list = Batch.to_data_list(ds)
        elif isinstance(ds, Data):
            ds_list = [ds]
        else:
            try:
                ds_list = list(ds)
            except Exception:
                ds_list = []
        for item in ds_list:
            try:
                labels.add(_get_label(item))
            except Exception:
                continue
    return max(labels) + 1 if labels else None

def get_balanced_cf(cf_data):
    cf_list, orig_list = _unpack_cf_pairs(cf_data)

    pairs_by_label = {0: [], 1: []}
    for cf_item, orig_item in zip(cf_list, orig_list):
        cf_label = _get_label(cf_item)
        orig_label = _get_label(orig_item)
        # if cf_label == orig_label:
        #     continue  # only keep pairs with differing labels
        if cf_label not in pairs_by_label:
            raise ValueError(f"Unsupported CF label value {cf_label}; expected binary classes 0 and 1.")
        if orig_label not in pairs_by_label:
            raise ValueError(f"Unsupported original label value {orig_label}; expected binary classes 0 and 1.")
        pairs_by_label[cf_label].append((cf_item, orig_item))

    min_cls = min(len(pairs_by_label[0]), len(pairs_by_label[1]))
    if min_cls == 0:
        raise ValueError("Cannot create a balanced set; one of the classes has zero valid CF/original mismatched pairs.")

    random.shuffle(pairs_by_label[0])
    random.shuffle(pairs_by_label[1])

    balanced_pairs = pairs_by_label[0][:min_cls] + pairs_by_label[1][:min_cls]
    random.shuffle(balanced_pairs)

    balanced_cf_list = [p[0] for p in balanced_pairs]
    balanced_orig_list = [p[1] for p in balanced_pairs]

    balanced_cf = Batch.from_data_list(balanced_cf_list)
    balanced_orig = Batch.from_data_list(balanced_orig_list)

    labels = [ _get_label(item) for item in balanced_cf_list ]
    assert labels.count(0) == labels.count(1), "Balanced counterfactual set is not even between classes 0 and 1."

    return balanced_cf, balanced_orig

def get_targeted_cf(cf_data):
    cf_list, orig_list = _unpack_cf_pairs(cf_data)

    targeted_pairs = []
    for cf_item, orig_item in zip(cf_list, orig_list):
        cf_label = _get_label(cf_item)
        orig_label = _get_label(orig_item)
        if cf_label not in (0, 1):
            raise ValueError(f"Unsupported CF label value {cf_label}; expected binary classes 0 and 1.")
        if orig_label not in (0, 1):
            raise ValueError(f"Unsupported original label value {orig_label}; expected binary classes 0 and 1.")
        if cf_label == 0 and orig_label == 1:
            targeted_pairs.append((cf_item, orig_item))

    if len(targeted_pairs) == 0:
        raise ValueError("No targeted counterfactual pairs found where CF label is 0 and original label is 1.")

    targeted_cf_list = [p[0] for p in targeted_pairs]
    targeted_orig_list = [p[1] for p in targeted_pairs]

    targeted_cf = Batch.from_data_list(targeted_cf_list)
    targeted_orig = Batch.from_data_list(targeted_orig_list)

    return targeted_cf, targeted_orig

class ListDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__('.')
        self.data, self.slices = self.collate(data_list)

def _ensure_label_tensor(data_obj, attr_name='y'):
    """Make sure label-like attributes are 1D tensors to appease PyG collation."""
    if not hasattr(data_obj, attr_name):
        return data_obj
    label = getattr(data_obj, attr_name)
    if label is None:
        return data_obj
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label)
    if label.dim() == 0:
        label = label.view(1)
    setattr(data_obj, attr_name, label)
    return data_obj

def flip_cf_labels(cf_data, from_label=0, to_label=1):
    """Flip counterfactual labels from one value to another without mutating input Batch."""
    if isinstance(cf_data, Batch):
        cf_data_list = Batch.to_data_list(cf_data)
    elif isinstance(cf_data, list):
        cf_data_list = list(cf_data)
    elif isinstance(cf_data, Data):
        cf_data_list = [cf_data]
    else:
        raise TypeError("Unsupported data container for flipping CF labels.")

    flipped_list = []
    for item in cf_data_list:
        cloned_item = item.clone()
        label_val = _get_label(cloned_item)
        if label_val == from_label:
            dtype = cloned_item.y.dtype if isinstance(cloned_item.y, torch.Tensor) else None
            cloned_item.y = torch.tensor([to_label], dtype=dtype)
        cloned_item = _ensure_label_tensor(cloned_item, 'y')
        flipped_list.append(cloned_item)

    return Batch.from_data_list(flipped_list)

def _load_base_model_for_uncertainty(dataset_name, model_name, input_dim, hidden_dim, output_dim, dropout, device):
    if model_name != 'GIN':
        raise ValueError("--base_conf currently requires a GIN model to access conformal prediction set sizes.")
    gin_layers = 6 if is_malnet_tiny(dataset_name) else 5
    model = GIN(gin_layers, 2, input_dim, hidden_dim, output_dim, dropout).to(device)
    base_model_path = './trained_model/{}_{}.pt'.format(dataset_name, model_name)
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model checkpoint required for --base_conf not found at {base_model_path}")
    state_dict = torch.load(base_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def _filter_pairs_by_uncertainty(cf_batch, orig_batch, val_data, dataset_name, model_name, hidden_dim, dropout, device, batch_size):
    input_dim = _infer_num_node_features(orig_batch, val_data)
    output_dim = _infer_num_classes(orig_batch, val_data)
    if input_dim is None or output_dim is None:
        raise ValueError("Unable to infer model input/output dimensions for base confidence filtering.")

    base_model = _load_base_model_for_uncertainty(dataset_name, model_name, input_dim, hidden_dim, output_dim, dropout, device)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    _, _, set_sizes = base_model.get_uncertainty(orig_batch, val_loader=val_loader, device=device, force_recalibrate=True)

    cf_list = Batch.to_data_list(cf_batch)
    orig_list = Batch.to_data_list(orig_batch)
    if len(cf_list) != len(set_sizes):
        raise ValueError(f"Mismatched CF pairs and uncertainty scores: {len(cf_list)} vs {len(set_sizes)}.")

    keep_indices = [idx for idx, sz in enumerate(set_sizes.view(-1).tolist()) if sz > 1]
    if len(keep_indices) == 0:
        print('No CF pairs retained after base confidence filtering (set size > 1).')
        return None, None

    filtered_cf = [cf_list[i] for i in keep_indices]
    filtered_orig = [orig_list[i] for i in keep_indices]

    return Batch.from_data_list(filtered_cf), Batch.from_data_list(filtered_orig)


def combine_train_cf(train_data,cf_data,orig_data=None):
    def _to_data_list(data_obj):
        if data_obj is None:
            return []
        if isinstance(data_obj, Batch):
            return Batch.to_data_list(data_obj)
        if isinstance(data_obj, list):
            return list(data_obj)
        if isinstance(data_obj, Data):
            return [data_obj]
        raise TypeError("Unsupported data container when combining train and counterfactual data.")

    cf_data_list = [_ensure_label_tensor(d, 'y') for d in _to_data_list(cf_data)]
    train_data_list = [_ensure_label_tensor(d, 'y') for d in train_data]
    combined_data_list = list(train_data_list)

    if orig_data is not None:
        orig_data_list = [_ensure_label_tensor(d, 'y') for d in _to_data_list(orig_data)]
        combined_data_list += orig_data_list

    combined_data_list += cf_data_list
    return ListDataset(combined_data_list)


##need a way to get hyperparameters from a saved config without getting them each time from the cmd

def train_with_cf():
    dataset_name = args.dataset
    model_name = args.model
    cf_type = args.cf_type
    cf_pct = args.cf_pct
    batch_size = args.batch_size
    num_epochs = args.epochs
    lr = args.learning_rate
    hidden_dim = args.hidden_dim
    dropout = args.dropout
    flip_label=args.flip_label
    targeted = args.targeted
    base_conf = args.base_conf

    train_data,test_data,val_data,cf_data,orig_data = fetch_train_cf(dataset_name,cf_type,model_name,cf_pct,targeted=targeted)
    
    # if flip_label: #this is only a hack for experimental purposes
    #     cf_data_list=Batch.to_data_list(cf_data)
    #     for i in cf_data_list:
    #         # print(i)
    #         i.y=i.y.view(-1)//2
    #     cf_data=Batch.from_data_list(cf_data_list)

    if base_conf and not targeted:
        print('--base_conf is only applied when --targeted is set; ignoring for untargeted training.')
        base_conf = False

    cf_pair_count = len(Batch.to_data_list(cf_data)) if isinstance(cf_data, Batch) else len(cf_data)
    original_cf_pairs = cf_pair_count

    if targeted and base_conf:
        filtered_cf, filtered_orig = _filter_pairs_by_uncertainty(
            cf_data,
            orig_data,
            val_data,
            dataset_name,
            model_name,
            hidden_dim,
            dropout,
            device,
            batch_size,
        )
        if filtered_cf is None or filtered_orig is None:
            cf_data, orig_data = [], []
            cf_pair_count = 0
        else:
            cf_data, orig_data = filtered_cf, filtered_orig
            cf_pair_count = len(Batch.to_data_list(cf_data)) if isinstance(cf_data, Batch) else len(cf_data)
        print(f'Base confidence filtering retained {cf_pair_count} / {original_cf_pairs} targeted CF pairs (set size > 1).')

    if targeted and cf_pair_count > 0:
        cf_data = flip_cf_labels(cf_data, from_label=0, to_label=1)

    combined_data = combine_train_cf(train_data,cf_data)#,orig_data
    print(f'Counterfactual/original pairs added to final train set: {cf_pair_count}')
    train_loader = DataLoader(combined_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)    

    input_dim = combined_data.num_node_features
    output_dim = combined_data.num_classes
    print('Total training samples after combining CFs: ', len(combined_data))

    if targeted and base_conf:
        save_dir = './trained_model/cf_lm_target_conf/{}/'.format(cf_type)
    elif targeted:
        save_dir = './trained_model/cf_lm_target/{}/'.format(cf_type)
    else:
        save_dir = './trained_model/cf_lm/{}/'.format(cf_type)
    os.makedirs(save_dir, exist_ok=True)
    results = main(model_name, dataset_name, device, num_epochs, lr,input_dim, hidden_dim,output_dim,dropout,train_loader,val_loader,test_loader, save_dir,True,cf_pct,uq=True,bdist=False,print_metrics=False)
    print('Test acc:{:04f}'.format(results['test_acc']))
    print('TPR: {:.4f}, FPR: {:.4f}'.format(results['tpr'], results['fpr']))

if __name__ == '__main__':

    set_seed(42)
    train_with_cf()
