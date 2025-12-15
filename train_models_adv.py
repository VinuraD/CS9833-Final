import argparse
import torch
from torch_geometric.data import DataLoader,Data
from torch_geometric.data import Batch, InMemoryDataset
from models import GIN, SAG, GUNet
from train_models import main, set_seed
from dataset_utils import load_tud_dataset, load_dataset_splits
import numpy as np

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
parser.add_argument('--flip_label',action='store_true',help='for experimental purposes, flips the true label')


args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(42)


def fetch_train_cf(dataset_name,cf_type,model_name,cf_pct):
    generated_cf_path=f'./cf/generated/{cf_type}/cf_{model_name}_{dataset_name}_{cf_pct}.pt'
    dataset = load_tud_dataset(dataset_name)
    splits = load_dataset_splits(dataset_name, len(dataset))

    train_data = dataset[splits['train']]
    test_data = dataset[splits['test']]
    val_data = dataset[splits['val']]

    cf_data = torch.load(generated_cf_path).to('cpu')

    return train_data,test_data,val_data,cf_data

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


def combine_train_cf(train_data,cf_data):
    cf_data_list = Batch.to_data_list(cf_data)
    cf_data_list = [_ensure_label_tensor(d, 'y') for d in cf_data_list]
    train_data_list = [_ensure_label_tensor(d, 'y') for d in train_data]
    combined_data_list = train_data_list + cf_data_list
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

    train_data,test_data,val_data,cf_data = fetch_train_cf(dataset_name,cf_type,model_name,cf_pct)
    
    if flip_label:
        cf_data_list=Batch.to_data_list(cf_data)
        for i in cf_data_list:
            # print(i)
            i.y=i.y.view(-1)//2
        cf_data=Batch.from_data_list(cf_data_list)

    combined_data = combine_train_cf(train_data,cf_data)
    train_loader = DataLoader(combined_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)    

    input_dim = combined_data.num_node_features
    output_dim = combined_data.num_classes
    print('Total training samples after combining CFs: ', len(combined_data))

    results = main(model_name, dataset_name, device, num_epochs, lr,input_dim, hidden_dim,output_dim,dropout,train_loader,val_loader,test_loader, './trained_model/cf/{}/'.format(cf_type),True,cf_pct,uq=True,bdist=True,print_metrics=False)
    print('Test acc:{:04f}'.format(results['test_acc']))
    print('TPR: {:.4f}, FPR: {:.4f}'.format(results['tpr'], results['fpr']))

if __name__ == '__main__':

    set_seed(42)
    train_with_cf()
