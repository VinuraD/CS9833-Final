import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

import argparse
import numpy as np
import random
from models import GIN, GCN, SAG, GUNet
from eval import *
from dataset_utils import (MALNET_TINY_MAX_NODES, is_malnet_tiny,
                           load_dataset_splits,
                           load_malnet_tiny_filtered_splits, load_tud_dataset)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser(description='graph classification model training')
    parser.add_argument('--dataset', type=str, default="NCI1",
                        help='Dataset to use (e.g., NCI1 or MalNetTiny)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32, help='32')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--model_path', type=str, default='./trained_model/')
    parser.add_argument('--model', type=str, default='GIN')
    parser.add_argument('--get_uncertainty', action='store_true', help='Whether to compute uncertainty of generated counterfactuals')
    parser.add_argument('--cal_rob',action='store_true',help='calculate decision boundary margins')
    args = parser.parse_args()
    return args 

def train(model, train_loader, device, lr):
    model.train()
    loss_all = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_loader.dataset)

def test(model, test_loader, device):
    model.eval()
    correct = 0
    for data in test_loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(test_loader.dataset)

def main(model_name, dataset_name, device, num_epochs, lr,input_dim,hidden_dim,output_dim, dropout, train_loader,val_loader,test_loader,model_path,adv=None,cf_pct=None,uq=False,bdist=False,print_metrics=True):
    set_seed(42)
    use_malnet_gin = is_malnet_tiny(dataset_name)
    if model_name=='GCN':
        model = GCN(5,input_dim,hidden_dim,output_dim,0.8,dropout).to(device)
    elif model_name=='GIN':
        gin_layers = 6 if use_malnet_gin else 5  # extra GINConv layer for MalNet
        model = GIN(gin_layers,2,input_dim,hidden_dim,output_dim,dropout).to(device)

    if adv==None:
        path = model_path + '{}_{}.pt'.format(dataset_name, model_name)
    else:
        path = model_path + '{}_{}_{}.pt'.format(dataset_name, model_name,cf_pct)

    best_loss, best_test_acc, best_epoch = 0, 0, 0
    for epoch in range(num_epochs):
        if epoch+1 % 50 == 0:
            lr = lr*0.5
        loss = train(model, train_loader, device, lr)
        train_acc = test(model, train_loader, device)
        val_acc = test(model, val_loader, device)
        if val_acc >= best_test_acc:
            best_loss, best_test_acc, best_epoch = loss, val_acc, epoch
            torch.save(model.state_dict(), path)
        print('Epoch:{:03d}, Loss:{:04f}, Train acc:{:04f}, Val acc:{:04f}'.format(epoch,loss,train_acc,val_acc))
    print('best loss:{:04f}, best acc:{:04f}, epoch:{:03d}'.format(best_loss,best_test_acc,best_epoch))

    if uq:
        print('Evaluating uncertainty on test set...')
        print('calibrating...')
        model._calibrate_conformal(val_loader, device)
        p_vals = []
        set_sizes = []
        for i in test_loader:
            i = i.to(device)
            _, p_value,set_size = model.get_uncertainty(i, device)
            p_vals.append(p_value[0].item())
            set_sizes.append(set_size[0].item())
        plot_uq(p_vals, set_sizes)
        print('Average set size: ', np.mean(set_sizes))

    if bdist:
        print('calculating distance to the decision boundary...')
        total_dist = 0
        counted = 0
        total_graphs = 0
        for batch in test_loader:
            batch = batch.to(device)
            for x0 in batch.to_data_list():
                total_graphs += 1
                y0 = x0.y[0]
                y_pred = model.predict(x0, device=device)
                if y0 == y_pred:
                    initial_theta, inital_F, initial_g, num_query,_ = find_coarse_perturb(model, x0, y0, device)
                    if inital_F.item() == float('inf'):
                        continue
                    total_dist += L0_norm(initial_g * initial_theta, device)
                    counted += 1
        if counted > 0:
            print('Average perturbation distance = {}, calculated using {} of {} graphs'.format(total_dist / counted, counted, total_graphs))
        else:
            print('Average perturbation distance = {}, no correctly classified graphs found'.format(float('inf')))

    
     # Load the best model for testing
    model.load_state_dict(torch.load(path))
    test_acc = test(model, test_loader, device)
    metrics = compute_binary_metrics(model, test_loader, device)
    # keep accuracy print consistent with previous behaviour
    if print_metrics:
        print('Test acc:{:04f}'.format(test_acc))
        print('TPR: {:.4f}, FPR: {:.4f}'.format(metrics['tpr'], metrics['fpr']))
    return {
        'test_acc': test_acc,
        'tpr': metrics['tpr'],
        'fpr': metrics['fpr'],
        'counts': {
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'tn': metrics['tn'],
            'fn': metrics['fn'],
        }
    }


if __name__ == '__main__':
    args = get_args()
    set_seed(42)
    dataset_name = args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    batch_size = args.batch_size
    num_epochs = args.epochs
    lr = args.learning_rate
    hidden_dim = args.hidden_dim
    dropout = args.dropout
    model_path = args.model_path
    model_name = args.model

    if is_malnet_tiny(dataset_name):
        filtered_dataset, splits = load_malnet_tiny_filtered_splits(
            max_nodes=MALNET_TINY_MAX_NODES, seed=42)
        train_index = splits['train']
        val_index = splits['val']
        test_index = splits['test']

        test_dataset = filtered_dataset[test_index]
        val_dataset = filtered_dataset[val_index]
        train_dataset = filtered_dataset[train_index]
        dataset_for_stats = filtered_dataset
    else:
        dataset = load_tud_dataset(dataset_name)
        splits = load_dataset_splits(dataset_name, len(dataset))
        train_index = splits['train']
        val_index = splits['val']
        test_index = splits['test']

        test_dataset = dataset[test_index]
        val_dataset = dataset[val_index]
        train_dataset = dataset[train_index]
        dataset_for_stats = dataset
    test_loader = DataLoader(test_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)
    train_loader = DataLoader(train_dataset, batch_size=32)

    input_dim = dataset_for_stats.num_node_features
    output_dim = dataset_for_stats.num_classes
    print('input dim: ', input_dim)
    print('output dim: ', output_dim)
    train_labels = []
    for g in train_dataset:
        y = g.y
        y_val = int(y[0]) if hasattr(y, 'dim') and y.dim() > 0 else int(y)
        train_labels.append(y_val)
    class_counts = np.bincount(train_labels, minlength=output_dim)
    print(f'Train size: {len(train_dataset)}, class distribution: {class_counts.tolist()}')
    uq= args.get_uncertainty
    calc_robustness=args.cal_rob
    results = main(model_name, dataset_name, device, num_epochs, lr, input_dim,hidden_dim,output_dim, dropout, train_loader,val_loader,test_loader,model_path,uq=uq,bdist=calc_robustness,print_metrics=False)
    print('Test acc:{:04f}'.format(results['test_acc']))
    print('TPR: {:.4f}, FPR: {:.4f}'.format(results['tpr'], results['fpr']))
