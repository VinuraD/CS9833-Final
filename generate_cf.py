from cf.cf_gnnexplainer import CFExplainer, DataInfo
from cf.clearexplainer import CLEARExplainer, train_clear_model
from cf.combinexexplainer import CombinexExplainer
from cf.randexplainer import RandExplainer
import argparse
import os
from collections import Counter
import random

import torch
from torch_geometric.data import DataLoader, Data, Batch
from models import GIN, GCN
import numpy as np
from eval import distance, validity, count_edges, plot_uq, idx_correct
import tqdm
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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='NCI1', help='Dataset to use (default: NCI1)')
parser.add_argument('--model', type=str, default='GIN', help='Model to attack (default: GIN)')
parser.add_argument(
    '--cf_type',
    type=str,
    default='cf_gnn',
    help="Type of counterfactual generation method (options: cf_gnn, clear, combinex, c2, gcf, rand)"
)
parser.add_argument('--get_pairs',action='store_true')
parser.add_argument('--cf_pct', type=float, default=0.25, help='Percentage of edges to perturb for cf generation (default: 0.25)')
parser.add_argument('--rand_trials', type=int, default=600, help='Number of random perturbations to sample for rand counterfactuals (default: 500)')
parser.add_argument(
    '--rand_budget',
    type=float,
    default=0.4,
    help='Maximum perturbation budget for rand counterfactuals (<=1 as ratio of possible edges, >1 as absolute) (default: 0.1)',
)
parser.add_argument('--get_uncertainty', action='store_true', help='Whether to compute uncertainty of generated counterfactuals')
parser.add_argument('--c2_epochs', type=int, default=300, help='Number of optimization epochs for C2Explainer (default: 300)')
parser.add_argument('--c2_lr', type=float, default=0.05, help='Learning rate for C2Explainer optimizer (default: 0.05)')
parser.add_argument('--c2_print_loss', action='store_true', help='Print per-epoch C2Explainer losses')
parser.add_argument('--c2_silent_mode', action='store_true', help='Silence verbose summaries from C2Explainer')
parser.add_argument('--c2_ar_mode', action='store_true', help='Enable AR mode in C2Explainer')
parser.add_argument('--c2_fpm', action='store_true', help='Enable feature perturbation matrix in C2Explainer')
parser.add_argument('--c2_subgraph_mode', action='store_true', help='Restrict C2Explainer to subgraph mode (no edge additions)')
parser.add_argument('--c2_wo_st', action='store_true', help='Disable straight-through trick in C2Explainer')
parser.add_argument('--c2_at_loss', action='store_true', help='Use adversarial training loss component in C2Explainer')
parser.add_argument('--c2_ent_loss', action='store_true', help='Include entropy regularization loss in C2Explainer')
parser.add_argument('--c2_repo', type=str, default=None, help='Optional path to the C2Explainer repository (defaults to Git/C2Explainer)')
parser.add_argument('--gcf_alpha', type=float, default=0.5, help='Weight for individual vs cumulative coverage for GCFExplainer (default: 0.5)')
parser.add_argument('--gcf_theta', type=float, default=0.05, help='Distance threshold used during GCFExplainer training (default: 0.05)')
parser.add_argument('--gcf_summary_theta', type=float, default=0.1, help='Distance threshold for GCFExplainer summaries (default: 0.1)')
parser.add_argument('--gcf_teleport', type=float, default=0.1, help='Teleport probability for the GCFExplainer random walk (default: 0.1)')
parser.add_argument('--gcf_max_steps', type=int, default=50000, help='Maximum random walk steps for GCFExplainer candidate search')
parser.add_argument('--gcf_k', type=int, default=100000, help='Maximum number of GCFExplainer candidates to retain')
parser.add_argument('--gcf_device_gnn', type=str, default='0', help='Device identifier for the GCFExplainer GNN model (default: 0)')
parser.add_argument('--gcf_device_neurosed', type=str, default='0', help='Device identifier for the GCFExplainer NeuroSED model (default: 0)')
parser.add_argument('--gcf_sample_size', type=int, default=10000, help='Neighbour sampling size for GCFExplainer')
parser.add_argument('--gcf_sample', action='store_true', help='Enable neighbour sampling mode for GCFExplainer')
parser.add_argument('--gcf_force', action='store_true', help='Force regeneration of GCFExplainer candidates even if cached results exist')
parser.add_argument('--gcf_run_summary', action='store_true', help='Run the summary phase after generating GCFExplainer candidates')
parser.add_argument('--gcf_repo', type=str, default=None, help='Optional path to the GCFExplainer repository (defaults to Git/GCFExplainer)')
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = args.dataset
model = args.model
cf_type = args.cf_type
uq_true= args.get_uncertainty


def _count_classes(dataset, num_classes):
    counts = Counter()
    for graph in dataset:
        label = int(graph.y.view(-1)[0])
        counts[label] += 1
    return {cls: counts.get(cls, 0) for cls in range(num_classes)}


def _compute_confusion_matrix(model, dataset, device, num_classes):
    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for graph in dataset:
            true_label = int(graph.y.view(-1)[0])
            pred_label = int(model.predict(graph, device=device).item())
            conf[true_label, pred_label] += 1
    return conf.cpu()

def get_model_and_dataset(dataset_name, model_name):
    if is_malnet_tiny(dataset_name):
        dataset, splits = load_malnet_tiny_filtered_splits(
            max_nodes=MALNET_TINY_MAX_NODES, seed=42)
    else:
        dataset = load_tud_dataset(dataset_name)
        splits = load_dataset_splits(dataset_name, len(dataset))

    train_dataset = dataset[splits['train']]
    val_dataset = dataset[splits['val']]
    test_dataset = dataset[splits['test']]

    input_dim = dataset.num_node_features
    output_dim = dataset.num_classes

    model_path = './trained_model/'
    if model_name == 'GCN':
        model = GCN(5, input_dim, 64, output_dim, 0.8, 0.5).to(device)
        load_path = model_path + '{}_{}.pt'.format(dataset_name, model_name)
    elif model_name == 'GIN':
        gin_layers = 6 if is_malnet_tiny(dataset_name) else 5  # MalNetTiny uses deeper GIN
        model = GIN(gin_layers, 2, input_dim, 64, output_dim, 0.5).to(device)
        load_path = model_path + '{}_{}.pt'.format(dataset_name, model_name)
    else:
        raise ValueError(f'Unsupported model type: {model_name}')

    model.load_state_dict(torch.load(load_path, map_location=device))
    return model, [train_dataset, test_dataset,val_dataset]


def generate_cf(cf_type: str, cf_pct: float, model_name: str, data: str):
    set_seed(42)
    model, loaders = get_model_and_dataset(data, model_name)
    train_dataset_full, test_dataset, val_dataset = loaders[0], loaders[1], loaders[2]
    num_classes = train_dataset_full.num_classes
    generated_cf_path = 'cf/generated/{}/cf_{}_{}_{}.pt'.format(cf_type, model_name, data, cf_pct)
    generated_cf_pairs_path = 'cf/generated/{}/pairs/cf_{}_{}_{}.pt'.format(cf_type, model_name, data, cf_pct)

    if cf_type == 'cf_gnn':
        datainfo = DataInfo(num_classes=num_classes)
        explainer = CFExplainer(datainfo)
    elif cf_type == 'clear':
        clear_model_root = './trained_model/clear'
        clear_exp_id = 0
        clear_epoch = 900
        weight_path = CLEARExplainer.weight_path(model_root=clear_model_root,
                                                 dataset_name=data,
                                                 exp_id=clear_exp_id,
                                                 epoch=clear_epoch,
                                                 disable_u=False)
        if not os.path.exists(weight_path):
            print(f'CLEAR weights not found at {weight_path}. Training CLEAR generator...')
            try:
                trained_path, clear_epoch = train_clear_model(dataset_name=data,
                                                              disable_u=False,
                                                              exp_id=clear_exp_id,
                                                              target_epoch=clear_epoch,
                                                              model_root=clear_model_root,
                                                              device=device)
            except Exception as exc:
                raise RuntimeError(
                    "Automatic CLEAR training failed. Ensure training dependencies and datasets are available."
                ) from exc
            weight_path = trained_path
            print(f'CLEAR weights saved at {weight_path}. Proceeding with counterfactual generation.')
        explainer = CLEARExplainer(dataset_name=data,
                                   train_dataset=train_dataset_full,
                                   model_root=clear_model_root,
                                   exp_id=clear_exp_id,
                                   epoch=clear_epoch,
                                   device=device)
    elif cf_type == 'c2':
        from cf.c2explainer import C2ExplainerWrapper

        explainer = C2ExplainerWrapper(model=model,
                                       device=device,
                                       repo_root=args.c2_repo,
                                       epochs=args.c2_epochs,
                                       lr=args.c2_lr,
                                       print_loss=args.c2_print_loss,
                                       silent_mode=args.c2_silent_mode,
                                       ar_mode=args.c2_ar_mode,
                                       fpm=args.c2_fpm,
                                       subgraph_mode=args.c2_subgraph_mode,
                                       wo_st=args.c2_wo_st,
                                       at_loss=args.c2_at_loss,
                                       ent_loss=args.c2_ent_loss)
    elif cf_type == 'gcf':
        from cf.gcfexplainer import GCFExplainerRunner

        gcf_runner = GCFExplainerRunner(dataset_name=data,
                                        model_name=model_name,
                                        output_path=generated_cf_path,
                                        repo_root=args.gcf_repo,
                                        alpha=args.gcf_alpha,
                                        theta=args.gcf_theta,
                                        summary_theta=args.gcf_summary_theta,
                                        teleport=args.gcf_teleport,
                                        max_steps=args.gcf_max_steps,
                                        max_candidates=args.gcf_k,
                                        device_gnn=args.gcf_device_gnn,
                                        device_neurosed=args.gcf_device_neurosed,
                                        sample_size=args.gcf_sample_size,
                                        sample=args.gcf_sample,
                                        run_summary=args.gcf_run_summary,
                                        cf_pct=cf_pct)
        gcf_runner.generate(force=args.gcf_force)
        return
    elif cf_type == 'combinex':
        explainer = CombinexExplainer(dataset=train_dataset_full,
                                      num_classes=num_classes,
                                      model=model,
                                      device=device)
    elif cf_type == 'rand':
        explainer = RandExplainer(device=device,
                                  trials=args.rand_trials,
                                  budget=args.rand_budget,
                                  distance_fn=distance)
    else:
        raise ValueError(f"Unsupported cf_type '{cf_type}'. Expected one of ['cf_gnn', 'clear', 'combinex', 'c2', 'gcf', 'rand'].")

    all_cf = []
    valid_cf=[]
    valid_cf_orig=[]
    correct_pred_idx = []
    train_class_counts = _count_classes(train_dataset_full, num_classes)
    print(f'Number of classes detected in train split: {num_classes}')
    print(f'Training class distribution: {train_class_counts}')
    train_confusion = _compute_confusion_matrix(model, train_dataset_full, device, num_classes)
    print('Confusion matrix on training data (rows=true, cols=pred):\n{}'.format(train_confusion.numpy()))

    test_class_counts = _count_classes(test_dataset, num_classes)
    print(f'Test class distribution: {test_class_counts}')
    test_confusion = _compute_confusion_matrix(model, test_dataset, device, num_classes)
    print('Confusion matrix on test data (rows=true, cols=pred):\n{}'.format(test_confusion.numpy()))

    for i, data_item in enumerate(train_dataset_full):
        if idx_correct(model, data_item, device):
            correct_pred_idx.append(i)
    filtered_train_dataset = train_dataset_full[correct_pred_idx]
    sample_count = max(1, int(len(filtered_train_dataset) * cf_pct))
    sample_count = min(sample_count, len(filtered_train_dataset))
    if sample_count == 0:
        output_dir = os.path.dirname(generated_cf_path)
        os.makedirs(output_dir, exist_ok=True)
        torch.save([], generated_cf_path)
        if args.get_pairs:
            pairs_dir = os.path.dirname(generated_cf_pairs_path)
            os.makedirs(pairs_dir, exist_ok=True)
            torch.save([], generated_cf_pairs_path)
            print('No valid counterfactual pairs were generated. Saved empty list at: {}'.format(generated_cf_pairs_path))
        print('No eligible training samples for counterfactual generation after filtering. Saved empty list at: {}'.format(generated_cf_path))
        return
    # Balance the random sample across the two classes as evenly as possible.
    if num_classes == 2:
        label_to_indices = {0: [], 1: []}
        for idx, graph in enumerate(filtered_train_dataset):
            y = graph.y.view(-1)[0] if isinstance(graph.y, torch.Tensor) else graph.y
            label = int(y.item()) if hasattr(y, 'item') else int(y)
            if label in label_to_indices:
                label_to_indices[label].append(idx)

        per_class = sample_count // 2
        remainder = sample_count - per_class * 2
        chosen_idx = []
        for cls in (0, 1):
            target_k = per_class + (1 if remainder > 0 else 0)
            remainder = max(0, remainder - 1)
            pool = label_to_indices.get(cls, [])
            k = min(target_k, len(pool))
            if k > 0:
                chosen_idx.extend(random.sample(pool, k))

        if len(chosen_idx) < sample_count:
            remaining_pool = list(set(range(len(filtered_train_dataset))) - set(chosen_idx))
            extra = min(sample_count - len(chosen_idx), len(remaining_pool))
            if extra > 0:
                chosen_idx.extend(random.sample(remaining_pool, extra))
        random_idx = chosen_idx
    else:
        random_idx = torch.multinomial(torch.ones(len(filtered_train_dataset)), sample_count, replacement=False).tolist()

    data_list = [filtered_train_dataset[i] for i in random_idx]
    print('Generating counterfactuals for {} out of {} training samples.'.format(len(data_list), len(filtered_train_dataset)))
    for d in data_list:
        y = d.y
        if y.numel() != 1:
            y = y.view(-1)[0]
        y_val = int(y.item()) if hasattr(y, 'item') else int(y)
        target_label = (y_val + 1) % num_classes
        d.targets = torch.tensor([target_label], dtype=torch.long)
    train_loader = DataLoader(data_list, batch_size=1, shuffle=False)

    sparsity = []
    val_score = 0
    added, deleted = 0, 0

    p_values=[]
    set_sizes=[]
    cf_class_counter = Counter()
    # if len(model.bucket_scores)==0:
    if uq_true:
        model._calibrate_conformal(DataLoader(val_dataset), device)
    pbar = tqdm.tqdm(total=len(train_loader), desc='Generating Counterfactuals')
    for batch in train_loader:
        batch = batch.to(device)
        model_pred = model.predict(batch, device=device)
        target_value = batch.targets.view(-1)[0]
        if model_pred.item() != target_value.item():
            cf = explainer.explain(batch, model)
            if uq_true and cf is not None:
                _,p_value,set_size = model.get_uncertainty(cf.to(device), device)
                p_values.append(p_value[0].item())
                set_sizes.append(set_size[0].item())
        else:
            print('Original prediction matches target label, skipping CF generation.')
            pbar.update(1)
            continue
        if cf is not None:
            cf_y = model.predict(cf, device=device)
            cf_label = int(cf_y.item()) if isinstance(cf_y, torch.Tensor) else int(cf_y)
            cf.y = torch.tensor([cf_label], device=cf_y.device if isinstance(cf_y, torch.Tensor) else None)
            all_cf.append(cf)
            dist = distance(cf, batch)
            sparsity.append(dist)
            target_value = int(batch.targets.view(-1)[0].item())
            valid = validity(cf_label, target_value)
            if valid:
                val_score += 1
                valid_cf.append(cf)
                orig_graph = batch.to_data_list()[0] if isinstance(batch, Batch) else batch
                valid_cf_orig.append(orig_graph)
            add, delete = count_edges(cf, batch)
            added += add
            deleted += delete
            # cf_class_counter[int(target_value.item())] += 1
            cf_class_counter[cf_label] += 1
        pbar.update(1)
    pbar.close()

    print('Total counterfactuals generated: {}'.format(len(all_cf)))
    cf_class_summary = {cls: cf_class_counter.get(cls, 0) for cls in range(num_classes)}
    print('Counterfactuals per target class: {}'.format(cf_class_summary))
    print('Average set size of counterfactuals: {:.2f}'.format(np.mean(set_sizes))) if uq_true and len(set_sizes)>0 else None
    output_dir = os.path.dirname(generated_cf_path)
    os.makedirs(output_dir, exist_ok=True)
    if len(all_cf) == 0:
        torch.save([], generated_cf_path)
        if args.get_pairs:
            pairs_dir = os.path.dirname(generated_cf_pairs_path)
            os.makedirs(pairs_dir, exist_ok=True)
            torch.save([], generated_cf_pairs_path)
            print('No valid counterfactual pairs were generated. Saved empty list at: {}'.format(generated_cf_pairs_path))
        print('No counterfactuals were generated. Saved an empty list at: {}'.format(generated_cf_path))
        return
    print('Average sparsity (edge changes): {:.2f}'.format(np.mean(sparsity)))
    print('Validity: {:.2f}%'.format(val_score / len(all_cf) * 100))
    print('average edges added: {:.2f}, average edges deleted: {:.2f}'.format(added / len(all_cf), deleted / len(all_cf)))

    all_cf = Batch.from_data_list(all_cf)
    torch.save(all_cf, generated_cf_path)
    print('Counterfactuals saved at: {}'.format(generated_cf_path))
    if args.get_pairs:
        pairs_dir = os.path.dirname(generated_cf_pairs_path)
        os.makedirs(pairs_dir, exist_ok=True)
        if len(valid_cf) == 0:
            torch.save([], generated_cf_pairs_path)
            print('No valid counterfactual pairs were generated. Saved empty list at: {}'.format(generated_cf_pairs_path))
        else:
            valid_cf_batch = Batch.from_data_list(valid_cf)
            orig_batch = Batch.from_data_list(valid_cf_orig)
            torch.save({'cf': valid_cf_batch, 'orig': orig_batch}, generated_cf_pairs_path)
            print('Valid counterfactual/original pairs saved at: {}'.format(generated_cf_pairs_path))


def main():
    generate_cf(cf_type=args.cf_type, cf_pct=args.cf_pct, model_name=args.model, data=args.dataset)


if __name__ == '__main__':
        
    set_seed(42)
    main()
