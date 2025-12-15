import ast
import os
import os.path as osp
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_tar, extract_zip)

try:
    from torch_geometric.datasets import MalNetTiny as _PyGMalNetTiny
except ImportError:  # pragma: no cover - optional dependency guard
    _PyGMalNetTiny = None

# Absolute project paths so dataset handling works regardless of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / 'dataset'
DATA_SPLIT_DIR = PROJECT_ROOT / 'data_split'

MALNET_TINY_NAMES = {'malnettiny', 'malnet_tiny', 'malnet-tiny'}
MALNET_TINY_MAX_NODES = 700
BENIGN_CLASS_NAMES = {'benign', 'benignware'}
MALNET_TINY_SPLITS = ('train', 'val', 'test')

# Flag indicates whether a TU dataset requires degree-as-attribute preprocessing.
TUD_DEGREE_ATTR: Dict[str, int] = {
    'MUTAG': 0,
    'PTC_FM': 0,
    'PROTEINS': 0,
    'NCI1': 0,
    'COIL-DEL': 0,
    'COLLAB': 1,
    'IMDB-BINARY': 1,
    'IMDB-MULTI': 1,
    'REDDIT-BINARY': 1,
    'REDDIT-MULTI5K': 1,
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dataset_root() -> str:
    _ensure_dir(DATASET_DIR)
    return str(DATASET_DIR)


def _split_paths(dataset_name: str) -> Dict[str, Path]:
    _ensure_dir(DATA_SPLIT_DIR)
    return {
        split: DATA_SPLIT_DIR / f'{dataset_name}_{split}_index.txt'
        for split in ('train', 'val', 'test')
    }


def load_tud_dataset(dataset_name: str) -> TUDataset:
    """
    Load a TU dataset from the local dataset directory with the proper transform.
    """
    if dataset_name not in TUD_DEGREE_ATTR:
        raise ValueError(
            f'Unsupported dataset "{dataset_name}". '
            f'Available options: {sorted(TUD_DEGREE_ATTR)}'
        )

    kwargs = dict(root=dataset_root(),
                  name=dataset_name,
                  use_edge_attr='False',
                  use_node_attr=True)
    if TUD_DEGREE_ATTR[dataset_name]:
        kwargs['pre_transform'] = T.Constant(1, True)
    return TUDataset(**kwargs)


def is_malnet_tiny(dataset_name: str) -> bool:
    """
    Identify whether the requested dataset name refers to MalNetTiny.
    """
    normalized = dataset_name.replace('-', '_').lower()
    return normalized in MALNET_TINY_NAMES


def _build_local_malnet_tiny():
    """
    Build a local MalNetTiny class (for PyG versions that lack it).
    Returns None if dependencies are missing.
    """
    if _PyGMalNetTiny is not None:
        return _PyGMalNetTiny

    try:
        class _LocalMalNetTiny(InMemoryDataset):
            data_url = ('http://malnet.cc.gatech.edu/graph-data/'
                        'malnet-graphs-tiny.tar.gz')
            split_url = 'http://malnet.cc.gatech.edu/split-info/split_info_tiny.zip'
            splits = list(MALNET_TINY_SPLITS)

            def __init__(self,
                         root: str,
                         split: Optional[str] = None,
                         transform: Optional[Callable] = None,
                         pre_transform: Optional[Callable] = None,
                         pre_filter: Optional[Callable] = None):
                if split not in {None, 'train', 'val', 'trainval', 'test'}:
                    raise ValueError(
                        f'Split "{split}" found, but expected either '
                        f'"train", "val", "trainval", "test" or None'
                    )
                super().__init__(root, transform, pre_transform, pre_filter)
                self.data, self.slices = torch.load(self.processed_paths[0])
                self.class_mapping = self._build_class_mapping()
                self.class_names = list(self.class_mapping.keys())

                if split is not None:
                    split_slices = torch.load(self.processed_paths[1])
                    if split == 'train':
                        self._indices = range(split_slices[0], split_slices[1])
                    elif split == 'val':
                        self._indices = range(split_slices[1], split_slices[2])
                    elif split == 'trainval':
                        self._indices = range(split_slices[0], split_slices[2])
                    elif split == 'test':
                        self._indices = range(split_slices[2], split_slices[3])

            @property
            def raw_file_names(self) -> List[str]:
                return ['malnet-graphs-tiny', osp.join('split_info_tiny', 'type')]

            @property
            def processed_file_names(self) -> List[str]:
                return ['data.pt', 'split_slices.pt']

            def download(self):
                path = download_url(self.data_url, self.raw_dir)
                extract_tar(path, self.raw_dir)
                os.unlink(path)

                path = download_url(self.split_url, self.raw_dir)
                extract_zip(path, self.raw_dir)
                os.unlink(path)

            def process(self):
                y_map: Dict[str, int] = {}
                data_list = []
                split_slices = [0]

                for split in MALNET_TINY_SPLITS:
                    with open(osp.join(self.raw_paths[1], f'{split}.txt'), 'r') as f:
                        filenames = [fn for fn in f.read().split('\n') if fn]
                        split_slices.append(split_slices[-1] + len(filenames))

                    for filename in filenames:
                        path = osp.join(self.raw_paths[0], f'{filename}.edgelist')
                        malware_type = filename.split('/')[0]
                        y = y_map.setdefault(malware_type, len(y_map))

                        with open(path, 'r') as f:
                            edges = [ln for ln in f.read().split('\n')[5:] if ln]

                        edge_index = [[int(s) for s in edge.split()] for edge in edges]
                        edge_index = torch.tensor(edge_index).t().contiguous()
                        num_nodes = int(edge_index.max()) + 1
                        x = torch.ones((num_nodes, 1), dtype=torch.float32)
                        data = Data(edge_index=edge_index, x=x, y=y, num_nodes=num_nodes)
                        data_list.append(data)

                if self.pre_filter is not None:
                    data_list = [data for data in data_list if self.pre_filter(data)]

                if self.pre_transform is not None:
                    data_list = [self.pre_transform(data) for data in data_list]

                torch.save(self.collate(data_list), self.processed_paths[0])
                torch.save(split_slices, self.processed_paths[1])

            def _build_class_mapping(self) -> Dict[str, int]:
                mapping: Dict[str, int] = {}
                split_dir = self.raw_paths[1]
                for split in MALNET_TINY_SPLITS:
                    path = osp.join(split_dir, f'{split}.txt')
                    if not os.path.exists(path):
                        continue
                    with open(path, 'r') as f:
                        filenames = [fn for fn in f.read().split('\n') if fn]
                    for filename in filenames:
                        malware_type = filename.split('/')[0]
                        mapping.setdefault(malware_type, len(mapping))
                return mapping

        return _LocalMalNetTiny
    except Exception:
        return None


MalNetTiny = _build_local_malnet_tiny()


def _graph_num_nodes(data_obj) -> int:
    if hasattr(data_obj, 'num_nodes') and data_obj.num_nodes is not None:
        return int(data_obj.num_nodes)
    if hasattr(data_obj, 'x') and data_obj.x is not None:
        return int(data_obj.x.size(0))
    raise ValueError('Graph object is missing node count information.')


def _find_benign_label(dataset: 'MalNetTiny') -> int:
    """
    Infer which label index corresponds to the benign class.
    Falls back to 0 if no mapping information is available.
    """
    def _matches(mapping) -> Optional[int]:
        for key, value in mapping.items():
            if isinstance(key, str) and key.lower() in BENIGN_CLASS_NAMES:
                return int(value)
            if isinstance(value, str) and value.lower() in BENIGN_CLASS_NAMES:
                return int(key)
        return None

    for attr in ('class_mapping', 'label_dict', 'class_dict', 'label_map'):
        mapping = getattr(dataset, attr, None)
        if isinstance(mapping, dict):
            benign = _matches(mapping)
            if benign is not None:
                return benign

    for attr in ('class_names', 'classes', 'labels'):
        names = getattr(dataset, attr, None)
        if isinstance(names, (list, tuple)):
            for idx, name in enumerate(names):
                if isinstance(name, str) and name.lower() in BENIGN_CLASS_NAMES:
                    return idx

    return 0


class _ListInMemoryDataset(InMemoryDataset):
    """
    Minimal InMemoryDataset wrapper around an in-memory list of graphs.
    """
    def __init__(self, data_list, num_classes: Optional[int] = None):
        super().__init__('.')
        self.data, self.slices = self.collate(data_list)
        self._num_classes = num_classes

    @property
    def num_classes(self):
        return self._num_classes


def load_malnet_tiny(split: str = 'train') -> 'MalNetTiny':
    """
    Load a MalNetTiny split (train/val/test) using the shared dataset root.
    """
    if MalNetTiny is None:
        raise ImportError('MalNetTiny dataset is unavailable; please install torch_geometric with MalNetTiny support.')
    split = split.lower()
    if split not in ('train', 'val', 'test'):
        raise ValueError('split must be one of {"train", "val", "test"}.')
    return MalNetTiny(root=dataset_root(), split=split)


def load_malnet_tiny_splits() -> Dict[str, 'MalNetTiny']:
    """
    Convenience helper returning all MalNetTiny splits keyed by split name.
    """
    return {split: load_malnet_tiny(split) for split in ('train', 'val', 'test')}


def load_malnet_tiny_filtered(max_nodes: int = MALNET_TINY_MAX_NODES, seed: int = 0) -> InMemoryDataset:
    """
    Load MalNetTiny graphs with controlled class balance and node budget.
    - Collect all benign graphs with num_nodes <= max_nodes_cap (default 500).
    - Collect malware graphs under the same node cap and limit their count to the benign count.
    Combines official splits and binarizes labels (0: benign, 1: malware).
    """
    datasets = load_malnet_tiny_splits()
    benign_label = _find_benign_label(datasets['train'])
    rng = random.Random(seed)
    node_limit = max_nodes
    benign_graphs = []
    malware_graphs = []
    for ds in datasets.values():
        for graph in ds:
            num_nodes = _graph_num_nodes(graph)
            if num_nodes <= node_limit:
                graph = graph.clone()
                if getattr(graph, 'x', None) is None:
                    graph.x = torch.ones((num_nodes, 1), dtype=torch.float32)
                raw_label = int(graph.y.view(-1)[0].item())
                binary_label = 0 if raw_label == benign_label else 1
                graph.y = graph.y.new_tensor([binary_label])
                if binary_label == 0:
                    benign_graphs.append(graph)
                else:
                    malware_graphs.append(graph)
    if not benign_graphs:
        raise ValueError(f'No benign MalNetTiny graphs found with <= {node_limit} nodes.')
    if not malware_graphs:
        raise ValueError(f'No malware MalNetTiny graphs found with <= {node_limit} nodes.')

    rng.shuffle(malware_graphs)
    malware_graphs = malware_graphs[:len(benign_graphs)]
    combined_graphs = benign_graphs + malware_graphs
    rng.shuffle(combined_graphs)
    num_classes = 2
    return _ListInMemoryDataset(combined_graphs, num_classes=num_classes)


def load_malnet_tiny_filtered_splits(max_nodes: int = MALNET_TINY_MAX_NODES,
                                     seed: int = 0) -> Tuple[InMemoryDataset, Dict[str, List[int]]]:
    """
    Return (dataset, splits) for the filtered MalNetTiny subset.
    Splits are deterministic 80/10/10 (train/val/test) similar to TU datasets.
    """
    dataset = load_malnet_tiny_filtered(max_nodes=max_nodes, seed=seed)
    split_dataset_name = f'MalNetTiny_sub{max_nodes}'
    splits = load_dataset_splits(split_dataset_name, len(dataset), seed=seed)
    return dataset, splits


def _read_split(path: Path) -> List[int]:
    return list(ast.literal_eval(path.read_text()))


def _write_split(path: Path, indices: List[int]) -> None:
    path.write_text(str(indices))


def create_splits(dataset_name: str,
                  dataset_len: int,
                  seed: int = 0) -> None:
    """
    Deterministically create train/val/test splits (80/10/10) compatible with existing code.
    """
    if dataset_len <= 0:
        raise ValueError('Dataset must contain at least one graph to create splits.')

    split_paths = _split_paths(dataset_name)
    if all(p.exists() for p in split_paths.values()):
        return

    indices = list(range(dataset_len))
    rng = random.Random(seed)
    rng.shuffle(indices)
    fold = max(1, dataset_len // 10)
    test_indices = indices[:fold]
    val_indices = indices[fold:2 * fold]
    train_indices = indices[2 * fold:]
    if not train_indices:
        train_indices = indices

    for split, idx in (('train', train_indices),
                       ('val', val_indices),
                       ('test', test_indices)):
        _write_split(split_paths[split], idx)


def load_dataset_splits(dataset_name: str,
                        dataset_len: Optional[int] = None,
                        create_if_missing: bool = True,
                        seed: int = 0) -> Dict[str, List[int]]:
    """
    Load dataset split indices, optionally creating them if they do not exist.
    """
    if is_malnet_tiny(dataset_name):
        raise ValueError('MalNetTiny provides built-in splits; use load_malnet_tiny_splits() or load_malnet_tiny_filtered_splits().')

    split_paths = _split_paths(dataset_name)
    missing = [p for p in split_paths.values() if not p.exists()]

    if missing:
        if not create_if_missing:
            names = ', '.join(p.name for p in missing)
            raise FileNotFoundError(
                f'Missing split files ({names}) for dataset {dataset_name}.'
            )
        if dataset_len is None:
            raise ValueError('dataset_len is required to create missing splits.')
        create_splits(dataset_name, dataset_len, seed=seed)

    return {split: _read_split(split_paths[split])
            for split in ('train', 'val', 'test')}


__all__ = [
    'DATASET_DIR',
    'DATA_SPLIT_DIR',
    'TUD_DEGREE_ATTR',
    'MALNET_TINY_MAX_NODES',
    'is_malnet_tiny',
    'load_malnet_tiny',
    'load_malnet_tiny_splits',
    'load_malnet_tiny_filtered',
    'load_malnet_tiny_filtered_splits',
    'create_splits',
    'dataset_root',
    'load_dataset_splits',
    'load_tud_dataset',
]
