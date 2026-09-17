import os
import os.path as osp
import pickle
import urllib.request
from collections import namedtuple
import numpy as np
import scipy.sparse as sp

Data = namedtuple('Data', ['x', 'y', 'adjacency_dict', 'train_mask', 'val_mask', 'test_mask'])


class PlanetoidData:
    download_url = 'https://raw.githubusercontent.com/kimiyoung/planetoid/master/data'

    def __init__(self, name='cora', data_root=None, rebuild=False):
        self.name = name.lower()
        if self.name not in {'cora', 'citeseer', 'pubmed'}:
            raise ValueError(self.name)
        self.data_root = data_root or osp.join(osp.dirname(__file__), self.name)
        self.filenames = [f'ind.{self.name}.{part}' for part in ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]
        self._download()
        self._data = self._process()

    @property
    def data(self):
        return self._data

    def _download(self):
        raw_dir = osp.join(self.data_root, 'raw')
        os.makedirs(raw_dir, exist_ok=True)
        for filename in self.filenames:
            path = osp.join(raw_dir, filename)
            if not osp.exists(path):
                urllib.request.urlretrieve(f'{self.download_url}/{filename}', path)

    @staticmethod
    def _read(path):
        if path.endswith('test.index'):
            return np.genfromtxt(path, dtype=np.int64)
        with open(path, 'rb') as f:
            value = pickle.load(f, encoding='latin1')
        return value.toarray() if hasattr(value, 'toarray') else value

    def _process(self):
        raw_dir = osp.join(self.data_root, 'raw')
        x_train, tx, allx, y_train, ty, ally, graph, test_index = [self._read(osp.join(raw_dir, filename)) for filename in self.filenames]
        test_index = np.asarray(test_index, dtype=np.int64)
        test_index_sorted = np.sort(test_index)
        if self.name == 'citeseer':
            full_range = np.arange(test_index_sorted.min(), test_index_sorted.max() + 1)
            tx_full = np.zeros((len(full_range), tx.shape[1]), dtype=tx.dtype)
            tx_full[test_index_sorted - full_range.min()] = tx
            ty_full = np.zeros((len(full_range), ty.shape[1]), dtype=ty.dtype)
            ty_full[test_index_sorted - full_range.min()] = ty
            tx = tx_full
            ty = ty_full
        x = np.concatenate((allx, tx), axis=0)
        y_onehot = np.concatenate((ally, ty), axis=0)
        x[test_index] = x[test_index_sorted]
        y_onehot[test_index] = y_onehot[test_index_sorted]
        y = y_onehot.argmax(axis=1).astype(np.int64)
        num_nodes = int(x.shape[0])
        train_index = np.arange(y_train.shape[0], dtype=np.int64)
        val_index = np.arange(y_train.shape[0], min(y_train.shape[0] + 500, num_nodes), dtype=np.int64)
        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        adjacency_dict = {i: list(dict.fromkeys(int(v) for v in graph.get(i, []) if int(v) != i)) for i in range(num_nodes)}
        return Data(np.asarray(x, dtype=np.float32), y, adjacency_dict, train_mask, val_mask, test_mask)


class CoraData(PlanetoidData):
    def __init__(self, data_root=None, rebuild=False):
        super().__init__('cora', data_root, rebuild)


class NPZGraphData:
    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        x = self._features(z).astype(np.float32)
        y = self._labels(z)
        if 'edge_index' in z:
            edge_index = np.asarray(z['edge_index'], dtype=np.int64)
            if edge_index.shape[0] != 2:
                edge_index = edge_index.T
        elif {'adj_data', 'adj_indices', 'adj_indptr', 'adj_shape'}.issubset(z.files):
            adj = sp.csr_matrix((z['adj_data'], z['adj_indices'], z['adj_indptr']), shape=tuple(z['adj_shape']))
            edge_index = np.vstack(adj.nonzero()).astype(np.int64)
        else:
            raise ValueError('NPZ file requires edge_index or CSR adjacency fields')
        n = int(x.shape[0])
        adjacency_dict = {i: [] for i in range(n)}
        for a, b in edge_index.T:
            a = int(a)
            b = int(b)
            if a != b:
                adjacency_dict[a].append(b)
                adjacency_dict[b].append(a)
        adjacency_dict = {k: list(dict.fromkeys(v)) for k, v in adjacency_dict.items()}
        train_mask, val_mask, test_mask = self._masks(z, n)
        self._data = Data(x, y, adjacency_dict, train_mask, val_mask, test_mask)

    @property
    def data(self):
        return self._data

    @staticmethod
    def _features(z):
        if 'x' in z:
            return np.asarray(z['x'])
        if 'attr_matrix' in z:
            value = z['attr_matrix']
            if isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
                value = value.item()
            return value.toarray() if hasattr(value, 'toarray') else np.asarray(value)
        fields = {'attr_data', 'attr_indices', 'attr_indptr', 'attr_shape'}
        if fields.issubset(z.files):
            return sp.csr_matrix((z['attr_data'], z['attr_indices'], z['attr_indptr']), shape=tuple(z['attr_shape'])).toarray()
        raise ValueError('NPZ file requires node features')

    @staticmethod
    def _labels(z):
        if 'y' in z:
            labels = np.asarray(z['y'])
        elif 'labels' in z:
            labels = np.asarray(z['labels'])
        else:
            raise ValueError('NPZ file requires labels')
        if labels.ndim > 1:
            labels = labels.argmax(axis=1)
        return labels.reshape(-1).astype(np.int64)

    @staticmethod
    def _mask(z, key, n):
        if key in z:
            return np.asarray(z[key], dtype=bool)
        index_key = key.replace('_mask', '_index')
        if index_key in z:
            mask = np.zeros(n, dtype=bool)
            mask[np.asarray(z[index_key], dtype=np.int64)] = True
            return mask
        return None

    @staticmethod
    def _masks(z, n):
        train_mask = NPZGraphData._mask(z, 'train_mask', n)
        val_mask = NPZGraphData._mask(z, 'val_mask', n)
        test_mask = NPZGraphData._mask(z, 'test_mask', n)
        if train_mask is not None and val_mask is not None and test_mask is not None:
            return train_mask, val_mask, test_mask
        rng = np.random.default_rng(0)
        order = rng.permutation(n)
        n_train = int(round(0.8 * n))
        n_val = int(round(0.1 * n))
        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)
        train_mask[order[:n_train]] = True
        val_mask[order[n_train:n_train + n_val]] = True
        test_mask[order[n_train + n_val:]] = True
        return train_mask, val_mask, test_mask


def load_dataset(name='cora', data_root=None):
    key = name.lower()
    if key in {'cora', 'citeseer', 'pubmed'}:
        return PlanetoidData(key, data_root).data
    if key in {'coauthor-cs', 'coauthor_cs', 'coauthorcs'}:
        path = data_root or osp.join(osp.dirname(__file__), 'coauthor_cs.npz')
        if not osp.exists(path):
            os.makedirs(osp.dirname(path) or '.', exist_ok=True)
            urllib.request.urlretrieve('https://github.com/shchur/gnn-benchmark/raw/master/data/npz/ms_academic_cs.npz', path)
        return NPZGraphData(path).data
    if key in {'ogbn-arxiv', 'ogbn_arxiv'}:
        from ogb.nodeproppred import NodePropPredDataset
        root = data_root or osp.join(osp.dirname(__file__), 'ogb')
        dataset = NodePropPredDataset(name='ogbn-arxiv', root=root)
        graph, labels = dataset[0]
        x = np.asarray(graph['node_feat'], dtype=np.float32)
        y = np.asarray(labels).reshape(-1).astype(np.int64)
        n = int(x.shape[0])
        adjacency_dict = {i: [] for i in range(n)}
        for a, b in np.asarray(graph['edge_index'], dtype=np.int64).T:
            a = int(a)
            b = int(b)
            if a != b:
                adjacency_dict[a].append(b)
                adjacency_dict[b].append(a)
        adjacency_dict = {k: list(dict.fromkeys(v)) for k, v in adjacency_dict.items()}
        split = dataset.get_idx_split()
        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)
        train_mask[np.asarray(split['train'], dtype=np.int64)] = True
        val_mask[np.asarray(split['valid'], dtype=np.int64)] = True
        test_mask[np.asarray(split['test'], dtype=np.int64)] = True
        return Data(x, y, adjacency_dict, train_mask, val_mask, test_mask)
    raise ValueError(name)
