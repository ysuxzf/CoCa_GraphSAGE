import numpy as np
import scipy.sparse as sp
from collections import namedtuple

Coa_CS = namedtuple('Coa_CS', ['features', 'labels', 'adj_list', 'train_index',
                               'val_index', 'test_index'])


class Coauthor:
    def __init__(self):
        self.dataset = np.load('ms_academic_cs.npz', allow_pickle=True)
        self.train_size = 10000
        self.val_size = 2000
        self.test_size = 1000
        self.Coa_cs = self.make_dataset()

    def make_dataset(self):
        loader = dict(self.dataset)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])
        attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                    shape=loader['attr_shape'])
        labels = loader['labels']
        num_samples = len(labels)
        adj_list = [[] for _ in range(num_samples)]

        for i in range(num_samples):
            row_start = adj_matrix.indptr[i]
            row_end = adj_matrix.indptr[i + 1]
            neighbors = adj_matrix.indices[row_start:row_end]
            adj_list[i].extend(neighbors)

        remaining_indices = list(range(num_samples))

        train_indices = np.random.choice(remaining_indices, self.train_size, replace=False)

        remaining_indices1 = np.setdiff1d(remaining_indices, train_indices)
        val_indices = np.random.choice(remaining_indices1, size=self.val_size, replace=False)

        forbidden_indices = np.concatenate((train_indices, val_indices))
        remaining_indices2 = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = np.random.choice(remaining_indices2, size=self.test_size, replace=False)

        assert len(set(train_indices)) == len(train_indices)
        assert len(set(val_indices)) == len(val_indices)
        assert len(set(test_indices)) == len(test_indices)
        assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
        assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
        assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))

        return Coa_CS(adj_list=adj_list, features=attr_matrix.toarray(), labels=labels,
                      train_index=train_indices, val_index=val_indices, test_index=test_indices)






