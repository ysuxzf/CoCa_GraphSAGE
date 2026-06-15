from math import factorial
import numpy as np
import torch
import argparse
from itertools import combinations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from Data import Coauthor

dataset = Coauthor().make_dataset()

labels = dataset.labels

def sampling(src_nodes, sample_num, neighbor_table, con):

    results = []
    # print(src_nodes)
    for sid in src_nodes:
        # print('sid', sid)
        res = np.random.choice(neighbor_table[sid], size=(sample_num,))
        # print(neighbor_table[sid])
        neighbor_node = np.asarray(res).flatten()
        con_us = []
        for i in neighbor_node:
            con_us.append(con[i])
        label_Y = labels[neighbor_node]
        label_yr = np.unique(label_Y)
        con_ur = np.unique(con_us)  # ur
        hopk_result1 = cs_backdoor(neighbor_node, label_Y, con_us, label_yr, con_ur, neighbor_node.size)
        results.append(hopk_result1)
    return np.asarray(results).flatten()  # flatten作用是把results降为一维


def cs_sampling(src_nodes, sample_nums, neighbor_table, con_us):

    sampling_result = [src_nodes]
    # print('sampling_result', sampling_result)
    for k, hopk_num in enumerate(sample_nums):
        # print('sampling_result[k]', k, sampling_result[k])
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table, con_us)
        sampling_result.append(hopk_result)
    return sampling_result


def rand_cat_fast(p, N):

    K = len(p)
    u = np.random.rand(N, 1)
    P = np.cumsum(p, axis=0)
    U = np.tile(P.T, (N, 1))
    c = np.tile(u, (1, K)) >= U
    c = c + 0
    x = np.sum(c, 1) + 1
    return x


def histcnd(Y, U, yr, ur):
    Nyu = np.zeros((15, 14), dtype=int)
    for index, i in enumerate(Y):
        Nyu[int(i)][int(U[index])] += 1
    return Nyu


def norm(value):

    if value.sum(0) != 0:
        value = value/value.sum(0)
    else:
        value = 0
    return value


def cs_backdoor(X, Y, U, yr, ur, M):

    M = M-1
    N = len(X)
    shapley_value = np.zeros((N, 1), dtype=float)
    for a in range(N):
        x = (X[a], )
        move_index = [a]
        R = np.delete(X, move_index)
        for i in range(0, N):
            for X_subsets in combinations(R, i):
                X_sub = X_subsets + x
                H = len(X_sub)
                X_index = np.asarray(X_sub).flatten()
                sub_label = labels[X_index]
                sub_yr = np.unique(sub_label)
                con_us = []
                for m in range(1, H):
                    con_us.append(U[m])
                con_ur = np.unique(con_us)
                s = len(X_subsets)
                weight = factorial(s) * factorial(N - s - 1) / factorial(N)

                Nyu = histcnd(sub_label, U, sub_yr, con_ur)
                pyu = Nyu/H
                pu = np.sum(pyu, axis=0, keepdims=True).T
                py_u = pyu/pu.T

                v = np.zeros((H, 1), dtype=float)
                index_i = []
                for k in sub_yr:
                    for index, j in enumerate(sub_label):
                        if j == k:
                            v[index][0] = j / (py_u[k-1][U[index]-1]).T / M
                            contrib = v[H-1][0]
                            shapley_value[a] += weight * contrib

    Nyu = histcnd(Y, U, yr, ur)
    Ns = np.sum(Nyu, axis=1)
    for i in yr:
        a = rand_cat_fast(shapley_value, Ns[i - 1])
        for i in a:
            index_i.append(i)
    x1 = []
    for i in index_i:
        if i == len(index_i) + 1:
            x1.append(X[len(index_i)-1])
        else:
            x1.append(X[i-1])
    Xw = np.array(x1)
    return Xw

