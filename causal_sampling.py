from itertools import combinations
from math import comb
import numpy as np


def _unique_neighbors(neighbor_table, sid):
    return np.asarray(list(dict.fromkeys(int(v) for v in neighbor_table.get(int(sid), []) if int(v) != int(sid))), dtype=np.int64)


def _softmax(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values)
    exp_values = np.exp(np.clip(shifted, -700.0, 700.0))
    total = exp_values.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(values.size, 1.0 / values.size, dtype=np.float64)
    return exp_values / total


def _conditional_kde_probability(v_t, context_nodes, labels):
    context_nodes = np.asarray(context_nodes, dtype=np.int64)
    if context_nodes.size == 0:
        return 1.0 / max(1, labels.shape[0])
    kernel_values = (labels[context_nodes] == labels[int(v_t)]).astype(np.float64)
    probability = float(kernel_values.mean())
    return max(probability, 1.0 / max(1, labels.shape[0]))


def _causal_weight(v_t, context_nodes, labels, M):
    p_hat = _conditional_kde_probability(v_t, context_nodes, labels)
    return 1.0 / (int(M) * p_hat)


def _marginal_causal_contribution(v_r, v_t, S_j, labels, M):
    S_j = tuple(int(v) for v in S_j)
    with_vr = tuple(dict.fromkeys(S_j + (int(v_r),)))
    return _causal_weight(v_t, with_vr, labels, M) - _causal_weight(v_t, S_j, labels, M)


def cs_backdoor(v_r, candidate_nodes, M, labels):
    candidate_nodes = np.asarray(list(dict.fromkeys(int(v) for v in candidate_nodes if int(v) != int(v_r))), dtype=np.int64)
    T = int(candidate_nodes.size)
    if T == 0:
        return np.empty(0, dtype=np.float64)
    M = min(max(1, int(M)), T)
    coalition_size = M - 1
    coefficient = 1.0 / (M * comb(T, M))
    cooperative_weights = np.zeros(T, dtype=np.float64)
    for index, v_t in enumerate(candidate_nodes):
        remaining = [int(v) for v in candidate_nodes if int(v) != int(v_t)]
        value = 0.0
        for S_j in combinations(remaining, coalition_size):
            value += coefficient * _marginal_causal_contribution(v_r, int(v_t), S_j, labels, M)
        cooperative_weights[index] = value
    return cooperative_weights


def _pad(selected, fanout, pad_node):
    selected = np.asarray(selected, dtype=np.int64)
    mask = np.ones(selected.size, dtype=np.float32)
    if selected.size < int(fanout):
        padding = int(fanout) - selected.size
        selected = np.concatenate([selected, np.full(padding, int(pad_node), dtype=np.int64)])
        mask = np.concatenate([mask, np.zeros(padding, dtype=np.float32)])
    return selected, mask


def sampling(src_nodes, sample_num, neighbor_table, labels, M, rng, pad_node):
    results = []
    masks = []
    for sid in np.asarray(src_nodes, dtype=np.int64):
        if int(sid) == int(pad_node):
            selected = np.full(int(sample_num), int(pad_node), dtype=np.int64)
            mask = np.zeros(int(sample_num), dtype=np.float32)
        else:
            candidates = _unique_neighbors(neighbor_table, sid)
            if candidates.size <= int(sample_num):
                selected, mask = _pad(candidates, int(sample_num), int(pad_node))
            else:
                cooperative_weights = cs_backdoor(int(sid), candidates, int(M), labels)
                probabilities = _softmax(cooperative_weights)
                selected = rng.choice(candidates, size=int(sample_num), replace=False, p=probabilities)
                mask = np.ones(int(sample_num), dtype=np.float32)
        results.append(selected)
        masks.append(mask)
    return np.concatenate(results), np.concatenate(masks)


def cs_sampling(src_nodes, sample_nums, neighbor_table, labels, cooperative_nums, rng, pad_node):
    sample_nums = [int(v) for v in sample_nums]
    if np.isscalar(cooperative_nums):
        cooperative_nums = [int(cooperative_nums)] * len(sample_nums)
    cooperative_nums = [int(v) for v in cooperative_nums]
    if len(sample_nums) != len(cooperative_nums):
        raise ValueError('sample_nums and cooperative_nums must have the same length')
    sampling_result = [np.asarray(src_nodes, dtype=np.int64)]
    mask_result = []
    for layer, fanout in enumerate(sample_nums):
        sampled, mask = sampling(sampling_result[layer], fanout, neighbor_table, labels, cooperative_nums[layer], rng, pad_node)
        sampling_result.append(sampled)
        mask_result.append(mask)
    return sampling_result, mask_result


multihop_sampling = cs_sampling
