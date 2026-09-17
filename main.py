import argparse
import csv
import json
import os
import random
import secrets
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from .Data import load_dataset
    from .Net import GraphSage
    from .causal_sampling import cs_sampling
except ImportError:
    from Data import load_dataset
    from Net import GraphSage
    from causal_sampling import cs_sampling


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True, warn_only=True)


def generate_random_seeds(num_runs):
    generator = secrets.SystemRandom()
    seeds = set()
    while len(seeds) < int(num_runs):
        seeds.add(generator.randrange(1, 2**32 - 1))
    return list(seeds)


def normalize_features(x):
    x = np.asarray(x, dtype=np.float32)
    sums = x.sum(axis=1, keepdims=True)
    sums = np.where(np.abs(sums) < 1e-12, 1.0, sums)
    return x / sums


def parse_int_list(text):
    return [int(v.strip()) for v in str(text).split(',') if v.strip()]


def tensorize(features, node_sets, masks, device):
    padded_features = np.vstack([features, np.zeros((1, features.shape[1]), dtype=np.float32)])
    node_tensors = [torch.from_numpy(padded_features[index]).float().to(device) for index in node_sets]
    mask_tensors = [torch.from_numpy(mask).float().to(device) for mask in masks]
    return node_tensors, mask_tensors


def train_once(data, seed, fanouts, cooperative_nums, hidden_dim, epochs, batches_per_epoch, batch_size, eval_batch_size, lr, weight_decay, device):
    set_seed(seed)
    x = normalize_features(data.x)
    train_index = np.where(data.train_mask)[0].astype(np.int64)
    test_index = np.where(data.test_mask)[0].astype(np.int64)
    input_dim = int(x.shape[1])
    num_classes = int(np.max(data.y)) + 1
    hidden_dims = [int(hidden_dim)] * (len(fanouts) - 1) + [num_classes]
    model = GraphSage(input_dim, hidden_dims, fanouts).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    criterion = nn.CrossEntropyLoss().to(device)
    pad_node = int(x.shape[0])
    batch_rng = np.random.default_rng((int(seed) + 1000003) % (2**32 - 1))
    train_sampling_rng = np.random.default_rng((int(seed) + 2000003) % (2**32 - 1))
    test_sampling_rng = np.random.default_rng((int(seed) + 3000017) % (2**32 - 1))
    for _ in range(int(epochs)):
        model.train()
        for _ in range(int(batches_per_epoch)):
            batch_nodes = batch_rng.choice(train_index, size=int(batch_size), replace=train_index.size < int(batch_size))
            node_sets, masks = cs_sampling(batch_nodes, fanouts, data.adjacency_dict, data.y, cooperative_nums, train_sampling_rng, pad_node)
            node_tensors, mask_tensors = tensorize(x, node_sets, masks, device)
            labels = torch.from_numpy(data.y[batch_nodes]).long().to(device)
            logits = model(node_tensors, mask_tensors)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, test_index.size, int(eval_batch_size)):
            batch_nodes = test_index[start:start + int(eval_batch_size)]
            node_sets, masks = cs_sampling(batch_nodes, fanouts, data.adjacency_dict, data.y, cooperative_nums, test_sampling_rng, pad_node)
            node_tensors, mask_tensors = tensorize(x, node_sets, masks, device)
            labels = torch.from_numpy(data.y[batch_nodes]).long().to(device)
            logits = model(node_tensors, mask_tensors)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(batch_nodes.size)
    return correct / max(1, total)


def run(args):
    data = load_dataset(args.dataset, args.data_root)
    fanouts = parse_int_list(args.fanouts)
    cooperative_nums = parse_int_list(args.cooperative_m)
    if len(cooperative_nums) == 1:
        cooperative_nums = cooperative_nums * len(fanouts)
    if len(fanouts) != len(cooperative_nums):
        raise ValueError('fanouts and cooperative_m must have the same number of layers')
    seeds = generate_random_seeds(args.num_runs)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for run_index, seed in enumerate(seeds, start=1):
        accuracy = train_once(data, seed, fanouts, cooperative_nums, args.hidden_dim, args.epochs, args.batches_per_epoch, args.batch_size, args.eval_batch_size, args.lr, args.weight_decay, device)
        row = {'run': int(run_index), 'accuracy': float(accuracy)}
        rows.append(row)
        print(json.dumps(row))
    values = np.asarray([row['accuracy'] for row in rows], dtype=np.float64)
    summary = {
        'dataset': args.dataset,
        'num_runs': int(args.num_runs),
        'mean_accuracy': float(values.mean()),
        'std_accuracy': float(values.std(ddof=1)) if values.size > 1 else 0.0,
        'fanouts': fanouts,
        'cooperative_m': cooperative_nums
    }
    with open(os.path.join(args.output_dir, f'{args.dataset}_per_run_results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run', 'accuracy'])
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(args.output_dir, f'{args.dataset}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cora', 'citeseer', 'pubmed', 'coauthor-cs', 'ogbn-arxiv'], default='cora')
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--num-runs', type=int, default=10)
    parser.add_argument('--fanouts', default='10,10')
    parser.add_argument('--cooperative-m', default='10,10')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batches-per-epoch', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', default='results')
    return parser


if __name__ == '__main__':
    run(build_parser().parse_args())
