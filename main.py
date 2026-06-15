import argparse
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from Net import GraphSage
from sampling_cs import cs_sampling
import time
from sklearn.metrics import f1_score
import torch.nn.functional as F
from Data import Coauthor
device = "cuda" if torch.cuda.is_available() else "cpu"

input_dim = 6805
hidden_dim = [6805, 15]
sampling_num = [10, 10]
batch_size = 50
epochs = 200
batch_per_epoch = 20
learning_rate = 0.01
patience = 50

data = Coauthor().make_dataset()
adj = data.adj_list
features = data.features
labels = data.labels

train_idx = data.train_index
train_labels = labels[train_idx]
val_idx = data.val_index
val_labels = labels[val_idx]
test_idx = data.test_index
test_labels = labels[test_idx]

model = GraphSage(input_dim=input_dim, hidden_dim=hidden_dim, num_neighbors_list=sampling_num).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

con_Us = np.random.randint(1, 3, 18333)

def train():
    model.train()
    model.train()
    start = time.time()
    val_loss = []
    val_acc = []
    best_acc = []
    sum_acc, sum_f1 = 0, 0
    loss_min = np.inf
    acc_max = np.inf
    loss_best = np.inf
    bad_counter = 0
    best_epoch = 0

    for e in range(epochs):
        start_time = time.time()
        for batch in range(batch_per_epoch):
            batch_index = np.random.choice(train_idx, size=(batch_size, ), replace=False)
            batch_sampling_result = cs_sampling(batch_index, sampling_num, adj, con_Us)
            batch_features = [torch.from_numpy(features[idx]).float().to(device) for idx in batch_sampling_result]
            batch_output = model(batch_features)
            batch_label = torch.from_numpy(labels[batch_index]).long().to(device)
            loss = criterion(batch_output, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        end_time = time.time()
        use_time = end_time - start_time

        val_sampling_result = cs_sampling(val_idx, sampling_num, adj, con_Us)
        val_features = [torch.from_numpy(features[idx]).float().to(device) for idx in val_sampling_result]
        val_output = model(val_features)
        val_label = torch.from_numpy(labels[val_idx]).long().to(device)
        val_loss = criterion(val_output, val_label)
        val_pred = val_output.max(1)[1]
        val_acc = torch.eq(val_pred, val_label).float().mean()
        print("epoch:{:04d}".format(e+1),
              "train_loss:{:.4f}".format(loss.item()),
              "val_val:{:.4f}".format(val_loss.item()),
              'acc_val:{:.4f}'.format(val_acc.item()))

        val_loss.append(val_loss.item())

        if val_loss[-1] <= loss_min:
            if val_loss[-1] <= loss_best:
                loss_best = val_loss[-1]
                best_epoch = e
                torch.save(model.state_dict(), 'Coauthor_cs.pkl')

            loss_min = np.min((val_loss[-1], loss_min))
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == patience:
            break

    print('loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('Coauthor_cs.pkl'))

    result = []
    for i in range(10):
        acc, f1 = test()
        sum_acc += acc
        sum_f1 += f1
        result.append(acc)
    avg_acc, std_acc = calculate_metrics(result)
    avg_f1 = sum_f1 / 10
    print(avg_acc,  std_acc)
    print( avg_f1)
    print( use_time)


def test():
    model.eval()
    with torch.no_grad():
        test_sampling_result = cs_sampling(test_idx, sampling_num, adj, con_Us)
        test_faetures = [torch.from_numpy(features[idx]).float().to(device) for idx in test_sampling_result]
        test_output = model(test_faetures)
        test_label = torch.from_numpy(labels[test_idx]).long().to(device)
        predict_y = test_output.max(1)[1]
        accuracy = torch.eq(predict_y, test_label).float().mean().item()
        f1 = f1_score(y_true=test_label, y_pred=predict_y, average='macro')
    return accuracy, f1


def calculate_metrics(results):
    avg = np.mean(results)
    std = np.std(results)
    return avg, std


if __name__ == "__main__":
    train()
