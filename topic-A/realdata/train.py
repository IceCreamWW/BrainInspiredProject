#!/usr/bin/env python
# coding: utf-8

import os
import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import string
import numpy as np
from datetime import datetime
import time
import sys


class Dataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.labels = labels
        self.features = features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class Net(nn.Module):
    def __init__(self, freeze_id, embedding_matrix):
        super(Net, self).__init__()

        self.freeze_id = freeze_id

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.load_state_dict({'weight': torch.from_numpy(embedding_matrix)})

        self.lstm = nn.LSTM(60, 128, 2, batch_first=True, dropout=0.6)
        self.linear1 = nn.Linear(128, 32)
        self.linear2 = nn.Linear(32, 2)

    def forward(self, token_ids):

        with torch.no_grad():
            self.embedding.weight[self.freeze_id] = 0

        tokens = self.embedding(token_ids)

        output, (hn, cn) = self.lstm(tokens)

        output = self.linear2(torch.relu(self.linear1(hn[-1])))
        return output


if __name__=="__main__":
    n_samples = int(sys.argv[1])

    # load data
    data = np.loadtxt("data.npy").astype(np.int)
    features = data[:,:-1]
    labels = data[:,-1]

    # load embedding
    embedding_matrix = np.loadtxt("embedding.npy")

    # load NIL id
    with open("freeze_id", "r") as fp:
        freeze_id = int(fp.read())


    features_train = features[:n_samples]
    labels_train = labels[:n_samples]

    # set random seed
    seed = int((time.time() * 1e5) % 1e6)
    torch.manual_seed(seed)

    max_epochs = 200 
    batch_size = 1024
    train_set = Dataset(features_train, labels_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True)

    model = Net(freeze_id, embedding_matrix)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # random initialization
    for parameter in model.parameters():
        parameter.data.normal_(0,0.5)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    history = []
    for epoch in range(max_epochs):
        loss_ = 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            pred = model(feats)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ += loss.item()

        print(f"epoch: {epoch}\t loss: {loss_ / len(train_loader)}")
        history.append(loss_ / len(train_loader))


    dirname = os.path.join("exp", str(n_samples), str(seed))
    os.makedirs(dirname, exist_ok=True)

    # save loss history
    path_to_history = os.path.join(dirname, "history.npy")
    np.savetxt(path_to_history, np.array(history))

    # save model status dict
    path_to_model = os.path.join(dirname, "mdl.th")
    torch.save(model.state_dict(), path_to_model)


