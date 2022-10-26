#!/usr/bin/env python3

# !/usr/bin/env python3

# !/usr/bin/env python3

# !/usr/bin/env python3

# !/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
import torch.nn.functional as F
import os
import random

import torch
import torch.nn as nn  ##########
import torch.utils.data as Data
from torch.autograd import Variable
from torch import optim
from sklearn.model_selection import train_test_split
import itertools

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_curve, roc_auc_score, auc, \
    precision_recall_curve
# from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
from util import calculateScore, get_k_fold_data,analyze,get_index

# ----->>
# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle


# -------------------------------------->>>>

# --------------------------------------------------->>>
class BahdanauAttention(nn.Module):
    """
    input: from RNN module h_1, ... , h_n (batch_size, seq_len, units*num_directions),
                                    h_n: (num_directions, batch_size, units)
    return: (batch_size, num_task, units)
    """

    def __init__(self, in_features, hidden_units, num_task):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.W2 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.V = nn.Linear(in_features=hidden_units, out_features=num_task)

    def forward(self, hidden_states, values):
        hidden_with_time_axis = torch.unsqueeze(hidden_states, dim=1)

        score = self.V(nn.Tanh()(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = nn.Softmax(dim=1)(score)
        values = torch.transpose(values, 1, 2)
        context_vector = torch.matmul(values, attention_weights)
        context_vector = torch.transpose(context_vector, 1, 2)
        return context_vector, attention_weights


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=device, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)
        # print(pos)

        embedding = self.pos_embed(pos)
        embedding = embedding + self.tok_embed(x)
        embedding = self.norm(embedding)
        return embedding

# https://github.com/Amanbhandula/AlphaPose/blob/master/train_sppe/src/models/layers/SE_module.py
class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(channel, int(channel / reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel / reduction), channel),
            nn.Sigmoid())
        # self.fc2 = nn.Sequential(
        # nn.Conv1d(channel, int(channel / reduction), 1, bias=False),
        # nn.ReLU(inplace=True),
        # nn.Conv1d(channel, int(channel / reduction), 1, bias=False),
        # nn.Sigmoid()
        # )

    def forward(self, x):
        b, c, _= x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y).view(b, c, 1)
        return x * y


class Emb_CNNGRU_ATT(nn.Module):
    def __init__(self):
        super(Emb_CNNGRU_ATT, self).__init__()
        kernel_size = 10
        max_len = 3075
        d_model = d_model
        vocab_size = 28

        self.embedding = Embedding(vocab_size, d_model, max_len)
        # ---------------->>>
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=d_model,  # input height
                out_channels=256,  # n_filters
                kernel_size=kernel_size, groups=4),
            # dilation =2),     #!!!
            # padding = int(kernel_size/2)),
            # padding=(kernel_size-1)/2
            nn.ReLU(),  # activation
            # nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5))

        self.lstm = torch.nn.LSTM(256, 128, 1, batch_first=True, bidirectional=True)  #
        # self.gru = torch.nn.GRU(256, 128, 1, batch_first=True, bidirectional=True)
        self.Attention = BahdanauAttention(in_features=256, hidden_units=10, num_task=1)
        self.Se_Attention = SELayer(256)

        self.fc_task = nn.Sequential(
            nn.Linear(256, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.classifier = nn.Linear(2, 1)

    # ---------------->>>
    def forward(self, x):
        x = self.embedding(x)
        print(x.shape)
        x = x.transpose(1, 2)
        batch_size, features, seq_len = x.size()

        x = self.conv1(x)
        print(x.shape)
        x = self.Se_Attention(x)
        print('*'*111)
        print(x.shape)
        x = x.transpose(1, 2)
        # rnn layer
        out, (h_n, c_n) = self.lstm(x)
        h_n = h_n.view(batch_size, out.size()[-1])
        context_vector, attention_weights = self.Attention(h_n, out)
        reduction_feature = self.fc_task(torch.mean(context_vector, 1))

        representation = reduction_feature
        logits_clsf = self.classifier(representation)
        logits_clsf1 = logits_clsf
        logits_clsf = torch.sigmoid(logits_clsf)

        return logits_clsf, representation








