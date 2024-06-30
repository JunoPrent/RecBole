# -*- coding: utf-8 -*-
# @Time   : 2020/6/25
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/9/16
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
BPR
################################################
Reference:
    Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." in UAI 2009.
"""

import torch
import torch.nn as nn
import pandas as pd
import csv
import json

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class BPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way."""
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(BPR, self).__init__(config, dataset)
        
        self.mainstream_labels = pd.read_csv(f"dataset/{config['dataset']}/{config['dataset']}.user", sep="\t", engine="python", index_col=0, header=0)["mainstream class (even groups)"]
        # self.item_info = pd.read_csv(f"dataset/{config['dataset']}/{config['dataset']}.item", sep="\t", engine="python", index_col=0, header=0)
        # self.inter_info = pd.read_csv(f"dataset/{config['dataset']}/{config['dataset']}.inter", sep="\t", engine="python", header=0)
        with open(f"dataset/remappings/{config['dataset']}.json") as f:
            remappings = json.load(f)

        del remappings["user_id"]["[PAD]"]
        del remappings["item_id"]["[PAD]"]

        # self.inter_info["user_id:token"] = self.inter_info["user_id:token"].map({int(k): v for (k, v) in remappings["user_id"].items()})
        # self.inter_info["item_id:token"] = self.inter_info["item_id:token"].map({int(k): v for (k, v) in remappings["item_id"].items()})
        self.mainstream_labels = pd.Series(self.mainstream_labels.values, index=[remappings["user_id"][str(i)] for i in self.mainstream_labels.index])

        # self.item_labels = self.item_info.loc[self.item_info["total ratings (%)"] > 0]["popular item"]
        # self.item_labels = pd.Series(self.item_labels.values, index=[remappings["item_id"][str(i)] for i in self.item_labels.index])
        self.lambdas = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.IPS = False
        # load parameters info
        self.embedding_size = config["embedding_size"]

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()

        self.train_data_dist = {user_idx: [] for user_idx in list(self.mainstream_labels.index)}
        self.train_items = []
        self.loss_path = f"dataset/training_losses/bpr/{config['dataset']}/0001-128_item.csv"

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):
        r"""Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        r"""Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding(item)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        
        self.loss.batch_labels = self.mainstream_labels.loc[user].values
        self.loss.batch_item_labels = self.item_labels.loc[pos_item].values
        self.loss.lambdas = self.lambdas
        self.loss.IPS = self.IPS
        self.loss.batch_item_counts = torch.tensor(self.train_item_counts.loc[pos_item].values)

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(
            user_e, neg_e
        ).sum(dim=1)
        loss = self.loss(pos_item_score, neg_item_score)

        tracking_data = list(zip([int(self.epoch_idx)] * len(user), user.numpy(), pos_item.numpy(), pos_item_score.data.numpy(), loss.data.numpy()))
        with open(self.loss_path, "a") as f:
            writer = csv.writer(f)
            writer.writerows(tracking_data)
            
        loss = loss.mean()

        if self.epoch_idx == 0:
        #     self.train_items += [int(i) for i in pos_item.tolist()]
            for u, i in list(zip(user.tolist(), pos_item.tolist())):
                self.train_data_dist[u].append(self.item_labels.loc[i])

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
