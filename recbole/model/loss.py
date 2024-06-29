# @Time   : 2020/6/26
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/7, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com


"""
recbole.model.loss
#######################
Common Loss in recommender system
"""

import torch
import torch.nn as nn


class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    
    def MAD(self, group_avgs):
        return sum(abs(group_avgs - group_avgs.mean())) / len(group_avgs)


    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score))
        # label_count = len(set(self.batch_labels))
        fairness = 0
        
        if any(self.lambdas):
            # # G = torch.tensor([[int(i == j) * (i+1) for j in self.batch_labels] for i in sorted(set(self.batch_labels))])
            # G = torch.tensor([[int(i == j) for j in self.batch_labels] for i in sorted(set(self.batch_labels))])
            Gi = torch.tensor([[int(i == j) for j in self.batch_item_labels] for i in ["H", "M", "T"]])
            # group_avgs = torch.mul(loss, G).sum(dim=1) / G.sum(dim=1)
            group_avgs_item = torch.mul(loss, Gi).sum(dim=1) / Gi.sum(dim=1)
            # print(group_avgs_item)

            # std
            if self.lambdas[0] > 0.0:
                # fairness += self.lambdas[0] * torch.std(group_avgs)
                fairness += self.lambdas[0] * torch.std(group_avgs_item)
            # entropy
            if self.lambdas[1] > 0.0:
                fairness += self.lambdas[1] * (len(group_avgs_item) - torch.exp(torch.distributions.Categorical(group_avgs_item).entropy()))
            # euclidean
            if self.lambdas[2] > 0.0:
                # fairness += self.lambdas[2] * torch.dist(group_avgs_item, torch.tensor([0., 0., 0.]), 2)
                fairness += self.lambdas[2] * torch.dist(group_avgs_item, group_avgs_item.mean().repeat(len(group_avgs_item)), 2)
            # kl-divergence
            if self.lambdas[3] > 0.0:
                kl_div = torch.nn.KLDivLoss(reduction="batchmean")
                # fairness += self.lambdas[3] * abs(kl_div(group_avgs_item.mean().repeat(len(group_avgs_item)), group_avgs_item))
                fairness += self.lambdas[3] * abs(kl_div(torch.tensor([0., 0., 0.]), group_avgs_item))
            # mad
            if self.lambdas[4] > 0.0:
                # print(self.MAD(torch.tensor([1.0, 2.0, 3.0])))
                fairness += self.lambdas[4] * self.MAD(group_avgs_item)
        
        if self.IPS:
            loss = torch.div(loss, self.batch_item_counts)
        
        return loss + fairness


class RegLoss(nn.Module):
    """RegLoss, L2 regularization on model parameters"""

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss


class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


class EmbMarginLoss(nn.Module):
    """EmbMarginLoss, regularization on embeddings"""

    def __init__(self, power=2):
        super(EmbMarginLoss, self).__init__()
        self.power = power

    def forward(self, *embeddings):
        dev = embeddings[-1].device
        cache_one = torch.tensor(1.0).to(dev)
        cache_zero = torch.tensor(0.0).to(dev)
        emb_loss = torch.tensor(0.0).to(dev)
        for embedding in embeddings:
            norm_e = torch.sum(embedding**self.power, dim=1, keepdim=True)
            emb_loss += torch.sum(torch.max(norm_e - cache_one, cache_zero))
        return emb_loss
