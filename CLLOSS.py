import os
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F

def norm(t):
	return t / t.norm(dim=1, keepdim=True)

def cos_sim(v1, v2):
	v1 = norm(v1)
	v2 = norm(v2)
	return v1 @ v2.t()

def inner_conduct(x):
    temp = torch.zeros(x.shape[0])
    for i in range(x.shape[0]):
        temp[i] = torch.dot(x[i],x[i])
    return temp.cuda()

def euclidean_dist(x, y):
    dist = x-y
    dist = torch.pow(dist, 2).sum(1, keepdim=True)
    dist = dist.sqrt()
    return dist.cuda()


tau = 0.5
def mm(x, y):
    if args.sparse:
        return torch.sparse.mm(x, y)
    else:
        return torch.mm(x, y)
def sim(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def batched_contrastive_loss(self, z1, z2, batch_size=4096):
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / self.tau)
    indices = torch.arange(0, num_nodes).to(device)
    losses = []

    for i in range(num_batches):
        mask = indices[i * batch_size:(i + 1) * batch_size]
        refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
        between_sim = f(self.sim(z1[mask], z2))  # [B, N]

        losses.append(-torch.log(
            between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
            / (refl_sim.sum(1) + between_sim.sum(1)
                - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

    loss_vec = torch.cat(losses)
    return loss_vec.mean()

def batched_contrastive_loss_1(z1, z2):
    device = z1.device
    f = lambda x: torch.exp(x / tau)
    losses = []

    between_sim = f(sim(z1, z2))  # [B, N] n

    losses.append(-torch.log(
        between_sim.diag() / ( between_sim.sum(1)- between_sim.diag())))

    loss_vec = torch.cat(losses)
    return loss_vec.mean()

def SupConLoss(temperature=0.07, contrast_mode='all', features=None, labels=None, mask=None):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
    device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)  # 1 indicates two items belong to same class
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]  # num of views
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # (bsz * views, dim)

    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature  # (bsz * views, dim)
        anchor_count = contrast_count  # num of views
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T), temperature)  # (bsz, bsz)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # (bsz, 1)
    logits = anchor_dot_contrast - logits_max.detach()  # (bsz, bsz) set max_value in logits to zero
    # logits = anchor_dot_contrast

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)  # (anchor_cnt * bsz, contrast_cnt * bsz)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
                                0)  # (anchor_cnt * bsz, contrast_cnt * bsz)
    mask = mask * logits_mask  # 1 indicates two items belong to same class and mask-out itself
    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask  # (anchor_cnt * bsz, contrast_cnt * bsz)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    # compute mean of log-likelihood over positive

    # if 0 in mask.sum(1):
    #     raise ValueError('Make sure there are at least two instances with the same class')
    # temp = mask.sum(1)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

    # loss
    # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()
    return loss

def triplet_loss_with_label(direct_embs, target_embs, target_dist, labels):
    batch_size = labels.size(0)
    cosine_matrix = cos_sim(direct_embs, direct_embs)
    mask = torch.eye(cosine_matrix.size(0)) > .5#获取对角线布尔矩阵，对角线元素为true，其他为false
    if torch.cuda.is_available():
        mask = mask.cuda()
    cosine_matrix = cosine_matrix.masked_fill_(mask, 0)#将对角线元素填充为0
    labels_expand = labels.expand(batch_size, batch_size)
    label_mask = labels_expand ^ labels_expand.T
    label_mask = label_mask > 0
    idx_p = cosine_matrix.masked_fill(label_mask, 0).max(dim=1)[1]
    idx_n = cosine_matrix.masked_fill(~label_mask, 0).max(dim=1)[1]

    with torch.no_grad():
        dist1_te = euclidean_dist(direct_embs, direct_embs[idx_p])
        dist2_te = euclidean_dist(direct_embs, direct_embs[idx_n])

        b = (dist1_te <= dist2_te)
        delta_single = torch.ones_like(b)
        delta_single[b] = -1
        delta_single = delta_single.float()
        delta_single = torch.squeeze(delta_single)

    d_i = inner_conduct(target_dist - target_dist[idx_p])
    d_j = inner_conduct(target_dist - target_dist[idx_n])
    diff_cap = d_i - d_j
    diff_cap_norm = torch.sigmoid(diff_cap.clamp(min=-5.0, max=5.0))
    dist_loss1 = (0.2 + delta_single * diff_cap_norm).clamp(min=0)
    return torch.sum(dist_loss1) * 0.3
