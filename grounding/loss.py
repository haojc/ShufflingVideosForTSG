import torch
import torch.nn.functional as F
import numpy as np
DELTA = 1e-4

def temporal_order_discrimination_loss(original_video_prob, pseudo_video_prob, criterion_domain):
    pred_original_single = original_video_prob.view(-1, original_video_prob.size()[-1])
    pred_pseudo_single = pseudo_video_prob.view(-1, pseudo_video_prob.size()[-1])

    # prepare labels
    original_video_label = torch.zeros(pred_original_single.size(0)).long()
    pseudo_video_label = torch.ones(pred_pseudo_single.size(0)).long()
    domain_label = torch.cat((original_video_label, pseudo_video_label), 0)

    domain_label = domain_label.cuda(non_blocking=True)

    pred_domain = torch.cat((pred_original_single, pred_pseudo_single), 0)

    loss_single = criterion_domain(pred_domain, domain_label)
    return loss_single

def span_ground_loss(start_prob, end_prob, framestamps):
    loss = 0
    for idx in range(len(framestamps)):
        start_gt, end_gt = framestamps[idx]
        loss = loss - torch.log(start_prob[idx][start_gt]) - torch.log(end_prob[idx][end_gt])

    return torch.div(loss, len(framestamps))

def BCE_loss(logits, labels, mask):
    labels = labels.type_as(logits)
    loss_per_location = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    mask = mask.type_as(logits)
    loss = (loss_per_location * mask).sum() / (mask.sum() + DELTA)

    return loss

def KL_divergence(prob1, prob2, epsilon=1e-4):
    KL = torch.sum(prob1 * torch.log((prob1+epsilon) / (prob2+epsilon)), dim=-1)
    return KL

def matching_KL_divergence(prob1, prob2, framestps1, framestps2):
    assert len(framestps1) == len(framestps2), '{:d}, {:d}'.format(len(framestps1),len(framestps2))

    loss = 0
    for idx in range(len(framestps1)):
        start_gt1, end_gt1 = framestps1[idx]
        start_gt2, end_gt2 = framestps2[idx]
        loss = loss + KL_divergence(prob1[idx][start_gt1:end_gt1+1], prob2[idx][start_gt2:end_gt2+1])

    return torch.div(loss, len(framestps1))

def span_pred(start_prob, end_prob):
    B, T = start_prob.size()
    start_matrix = start_prob.unsqueeze(dim=-1).expand(B, T, T)
    end_matrix = end_prob.unsqueeze(dim=-1).expand(B, T, T).permute(0, 2, 1)
    prob_matrix = (start_matrix + end_matrix).triu(diagonal=0)

    row_max, row_max_idx = prob_matrix.max(dim=2)
    prob_max, colum_max_idx = row_max.max(dim=1)

    idx = torch.arange(0, B)
    idx = torch.stack((idx, colum_max_idx), dim=0).numpy()

    start = colum_max_idx
    end = row_max_idx[idx]

    pred_time = torch.cat((start.unsqueeze(dim=-1), end.unsqueeze(dim=-1)),dim=-1)

    return pred_time, prob_max

def compute_mean_iou(seg1, seg2):
    """
    :param seg1: batch, 2 in (s, e) format
    :param seg2: batch, 2 in (s, e) format
    :return:
        miou: scalar
    """
    # assert not isinstance(seg1, Variable)
    # assert not isinstance(seg2, Variable)
    seg1_s, seg1_e = seg1.chunk(2, dim=1)  # batch, 1
    seg2_s, seg2_e = seg2.chunk(2, dim=1)  # batch, 1
    min_end, _ = torch.cat([seg1_e, seg2_e], dim=1).min(1)  # batch
    max_end, _ = torch.cat([seg1_e, seg2_e], dim=1).max(1)
    min_beg, _ = torch.cat([seg1_s, seg2_s], dim=1).min(1)
    max_beg, _ = torch.cat([seg1_s, seg2_s], dim=1).max(1)
    intersection = min_end - max_beg
    intersection, _ = torch.stack([intersection, torch.zeros_like(intersection)], dim=1).max(1)  # batch
    union = max_end - min_beg  # batch
    iou = intersection / (union + DELTA) # batch
    return iou.mean()

