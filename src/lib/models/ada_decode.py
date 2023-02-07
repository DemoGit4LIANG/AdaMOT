from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from pt_soft_nms import soft_nms

from .utils import _gather_feat, _tranpose_and_gather_feat


def _max_pool_nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def adamot_decode(cls_logits, bbox, K=100, nms=True):
    # [B, H, W, C] ==> [B, C, H, W]
    cls_logits = cls_logits.permute(0, 3, 1, 2)
    bbox = bbox.permute(0, 3, 1, 2)
    batch, cat, height, width = cls_logits.shape
    cls = torch.sigmoid(cls_logits)

    cls = _max_pool_nms(cls, kernel=3)  # perform nms on heatmaps
    scores, inds, clses, ys, xs = _topk(cls, K=K)

    bbox = _tranpose_and_gather_feat(bbox, inds)
    bbox = bbox.view(batch, K, 4)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = bbox

    # if nms:
    #     _, keep = soft_nms(bboxes[0], scores.flatten(), 0.5, 0.1)
    #     clses = clses[:, keep]
    #     scores = scores[:, keep]
    #     bboxes = bboxes[:, keep]
    #     inds = inds[:, keep]

    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections, inds
