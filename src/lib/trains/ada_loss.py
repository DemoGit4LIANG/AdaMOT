import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.lib.utils.ada_utils import generalized_box_iou

import numpy as np


class AdaMotLoss(nn.Module):

    def __init__(self, opt, matcher):
        super(AdaMotLoss, self).__init__()
        self.matcher = matcher
        self.cls_weight = 2.0
        self.l1_weight = 5.
        self.giou_weight = 2.
        self.img_whwh = matcher.img_whwh

        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

        self.id_aware = opt.id_aware
        self.loss_cnt = 0


        if self.id_aware:
            print('===> using DisFL...')
        else:
            print('### not using DisFL...')

    def get_id_loss_fn(self):
        return {
            'classifier': self.classifier,
            'emb_scale': self.emb_scale,
            's_det': self.s_det,
            's_id': self.s_id
        }

    def forward(self, batch_pred, batch, cur_epoch=1e5):
        targets = batch['label']
        batch_pred = batch_pred[-1]
        batch_cls_preds, batch_bbox_preds = batch_pred['cls_pred_logits'], batch_pred['bbox_preds']
        batch_id_preds = batch_pred['id_emb_logits']

        match_inds, id_aware_inds = self.matcher(batch_pred, targets=targets, loss_fn=self.get_id_loss_fn(),
                                                 cur_epoch=cur_epoch)
        matches = self._permute_all_inds(match_inds)

        if self.id_aware and cur_epoch >= 1:
            id_matches = self._permute_id_aware_inds(id_aware_inds)

        num_obj = len(matches['batch_inds'])
        num_obj = torch.tensor(num_obj).cuda()

        b, h, w, num_cls = batch_cls_preds.shape
        # [b, h, w, c] => [b, c, h, w]
        batch_cls_preds = batch_cls_preds.view(b, h * w, -1)
        # [b, h, w, 4] => [b, hw, 4]
        batch_box_preds = batch_bbox_preds.view(b, h * w, -1)
        # [b, h, w, emb_dim] => [b, hw, emb_dim]
        batch_id_preds = batch_id_preds.view(b, h * w, -1)

        if self.id_aware and cur_epoch >= 1 and len(id_matches) > 0:
            cls_loss = self.compute_DisFL_loss(batch_cls_preds, id_matches)
        else:
            cls_loss = self.compute_label_loss(batch_cls_preds, matches)

        box_losses = self.compute_box_loss(batch_box_preds, targets=targets, matches=matches)
        l1_loss, giou_loss = box_losses['l1_loss'], box_losses['giou_loss']
        id_loss = self.compute_id_loss(batch_id_preds, matches)

        cls_loss = cls_loss * self.cls_weight
        l1_loss = l1_loss * self.l1_weight
        giou_loss = giou_loss * self.giou_weight

        det_loss = cls_loss + l1_loss + giou_loss
        loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
        loss *= 0.5

        loss_stats = {'loss': loss, 'cls_loss': cls_loss,
                      'l1_loss': l1_loss, 'giou_loss': giou_loss, 'id_loss': id_loss, 'num_obj': num_obj}

        return loss, loss_stats

    def _permute_all_inds(self, matched_indices):
        '''
        :param matched_indices:
        :return:
        '''
        batch_inds, batch_src_inds, batch_gt_inds, batch_gt_cls, batch_gt_ids = [], [], [], [], []
        for batch_i, (src_inds, gt_inds, gt_cls, gt_ids) in enumerate(matched_indices):
            batch_inds.append(torch.full_like(src_inds, batch_i))
            batch_src_inds.append(src_inds)
            batch_gt_inds.append(gt_inds)
            batch_gt_cls.append(gt_cls)
            batch_gt_ids.append(gt_ids)

        return {'batch_inds': torch.cat(batch_inds),
                'batch_src_inds': torch.cat(batch_src_inds),
                'batch_gt_inds': torch.cat(batch_gt_inds),
                'batch_gt_cls': torch.cat(batch_gt_cls),
                'batch_gt_ids': torch.cat(batch_gt_ids)}

    def _permute_id_aware_inds(self, id_awares):
        '''
        :param matched_indices:
        :return:
        '''
        batch_inds, batch_src_inds, batch_id_aware_w = [], [], []
        for batch_i, (src_inds, id_aware_w) in enumerate(id_awares):
            batch_inds.append(torch.full_like(src_inds, batch_i))
            batch_src_inds.append(src_inds)
            batch_id_aware_w.append(id_aware_w)

        return {'batch_inds': torch.cat(batch_inds),
                'batch_src_inds': torch.cat(batch_src_inds),
                'batch_id_aware_w': torch.cat(batch_id_aware_w)}


    def compute_label_loss(self, batch_cls_pred, matches):
        '''
        :param batch_cls_pred: [bs, hw, num_cls]
        :param targets_list:
        :param match_indices:
        :return:
        '''
        batch_inds, batch_src_inds, batch_gt_cls = matches['batch_inds'], matches['batch_src_inds'], matches[
            'batch_gt_cls']

        gt_tensors = torch.zeros_like(batch_cls_pred)
        gt_tensors[batch_inds, batch_src_inds, batch_gt_cls] = 1.0
        gt_tensors = gt_tensors.flatten(0, 1)
        batch_cls_pred = batch_cls_pred.flatten(0, 1)

        return sigmoid_focal_loss(batch_cls_pred, gt_tensors, alpha=0.25, gamma=2.0, reduction='sum')
        # return focal_loss_logits(torch.sigmoid(batch_cls_pred), gt_tensors, alpha=0.25, gamma=2.0, reduction='sum')

    def compute_DisFL_loss(self, batch_cls_pred, id_matches):
        '''
        :param batch_cls_pred: [bs, hw, num_cls]
        :param targets_list:
        :param match_indices:
        :return:
        '''
        gt_tensors = torch.zeros_like(batch_cls_pred)
        batch_inds2, batch_src_inds2, batch_id_aware_w = id_matches['batch_inds'], id_matches['batch_src_inds'], \
                                                         id_matches[
                                                             'batch_id_aware_w']

        gt_tensors[batch_inds2, batch_src_inds2, 0] = batch_id_aware_w

        loss, cnt = discriminative_focal_loss(torch.sigmoid(batch_cls_pred), gt_tensors, gamma=2.0, reduction='sum',
                                         cnt=self.loss_cnt)
        self.loss_cnt = cnt
        return loss

    def compute_id_loss(self, batch_id_preds, matches):
        '''
        :param batch_id_pred: [bs, hw, emb_dim]
        :param targets_list:
        :param match_indices:
        :return:
        '''
        batch_inds, batch_src_inds, batch_gt_ids = matches['batch_inds'], matches['batch_src_inds'], matches[
            'batch_gt_ids']

        id_embs = batch_id_preds[batch_inds, batch_src_inds]
        id_embs = self.emb_scale * F.normalize(id_embs)
        id_preds = self.classifier(id_embs).contiguous()

        mask = batch_gt_ids != -1
        batch_gt_ids = batch_gt_ids[mask]
        id_preds = id_preds[mask]

        id_loss = self.IDLoss(id_preds, batch_gt_ids.cuda())

        return id_loss

    def compute_box_loss(self, batch_box_pred, targets, matches):
        '''
        :param batch_box_pred: [bs, hw, 4]
        :param target_list:
        :param match_indices:
        :return:
        '''

        batch_inds, batch_src_inds, batch_gt_inds = matches['batch_inds'], matches['batch_src_inds'], matches[
            'batch_gt_inds']
        target_list = []

        for targets_per_img in targets:
            valid_inds = targets_per_img[:, 0] != -1.
            targets_per_img = targets_per_img[valid_inds]
            target_list.append(targets_per_img)

        # [num_gt, 4]
        box_pred = batch_box_pred[batch_inds, batch_src_inds]
        target_list = [target_list[b_ind][gt_ind][2:] for b_ind, gt_ind in zip(batch_inds, batch_gt_inds)]
        x1y1x2y2_gt_norm = torch.cat(target_list).view(-1, 4)

        box_pred_norm = box_pred / self.img_whwh.to(box_pred.device).expand_as(box_pred)

        giou_loss = 1 - torch.diag(generalized_box_iou(box_pred_norm, x1y1x2y2_gt_norm))
        # giou_loss = giou_loss.sum() / num_obj
        giou_loss = giou_loss.sum()

        # l1_loss = F.l1_loss(box_pred_norm + 1., x1y1x2y2_gt_norm + 1., reduction='none')
        l1_loss = F.l1_loss(box_pred_norm, x1y1x2y2_gt_norm, reduction='none')
        # l1_loss = l1_loss.sum() / num_obj
        l1_loss = l1_loss.sum()

        box_losses = {'l1_loss': l1_loss,
                      'giou_loss': giou_loss
                      }

        return box_losses


def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = -1,
        gamma: float = 2,
        reduction: str = "none",
) -> torch.Tensor:
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def focal_loss_logits(preds, targets, alpha=0.25, gamma=2.0, reduction='sum'):
    '''
    :param preds: [bs, hw, num_cls]
    :param targets: [bs, hw, num_cls]
    :param alpha:
    :param gamma:
    :param reduction:
    :return:
    '''
    assert reduction in ('none', 'sum')
    pos_mask = (targets > 0.).float()
    neg_mask = (targets == 0.).float()

    pos_loss = alpha * ((1 - preds) ** gamma) * (-(preds + 1e-8).log()) * pos_mask
    neg_loss = (1 - alpha) * (preds ** gamma) * (-(1 - preds + 1e-8).log()) * neg_mask

    with open('./cls.txt', 'a') as f_cls:
        cls_score = preds[targets > 0.]
        np.savetxt(f_cls, cls_score.detach().cpu().numpy(), delimiter=',', fmt='%.4f')

    with open('./pos_loss.txt', 'a') as f_pos_loss:
        p_loss = pos_loss[targets > 0.]
        np.savetxt(f_pos_loss, p_loss.detach().cpu().numpy(), delimiter=',', fmt='%.4f')

    total_loss = pos_loss + neg_loss

    return torch.sum(total_loss) if reduction == 'sum' else total_loss


# def discriminative_focal_loss(preds, targets, gamma=2.0, reduction='sum'):
#     '''
#     :param preds: [bs, hw, num_cls]
#     :param targets: [bs, hw, num_cls]
#     :param alpha:
#     :param gamma:
#     :param reduction:
#     :return:
#     '''
#     assert reduction in ('none', 'sum')
#     pos_mask = (targets > 0.).float()
#     neg_mask = (targets == 0.).float()
#
#     num_pos = pos_mask.sum()
#     alpha = (targets.sum() / num_pos) ** 2
#     beta = 3 * alpha
#
#     pos_loss = ((targets * (1. - preds))**gamma) * (-torch.log(preds + 1e-8)) * pos_mask
#     neg_loss = beta * (preds ** gamma) * (-torch.log(1 - preds + 1e-8)) * neg_mask
#     total_loss = 1/(alpha + beta) * (pos_loss + neg_loss)
#
#     return torch.sum(total_loss) if reduction == 'sum' else total_loss

def discriminative_focal_loss(preds, targets, gamma=2.0, reduction='sum', cnt=0):
    '''
    :param preds: [bs, hw, num_cls]
    :param targets: [bs, hw, num_cls]
    :param alpha:
    :param gamma:
    :param reduction:
    :return:
    '''
    assert reduction in ('none', 'sum')
    pos_mask = (targets > 0.).float()
    neg_mask = (targets == 0.).float()

    num_pos = pos_mask.sum()
    alpha = (targets.sum() / num_pos) ** 2
    beta = 3 * alpha

    pos_loss = ((targets * (1. - preds)) ** gamma) * (-torch.log(preds + 1e-8)) * pos_mask
    neg_loss = beta * (preds ** gamma) * (-torch.log(1 - preds + 1e-8)) * neg_mask
    total_loss = 1 / (alpha + beta) * (pos_loss + neg_loss)

    # def re_sigmoid(x):
    #     return -torch.log((1 - x) / x)
    #
    # cls_score = preds[targets > 0.]
    # p_loss = pos_loss[targets > 0.]
    # p_targets = targets[pos_mask > 0.]
    # p_targets = re_sigmoid(p_targets)

    # if random.random() > 0.2:
    #     m1 = p_targets > 0.5
    #     cls_score = cls_score[m1]
    #     p_loss = p_loss[m1]
    #     p_targets = p_targets[m1]
    #
    #     m2 = p_loss > 0.6
    #     cls_score = cls_score[m2]
    #     p_loss = p_loss[m2]
    #     p_targets = p_targets[m2]

        # m2 = torch.abs(p_targets - p_loss) < 0.4
        # cls_score = cls_score[m2]
        # p_loss = p_loss[m2]
        # p_targets = p_targets[m2]

    # with open('./idw_cls.txt', 'a') as f_idw_cls:
    #     # cls_score = preds[targets > 0.]
    #     np.savetxt(f_idw_cls, cls_score.detach().cpu().numpy(), delimiter=',', fmt='%.4f')
    #
    # with open('./idw_pos_loss.txt', 'a') as f_idw_pos_loss:
    #     # p_loss = pos_loss[targets > 0.]
    #     np.savetxt(f_idw_pos_loss, p_loss.detach().cpu().numpy(), delimiter=',', fmt='%.4f')
    #
    # with open('./idw_reid_score.txt', 'a') as f_idw_reid_score:
    #     # p_targets = targets[pos_mask > 0.]
    #     # p_targets = re_sigmoid(p_targets)
    #     np.savetxt(f_idw_reid_score, p_targets.detach().cpu().numpy(), delimiter=',', fmt='%.4f')
    #
    # cnt += len(cls_score)
    # if cnt > 2000:
    #     raise Exception()


    return torch.sum(total_loss) if reduction == 'sum' else total_loss, cnt
