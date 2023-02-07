import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.lib.utils.ada_utils import generalized_box_iou

import random


class IdentityAwareLabelAssignment(nn.Module):

    def __init__(self, opt):
        super(IdentityAwareLabelAssignment, self).__init__()

        self.device = opt.device
        '''
            img_whwh: [[w, h, w, h]]
        '''
        input_wh = opt.img_size
        self.img_whwh = torch.tensor(input_wh, dtype=torch.float32).repeat(1, 2)

        self.cls_weight = 2.5
        self.l1_weight = 5.
        self.giou_weight = 2.

        self.alpha = 0.25
        self.gamma = 2.0

        self.k = opt.pos

        self.id_aware = opt.id_aware
        if self.id_aware:
            print('===> ID-aware loss...')
            self.IDLoss = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
        print('===> pos numbner:', self.k)

        self.tmp_cnt = 0

    # def reset_img_size(self, img_wh):
    #     self.img_whwh = torch.tensor(img_wh, dtype=torch.float32).repeat(1, 2).to(self.device)

    @torch.no_grad()
    def get_foreground_inds(self, gt_box_norm, hm_w, hm_h):
        foreground = np.zeros((hm_h, hm_w))
        gt_box_hm = (gt_box_norm * np.array([hm_w, hm_h, hm_w, hm_h]))
        gt_box_hm[:, :2] = np.floor(gt_box_hm[:, :2])
        gt_box_hm[:, 2:] = np.ceil(gt_box_hm[:, 2:])
        # TODO: debug
        gt_box_hm[:, 0::2] = np.clip(gt_box_hm[:, 0::2], 0., hm_w - 1)
        gt_box_hm[:, 1::2] = np.clip(gt_box_hm[:, 1::2], 0., hm_h - 1)

        gt_box_hm = gt_box_hm.astype(np.long)

        for i, box in enumerate(gt_box_hm):
            x1, x2 = box[0], box[2]
            y1, y2 = box[1], box[3]
            foreground[y1:y2 + 1, x1:x2 + 1] = 1.0

        foreground = foreground.reshape(-1)
        foreground_inds = foreground.nonzero()[0]

        return foreground_inds

    @torch.no_grad()
    def forward(self, preds, targets, loss_fn=None, cur_epoch=1e5):
        '''
        :param cls_pred: [BS, HW, num_cls]
        :param box_pred: [BS, HW, 4]
        :param targets: list of array(targets in an image)
        :return:
        '''

        btach_cls_pred_logits = preds['cls_pred_logits']
        batch_bbox_preds = preds['bbox_preds']
        batch_id_emb_logits = preds['id_emb_logits']

        bs, h, w, num_cls = btach_cls_pred_logits.shape

        btach_cls_pred_logits = btach_cls_pred_logits.view(bs, h * w, -1)
        batch_bbox_preds = batch_bbox_preds.view(bs, h * w, -1)
        batch_id_emb_logits = batch_id_emb_logits.view(bs, h * w, -1)

        inds = []
        id_aware_inds_weights = []

        self.img_whwh = self.img_whwh.to(batch_bbox_preds.device)

        id_classifier = loss_fn['classifier']
        emb_scale = loss_fn['emb_scale']

        for batch_i, targets_per_img in enumerate(targets):
            '''
                [cls, id, x1, y1, x2, y2]
                targets_per_img
            '''
            valid_inds = targets_per_img[:, 0] != -1.
            targets_per_img = targets_per_img[valid_inds]
            gt_cls = targets_per_img[:, 0].long()  # [c1, c2, ..., c_N]
            gt_id = targets_per_img[:, 1].long()  # [id1, id2, ..., id_N]
            gt_box_norm = targets_per_img[:, 2:]  # [BS, 4]

            foreground_inds = self.get_foreground_inds(gt_box_norm.cpu().numpy(), hm_w=w, hm_h=h)

            num_gt = len(gt_cls)

            cls_pred = torch.sigmoid(btach_cls_pred_logits[batch_i])  # [HW, num_cls]
            box_pred_abs = batch_bbox_preds[batch_i]  # [HW, 4]

            iw_cls_cost = 0.

            if self.id_aware and cur_epoch >= 1:
                try:
                    id_emd_logits = batch_id_emb_logits[batch_i]  # [HW, emd_dim]
                    id_emd_logits = emb_scale * F.normalize(id_emd_logits)
                    id_preds = id_classifier(id_emd_logits).contiguous()  # [HW, n_id]
                    id_preds = torch.softmax(id_preds, dim=1)
                    max_id_preds, _ = id_preds[:, gt_id].max(dim=-1, keepdim=True)
                    id_w = max_id_preds.sqrt()
                    w_cls_pred = cls_pred * id_w
                    neg_cls_cost = (1 - self.alpha) * (w_cls_pred ** self.gamma) * (-(1 - w_cls_pred + 1e-8).log())
                    pos_cls_cost = self.alpha * ((1 - w_cls_pred) ** self.gamma) * (-(w_cls_pred + 1e-8).log())
                except:
                    neg_cls_cost = (1 - self.alpha) * (cls_pred ** self.gamma) * (-(1 - cls_pred + 1e-8).log())
                    pos_cls_cost = self.alpha * ((1 - cls_pred) ** self.gamma) * (-(cls_pred + 1e-8).log())

                iw_cls_cost = pos_cls_cost[:, gt_cls] - neg_cls_cost[:, gt_cls]  # [HW, num_gt]

            neg_cls_cost = (1 - self.alpha) * (cls_pred ** self.gamma) * (-(1 - cls_pred + 1e-8).log())
            pos_cls_cost = self.alpha * ((1 - cls_pred) ** self.gamma) * (-(cls_pred + 1e-8).log())
            # TODO: compute the cls losses, using focal loss
            cls_cost = pos_cls_cost[:, gt_cls] - neg_cls_cost[:, gt_cls]  # [HW, num_gt]
            if self.id_aware:
                cls_cost = 0.75 * cls_cost + 0.25 * iw_cls_cost
                # cls_cost = 1 * cls_cost + 0 * iw_cls_cost
            # TODO: L1 box losses
            '''
                Note here
            '''
            box_pred_norm = box_pred_abs / self.img_whwh.expand_as(box_pred_abs)
            # box_gt_norm = gt_box / self.img_whwh.llexpand_as(gt_box)

            l1_box_cost = torch.cdist(box_pred_norm, gt_box_norm, p=1)  # [HW, num_gt]

            # TODO: Compute the giou cost between boxes
            giou_cost = -generalized_box_iou(box_pred_norm, gt_box_norm)  # [HW, num_gt]
            # giou_cost = -generalized_box_iou(box_pred_abs, gt_box_abs)  # [HW, num_gt]

            # TODO: cost-matrix C
            C = self.cls_weight * cls_cost + self.l1_weight * l1_box_cost + self.giou_weight * giou_cost  # [HW, num_gt]

            C_fg = C[foreground_inds]
            C_fg = C_fg.repeat(1, self.k)

            src_inds, gt_inds = linear_sum_assignment(C_fg.cpu().numpy(), maximize=False)
            src_inds = foreground_inds[src_inds]
            src_inds = torch.as_tensor(src_inds, dtype=torch.int64).cuda()
            gt_inds = gt_inds % num_gt
            gt_inds = torch.as_tensor(gt_inds, dtype=torch.int64, device=C.device)
            gt_cls = gt_cls[gt_inds]
            gt_id = gt_id[gt_inds]

            if self.id_aware and cur_epoch >= 1 and len(gt_id) > 0:
                id_aware_src_inds, id_aware_weights = [], []
                unique = gt_id.unique()
                for _id in unique:
                    group_by_id = src_inds[_id == gt_id]
                    id_emd_logits = batch_id_emb_logits[batch_i, group_by_id]
                    id_emb_norm = F.normalize(id_emd_logits)
                    id_emd_norm_scale = emb_scale * id_emb_norm
                    id_preds = id_classifier(id_emd_norm_scale).contiguous()

                    id_aware_w = torch.softmax(id_preds, dim=1)[:, _id]

                    # max_ind = cls_pred[group_by_id].argmax()
                    # max_cls_score = cls_pred[group_by_id][max_ind]
                    # max_id_score = id_aware_w[max_ind]
                    #
                    # if torch.abs(max_cls_score - max_id_score) > 0.2:
                    #     if random.random() > 0.5:
                    #         with open('./id_score.txt', 'a') as id_f:
                    #             np.savetxt(id_f, max_id_score.view(1, -1).cpu().numpy(), delimiter=',', fmt='%.4f')
                    #         with open('./cls_score.txt', 'a') as cls_f:
                    #             np.savetxt(cls_f, max_cls_score.view(1, -1).cpu().numpy(), delimiter=',', fmt='%.4f')
                    #
                    #         self.tmp_cnt += 1
                    #         if self.tmp_cnt > 2000:
                    #             raise Exception
                    # else:
                    #     with open('./id_score.txt', 'a') as id_f:
                    #         np.savetxt(id_f, max_id_score.view(1, -1).cpu().numpy(), delimiter=',', fmt='%.4f')
                    #     with open('./cls_score.txt', 'a') as cls_f:
                    #         np.savetxt(cls_f, max_cls_score.view(1, -1).cpu().numpy(), delimiter=',', fmt='%.4f')
                    #
                    #     self.tmp_cnt += 1
                    #     if self.tmp_cnt > 2000:
                    #         raise Exception


                    id_aware_src_inds.append(group_by_id)
                    id_aware_weights.append(torch.sigmoid(id_aware_w))

                id_aware_src_inds = torch.cat(id_aware_src_inds)
                id_aware_weights = torch.cat(id_aware_weights)
                id_aware_inds_weights.append((id_aware_src_inds, id_aware_weights))

            inds.append((src_inds, gt_inds, gt_cls, gt_id))

        return [(i, torch.as_tensor(j, dtype=torch.int64),
                 torch.as_tensor(k, dtype=torch.int64), torch.as_tensor(l, dtype=torch.int64)) for i, j, k, l in
                inds], id_aware_inds_weights

