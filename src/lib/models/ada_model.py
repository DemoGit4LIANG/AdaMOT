import math
import torch
from torch import nn
from src.lib.models.networks.backnones.pose_dla_dcn import get_pose_net
from src.lib.models.networks.backnones.resnet_dcn import get_resnet_dcn


class AdaptiveNet(nn.Module):

    def __init__(self, device=None, backbone_name='resnet', layers=50, num_class=80, stride=4, head_conv=256, gn=False,
                 is_train=True, topK=100, reid_dim=128):
        super(AdaptiveNet, self).__init__()
        self.device = device
        self.num_class = num_class
        self.stride = stride
        self.head_conv = head_conv
        self.gn = gn
        self.is_train = is_train
        self.topK = topK
        self.scale = None
        self.emb_dim = reid_dim

        assert backbone_name in ('resnet', 'dla')
        if backbone_name == 'resnet':
            self.backbone = get_resnet_dcn(num_layers=layers)
            print('# Using resnet{} with dcn as backbone...'.format(layers))
        else:
            self.backbone = get_pose_net(num_layers=layers, down_ratio=4)
            print('# Using dla{} with dcn backbone...'.format(layers))

        cls_head = [
            nn.Conv2d(64, self.head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, self.num_class,
                      kernel_size=1, stride=1,
                      padding=0, bias=True)
        ]
        ltrb_head = [
            nn.Conv2d(64, self.head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, 4,
                      kernel_size=1, stride=1,
                      padding=0, bias=True)
        ]
        id_head = [
            nn.Conv2d(64, self.head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, self.emb_dim,
                      kernel_size=1, stride=1,
                      padding=0, bias=True)
        ]

        if self.gn:
            cls_head.insert(1, nn.GroupNorm(32, self.head_conv))
            ltrb_head.insert(1, nn.GroupNorm(32, self.head_conv))
            id_head.insert(1, nn.GroupNorm(32, self.head_conv))

        self.cls_head = nn.Sequential(*cls_head)
        self.ltrb_head = nn.Sequential(*ltrb_head)
        self.id_head = nn.Sequential(*id_head)

        prior_prob = 0.01
        self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters(self.cls_head, self.ltrb_head)

    def _reset_parameters(self, *heads):
        # init all parameters.
        for head in heads:
            for p in head.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        # initialize the bias for focal loss.
        nn.init.constant_(self.cls_head[-1].bias, self.bias_value)

    def reset_scale(self, img_w, img_h):
        self.scale = torch.tensor([img_w, img_h, img_w, img_h], device=self.device)[None]

    def forward(self, imgs):
        hm4 = self.backbone(imgs)
        B, _, H, W = hm4.shape
        # heads
        cls_pred_logits = self.cls_head(hm4)  # [B, num_cls, H, W]
        ltrb_pred_logits = self.ltrb_head(hm4)
        id_pred_logits = self.id_head(hm4)
        ltrb_preds = torch.relu(ltrb_pred_logits)  # [B, 4, H, W]
        box_preds = self.apply_ltrb(self.gen_grids(hm4)[None], ltrb_preds)

        # [B, num_cls, H, W] -> [B, H， W, num_cls]
        cls_pred_logits = cls_pred_logits.permute(0, 2, 3, 1).contiguous()
        # [B, 4, H, W] -> [B, H，W, 4]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        # [B, dim, H, W] -> [B, H，W, dim]
        id_emb_logits = id_pred_logits.permute(0, 2, 3, 1).contiguous()
        preds = {
            'cls_pred_logits': cls_pred_logits,
            'bbox_preds': box_preds,
            'id_emb_logits': id_emb_logits
        }

        return [preds]

    @torch.no_grad()
    def gen_grids(self, heatmap):
        hs, ws = heatmap.shape[-2:]
        device = heatmap.device

        shifts_x = torch.arange(
            0, ws * self.stride, step=self.stride,
            dtype=torch.float32, device=device)

        shifts_y = torch.arange(
            0, hs * self.stride, step=self.stride,
            dtype=torch.float32, device=device)

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        grids = torch.stack((shift_x, shift_y), dim=1) + self.stride // 2

        # [2, H, W]
        grids = grids.reshape(hs, ws, 2).permute(2, 0, 1)

        return grids

    def apply_ltrb(self, grids, pred_ltrb):
        """
        :param locations:  (1, 2, H, W)
        :param pred_ltrb:  (N, 4, H, W)
        """

        pred_boxes = torch.zeros_like(pred_ltrb)
        pred_boxes[:, 0, :, :] = grids[:, 0, :, :] - pred_ltrb[:, 0, :, :]  # x1
        pred_boxes[:, 1, :, :] = grids[:, 1, :, :] - pred_ltrb[:, 1, :, :]  # y1
        pred_boxes[:, 2, :, :] = grids[:, 0, :, :] + pred_ltrb[:, 2, :, :]  # x2
        pred_boxes[:, 3, :, :] = grids[:, 1, :, :] + pred_ltrb[:, 3, :, :]  # y2

        return pred_boxes
