import torch
import numpy as np
from torchvision.ops.boxes import box_area


def cxcywh_to_x1y1x2y2(cxcywh):
    '''
    :param cxcywh: [N, cxcywh]
    :return:  [N, x1x2y1y2]
    '''
    assert isinstance(cxcywh, torch.Tensor) or isinstance(cxcywh, np.ndarray)
    if isinstance(cxcywh, torch.Tensor):
        x1y1x2y2 = torch.zeros_like(cxcywh)
        x1y1x2y2[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2
        x1y1x2y2[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2
        x1y1x2y2[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2
        x1y1x2y2[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2

        return x1y1x2y2
    else:
        x1y1x2y2 = np.zeros_like(cxcywh)
        x1y1x2y2[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2.
        x1y1x2y2[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2.
        x1y1x2y2[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2.
        x1y1x2y2[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2.

        return x1y1x2y2


def compute_IoU(box1, box2):
    '''
    :param box1: [N, x1y1x2y2], relative
    :param box2: [N, x1y1x2y2], relative
    :return: [iou1, iou2, ..., iouN]
    '''
    assert isinstance(box1, torch.Tensor) or \
           isinstance(box1, np.ndarray)

    if isinstance(box1, torch.Tensor):
        tl = torch.max(box1[:, :2], box2[:, :2])
        br = torch.min(box1[:, 2:], box2[:, 2:])

        # torch.prod()
        area_box1 = torch.prod(box1[:, 2:] - box1[:, :2], dim=-1)
        area_box2 = torch.prod(box2[:, 2:] - box2[:, :2], dim=-1)

        en = (tl < br).type(tl.type()).prod(dim=-1)
        area_inter = torch.prod(br - tl, dim=-1) * en

        return area_inter / (area_box1 + area_box2 - area_inter)

    else:
        # TODO
        pass


########################################################################
# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def iou_score(bboxes_a, bboxes_b):
    """
        bbox_1 : [B*N, 4] = [x1, y1, x2, y2]
        bbox_2 : [B*N, 4] = [x1, y1, x2, y2]
    """
    tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en  # * ((tl < br).all())
    print(torch.prod(br - tl, dim=-1))
    return area_i / (area_a + area_b - area_i)


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def nms(dets, scores):
    # dets = dets.cpu().numpy()
    # scores = scores.cpu().numpy()
    """"Pure Python NMS baseline."""
    x1 = dets[:, 0]  # xmin
    y1 = dets[:, 1]  # ymin
    x2 = dets[:, 2]  # xmax
    y2 = dets[:, 3]  # ymax

    areas = (x2 - x1) * (y2 - y1)  # the size of bbox
    order = scores.argsort()[::-1]  # sort bounding boxes by decreasing order

    keep = []  # store the final bounding boxes
    while order.size > 0:
        i = order[0]  # the index of the bbox with highest confidence
        keep.append(i)  # save it to keep
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-28, xx2 - xx1)
        h = np.maximum(1e-28, yy2 - yy1)
        inter = w * h

        # Cross Area / (bbox + particular area - Cross Area)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # reserve all the boundingbox whose ovr less than thresh
        inds = np.where(ovr <= 0.45)[0]
        order = order[inds + 1]

    return keep


if __name__ == '__main__':
    cxcywh1 = torch.tensor([8, 4, 5, 6], dtype=torch.float32).unsqueeze(0)
    x1y1x2y2_1 = cxcywh_to_x1y1x2y2(cxcywh1)
    x1y1x2y2_1 = x1y1x2y2_1.repeat(2, 1)
    print(compute_IoU(x1y1x2y2_1, x1y1x2y2_1))

    t = torch.tensor([2, 1, 3])
    max_ind = torch.argmax(t)
    print(t[max_ind])
