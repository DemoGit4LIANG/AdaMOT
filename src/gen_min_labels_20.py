import os.path as osp
import os
import numpy as np


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


seq_root = '/home/allen/data/datasets/mot/MOT20/images/train'
label_root = '/home/allen/data/datasets/mot/MOT20min/images/train'
# mkdirs(label_root)
seqs = [s for s in os.listdir(seq_root)]

tid_curr = 0
tid_last = -1
for seq in seqs:
    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

    seq_label_root = osp.join(label_root, seq, 'gt')
    mkdirs(seq_label_root)

    for fid, tid, x, y, w, h, mark, label, vis in gt:
        if int(fid) > 200: continue
        label_str = '{:d},{:d},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n'.format(
            int(fid), int(tid), x, y, w, h, mark, label, vis)
        with open(os.path.join(seq_label_root, 'gt.txt'), 'a') as f:
            f.write(label_str)

    # 6: vehicle
    # 7: static person
    # 11: occluded full
    # 13: crowd
    # label_set = set()
    # for fid, tid, x, y, w, h, conf, cls, vis in gt:
    #     fid = int(fid)
    #     tid = int(tid)
    #     if not tid == tid_last:
    #         tid_curr += 1
    #         tid_last = tid
    #     x += w / 2
    #     y += h / 2
    #     # if cls > 1:
    #     #     label_set.add(cls)
    #     label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
    #     label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
    #         tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height, cls, vis)
    #     with open(label_fpath, 'a') as f:
    #         f.write(label_str)
    #
    # print(label_set)