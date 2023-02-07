# -*- coding: utf-8 -*-
import os
import numpy as np
import shutil

def gen_split(input_dir, output_dir, split_ratio=0.8):
    print('input_dir:', input_dir)
    print('output_dir:', output_dir)
    files = os.listdir(input_dir)
    n_files = len(files)
    print('original file num:', n_files)
    splitted_files = np.random.choice(files, size=int(n_files * split_ratio), replace=False)
    print('splited file num:', len(splitted_files))

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)
    for fname in splitted_files:
        shutil.copy(input_dir+'/'+fname, output_dir)


def gen_split4datafile(input, output, split_ratio=0.8):
    print('input:', input)
    print('output:', output)
    root_path = '/home/allen/data/datasets/mot/JDE/'

    with open(input, 'r') as in_f:
        lines = in_f.readlines()
        n_files = len(lines)
        splitted_files = np.random.choice(lines, size=int(n_files * split_ratio), replace=False)

    tmp = 0
    results = ''
    for line in splitted_files:
        f_path = line.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt').replace('\n', '')
        path = root_path + f_path
        with open(path, 'r') as in_f:
            lines = in_f.readlines()
            if len(lines) > 200:
                print('too many person, delete:', path)
            else:
                tmp += 1
                results += f_path
                results += '\n'
    with open(output, 'w') as out_f:
            out_f.write(results)

    print('read %d lines, write %d lines' % (n_files, tmp))

gen_split4datafile('/home/allen/py_projs/AdaMOT/src/data/crowdhuman.val', '/home/allen/py_projs/AdaMOT/src/data/crowdhuman.val_f200', split_ratio=1.)

# input = '/home/allen/data/datasets/mot/JDE/crowdhuman/labels_with_ids/train'
# output = '/home/allen/data/datasets/mot/JDE/crowdhuman/labels_with_ids/train80'
# gen_split(input, output, split_ratio=0.8)