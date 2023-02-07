import os
import shutil

src_path = '/home/allen/data/datasets/mot/MOT17/images/results/MOT17_test'
for f_name in os.listdir(src_path):
    if f_name.startswith('MOT'):
        task, seq, det = f_name[:-4].split('-')
        shutil.copy(os.path.join(src_path, f_name), os.path.join(src_path, task + '-' + seq + '-FRCNN' + '.txt'))
        shutil.copy(os.path.join(src_path, f_name), os.path.join(src_path, task + '-' + seq + '-DPM' + '.txt'))
