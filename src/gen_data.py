
import os

root_path = '/home/allen/data/MOT20P/images/train'

prefix = 'MOT20P/images/train/'

save_path = './mot20p.train'

with open(save_path, 'a') as f:
    for dir_name in os.listdir(root_path):
        img_seq = os.listdir(os.path.join(root_path, dir_name, 'img1'))
        img_seq = sorted(img_seq)
        for seq in img_seq:
            record = os.path.join(prefix, dir_name, 'img1', seq)
            print(record)
            f.write(record)
            f.write('\n')