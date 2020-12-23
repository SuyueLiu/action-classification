"""Get image name and its corresponding label"""

import os
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm
import argparse


def data_split(img_root_dir, train_raito=0.85, val_ratio=0.05, shuffle=True):
    test_ratio = 1 - train_raito - val_ratio
    img_names = os.listdir(img_root_dir)
    if '.DS_Store' in img_names:
        img_names.remove('.DS_Store')
    if shuffle:
        np.random.shuffle(img_names)
    train_end = round(len(img_names) * train_raito)
    val_end = round(len(img_names) * val_ratio + train_end)
    test_end = round(len(img_names) * test_ratio + val_end)

    return img_names[:train_end], img_names[train_end:val_end], img_names[val_end:test_end]


def move_imgs(img_root_dir, img_out_dir):
    train_imgs, val_imgs, test_imgs = data_split(img_root_dir)
    train_out_dir = os.path.join(img_out_dir, 'train')
    val_out_dir = os.path.join(img_out_dir, 'val')
    test_out_dir = os.path.join(img_out_dir, 'test')
    if not os.path.exists(train_out_dir):
        os.makedirs(train_out_dir)
    if not os.path.exists(val_out_dir):
        os.makedirs(val_out_dir)
    if not os.path.exists(test_out_dir):
        os.makedirs(test_out_dir)

    for img_name in tqdm(train_imgs, desc='moving training images'):
        img_path = os.path.join(img_root_dir, img_name)
        shutil.copy(img_path, train_out_dir)
    for img_name in tqdm(val_imgs, desc='moving val images'):
        img_path = os.path.join(img_root_dir, img_name)
        shutil.copy(img_path, val_out_dir)
    for img_name in tqdm(test_imgs, desc='moving test images'):
        img_path = os.path.join(img_root_dir, img_name)
        shutil.copy(img_path, test_out_dir)


def prepare(data_dir):
    names = ['take', 'cut', 'wash']
    data_pair = {}
    img_names = os.listdir(data_dir)
    for img_name in img_names:
        if img_name != '.DS_Store':
            label = img_name.split('_')[0]
            data_pair[img_name] = float(names.index(label))
    file_dir = os.path.abspath(os.path.join(data_dir, '..'))
    mode = data_dir.split('/')[-1]
    data_file_path = os.path.join(file_dir, mode+'.csv')
    return data_pair, data_file_path


def main():
    for action in os.listdir(opt.img_dir):
        if action != '.DS_Store':
            move_imgs(os.path.join(opt.img_dir, action), opt.out_dir)

    for dir in os.listdir(opt.out_dir):
        if dir != '.DS_Store':
            data_pair, data_file_path = prepare(os.path.join(opt.out_dir, dir))
            # write to csv
            data_pair = pd.DataFrame.from_dict(data_pair, orient='index')
            data_pair.to_csv(data_file_path, index=True)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--img_dir', type=str,
                       default='/Users/aaron/Desktop/code/dissertation-diary/hands_action_crop',
                       help='path to raw image')
    parse.add_argument('--out_dir', type=str, default='./data/hands_actions', help='directory of output')

    opt = parse.parse_args()

    main()

