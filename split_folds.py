#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5 折分层交叉验证数据划分脚本。

将 1,200 张图按四级严重程度(0/1/2/3)分层划分为 5 折。
每折内部再按 8:2 分 train/val(test 之外的 4 折合并后分层分出 val),
其中 val 用于 ELT 秩调整触发与模型选择(嵌套验证),test 仅用于最终评估。

同时把分散在 train_images/valid_images 与 train_masks1/valid_masks1 的
图片/掩膜软链到统一的 all_images/ 与 all_masks/ 目录,便于训练脚本按文件名加载。

用法:
    python split_folds.py [--data_root DIR] [--n_folds 5] [--seed 66]
"""
import os
import sys
import argparse
from collections import defaultdict, Counter

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def read_cls(path):
    """读取分类文件: img_name|label"""
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, label = line.split('|')
            data[name] = int(label)
    return data


def read_seg(path):
    """读取分割文件: img_name|mask_name"""
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, mask = line.split('|')
            data[name] = mask
    return data


def read_vqa(path):
    """读取 VQA 文件: img_name|question|answer -> {img_name: [(q, a), ...]}"""
    data = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, question, answer = line.split('|')
            data[name].append((question, answer))
    return data


def link_dir(src_dirs, dst_dir):
    """把多个源目录下的文件软链到统一目录(文件名唯一,跳过已存在)。"""
    os.makedirs(dst_dir, exist_ok=True)
    n_linked, n_skip = 0, 0
    for src in src_dirs:
        if not os.path.isdir(src):
            print(f"  [警告] 源目录不存在: {src}")
            continue
        for fn in sorted(os.listdir(src)):
            src_path = os.path.join(src, fn)
            dst_path = os.path.join(dst_dir, fn)
            if not os.path.lexists(dst_path):
                os.symlink(os.path.abspath(src_path), dst_path)
                n_linked += 1
            else:
                n_skip += 1
    return n_linked, n_skip


def write_split(out_dir, names, cls_map, seg_map, vqa_map, split_name):
    """为某个 split(train/val/test)写 cls/seg/vqa 三个 txt 文件。"""
    with open(os.path.join(out_dir, f'{split_name}_cls.txt'), 'w', encoding='utf-8') as f:
        for n in names:
            f.write(f'{n}|{cls_map[n]}\n')
    with open(os.path.join(out_dir, f'{split_name}_seg.txt'), 'w', encoding='utf-8') as f:
        for n in names:
            f.write(f'{n}|{seg_map[n]}\n')
    with open(os.path.join(out_dir, f'{split_name}_vqa.txt'), 'w', encoding='utf-8') as f:
        for n in names:
            for q, a in vqa_map[n]:
                f.write(f'{n}|{q}|{a}\n')


def main():
    parser = argparse.ArgumentParser(description='5 折分层交叉验证数据划分')
    parser.add_argument('--data_root', type=str,
                        default='./data/PALM')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=66)
    args = parser.parse_args()

    data_root = args.data_root
    n_folds = args.n_folds
    seed = args.seed

    # ---- 读取并合并 train/valid ----
    cls_map = {}
    cls_map.update(read_cls(os.path.join(data_root, 'train', 'train_cls.txt')))
    cls_map.update(read_cls(os.path.join(data_root, 'valid', 'valid_cls.txt')))

    seg_map = {}
    seg_map.update(read_seg(os.path.join(data_root, 'train', 'train_seg.txt')))
    seg_map.update(read_seg(os.path.join(data_root, 'valid', 'valid_seg.txt')))

    vqa_map = read_vqa(os.path.join(data_root, 'train', 'train_vqa.txt'))
    vqa_map.update(read_vqa(os.path.join(data_root, 'valid', 'valid_vqa.txt')))

    # 三任务交集(与 UnifiedDataset.image_names 逻辑一致)
    inter = sorted(set(cls_map) & set(seg_map) & set(vqa_map))
    print(f'[划分] 三任务交集图像数: {len(inter)}')

    # 类别分布核对
    cls_dist = Counter(cls_map[n] for n in inter)
    print(f'[划分] 类别分布(0/1/2/3): {dict(sorted(cls_dist.items()))}')

    # ---- 统一图片/掩膜目录(软链) ----
    print('[软链] 统一图片与掩膜目录 ...')
    all_images_dir = os.path.join(data_root, 'all_images')
    all_masks_dir = os.path.join(data_root, 'all_masks')
    n_img, _ = link_dir(
        [os.path.join(data_root, 'train', 'train_images'),
         os.path.join(data_root, 'valid', 'valid_images')],
        all_images_dir)
    n_mask, _ = link_dir(
        [os.path.join(data_root, 'train', 'train_masks1'),
         os.path.join(data_root, 'valid', 'valid_masks1')],
        all_masks_dir)
    print(f'[软链] 图片 {n_img} 张, 掩膜 {n_mask} 张 (跳过已存在)')

    # ---- 分层 5 折 ----
    X = np.array(inter)
    y = np.array([cls_map[n] for n in inter])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds_root = os.path.join(data_root, 'folds')
    os.makedirs(folds_root, exist_ok=True)

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(X, y)):
        test_names = X[test_idx]
        trainval_names = X[trainval_idx]
        trainval_labels = y[trainval_idx]

        # 内部按 8:2 分层分 val
        train_names, val_names = train_test_split(
            trainval_names, test_size=0.2, stratify=trainval_labels,
            random_state=seed)

        fold_dir = os.path.join(folds_root, f'fold{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        write_split(fold_dir, sorted(train_names), cls_map, seg_map, vqa_map, 'train')
        write_split(fold_dir, sorted(val_names), cls_map, seg_map, vqa_map, 'val')
        write_split(fold_dir, sorted(test_names), cls_map, seg_map, vqa_map, 'test')

        # 打印每折统计
        def _dist(names):
            return dict(sorted(Counter(cls_map[n] for n in names).items()))
        print(f'[fold{fold}] train={len(train_names)} {_dist(train_names)} '
              f'val={len(val_names)} {_dist(val_names)} '
              f'test={len(test_names)} {_dist(test_names)}')

    print(f'[完成] 5 折划分已写入: {folds_root}')
    print(f'[完成] 统一图片目录: {all_images_dir}')
    print(f'[完成] 统一掩膜目录: {all_masks_dir}')


if __name__ == '__main__':
    main()
