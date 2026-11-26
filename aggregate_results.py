#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总 5 折交叉验证结果: 读取各折 output_dir/fold{i}/test_results.json,
计算各指标 mean±std 并输出 summary.json。
"""
import os
import json
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='汇总 5 折交叉验证结果')
    parser.add_argument('--cv_dir', type=str, default='output_dir/cv')
    parser.add_argument('--n_folds', type=int, default=5)
    args = parser.parse_args()

    metrics = ['seg_miou', 'cls_accuracy', 'cls_precision', 'cls_recall', 'cls_f1',
               'vqa_accuracy', 'vqa_precision', 'vqa_recall', 'vqa_f1']
    collected = {m: [] for m in metrics}
    missing = []

    for fold in range(args.n_folds):
        path = os.path.join(args.cv_dir, f'fold{fold}', 'test_results.json')
        if not os.path.exists(path):
            missing.append(fold)
            continue
        with open(path, 'r', encoding='utf-8') as f:
            r = json.load(f)
        for m in metrics:
            if m in r:
                collected[m].append(r[m])

    if missing:
        print(f'[警告] 缺失折: {missing}')

    summary = {}
    print('=' * 60)
    print('5 折交叉验证汇总 (mean ± std)')
    print('=' * 60)
    for m in metrics:
        arr = np.array(collected[m])
        if len(arr) == 0:
            continue
        mean = float(arr.mean())
        std = float(arr.std())
        summary[m] = {'mean': mean, 'std': std, 'values': [float(v) for v in arr]}
        vals = ', '.join(f'{v:.4f}' for v in arr)
        print(f'{m:16s}: {mean:.4f} ± {std:.4f}   [{vals}]')

    out = os.path.join(args.cv_dir, 'summary.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print('-' * 60)
    print(f'汇总已写入: {out}')


if __name__ == '__main__':
    main()
