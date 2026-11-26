import os
import sys
import argparse
import logging
from itertools import cycle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import math
from transformers import BertTokenizer
import torch.nn.functional as F
from model.models_sum_sam import eye_multi_task
from Segmentation.utils.evalution_segmentation import eval_semantic_segmentation, get_dice

# 设置日志级别
logging.getLogger("real_accelerator").setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.INFO)

# 规范化参数配置
params = {
    "data_root": "./data/PALM",
    "train_image_path": "train/train_images",
    "train_mask_path": "train/train_masks1",
    "valid_image_path": "valid/valid_images",
    "valid_mask_path": "valid/valid_masks1",
    "train_class_file": "train/train_cls.txt",
    "valid_class_file": "valid/valid_cls.txt",
    "train_seg_file": "train/train_seg.txt",
    "valid_seg_file": "valid/valid_seg.txt",
    "train_vqa_file": "train/train_vqa.txt",
    "valid_vqa_file": "valid/valid_vqa.txt",
    "class_dict_file": "class_dict1.csv",
    "vqa_class": 10,
    "convert_arr": ['Normal', 'Not visible', 'Center', 'Left side',"Right side", 'No','Yes', 'Severe', 'Mild', "Moderate"],
    "batch_size": 60,
    "cls_num_class": 4,
    "seg_num_class": 2,
    "num_epochs": 100,
    "lr": 0.00005,
    "warmup_epochs": 5,
    "image_size": 224,
    "weight_decay": 1e-8,
    "device": "cuda",

    # LoRA / ELT / HPHA 控制变量参数
    "initial_lora_rank": 64,
    "use_elt": False,
    "min_lora_rank": 4,
    "rank_patience": 10,
    "info_retention_ratio": 0.99,
    "attention": "hpha",  # 'hpha' | 'self' | 'none'

    "output_dir": "./output_dir/LoRA_64_GLSA_SAM",
    "save_path": "./output_dir/LoRA_64_GLSA_SAM/saved_model.pth",

    "bert_model_path": "./vqa/bert-medvqa"
}


# 设置随机种子
def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from collections import defaultdict


# 自定义数据集
class UnifiedDataset(Dataset):
    def __init__(self, root_dir, class_file, seg_file, vqa_file, image_path, mask_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_path = image_path
        self.mask_path = mask_path
        self.class_data = []

        with open(os.path.join(root_dir, class_file), 'r') as f:
            for line in f:
                img_name, label = line.strip().split('|')
                self.class_data.append((img_name, int(label)))

        self.seg_data = []
        with open(os.path.join(root_dir, seg_file), 'r') as f:
            for line in f:
                img_name, mask_name = line.strip().split('|')
                self.seg_data.append((img_name, mask_name))

        self.vqa_data = defaultdict(list)
        with open(os.path.join(root_dir, vqa_file), 'r') as f:
            for line in f:
                img_name, question, answer = line.strip().split('|')
                answer_idx = params["convert_arr"].index(answer)
                self.vqa_data[img_name].append((question, answer_idx))

        self.image_names = sorted(set([x[0] for x in self.class_data]) &
                                  set([x[0] for x in self.seg_data]) &
                                  set(self.vqa_data.keys()))

        self.data = []
        for img_name in self.image_names:
            class_label = next(label for name, label in self.class_data if name == img_name)
            mask_name = next(mask for name, mask in self.seg_data if name == img_name)
            # 改为使用实际的问答对数量
            actual_vqa_count = len(self.vqa_data[img_name])
            for vqa_idx in range(actual_vqa_count):
                self.data.append((img_name, class_label, mask_name, vqa_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, class_label, mask_name, vqa_idx = self.data[idx]
        img_path = os.path.join(self.root_dir, self.image_path, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        mask_path = os.path.join(self.root_dir, self.mask_path , mask_name)
        mask = Image.open(mask_path).convert('L')
        mask = transforms.Resize((params["image_size"], params["image_size"]))(
            transforms.ToTensor()(mask)
        )
        mask = (mask > 0).long().squeeze()
        vqa_question, vqa_answer = self.vqa_data[img_name][vqa_idx]
        return {
            'image': image,
            'class_label': torch.tensor(class_label, dtype=torch.long),
            'mask': mask,
            'vqa_question': vqa_question,
            'vqa_answer': torch.tensor(vqa_answer, dtype=torch.long),
            'img_name': img_name
        }


# 统一验证函数
def validate_unified(model, val_loader, device, criterion, tokenizer, epoch):
    """
    统一验证函数：一次前向传播得到所有三个任务的输出，然后分别计算指标
    """
    model.eval()

    total_miou = 0.0
    cls_predictions = []
    cls_labels = []
    vqa_total_loss = 0.0
    vqa_label_true = []
    vqa_label_pred = []
    vqa_file_names = []

    batch_count = 0

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc="Unified Validation", colour='cyan', ncols=100)
        for batch in val_bar:
            batch_count += 1

            images = batch['image'].to(device)
            class_labels = batch['class_label'].to(device)
            masks = batch['mask'].to(device)
            vqa_labels = batch['vqa_answer'].to(device)
            questions = batch['vqa_question']
            img_names = batch['img_name']

            vqa_inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=80)
            vqa_inputs = {k: v.to(device) for k, v in vqa_inputs.items()}

            # 一次前向传播得到所有输出
            seg_outputs, cls_outputs, vqa_outputs = model(input=images, vqa_text_inputs=vqa_inputs)

            # ===== 分割任务指标计算 =====
            preout = seg_outputs.max(dim=1)[1].cpu().numpy()
            gtout = masks.cpu().numpy()
            pre_label = [i for i in preout]
            true_label = [i for i in gtout]
            seg_metrics = eval_semantic_segmentation(pre_label, true_label, preout, gtout)
            total_miou += seg_metrics['miou']

            # ===== 分类任务指标计算 =====
            if cls_outputs.dim() > 2:
                cls_outputs = cls_outputs.squeeze()
            _, cls_predicted = torch.max(cls_outputs, 1)
            cls_predictions.extend(cls_predicted.cpu().numpy())
            cls_labels.extend(class_labels.cpu().numpy())

            # ===== VQA任务指标计算 =====
            vqa_loss = criterion(vqa_outputs, vqa_labels)
            vqa_total_loss += vqa_loss.item()
            _, vqa_predicted = torch.max(F.softmax(vqa_outputs, dim=1), 1)
            vqa_label_true.extend(vqa_labels.cpu().numpy())
            vqa_label_pred.extend(vqa_predicted.cpu().numpy())
            vqa_file_names.extend(img_names)

            # 更新进度条显示
            current_miou = total_miou / batch_count
            current_cls_acc = accuracy_score(cls_labels, cls_predictions) if len(cls_labels) > 0 else 0
            current_vqa_acc = accuracy_score(vqa_label_true, vqa_label_pred) if len(vqa_label_true) > 0 else 0

            val_bar.set_postfix({
                'MIoU': current_miou,
                'Cls_Acc': current_cls_acc,
                'VQA_Acc': current_vqa_acc,
                'VQA_Loss': vqa_total_loss / batch_count
            }, refresh=True)

    # ===== 计算最终指标 =====

    # 分割指标
    final_miou = total_miou / len(val_loader)

    # 分类指标
    cls_conf_matrix = confusion_matrix(cls_labels, cls_predictions)
    cls_metrics = {
        'accuracy': accuracy_score(cls_labels, cls_predictions),
        'conf_matrix': cls_conf_matrix,
        'precision_macro': precision_score(cls_labels, cls_predictions, average='macro', zero_division=0),
        'precision_per_class': precision_score(cls_labels, cls_predictions, average=None, zero_division=0),
        'recall_macro': recall_score(cls_labels, cls_predictions, average='macro', zero_division=0),
        'recall_per_class': recall_score(cls_labels, cls_predictions, average=None, zero_division=0),
        'f1_macro': f1_score(cls_labels, cls_predictions, average='macro', zero_division=0),
        'f1_per_class': f1_score(cls_labels, cls_predictions, average=None, zero_division=0),
        'per_class_accuracy': np.nan_to_num(cls_conf_matrix.diagonal() / cls_conf_matrix.sum(axis=1), nan=0.0)
    }

    # VQA指标
    all_vqa_labels = np.array(vqa_label_true)
    all_vqa_preds = np.array(vqa_label_pred)
    unique_labels = np.unique(all_vqa_labels)
    missing_labels = np.setdiff1d(np.arange(params['vqa_class']), unique_labels)
    missing_label_names = [params['convert_arr'][i] for i in missing_labels]
    vqa_conf_matrix = confusion_matrix(all_vqa_labels, all_vqa_preds, labels=np.arange(params['vqa_class']))

    # 保存VQA评估结果到CSV
    vqa_df = pd.DataFrame({
        "Img": vqa_file_names,
        "Ground Truth": [params['convert_arr'][l] for l in all_vqa_labels],
        "Prediction": [params['convert_arr'][p] for p in all_vqa_preds]
    })
    os.makedirs(os.path.join(params["output_dir"], 'text_files'), exist_ok=True)
    vqa_df.to_csv(os.path.join(params["output_dir"], f'{epoch}_vqa_eval.csv'), index=False)

    vqa_metrics = {
        'accuracy': accuracy_score(all_vqa_labels, all_vqa_preds),
        'precision_macro': precision_score(all_vqa_labels, all_vqa_preds, average='macro', zero_division=0),
        'precision_per_class': precision_score(all_vqa_labels, all_vqa_preds, average=None,
                                               labels=np.arange(params['vqa_class']), zero_division=0),
        'recall_macro': recall_score(all_vqa_labels, all_vqa_preds, average='macro', zero_division=0),
        'recall_per_class': recall_score(all_vqa_labels, all_vqa_preds, average=None,
                                         labels=np.arange(params['vqa_class']), zero_division=0),
        'f1_macro': f1_score(all_vqa_labels, all_vqa_preds, average='macro', zero_division=0),
        'f1_per_class': f1_score(all_vqa_labels, all_vqa_preds, average=None,
                                 labels=np.arange(params['vqa_class']), zero_division=0),
        'conf_matrix': vqa_conf_matrix,
        'loss': vqa_total_loss / len(val_loader),
        'missing_labels': missing_labels,
        'missing_label_names': missing_label_names
    }

    return final_miou, cls_metrics, vqa_metrics


# 计算分割任务损失
def compute_seg_loss(outputs, masks, criterion):
    preout = outputs.max(dim=1)[1].detach().cpu().numpy()
    gtout = masks.detach().cpu().numpy()
    dice_loss = 1 - get_dice(preout, gtout)
    ce_loss = criterion(outputs, masks)
    return ce_loss + dice_loss


def train_and_validate(gpus, params, tasks):
    torch.cuda.set_device(gpus[0])
    device = torch.device("cuda")
    params['device'] = device
    os.makedirs(params["output_dir"], exist_ok=True)

    # 数据预处理
    preprocess = transforms.Compose([
        transforms.Resize((params["image_size"], params["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 数据集和数据加载器（支持 fold_dir 5 折交叉验证模式）
    fold_dir = params.get("fold_dir", None)
    limit = params.get("limit", None)
    image_path = params.get("image_path", params["train_image_path"])
    mask_path = params.get("mask_path", params["train_mask_path"])
    num_workers = 0 if (limit and limit > 0) else 8

    def _build_dataset(split):
        if fold_dir:
            class_file = os.path.join(fold_dir, f"{split}_cls.txt")
            seg_file = os.path.join(fold_dir, f"{split}_seg.txt")
            vqa_file = os.path.join(fold_dir, f"{split}_vqa.txt")
            img_path = image_path
            msk_path = mask_path
        else:
            prefix = "train" if split == "train" else "valid"
            class_file = params[f"{prefix}_class_file"]
            seg_file = params[f"{prefix}_seg_file"]
            vqa_file = params[f"{prefix}_vqa_file"]
            img_path = params[f"{prefix}_image_path"]
            msk_path = params[f"{prefix}_mask_path"]
        ds = UnifiedDataset(
            root_dir=params["data_root"],
            class_file=class_file,
            seg_file=seg_file,
            vqa_file=vqa_file,
            image_path=img_path,
            mask_path=msk_path,
            transform=preprocess,
        )
        if limit and limit > 0:
            ds.data = ds.data[:limit]
        return ds

    train_dataset = _build_dataset("train")
    val_dataset = _build_dataset("val")
    test_dataset = _build_dataset("test") if fold_dir else None

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = (DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False,
                              num_workers=num_workers, pin_memory=True)
                   if test_dataset is not None else None)

    # 初始化模型（使用 SAM 主干）
    model = eye_multi_task(params=params).to(device)
    if len(gpus) > 1:
        print(f"Using DataParallel with GPUs: {gpus}")
        model = nn.DataParallel(model, device_ids=gpus)

    criterion = nn.CrossEntropyLoss().to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    # 学习率调度
    def lr_schedule(epoch):
        if epoch < params["warmup_epochs"]:
            return (epoch + 1) / params["warmup_epochs"]
        return 0.5 * (1 + math.cos(
            (epoch - params["warmup_epochs"]) * math.pi /
            (params["num_epochs"] - params["warmup_epochs"])))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)

    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(params["bert_model_path"])

    # 训练记录
    best_metric = -float('inf')
    best_epoch = 0
    best_metrics = {}
    metrics_data = []
    train_seg_miou_history = []
    val_seg_miou_history = []
    train_cls_acc_history = []
    val_cls_acc_history = []
    train_vqa_acc_history = []
    val_vqa_acc_history = []

    # ELT 动态秩调整状态
    no_improve_count = 0
    current_rank = params.get("initial_lora_rank", 64)
    rank_adjustment_history = []

    for epoch in range(params["num_epochs"]):
        model.train()
        train_loss = 0.0
        seg_miou = 0.0
        cls_correct = 0
        cls_total = 0
        vqa_correct = 0
        vqa_total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", colour='blue')
        for batch in train_bar:
            images = batch['image'].to(device)
            class_labels = batch['class_label'].to(device)
            masks = batch['mask'].to(device)
            vqa_labels = batch['vqa_answer'].to(device)
            questions = batch['vqa_question']
            vqa_inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=80)
            vqa_inputs = {k: v.to(device) for k, v in vqa_inputs.items()}

            optimizer.zero_grad()

            # 一次前向传播得到所有输出 (tasks 控制跳过未启用任务的头, 避免 OOM)
            seg_outputs, cls_outputs, vqa_outputs = model(input=images, vqa_text_inputs=vqa_inputs, tasks=tasks)

            # 计算损失（仅启用任务参与梯度）
            total_loss = 0.0
            if 'seg' in tasks:
                total_loss = total_loss + compute_seg_loss(seg_outputs, masks, criterion)
            if 'cls' in tasks:
                total_loss = total_loss + criterion(cls_outputs, class_labels)
            if 'vqa' in tasks:
                total_loss = total_loss + criterion(vqa_outputs, vqa_labels)

            # 一次反向传播和优化
            total_loss.backward()
            optimizer.step()

            # 计算训练指标（仅启用任务）
            if 'seg' in tasks:
                preout = seg_outputs.max(dim=1)[1].detach().cpu().numpy()
                gtout = masks.detach().cpu().numpy()
                pre_label = [i for i in preout]
                true_label = [i for i in gtout]
                metrics = eval_semantic_segmentation(pre_label, true_label, preout, gtout)
                seg_miou += metrics['miou']

            if 'cls' in tasks:
                _, cls_pred = torch.max(cls_outputs, 1)
                cls_correct += (cls_pred == class_labels).sum().item()
                cls_total += class_labels.size(0)

            if 'vqa' in tasks:
                _, vqa_pred = torch.max(F.softmax(vqa_outputs, dim=1), 1)
                vqa_correct += (vqa_pred == vqa_labels).sum().item()
                vqa_total += vqa_labels.size(0)

            train_loss += total_loss.item()

            # 更新进度条
            postfix_info = {
                "Loss": total_loss.item(),
                "Seg_mIoU": seg_miou / (train_bar.n + 1),
                "Cls_Acc": cls_correct / max(1, cls_total),
                "VQA_Acc": vqa_correct / max(1, vqa_total)
            }
            train_bar.set_postfix(postfix_info)

        # 计算平均训练指标
        avg_train_loss = train_loss / len(train_loader)
        train_seg_miou = seg_miou / len(train_loader)
        train_cls_acc = cls_correct / cls_total if cls_total > 0 else 0
        train_vqa_acc = vqa_correct / vqa_total if vqa_total > 0 else 0

        # ===== 使用统一验证函数 =====
        val_seg_miou, val_cls_metrics, val_vqa_metrics = validate_unified(
            model, val_loader, device, criterion, tokenizer, epoch
        )

        # 记录历史数据
        train_seg_miou_history.append(train_seg_miou)
        val_seg_miou_history.append(val_seg_miou)
        train_cls_acc_history.append(train_cls_acc)
        val_cls_acc_history.append(val_cls_metrics['accuracy'])
        train_vqa_acc_history.append(train_vqa_acc)
        val_vqa_acc_history.append(val_vqa_metrics['accuracy'])

        # 打印结果
        print(f"Epoch {epoch + 1}/{params['num_epochs']}, Train Loss: {avg_train_loss:.4f}")
        print(f"Train Seg mIoU: {train_seg_miou:.4f}, Val Seg mIoU: {val_seg_miou:.4f}")
        print(f"Train Cls Acc: {train_cls_acc:.4f}, Val Cls Acc: {val_cls_metrics['accuracy']:.4f}")
        print(f"Train VQA Acc: {train_vqa_acc:.4f}, Val VQA Acc: {val_vqa_metrics['accuracy']:.4f}")
        print(f"Missing VQA labels: {val_vqa_metrics['missing_labels']}")
        print(f"Missing VQA label names: {val_vqa_metrics['missing_label_names']}")

        # 保存最佳模型（combined_metric 只累加启用任务）
        combined_metric = 0.0
        if 'seg' in tasks:
            combined_metric += val_seg_miou
        if 'cls' in tasks:
            combined_metric += val_cls_metrics.get('accuracy', 0)
        if 'vqa' in tasks:
            combined_metric += val_vqa_metrics.get('accuracy', 0)

        if combined_metric > best_metric:
            best_metric = combined_metric
            best_epoch = epoch + 1
            no_improve_count = 0
            best_metrics = {
                'epoch': best_epoch,
                'seg_miou': val_seg_miou,
                'cls_metrics': val_cls_metrics,
                'vqa_metrics': val_vqa_metrics
            }
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, params["save_path"])
        else:
            no_improve_count += 1

        print(f"Current Best Metric: {best_metric:.4f}, Epoch: {best_epoch}")

        # ===== ELT：基于信息熵的动态 LoRA 秩调整 =====
        if params.get("use_elt", False):
            if no_improve_count >= params["rank_patience"] and current_rank > params["min_lora_rank"]:
                try:
                    if isinstance(model, nn.DataParallel):
                        new_rank = model.module.get_entropy_based_rank(
                            info_retention_ratio=params["info_retention_ratio"])
                    else:
                        new_rank = model.get_entropy_based_rank(
                            info_retention_ratio=params["info_retention_ratio"])

                    new_rank = max(params["min_lora_rank"], min(new_rank, current_rank - 1))

                    if new_rank != current_rank:
                        print(f"\n=== ELT: 连续{params['rank_patience']}轮无提升，调整 LoRA rank: {current_rank} -> {new_rank} ===")
                        rank_adjustment_history.append({
                            'epoch': epoch + 1,
                            'old_rank': current_rank,
                            'new_rank': new_rank,
                            'metric_before': combined_metric,
                            'method': 'information_entropy'
                        })
                        if isinstance(model, nn.DataParallel):
                            model.module.update_lora_ranks(new_rank)
                        else:
                            model.update_lora_ranks(new_rank)
                        current_rank = new_rank
                        no_improve_count = 0
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.5
                except Exception as e:
                    print(f"ELT rank adjustment failed: {e}")

        scheduler.step()

        # 记录epoch指标
        epoch_metrics = {
            'Epoch': epoch + 1,
            'Train_Loss': avg_train_loss,
            'Train_Seg_mIoU': train_seg_miou,
            'Val_Seg_mIoU': val_seg_miou,
            'Train_Cls_Accuracy': train_cls_acc,
            'Val_Cls_Accuracy': val_cls_metrics['accuracy'],
            'Val_Cls_Precision_Macro': val_cls_metrics['precision_macro'],
            'Val_Cls_Recall_Macro': val_cls_metrics['recall_macro'],
            'Val_Cls_F1_Macro': val_cls_metrics['f1_macro'],
            'Train_VQA_Accuracy': train_vqa_acc,
            'Val_VQA_Accuracy': val_vqa_metrics['accuracy'],
            'Val_VQA_Precision_Macro': val_vqa_metrics['precision_macro'],
            'Val_VQA_Recall_Macro': val_vqa_metrics['recall_macro'],
            'Val_VQA_F1_Macro': val_vqa_metrics['f1_macro'],
        }
        metrics_data.append(epoch_metrics)

    # 保存训练指标到CSV
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(os.path.join(params["output_dir"], 'metrics.csv'), index=False)

    # 绘制训练曲线
    plt.figure(figsize=(12, 8))

    plt.plot(range(1, params["num_epochs"] + 1), train_seg_miou_history, label='Train Seg mIoU')
    plt.plot(range(1, params["num_epochs"] + 1), val_seg_miou_history, label='Val Seg mIoU')

    plt.plot(range(1, params["num_epochs"] + 1), train_cls_acc_history, label='Train Cls Accuracy')
    plt.plot(range(1, params["num_epochs"] + 1), val_cls_acc_history, label='Val Cls Accuracy')

    plt.plot(range(1, params["num_epochs"] + 1), train_vqa_acc_history, label='Train VQA Accuracy')
    plt.plot(range(1, params["num_epochs"] + 1), val_vqa_acc_history, label='Val VQA Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Training and Validation Metrics (SAM Backbone)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(params["output_dir"], 'curves.png'))
    plt.close()

    # 保存最佳模型指标
    with open(os.path.join(params["output_dir"], 'best_metrics.txt'), 'w') as f:
        f.write(f"Best Model (Epoch {best_metrics['epoch']})\n\n")

        f.write(f"Segmentation mIoU: {best_metrics['seg_miou']:.4f}\n\n")

        f.write("Classification Metrics:\n")
        f.write(f"Accuracy: {best_metrics['cls_metrics']['accuracy']:.4f}\n")
        f.write(f"Precision (macro): {best_metrics['cls_metrics']['precision_macro']:.4f}\n")
        f.write(f"Recall (macro): {best_metrics['cls_metrics']['recall_macro']:.4f}\n")
        f.write(f"F1 Score (macro): {best_metrics['cls_metrics']['f1_macro']:.4f}\n")
        f.write(f"Confusion Matrix:\n{best_metrics['cls_metrics']['conf_matrix']}\n\n")

        f.write("VQA Metrics:\n")
        f.write(f"Accuracy: {best_metrics['vqa_metrics']['accuracy']:.4f}\n")
        f.write(f"Precision (macro): {best_metrics['vqa_metrics']['precision_macro']:.4f}\n")
        f.write(f"Recall (macro): {best_metrics['vqa_metrics']['recall_macro']:.4f}\n")
        f.write(f"F1 Score (macro): {best_metrics['vqa_metrics']['f1_macro']:.4f}\n")
        f.write(f"Confusion Matrix:\n{best_metrics['vqa_metrics']['conf_matrix']}\n")

    print(f"Training completed, best metric: {best_metric:.4f}, model saved to {params['save_path']}")

    # ===== 若存在独立测试集，则加载最佳模型在 test 上评估并写出 JSON（供 5 折汇总） =====
    if test_loader is not None:
        state_dict = torch.load(params["save_path"], map_location=device)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        model.eval()
        test_seg_miou, test_cls_metrics, test_vqa_metrics = validate_unified(
            model, test_loader, device, criterion, tokenizer, epoch="test"
        )
        import json
        test_results = {
            "fold": params.get("fold", None),
            "best_epoch": best_epoch,
            "tasks": tasks,
            "final_rank": current_rank if params.get("use_elt", False) else params.get("initial_lora_rank", 64),
            "seg_miou": float(test_seg_miou) if 'seg' in tasks else None,
            "cls_accuracy": float(test_cls_metrics["accuracy"]) if 'cls' in tasks else None,
            "cls_precision": float(test_cls_metrics["precision_macro"]) if 'cls' in tasks else None,
            "cls_recall": float(test_cls_metrics["recall_macro"]) if 'cls' in tasks else None,
            "cls_f1": float(test_cls_metrics["f1_macro"]) if 'cls' in tasks else None,
            "vqa_accuracy": float(test_vqa_metrics["accuracy"]) if 'vqa' in tasks else None,
            "vqa_precision": float(test_vqa_metrics["precision_macro"]) if 'vqa' in tasks else None,
            "vqa_recall": float(test_vqa_metrics["recall_macro"]) if 'vqa' in tasks else None,
            "vqa_f1": float(test_vqa_metrics["f1_macro"]) if 'vqa' in tasks else None,
        }
        with open(os.path.join(params["output_dir"], "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=2)
        print(f"Test results: {test_results}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Task-Specific Gradient Update (SAM Backbone)")
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPU IDs')
    parser.add_argument('--tasks', type=str, default='seg,cls,vqa',
                        help='Comma-separated list of tasks: seg,cls,vqa')
    parser.add_argument('--fold', type=int, default=None, help='Fold index for 5-fold CV')
    parser.add_argument('--fold_dir', type=str, default=None,
                        help='Fold dir relative to data_root, e.g. folds/fold0')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--mask_path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit samples per split (for smoke test)')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=66)
    # ELT / HPHA 控制变量
    parser.add_argument('--elt', action='store_true', help='Enable Entropy-Driven LoRA Tuning (ELT)')
    parser.add_argument('--attention', type=str, default=None, choices=['hpha', 'self', 'none'],
                        help='Attention module: hpha (self+spatial), self, none')
    parser.add_argument('--init_rank', type=int, default=None, help='Initial LoRA rank')
    parser.add_argument('--min_rank', type=int, default=None, help='Minimum LoRA rank for ELT')
    parser.add_argument('--rank_patience', type=int, default=None, help='Epochs without improvement before ELT rank reduction')
    parser.add_argument('--info_retention_ratio', type=float, default=None, help='Information retention ratio for ELT')
    args = parser.parse_args()

    # 用命令行参数覆盖 params
    if args.data_root:
        params["data_root"] = args.data_root
    if args.image_path:
        params["image_path"] = args.image_path
    if args.mask_path:
        params["mask_path"] = args.mask_path
    if args.epochs:
        params["num_epochs"] = args.epochs
    if args.batch_size:
        params["batch_size"] = args.batch_size
    if args.limit:
        params["limit"] = args.limit
    if args.fold_dir:
        params["fold_dir"] = args.fold_dir
    if args.fold is not None:
        params["fold"] = args.fold
        if not args.fold_dir:
            params["fold_dir"] = f"folds/fold{args.fold}"
        if not args.output_dir:
            params["output_dir"] = f"./output_dir/cv/fold{args.fold}"
            params["save_path"] = os.path.join(params["output_dir"], "saved_model.pth")
    if args.output_dir:
        params["output_dir"] = args.output_dir
        params["save_path"] = os.path.join(args.output_dir, "saved_model.pth")

    # ELT / HPHA 控制变量覆盖
    if args.elt:
        params["use_elt"] = True
    if args.attention:
        params["attention"] = args.attention
    if args.init_rank:
        params["initial_lora_rank"] = args.init_rank
    if args.min_rank:
        params["min_lora_rank"] = args.min_rank
    if args.rank_patience:
        params["rank_patience"] = args.rank_patience
    if args.info_retention_ratio:
        params["info_retention_ratio"] = args.info_retention_ratio

    gpus = [int(gpu) for gpu in args.gpus.split(',')]
    tasks = [task.strip() for task in args.tasks.split(',')]
    valid_tasks = {'seg', 'cls', 'vqa'}
    tasks = [task for task in tasks if task in valid_tasks]
    if not tasks:
        print("No valid tasks specified. Choose from: seg, cls, vqa")
        return
    if not torch.cuda.is_available():
        print("CUDA不可用，请检查环境")
        return
    if len(gpus) > torch.cuda.device_count():
        print(f"请求的GPU数量 ({len(gpus)}) 超过可用GPU数量 ({torch.cuda.device_count()})")
        return
    print(f"Using GPUs: {gpus}, Tasks: {tasks}")
    print(f"Config: ELT={params['use_elt']}, attention={params['attention']}, "
          f"init_rank={params['initial_lora_rank']}, min_rank={params['min_lora_rank']}, "
          f"rank_patience={params['rank_patience']}, info_retention_ratio={params['info_retention_ratio']}")
    print("使用 SAM ViT-B 主干 + LoRA + 多任务学习")
    set_seed(args.seed)
    model = train_and_validate(gpus, params, tasks)


if __name__ == "__main__":
    main()
