#!/usr/bin/env python3
"""
Linear Probe Trainer - 训练1024个linear probe网络

每个linear probe对应一个attention head，学习从head输出预测语义先验偏置强度。

架构：
- 输入: head_vector (128维)
- 输出: g_u (标量，-1到1之间)
- 模型: Linear(128, 1) + Tanh
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Optional, Tuple
import json
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import random
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))

class HeadGroundTruthDataset(Dataset):
    """Head真值对数据集"""

    def __init__(self, pairs: List[Dict], dtype: torch.dtype = torch.float32):
        """
        Args:
            pairs: 真值对列表，每个元素包含 'head_vector' 和 'g_u'
            dtype: 数据类型，默认 float32，可设置为 float64 用于高精度训练
        """
        self.pairs = pairs
        self.dtype = dtype
        # 过滤掉无效数据
        self.valid_pairs = []
        for pair in pairs:
            if 'head_vector' in pair and 'g_u' in pair:
                head_vector = pair['head_vector']
                g_u = pair['g_u']
                # 检查 head_vector 是否有效
                if isinstance(head_vector, list) and len(head_vector) > 0:
                    # 检查 g_u 是否为有效数值（不是 nan 或 inf）
                    if isinstance(g_u, (int, float)):
                        if not (np.isnan(g_u) or np.isinf(g_u)):
                            self.valid_pairs.append(pair)
                    else:
                        # 如果不是数值类型，跳过
                        continue

        print(f"  有效样本数: {len(self.valid_pairs)}/{len(pairs)}")
        print(f"  数据类型: {dtype}")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        pair = self.valid_pairs[idx]
        head_vector = pair['head_vector']

        # 处理 head_vector 可能是嵌套列表的情况
        # 如果 head_vector 是 [[...]]，需要展平为 [...]
        if isinstance(head_vector, list) and len(head_vector) > 0:
            if isinstance(head_vector[0], list):
                # 嵌套列表，展平为一维列表
                head_vector = head_vector[0]

        head_vector = torch.tensor(head_vector, dtype=self.dtype)
        # 确保 head_vector 是一维的 [head_dim]
        if head_vector.dim() > 1:
            head_vector = head_vector.squeeze()

        g_u = torch.tensor(pair['g_u'], dtype=self.dtype)
        return head_vector, g_u


class LinearProbe(nn.Module):
    """单个Linear Probe网络"""

    def __init__(self, input_dim: int = 128, hidden_dim: Optional[int] = None, use_dropout: bool = False):
        """
        Args:
            input_dim: 输入维度（head维度，默认128）
            hidden_dim: 隐藏层维度（None表示使用简单线性映射）
            use_dropout: 是否使用Dropout
        """
        super(LinearProbe, self).__init__()

        if hidden_dim is None:
            # 简单线性映射: Linear(128, 1) + Tanh
            self.linear = nn.Linear(input_dim, 1)
            self.tanh = nn.Tanh()
            self.use_hidden = False
        else:
            # 单隐藏层: Linear(128, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, 1) -> Tanh
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2) if use_dropout else nn.Identity()
            self.fc2 = nn.Linear(hidden_dim, 1)
            self.tanh = nn.Tanh()
            self.use_hidden = True

    def forward(self, x):
        if self.use_hidden:
            h = self.relu(self.fc1(x))
            h = self.dropout(h)
            return self.tanh(self.fc2(h))
        else:
            return self.tanh(self.linear(x))


class LinearProbeTrainer:
    """训练1024个Linear Probe网络的类"""

    def __init__(
        self,
        num_layers: int = 32,
        num_heads: int = 32,
        input_dim: int = 128,
        hidden_dim: Optional[int] = None,
        use_dropout: bool = False,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32
    ):
        """
        Args:
            num_layers: 模型层数
            num_heads: 每层的head数
            input_dim: head维度
            hidden_dim: 隐藏层维度（None表示使用简单线性映射）
            use_dropout: 是否使用Dropout
            device: 设备
            dtype: 数据类型，默认 float32，可设置为 float64 用于高精度训练
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_dropout = use_dropout
        self.device = device
        self.dtype = dtype

        # 创建1024个linear probe
        self.probes = nn.ModuleDict()
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                key = f"layer_{layer_idx}_head_{head_idx}"
                probe = LinearProbe(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    use_dropout=use_dropout
                )
                # 将模型转换为指定精度
                probe = probe.to(dtype=dtype).to(device)
                self.probes[key] = probe

        print(f"✓ 创建了 {len(self.probes)} 个linear probe")
        print(f"  数据类型: {dtype}")
        if hidden_dim is None:
            print(f"  架构: Linear({input_dim}, 1) + Tanh")
        else:
            print(f"  架构: Linear({input_dim}, {hidden_dim}) -> ReLU -> Dropout -> Linear({hidden_dim}, 1) -> Tanh")

    def load_ground_truth_data(self, ground_truth_dir: str) -> Dict[str, List[Dict]]:
        """
        从目录加载真值对数据

        Args:
            ground_truth_dir: 真值对文件目录（包含 layer_X_head_Y.json 文件）

        Returns:
            Dict[str, List[Dict]]: {key: pairs} 映射，key格式为 "layer_X_head_Y"
        """
        ground_truth_dir = Path(ground_truth_dir)
        data_by_probe = {}

        print(f"\n加载真值对数据从: {ground_truth_dir}")

        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                key = f"layer_{layer_idx}_head_{head_idx}"
                filename = f"layer_{layer_idx}_head_{head_idx}.json"
                filepath = ground_truth_dir / filename

                if filepath.exists():
                    with open(filepath, 'r', encoding='utf-8') as f:
                        pairs = json.load(f)
                    data_by_probe[key] = pairs
                else:
                    data_by_probe[key] = []
                    print(f"  ⚠️  文件不存在: {filepath}")

        # 统计信息
        total_pairs = sum(len(pairs) for pairs in data_by_probe.values())
        non_empty_probes = sum(1 for pairs in data_by_probe.values() if len(pairs) > 0)
        print(f"✓ 加载完成:")
        print(f"  总真值对数量: {total_pairs}")
        print(f"  有数据的probe数量: {non_empty_probes}/{len(data_by_probe)}")

        return data_by_probe

    def train_probe(
        self,
        probe: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        patience: int = 10,
        verbose: bool = False,
        print_interval: int = 10,
        head_key: Optional[str] = None
    ) -> Dict:
        """
        训练单个probe

        Args:
            probe: Linear probe模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            num_epochs: 最大训练轮数
            lr: 学习率
            weight_decay: L2正则化系数
            patience: 早停patience
            verbose: 是否输出详细信息
            print_interval: 每N个epoch打印一次loss（0表示不打印）
            head_key: head的标识符（用于打印，可选）

        Returns:
            Dict: 训练结果统计
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            # 训练阶段
            probe.train()
            train_loss = 0.0
            train_count = 0

            for head_vector, g_u in train_loader:
                head_vector = head_vector.to(self.device, dtype=self.dtype)
                g_u = g_u.to(self.device, dtype=self.dtype).unsqueeze(1)  # [batch_size, 1]

                optimizer.zero_grad()
                pred = probe(head_vector)
                loss = criterion(pred, g_u)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * head_vector.size(0)
                train_count += head_vector.size(0)

            avg_train_loss = train_loss / train_count if train_count > 0 else 0.0
            train_losses.append(avg_train_loss)

            # 验证阶段
            if val_loader is not None:
                probe.eval()
                val_loss = 0.0
                val_count = 0

                with torch.no_grad():
                    for head_vector, g_u in val_loader:
                        head_vector = head_vector.to(self.device, dtype=self.dtype)
                        g_u = g_u.to(self.device, dtype=self.dtype).unsqueeze(1)

                        pred = probe(head_vector)
                        loss = criterion(pred, g_u)

                        val_loss += loss.item() * head_vector.size(0)
                        val_count += head_vector.size(0)

                avg_val_loss = val_loss / val_count if val_count > 0 else float('inf')
                val_losses.append(avg_val_loss)

                # 学习率调度
                scheduler.step(avg_val_loss)

                # 打印loss（如果启用）
                if print_interval > 0 and (epoch + 1) % print_interval == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    head_info = f"[{head_key}] " if head_key else ""
                    print(f"    {head_info}Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.6f}, "
                          f"Val Loss = {avg_val_loss:.6f}, LR = {current_lr:.6f}")

                # 早停检查
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose or print_interval > 0:
                            head_info = f"[{head_key}] " if head_key else ""
                            print(f"    {head_info}早停于epoch {epoch+1}, best_val_loss = {best_val_loss:.6f}")
                        break
            else:
                # 没有验证集，只使用训练损失
                avg_val_loss = avg_train_loss
                val_losses.append(avg_val_loss)

                # 打印loss（如果启用）
                if print_interval > 0 and (epoch + 1) % print_interval == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    head_info = f"[{head_key}] " if head_key else ""
                    print(f"    {head_info}Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.6f}, "
                          f"LR = {current_lr:.6f}")

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss if val_loader is not None else train_losses[-1],
            'num_epochs': epoch + 1
        }

    def train_all(
        self,
        ground_truth_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        batch_size: int = 64,
        num_epochs: int = 100,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        patience: int = 10,
        save_dir: Optional[str] = None,
        print_interval: int = 10
    ):
        """
        训练所有1024个linear probe

        Args:
            ground_truth_dir: 真值对文件目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            batch_size: 批量大小
            num_epochs: 最大训练轮数
            lr: 学习率
            weight_decay: L2正则化系数
            patience: 早停patience
            save_dir: 模型保存目录（如果为None则不保存）
            print_interval: 每N个epoch打印一次loss（0表示不打印，默认10）
        """
        # 加载数据
        data_by_probe = self.load_ground_truth_data(ground_truth_dir)

        # 训练结果统计
        results = {}
        all_train_losses = []
        all_val_losses = []

        print(f"\n开始训练 {len(self.probes)} 个linear probe...")
        print(f"数据划分: 训练集 {train_ratio*100:.1f}%, 验证集 {val_ratio*100:.1f}%, 测试集 {test_ratio*100:.1f}%")
        print(f"训练参数: batch_size={batch_size}, lr={lr}, weight_decay={weight_decay}, patience={patience}")
        print(f"数据类型: {self.dtype}")
        if print_interval > 0:
            print(f"Loss打印间隔: 每 {print_interval} 个epoch")

        # 训练每个probe
        trained_count = 0
        for key, probe in tqdm(self.probes.items(), desc="训练进度"):
            pairs = data_by_probe.get(key, [])

            if len(pairs) == 0:
                results[key] = {
                    'status': 'no_data',
                    'num_samples': 0
                }
                continue

            # 创建数据集（使用指定的精度）
            dataset = HeadGroundTruthDataset(pairs, dtype=self.dtype)

            # 打印当前训练的 head 信息（每10个head打印一次，或第一个）
            if trained_count == 0 or (trained_count + 1) % 10 == 0:
                print(f"\n[{trained_count + 1}/1024] 训练 {key} (样本数: {len(dataset)})...")

            if len(dataset) == 0:
                results[key] = {
                    'status': 'no_valid_data',
                    'num_samples': 0
                }
                continue

            # 划分数据集
            total_size = len(dataset)
            train_size = int(train_ratio * total_size)
            val_size = int(val_ratio * total_size)
            test_size = total_size - train_size - val_size

            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )

            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # 训练
            train_result = self.train_probe(
                probe=probe,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                lr=lr,
                weight_decay=weight_decay,
                patience=patience,
                verbose=False,
                print_interval=print_interval,
                head_key=key
            )

            # 评估测试集
            probe.eval()
            test_loss = 0.0
            test_count = 0
            test_predictions = []
            test_targets = []

            with torch.no_grad():
                for head_vector, g_u in test_loader:
                    head_vector = head_vector.to(self.device, dtype=self.dtype)
                    g_u = g_u.to(self.device, dtype=self.dtype).unsqueeze(1)

                    pred = probe(head_vector)
                    loss = nn.MSELoss()(pred, g_u)

                    # 检查 loss 是否为 nan 或 inf
                    if not (np.isnan(loss.item()) or np.isinf(loss.item())):
                        test_loss += loss.item() * head_vector.size(0)
                        test_count += head_vector.size(0)

                        # 检查预测值和目标值是否为 nan/inf
                        pred_np = pred.cpu().numpy().flatten()
                        g_u_np = g_u.cpu().numpy().flatten()

                        # 只保留有效的预测值和目标值（不是 nan 或 inf）
                        valid_mask = ~(np.isnan(pred_np) | np.isinf(pred_np) | np.isnan(g_u_np) | np.isinf(g_u_np))
                        if np.any(valid_mask):
                            test_predictions.extend(pred_np[valid_mask].tolist())
                            test_targets.extend(g_u_np[valid_mask].tolist())

            # 处理空测试集或所有值都是 nan/inf 的情况
            if test_count == 0 or len(test_predictions) == 0:
                avg_test_loss = float('nan')
                r2 = float('nan')
                mae = float('nan')
            else:
                avg_test_loss = test_loss / test_count

                # 计算R²
                test_predictions = np.array(test_predictions)
                test_targets = np.array(test_targets)

                # 再次检查是否有 nan/inf（防御性编程）
                valid_mask = ~(np.isnan(test_predictions) | np.isinf(test_predictions) |
                              np.isnan(test_targets) | np.isinf(test_targets))
                if np.sum(valid_mask) == 0:
                    # 所有值都是 nan/inf
                    avg_test_loss = float('nan')
                    r2 = float('nan')
                    mae = float('nan')
                else:
                    test_predictions = test_predictions[valid_mask]
                    test_targets = test_targets[valid_mask]

                    ss_res = np.sum((test_targets - test_predictions) ** 2)
                    ss_tot = np.sum((test_targets - np.mean(test_targets)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                    # 检查 r2 是否为 nan/inf
                    if np.isnan(r2) or np.isinf(r2):
                        r2 = float('nan')

                    # 计算MAE
                    mae = np.mean(np.abs(test_predictions - test_targets))

                    # 检查 mae 是否为 nan/inf
                    if np.isnan(mae) or np.isinf(mae):
                        mae = float('nan')

            results[key] = {
                'status': 'trained',
                'num_samples': total_size,
                'train_size': train_size,
                'val_size': val_size,
                'test_size': test_size,
                'best_val_loss': train_result['best_val_loss'],
                'test_loss': avg_test_loss,
                'test_r2': r2,
                'test_mae': mae,
                'num_epochs': train_result['num_epochs'],
                'train_losses': train_result['train_losses'],  # 保存完整的loss历史
                'val_losses': train_result['val_losses']  # 保存完整的loss历史
            }

            all_train_losses.append(train_result['train_losses'][-1])
            all_val_losses.append(train_result['val_losses'][-1])

            trained_count += 1

        # 打印统计信息
        print(f"\n训练完成!")
        print(f"=" * 80)

        trained_count = sum(1 for r in results.values() if r.get('status') == 'trained')
        no_data_count = sum(1 for r in results.values() if r.get('status') == 'no_data')
        no_valid_count = sum(1 for r in results.values() if r.get('status') == 'no_valid_data')

        print(f"训练统计:")
        print(f"  成功训练: {trained_count}/{len(self.probes)}")
        print(f"  无数据: {no_data_count}")
        print(f"  无有效数据: {no_valid_count}")

        if trained_count > 0:
            test_losses = [r['test_loss'] for r in results.values() if r.get('status') == 'trained']
            test_r2s = [r['test_r2'] for r in results.values() if r.get('status') == 'trained']
            test_maes = [r['test_mae'] for r in results.values() if r.get('status') == 'trained']

            # 过滤掉 nan 和 inf 值
            test_losses_valid = [x for x in test_losses if not (np.isnan(x) or np.isinf(x))]
            test_r2s_valid = [x for x in test_r2s if not (np.isnan(x) or np.isinf(x))]
            test_maes_valid = [x for x in test_maes if not (np.isnan(x) or np.isinf(x))]

            print(f"\n测试集性能:")
            if len(test_losses_valid) > 0:
                print(f"  平均MSE: {np.mean(test_losses_valid):.6f} ± {np.std(test_losses_valid):.6f} (有效样本: {len(test_losses_valid)}/{len(test_losses)})")
            else:
                print(f"  平均MSE: nan (所有样本都无效)")

            if len(test_r2s_valid) > 0:
                print(f"  平均R²: {np.mean(test_r2s_valid):.4f} ± {np.std(test_r2s_valid):.4f} (有效样本: {len(test_r2s_valid)}/{len(test_r2s)})")
            else:
                print(f"  平均R²: nan (所有样本都无效)")

            if len(test_maes_valid) > 0:
                print(f"  平均MAE: {np.mean(test_maes_valid):.6f} ± {np.std(test_maes_valid):.6f} (有效样本: {len(test_maes_valid)}/{len(test_maes)})")
            else:
                print(f"  平均MAE: nan (所有样本都无效)")

            # 如果有很多无效样本，打印警告
            if len(test_losses_valid) < len(test_losses):
                nan_count = len(test_losses) - len(test_losses_valid)
                print(f"\n  ⚠️  警告: {nan_count}/{len(test_losses)} 个probe的测试集指标为nan/inf")
                print(f"     可能原因:")
                print(f"     1. 测试集为空（数据量太少，分割后测试集为空）")
                print(f"     2. 数据中包含nan/inf值（g_u或head_vector）")
                print(f"     3. 模型预测产生了nan/inf值")

        # 保存模型和生成可视化
        subdir_name = None
        save_path = None
        if save_dir is not None:
            # 从 ground_truth_dir 提取最后一级目录名
            # 例如: "train/coco_train_json/coco_train_20_generate_spp_gt_pair" -> "coco_train_20_generate_spp_gt_pair"
            ground_truth_path = Path(ground_truth_dir)
            subdir_name = ground_truth_path.name  # 获取最后一级目录名

            # 在 save_dir 下创建子目录
            save_path = Path(save_dir) / subdir_name
            save_path.mkdir(parents=True, exist_ok=True)

            # 保存所有模型
            for key, probe in self.probes.items():
                model_path = save_path / f"{key}.pth"
                torch.save(probe.state_dict(), model_path)

            # 保存训练结果（不包含完整的loss历史，只保留统计信息）
            results_for_save = {}
            for key, result in results.items():
                result_copy = result.copy()
                # 移除完整的loss历史（太大），只保留最后的值
                if 'train_losses' in result_copy:
                    result_copy['final_train_loss'] = result_copy['train_losses'][-1] if result_copy['train_losses'] else None
                    result_copy['final_val_loss'] = result_copy['val_losses'][-1] if result_copy['val_losses'] else None
                    del result_copy['train_losses']
                    del result_copy['val_losses']
                results_for_save[key] = result_copy

            results_path = save_path / "training_results.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results_for_save, f, indent=2, ensure_ascii=False)

            print(f"\n模型已保存到: {save_path}")
            print(f"  基础目录: {save_dir}")
            print(f"  子目录: {subdir_name}")

            # 生成loss可视化图表
            self._plot_sample_loss_curves(results, save_path, subdir_name)

            # 生成汇总文档
            self._generate_summary_report(results, save_path, subdir_name, trained_count, no_data_count, no_valid_count)

        return results

    def _plot_sample_loss_curves(self, results: Dict, save_path: Path, subdir_name: str):
        """
        从第8、16、24、32层中每层随机选择一个head，绘制loss变化折线图

        Args:
            results: 训练结果字典
            save_path: 保存路径
            subdir_name: 子目录名称
        """
        # 目标层索引（0-based，所以是7, 15, 23, 31）
        target_layers = [7, 15, 23, 31]

        # 收集每个目标层中成功训练的head
        heads_by_layer = {layer: [] for layer in target_layers}
        for key, result in results.items():
            if result.get('status') == 'trained' and 'train_losses' in result:
                # 解析key: "layer_X_head_Y"
                parts = key.split('_')
                layer_idx = int(parts[1])
                if layer_idx in target_layers:
                    heads_by_layer[layer_idx].append((key, result))

        # 从每层随机选择一个head
        selected_heads = []
        for layer_idx in target_layers:
            available_heads = heads_by_layer[layer_idx]
            if len(available_heads) > 0:
                selected_key, selected_result = random.choice(available_heads)
                selected_heads.append((layer_idx, selected_key, selected_result))

        if len(selected_heads) == 0:
            print("  ⚠️  无法生成loss曲线图：没有找到成功训练的head")
            return

        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Training Loss Curves - {subdir_name}', fontsize=16, fontweight='bold')

        # 填充子图（如果不足4个，留空）
        for idx, (layer_idx, key, result) in enumerate(selected_heads[:4]):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            train_losses = result.get('train_losses', [])
            val_losses = result.get('val_losses', [])

            if len(train_losses) > 0:
                epochs = range(1, len(train_losses) + 1)
                ax.plot(epochs, train_losses, label='Train Loss', linewidth=2, color='blue')

                if len(val_losses) > 0 and len(val_losses) == len(train_losses):
                    ax.plot(epochs, val_losses, label='Val Loss', linewidth=2, color='red')

                ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
                ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
                head_idx = key.split('_')[3]
                ax.set_title(f'Layer {layer_idx + 1}, Head {head_idx}', fontsize=14, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=10)
            else:
                ax.text(0.5, 0.5, 'No Loss Data', ha='center', va='center', fontsize=12)
                head_idx = key.split('_')[3]
                ax.set_title(f'Layer {layer_idx + 1}, Head {head_idx}', fontsize=14, fontweight='bold')

        # 如果不足4个，隐藏多余的子图
        for idx in range(len(selected_heads), 4):
            row = idx // 2
            col = idx % 2
            axes[row, col].axis('off')

        plt.tight_layout()

        # 保存图片
        plot_path = save_path / f"{subdir_name}_loss_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Loss曲线图已保存到: {plot_path}")
        selected_info = [f'Layer {l+1}, Head {k.split("_")[3]}' for l, k, _ in selected_heads[:4]]
        print(f"  选择的head: {selected_info}")

    def _generate_summary_report(self, results: Dict, save_path: Path, subdir_name: str,
                                trained_count: int, no_data_count: int, no_valid_count: int):
        """
        生成训练结果汇总文档

        Args:
            results: 训练结果字典
            save_path: 保存路径
            subdir_name: 子目录名称（用于文档命名）
            trained_count: 成功训练的数量
            no_data_count: 无数据的数量
            no_valid_count: 无有效数据的数量
        """
        report_path = save_path / f"{subdir_name}_training_summary.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Linear Probe Training Summary Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # 训练统计
            f.write("Training Statistics\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Probes: {len(self.probes)}\n")
            f.write(f"Successfully Trained: {trained_count}/{len(self.probes)}\n")
            f.write(f"No Data: {no_data_count}\n")
            f.write(f"No Valid Data: {no_valid_count}\n")
            f.write("\n")

            # 测试集性能统计
            if trained_count > 0:
                test_losses = [r['test_loss'] for r in results.values() if r.get('status') == 'trained']
                test_r2s = [r['test_r2'] for r in results.values() if r.get('status') == 'trained']
                test_maes = [r['test_mae'] for r in results.values() if r.get('status') == 'trained']

                f.write("Test Set Performance (All Trained Heads)\n")
                f.write("-" * 80 + "\n")
                f.write(f"Average MSE: {np.mean(test_losses):.6f} ± {np.std(test_losses):.6f}\n")
                f.write(f"Average R²: {np.mean(test_r2s):.4f} ± {np.std(test_r2s):.4f}\n")
                f.write(f"Average MAE: {np.mean(test_maes):.6f} ± {np.std(test_maes):.6f}\n")
                f.write("\n")

                # 按层统计
                f.write("Performance by Layer\n")
                f.write("-" * 80 + "\n")
                layer_stats = defaultdict(lambda: {'count': 0, 'mse': [], 'r2': [], 'mae': []})
                for key, result in results.items():
                    if result.get('status') == 'trained':
                        parts = key.split('_')
                        layer_idx = int(parts[1])
                        layer_stats[layer_idx]['count'] += 1
                        layer_stats[layer_idx]['mse'].append(result['test_loss'])
                        layer_stats[layer_idx]['r2'].append(result['test_r2'])
                        layer_stats[layer_idx]['mae'].append(result['test_mae'])

                # 只显示有数据的层
                for layer_idx in sorted(layer_stats.keys()):
                    stats = layer_stats[layer_idx]
                    if stats['count'] > 0:
                        f.write(f"Layer {layer_idx + 1:2d}: {stats['count']:3d} heads | "
                               f"MSE: {np.mean(stats['mse']):.6f} ± {np.std(stats['mse']):.6f} | "
                               f"R²: {np.mean(stats['r2']):.4f} ± {np.std(stats['r2']):.4f} | "
                               f"MAE: {np.mean(stats['mae']):.6f} ± {np.std(stats['mae']):.6f}\n")
                f.write("\n")

                # 样本数量统计
                f.write("Sample Size Statistics\n")
                f.write("-" * 80 + "\n")
                sample_sizes = [r['num_samples'] for r in results.values() if r.get('status') == 'trained']
                if len(sample_sizes) > 0:
                    f.write(f"Total Samples: {sum(sample_sizes)}\n")
                    f.write(f"Average Samples per Head: {np.mean(sample_sizes):.1f} ± {np.std(sample_sizes):.1f}\n")
                    f.write(f"Min Samples: {min(sample_sizes)}\n")
                    f.write(f"Max Samples: {max(sample_sizes)}\n")
                    f.write("\n")

            # 训练参数
            f.write("Training Parameters\n")
            f.write("-" * 80 + "\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Data Type: {self.dtype}\n")
            f.write(f"Input Dimension: {self.input_dim}\n")
            if self.hidden_dim is None:
                f.write(f"Architecture: Linear({self.input_dim}, 1) + Tanh\n")
            else:
                f.write(f"Architecture: Linear({self.input_dim}, {self.hidden_dim}) -> ReLU -> Dropout -> Linear({self.hidden_dim}, 1) + Tanh\n")
            f.write("\n")

            # 文件信息
            f.write("Output Files\n")
            f.write("-" * 80 + "\n")
            f.write(f"Model Directory: {save_path}\n")
            f.write(f"Training Results JSON: {save_path / 'training_results.json'}\n")
            f.write(f"Loss Curves Plot: {save_path / f'{subdir_name}_loss_curves.png'}\n")
            f.write(f"Summary Report: {report_path}\n")
            f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("End of Report\n")
            f.write("=" * 80 + "\n")

        print(f"✓ 训练汇总文档已保存到: {report_path}")

    def load_models(self, model_dir: str):
        """加载已训练的模型"""
        model_dir = Path(model_dir)
        loaded_count = 0

        for key, probe in self.probes.items():
            model_path = model_dir / f"{key}.pth"
            if model_path.exists():
                probe.load_state_dict(torch.load(model_path, map_location=self.device))
                loaded_count += 1

        print(f"✓ 加载了 {loaded_count}/{len(self.probes)} 个模型")

    def predict(self, layer_idx: int, head_idx: int, head_vector: torch.Tensor) -> float:
        """
        使用指定的probe进行预测

        Args:
            layer_idx: 层索引
            head_idx: head索引
            head_vector: head向量 [head_dim] 或 [batch_size, head_dim]

        Returns:
            float或tensor: 预测的g_u值
        """
        key = f"layer_{layer_idx}_head_{head_idx}"
        probe = self.probes.get(key)

        if probe is None:
            raise ValueError(f"Probe {key} not found")

        probe.eval()
        with torch.no_grad():
            if head_vector.dim() == 1:
                head_vector = head_vector.unsqueeze(0)
            head_vector = head_vector.to(self.device, dtype=self.dtype)
            pred = probe(head_vector)
            return pred.cpu().item() if pred.size(0) == 1 else pred.cpu()


if __name__ == "__main__":
    # 示例用法
    import argparse

    parser = argparse.ArgumentParser(description="训练1024个Linear Probe")
    parser.add_argument("--ground-truth-dir", type=str, default=os.path.join(current_dir, "coco_train_json/coco_train_200_generate_spp_gt_pair_np_log"),
                       help="真值对文件目录")
    parser.add_argument("--save-dir", type=str, default=os.path.join(current_dir, "ckpt"),
                       help="模型保存目录")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="设备")
    parser.add_argument("--hidden-dim", type=int, default=16,
                       help="隐藏层维度（None表示使用简单线性映射）")
    parser.add_argument("--use-dropout", action="store_true",
                       help="使用Dropout")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="批量大小")
    parser.add_argument("--num-epochs", type=int, default=200,
                       help="最大训练轮数")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="L2正则化系数")
    parser.add_argument("--patience", type=int, default=30,
                       help="早停patience")
    parser.add_argument("--dtype", type=str, default="float32",
                       choices=["float32", "float64"],
                       help="训练精度: float32 (默认) 或 float64 (高精度，适用于小数值)")
    parser.add_argument("--print-interval", type=int, default=10,
                       help="每N个epoch打印一次loss（0表示不打印，默认10）")

    args = parser.parse_args()

    # 转换 dtype 字符串为 torch.dtype
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64
    }
    dtype = dtype_map.get(args.dtype, torch.float32)

    # 创建trainer
    trainer = LinearProbeTrainer(
        num_layers=32,
        num_heads=32,
        input_dim=128,
        hidden_dim=args.hidden_dim,
        use_dropout=args.use_dropout,
        device=args.device,
        dtype=dtype
    )

    # 训练
    trainer.train_all(
        ground_truth_dir=args.ground_truth_dir,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        print_interval=args.print_interval
    )
