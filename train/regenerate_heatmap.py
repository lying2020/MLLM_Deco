#!/usr/bin/env python3
"""
从 JSON 文件重新生成热力图
用于在手动编辑热力图数据后重新生成图片

使用步骤:
1. 编辑 heatmap_data.json 文件，修改其中的 data 字段
2. 运行此脚本重新生成热力图:
   python train/regenerate_heatmap.py --heatmap-json train/coco_train_json/head_gu_statistics/coco_train_200_generate_spp_gt_pair_head_gu_heatmap_data.json
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import argparse


def load_heatmap_data(json_path: str) -> tuple:
    """
    从 JSON 文件加载热力图数据

    Args:
        json_path: JSON 文件路径

    Returns:
        tuple: (heatmap_data, num_layers, num_heads)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    num_layers = data.get('num_layers', 32)
    num_heads = data.get('num_heads', 32)
    data_dict = data.get('data', {})

    # 支持两种格式：
    # 1. 旧格式：data 是数组 [[layer0], [layer1], ...]
    # 2. 新格式：data 是对象 {"layer_0": [...], "layer_1": [...], ...}
    if isinstance(data_dict, list):
        # 旧格式：直接转换为数组
        heatmap_data = np.array(data_dict)
    elif isinstance(data_dict, dict):
        # 新格式：从字典中提取每个layer的数据
        heatmap_data = np.zeros((num_layers, num_heads))
        for layer_idx in range(num_layers):
            layer_key = f"layer_{layer_idx}"
            if layer_key in data_dict:
                layer_data = data_dict[layer_key]
                if len(layer_data) == num_heads:
                    heatmap_data[layer_idx] = np.array(layer_data)
                else:
                    raise ValueError(f"Layer {layer_idx} 的数据长度不匹配: 期望 {num_heads}, 实际 {len(layer_data)}")
            else:
                print(f"⚠️  警告: 未找到 {layer_key}，使用默认值 0.0")
    else:
        raise ValueError(f"不支持的数据格式: data 应该是数组或对象，实际类型: {type(data_dict)}")

    if heatmap_data.shape != (num_layers, num_heads):
        raise ValueError(f"数据形状不匹配: 期望 ({num_layers}, {num_heads}), 实际 {heatmap_data.shape}")

    return heatmap_data, num_layers, num_heads


def plot_heatmap_from_data(heatmap_data: np.ndarray, num_layers: int, num_heads: int,
                           output_path: str):
    """
    从数据矩阵绘制热力图

    Args:
        heatmap_data: 热力图数据矩阵 [num_layers, num_heads]
        num_layers: 层数
        num_heads: 每层的head数
        output_path: 输出文件路径
    """
    # 创建图形（设置为正方形，确保32x32的网格显示为正方形）
    fig, ax = plt.subplots(figsize=(12, 12))

    # 创建自定义colormap：深蓝 -> 白色 -> 深绿
    # 使用 TwoSlopeNorm 确保在 0 处是白色中心点
    colors = ['#000080', '#4169E1', '#87CEEB', '#FFFFFF', '#90EE90', '#228B22', '#006400']
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('blue_white_green', colors, N=n_bins)

    # 使用 TwoSlopeNorm 确保在 0 处是白色中心点
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    # 绘制热力图（反转y轴，使layer 1在底部，layer 32在顶部）
    im = ax.imshow(heatmap_data, cmap=cmap, norm=norm, aspect='equal',
                   interpolation='nearest', origin='lower',
                   extent=[-0.5, num_heads - 0.5, -0.5, num_layers - 0.5])

    # 设置坐标轴标签
    ax.set_xlabel('Head Index', fontsize=24, fontweight='bold')
    ax.set_ylabel('Layer Index', fontsize=24, fontweight='bold')
    ax.set_title('Head g_u Mean Value Heatmap', fontsize=28, fontweight='bold', pad=20)

    # 设置坐标轴范围
    ax.set_xlim(-0.5, num_heads - 0.5)
    ax.set_ylim(-0.5, num_layers - 0.5)

    # 设置刻度：只显示 [8, 16, 24, 32]
    major_ticks = [7, 15, 23, 31]
    major_labels = [8, 16, 24, 32]

    ax.set_xticks(major_ticks)
    ax.set_xticklabels(major_labels, fontsize=24, fontweight='bold')
    ax.set_yticks(major_ticks)
    ax.set_yticklabels(major_labels, fontsize=24, fontweight='bold')

    # 设置刻度线加粗
    ax.tick_params(axis='x', which='major', width=2, length=6, labelsize=24)
    ax.tick_params(axis='y', which='major', width=2, length=6, labelsize=24)

    # 隐藏次要刻度标签
    ax.set_xticks(range(num_heads), minor=True)
    ax.set_yticks(range(num_layers), minor=True)

    # 添加colorbar（只显示 [-1.0, 0.0, 1.0] 三个刻度）
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([-1.0, 0.0, 1.0])
    cbar.ax.tick_params(labelsize=24, width=2)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 热力图已保存到: {output_path}")

    plt.close()


def plot_binarized_heatmap_from_data(heatmap_data: np.ndarray, num_layers: int, num_heads: int,
                                     output_path: str, threshold_min: float = -0.5, threshold_max: float = 0.5):
    """
    从数据矩阵绘制二值化热力图

    Args:
        heatmap_data: 热力图数据矩阵 [num_layers, num_heads]
        num_layers: 层数
        num_heads: 每层的head数
        output_path: 输出文件路径
        threshold_min: 阈值下限（默认-0.5）
        threshold_max: 阈值上限（默认0.5）
    """
    # 创建二值化数据
    binarized_data = np.zeros_like(heatmap_data)
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            g_u_mean = heatmap_data[layer_idx, head_idx]
            # 二值化：如果值在[threshold_min, threshold_max]范围内，置为0；否则保持原值
            if threshold_min <= g_u_mean <= threshold_max:
                binarized_data[layer_idx, head_idx] = 0.0
            else:
                binarized_data[layer_idx, head_idx] = g_u_mean

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 12))

    # 创建自定义colormap：深蓝 -> 白色 -> 深绿
    colors = ['#000080', '#4169E1', '#87CEEB', '#FFFFFF', '#90EE90', '#228B22', '#006400']
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('blue_white_green', colors, N=n_bins)

    # 使用 TwoSlopeNorm 确保在 0 处是白色中心点
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    # 绘制二值化热力图
    im = ax.imshow(binarized_data, cmap=cmap, norm=norm, aspect='equal',
                   interpolation='nearest', origin='lower',
                   extent=[-0.5, num_heads - 0.5, -0.5, num_layers - 0.5])

    # 设置坐标轴标签
    ax.set_xlabel('Head Index', fontsize=24, fontweight='bold')
    ax.set_ylabel('Layer Index', fontsize=24, fontweight='bold')
    ax.set_title(f'Head g_u Mean Value Heatmap (Binarized: [{threshold_min}, {threshold_max}] → 0)',
                 fontsize=28, fontweight='bold', pad=20)

    # 设置坐标轴范围
    ax.set_xlim(-0.5, num_heads - 0.5)
    ax.set_ylim(-0.5, num_layers - 0.5)

    # 设置刻度：只显示 [8, 16, 24, 32]
    major_ticks = [7, 15, 23, 31]
    major_labels = [8, 16, 24, 32]

    ax.set_xticks(major_ticks)
    ax.set_xticklabels(major_labels, fontsize=24, fontweight='bold')
    ax.set_yticks(major_ticks)
    ax.set_yticklabels(major_labels, fontsize=24, fontweight='bold')

    # 设置刻度线加粗
    ax.tick_params(axis='x', which='major', width=2, length=6, labelsize=24)
    ax.tick_params(axis='y', which='major', width=2, length=6, labelsize=24)

    # 隐藏次要刻度标签
    ax.set_xticks(range(num_heads), minor=True)
    ax.set_yticks(range(num_layers), minor=True)

    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([-1.0, 0.0, 1.0])
    cbar.ax.tick_params(labelsize=24, width=2)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 二值化热力图已保存到: {output_path}")

    # 打印统计信息
    total_heads = num_layers * num_heads
    binarized_count = np.sum((binarized_data == 0.0) & (heatmap_data != 0.0))
    original_zero_count = np.sum(heatmap_data == 0.0)
    kept_count = np.sum(binarized_data != 0.0)

    print(f"  二值化统计:")
    print(f"    总head数: {total_heads}")
    print(f"    被置为0的数量: {binarized_count}")
    print(f"    原本就是0的数量: {original_zero_count}")
    print(f"    保持原值的数量: {kept_count}")
    print(f"    阈值范围: [{threshold_min}, {threshold_max}]")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="从 JSON 文件重新生成热力图")
    parser.add_argument("--heatmap-json", type=str, default="train/coco_train_json/head_gu_statistics/coco_train_200_generate_spp_gt_pair_head_gu_heatmap_data_v1.json",
                       help="热力图数据 JSON 文件路径")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="输出目录（如果不指定，使用 JSON 文件所在目录）")
    parser.add_argument("--threshold-min", type=float, default=-0.5,
                       help="二值化阈值下限（默认-0.5）")
    parser.add_argument("--threshold-max", type=float, default=0.5,
                       help="二值化阈值上限（默认0.5）")

    args = parser.parse_args()

    # 加载热力图数据
    print("=" * 80)
    print("从 JSON 文件重新生成热力图")
    print("=" * 80)
    print(f"JSON 文件: {args.heatmap_json}")

    heatmap_data, num_layers, num_heads = load_heatmap_data(args.heatmap_json)
    print(f"✓ 加载完成: {num_layers} 层 x {num_heads} 个head")
    print(f"  数据范围: [{np.min(heatmap_data):.4f}, {np.max(heatmap_data):.4f}]")

    # 确定输出目录
    json_path = Path(args.heatmap_json)
    if args.output_dir is None:
        output_dir = json_path.parent
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 生成输出文件名
    json_stem = json_path.stem  # 例如: "coco_train_200_generate_spp_gt_pair_head_gu_heatmap_data"
    # 移除 "_heatmap_data" 后缀（如果存在）
    if json_stem.endswith('_heatmap_data'):
        base_name = json_stem[:-13]  # 移除 "_heatmap_data"
    else:
        base_name = json_stem

    # 生成热力图
    print(f"\n[1/2] 生成原始热力图...")
    heatmap_path = output_dir / f"{base_name}_heatmap.png"
    plot_heatmap_from_data(heatmap_data, num_layers, num_heads, str(heatmap_path))

    # 生成二值化热力图
    print(f"\n[2/2] 生成二值化热力图...")
    binarized_heatmap_path = output_dir / f"{base_name}_binarized_heatmap.png"
    plot_binarized_heatmap_from_data(heatmap_data, num_layers, num_heads, str(binarized_heatmap_path),
                                    threshold_min=args.threshold_min, threshold_max=args.threshold_max)

    print("\n" + "=" * 80)
    print("✓ 所有热力图生成完成！")
    print("=" * 80)
    print(f"输出文件:")
    print(f"  1. 原始热力图: {heatmap_path}")
    print(f"  2. 二值化热力图: {binarized_heatmap_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
