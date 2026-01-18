#!/usr/bin/env python3
"""
分析 head 级别的 g_u 统计信息
从真值对文件中提取 g_u 值，生成可视化图表

生成图表：
1. Heatmap: 32层 x 32个head 的 g_u 平均值热力图
2. 柱状图: 1024个head的g_u平均值分布
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import argparse


def load_head_ground_truth_data(ground_truth_dir: str, num_layers: int = 32, num_heads: int = 32) -> Dict[Tuple[int, int], List[float]]:
    """
    加载所有head的真值对数据，提取g_u值

    Args:
        ground_truth_dir: 真值对文件目录
        num_layers: 层数
        num_heads: 每层的head数

    Returns:
        Dict[Tuple[int, int], List[float]]: {(layer_idx, head_idx): [g_u_values]}
    """
    ground_truth_dir = Path(ground_truth_dir)
    g_u_by_head = defaultdict(list)

    print(f"正在加载真值对数据从: {ground_truth_dir}")

    total_files = 0
    loaded_files = 0
    empty_files = 0

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            filename = f"layer_{layer_idx}_head_{head_idx}.json"
            filepath = ground_truth_dir / filename
            total_files += 1

            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        pairs = json.load(f)

                    if len(pairs) > 0:
                        # 提取所有g_u值，过滤掉nan和None
                        g_u_values = []
                        for pair in pairs:
                            if 'g_u' in pair:
                                g_u = pair.get('g_u')
                                if g_u is not None and not (isinstance(g_u, float) and np.isnan(g_u)):
                                    g_u_values.append(float(g_u))

                        if len(g_u_values) > 0:
                            g_u_by_head[(layer_idx, head_idx)] = g_u_values
                            loaded_files += 1
                        else:
                            empty_files += 1
                    else:
                        empty_files += 1
                except Exception as e:
                    print(f"  ⚠️  读取文件失败: {filepath}, 错误: {e}")
                    empty_files += 1
            else:
                empty_files += 1

    print(f"✓ 加载完成:")
    print(f"  总文件数: {total_files}")
    print(f"  成功加载: {loaded_files}")
    print(f"  空文件/不存在: {empty_files}")
    print(f"  有数据的head数: {len(g_u_by_head)}")

    return g_u_by_head


def compute_head_statistics(g_u_by_head: Dict[Tuple[int, int], List[float]]) -> Dict[Tuple[int, int], Dict]:
    """
    计算每个head的统计信息

    Args:
        g_u_by_head: {(layer_idx, head_idx): [g_u_values]}

    Returns:
        Dict[Tuple[int, int], Dict]: {(layer_idx, head_idx): {'mean': float, 'std': float, 'count': int}}
    """
    stats = {}

    for (layer_idx, head_idx), g_u_values in g_u_by_head.items():
        if len(g_u_values) > 0:
            # 过滤掉nan值
            g_u_values_clean = [v for v in g_u_values if not (isinstance(v, float) and np.isnan(v))]
            if len(g_u_values_clean) > 0:
                stats[(layer_idx, head_idx)] = {
                    'mean': np.mean(g_u_values_clean),
                    'std': np.std(g_u_values_clean),
                    'min': np.min(g_u_values_clean),
                    'max': np.max(g_u_values_clean),
                    'count': len(g_u_values_clean)
                }

    return stats


def plot_heatmap(stats: Dict[Tuple[int, int], Dict], num_layers: int = 32, num_heads: int = 32,
                 output_path: str = None):
    """
    绘制32层 x 32个head的g_u平均值热力图

    Args:
        stats: {(layer_idx, head_idx): {'mean': float, ...}}
        num_layers: 层数
        num_heads: 每层的head数
        output_path: 输出文件路径
    """
    # 创建矩阵：行是layer，列是head
    heatmap_data = np.zeros((num_layers, num_heads))

    # 填充数据
    for (layer_idx, head_idx), stat in stats.items():
        if 0 <= layer_idx < num_layers and 0 <= head_idx < num_heads:
            heatmap_data[layer_idx, head_idx] = stat['mean']

    # 创建图形（设置为正方形，确保32x32的网格显示为正方形）
    fig, ax = plt.subplots(figsize=(12, 12))

    # 创建自定义colormap：深蓝到深绿
    # 深蓝 (dark blue) -> 浅蓝 -> 白色 -> 浅绿 -> 深绿 (dark green)
    colors = ['#000080', '#4169E1', '#87CEEB', '#90EE90', '#228B22', '#006400']
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('blue_to_green', colors, N=n_bins)

    # 绘制热力图（反转y轴，使layer 1在底部，layer 32在顶部）
    # 使用 aspect='equal' 确保每个单元格都是正方形
    im = ax.imshow(heatmap_data, cmap=cmap, aspect='equal', vmin=-1.0, vmax=1.0,
                   interpolation='nearest', origin='lower',
                   extent=[-0.5, num_heads - 0.5, -0.5, num_layers - 0.5])

    # 设置坐标轴标签（字体放大2倍）
    ax.set_xlabel('Head Index', fontsize=24, fontweight='bold')
    ax.set_ylabel('Layer Index', fontsize=24, fontweight='bold')
    # (Dark Blue: Suppression, Dark Green: Enhancement)
    ax.set_title('Head g_u Mean Value Heatmap', fontsize=28, fontweight='bold', pad=20)

    # 设置坐标轴范围，确保显示完整的32x32网格
    ax.set_xlim(-0.5, num_heads - 0.5)
    ax.set_ylim(-0.5, num_layers - 0.5)

    # 设置刻度：只显示 [8, 16, 24, 32]，索引从1开始计数
    # 横轴：head索引+1，只显示 [8, 16, 24, 32]
    major_ticks = [7, 15, 23, 31]  # 对应索引 8, 16, 24, 32（因为从0开始，所以是7, 15, 23, 31）
    major_labels = [8, 16, 24, 32]  # 显示的标签（索引+1）

    ax.set_xticks(major_ticks)
    ax.set_xticklabels(major_labels, fontsize=24, fontweight='bold')  # 字体放大：20->24
    ax.set_yticks(major_ticks)
    ax.set_yticklabels(major_labels, fontsize=24, fontweight='bold')  # 字体放大：20->24

    # 设置刻度线加粗
    ax.tick_params(axis='x', which='major', width=2, length=6, labelsize=24)
    ax.tick_params(axis='y', which='major', width=2, length=6, labelsize=24)

    # 隐藏次要刻度标签（但保留网格线）
    ax.set_xticks(range(num_heads), minor=True)
    ax.set_yticks(range(num_layers), minor=True)

    # 添加colorbar（只显示 [-1.0, 0.0, 1.0] 三个刻度，不显示标签）
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([-1.0, 0.0, 1.0])  # 只显示三个刻度
    cbar.ax.tick_params(labelsize=24, width=2)  # colorbar刻度字体大小和线宽
    # 不设置label，移除colorbar标签

    # 添加网格线（可选）
    ax.set_xticks(np.arange(-0.5, num_heads, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_layers, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if output_path is None:
        output_path = 'head_gu_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 热力图已保存到: {output_path}")

    plt.close()


def plot_histogram(stats: Dict[Tuple[int, int], Dict], output_path: str = None, bin_width: float = 0.1):
    """
    绘制1024个head的g_u平均值分布柱状图

    Args:
        stats: {(layer_idx, head_idx): {'mean': float, ...}}
        output_path: 输出文件路径
        bin_width: 柱状图区间宽度（默认0.1）
    """
    # 提取所有head的g_u平均值
    g_u_means = [stat['mean'] for stat in stats.values()]

    if len(g_u_means) == 0:
        print("⚠️  警告: 没有数据可绘制")
        return

    # 创建区间（bin宽度改为0.05）
    actual_bin_width = 0.05  # 使用0.05作为bin宽度，而不是参数中的bin_width
    bins = np.arange(-1.0, 1.0 + actual_bin_width, actual_bin_width)

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制柱状图
    n, bins, patches = ax.hist(g_u_means, bins=bins, edgecolor='black', linewidth=0.5, alpha=0.7)

    # 设置颜色：负值用蓝色系，正值用绿色系
    for i, (patch, bin_left) in enumerate(zip(patches, bins[:-1])):
        if bin_left < 0:
            # 蓝色系（抑制）
            patch.set_facecolor(plt.cm.Blues(0.3 + 0.7 * (abs(bin_left) / 1.0)))
        else:
            # 绿色系（增强）
            patch.set_facecolor(plt.cm.Greens(0.3 + 0.7 * (bin_left / 1.0)))

    # 设置坐标轴（字体放大2倍：12->24, 14->28）
    ax.set_xlabel('g_u Mean Value', fontsize=24, fontweight='bold')
    ax.set_ylabel('Number of Heads', fontsize=24, fontweight='bold')
    ax.set_title(f'Distribution of g_u Mean Values for 1024 Heads\n(Bin Width: 0.05)',
                 fontsize=28, fontweight='bold', pad=20)

    # 设置x轴范围：只显示有数据的那部分范围
    data_min = np.min(g_u_means)
    data_max = np.max(g_u_means)
    # 扩展一点范围以便更好地显示
    x_range_padding = (data_max - data_min) * 0.05
    ax.set_xlim(data_min - x_range_padding, data_max + x_range_padding)

    # 设置x轴刻度：显示 [-0.3, 0.0, 0.3, 0.6, 0.9]
    # 但只显示在数据范围内的刻度
    desired_ticks = [-0.3, 0.0, 0.3, 0.6, 0.9]
    # 过滤出在数据范围内的刻度
    x_min = data_min - x_range_padding
    x_max = data_max + x_range_padding
    visible_ticks = [tick for tick in desired_ticks if x_min <= tick <= x_max]
    ax.set_xticks(visible_ticks)
    ax.set_xticklabels([f'{tick:.1f}' for tick in visible_ticks], fontsize=24, fontweight='bold')

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # 添加统计信息
    mean_val = np.mean(g_u_means)
    std_val = np.std(g_u_means)
    median_val = np.median(g_u_means)

    stats_text = f'Mean: {mean_val:.4f}, Std: {std_val:.4f}, Median: {median_val:.4f}, Total Heads: {len(g_u_means)}'
    print(f"stats_text: {stats_text}")
    # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=20,
    #         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 添加垂直线标记均值
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
    ax.legend(loc='upper right', fontsize=18)

    # 设置刻度字体大小和加粗（放大2倍：20->24）
    ax.tick_params(axis='x', labelsize=24, width=2, length=6)
    ax.tick_params(axis='y', labelsize=24, width=2, length=6)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if output_path is None:
        output_path = 'head_gu_histogram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 柱状图已保存到: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="分析head级别的g_u统计信息")
    parser.add_argument("--ground-truth-dir", type=str,
                       default="train/coco_train_500_head_ground_truth",
                       help="真值对文件目录")
    parser.add_argument("--num-layers", type=int, default=32,
                       help="模型层数")
    parser.add_argument("--num-heads", type=int, default=32,
                       help="每层的head数")
    parser.add_argument("--bin-width", type=float, default=0.1,
                       help="柱状图区间宽度")

    args = parser.parse_args()

    # 确定输出目录
    ground_truth_dir = Path(args.ground_truth_dir)
    output_dir = os.path.join(ground_truth_dir.parent, "head_gu_statistics")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Head g_u 统计分析")
    print("=" * 80)
    print(f"真值对目录: {ground_truth_dir}")
    print(f"输出目录: {output_dir}")
    print(f"模型配置: {args.num_layers} 层, 每层 {args.num_heads} 个head")
    print("=" * 80)

    # 加载数据
    print("\n[1/3] 加载真值对数据...")
    g_u_by_head = load_head_ground_truth_data(
        str(ground_truth_dir),
        args.num_layers,
        args.num_heads
    )

    if len(g_u_by_head) == 0:
        print("⚠️  错误: 没有找到任何真值对数据！")
        return

    # 计算统计信息
    print("\n[2/3] 计算统计信息...")
    stats = compute_head_statistics(g_u_by_head)

    # 打印简要统计
    all_means = [stat['mean'] for stat in stats.values() if not np.isnan(stat['mean'])]
    print(f"\n统计摘要:")
    print(f"  有数据的head数: {len(stats)}")
    if len(all_means) > 0:
        print(f"  g_u平均值范围: [{np.min(all_means):.4f}, {np.max(all_means):.4f}]")
        print(f"  g_u平均值均值: {np.mean(all_means):.4f}")
        print(f"  g_u平均值标准差: {np.std(all_means):.4f}")
        print(f"  g_u平均值中位数: {np.median(all_means):.4f}")
    else:
        print(f"  ⚠️  警告: 所有head的g_u平均值都是nan！")

    # 生成图表
    print("\n[3/3] 生成可视化图表...")

    # 从ground_truth_dir中提取目录名，用于生成文件名
    ground_truth_dir_name = ground_truth_dir.name  # 获取目录名，例如 "coco_train_500_head_ground_truth"

    # 1. 热力图
    heatmap_path = os.path.join(output_dir, f"{ground_truth_dir_name}_head_gu_heatmap.png")
    print(f"\n生成热力图...")
    plot_heatmap(stats, args.num_layers, args.num_heads, str(heatmap_path))

    # 2. 柱状图
    histogram_path = os.path.join(output_dir, f"{ground_truth_dir_name}_head_gu_histogram.png")
    print(f"\n生成柱状图...")
    plot_histogram(stats, str(histogram_path), args.bin_width)

    # 保存统计信息到JSON文件
    stats_json_path = os.path.join(output_dir, f"{ground_truth_dir_name}_head_gu_statistics.json")
    stats_for_json = {}
    for (layer_idx, head_idx), stat in stats.items():
        key = f"layer_{layer_idx}_head_{head_idx}"
        stats_for_json[key] = {
            'layer': layer_idx,
            'head': head_idx,
            'mean': float(stat['mean']),
            'std': float(stat['std']),
            'min': float(stat['min']),
            'max': float(stat['max']),
            'count': int(stat['count'])
        }

    with open(stats_json_path, 'w', encoding='utf-8') as f:
        json.dump(stats_for_json, f, indent=2, ensure_ascii=False)
    print(f"✓ 统计信息已保存到: {stats_json_path}")

    print("\n" + "=" * 80)
    print("✓ 所有分析完成！")
    print("=" * 80)
    print(f"输出文件:")
    print(f"  1. 热力图: {heatmap_path}")
    print(f"  2. 柱状图: {histogram_path}")
    print(f"  3. 统计信息: {stats_json_path}")


if __name__ == "__main__":
    main()
