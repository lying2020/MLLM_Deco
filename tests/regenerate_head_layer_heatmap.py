#!/usr/bin/env python3
"""
从JSON文件重新生成Head-Layer Heatmap图例

用法:
    python regenerate_head_layer_heatmap.py <json_file> [--output-dir <output_dir>]

功能:
    1. 读取JSON文件（包含head-layer heatmap数据）
    2. 重新生成完全相同的图例（heatmap）
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from datetime import datetime


def load_json_data(json_file):
    """加载JSON数据文件"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def regenerate_heatmap(data, output_dir, output_filename_prefix):
    """
    从数据重新生成head-layer heatmap图例

    Args:
        data: 数据字典
        output_dir: 输出目录
        output_filename_prefix: 输出文件名前缀（不含扩展名）
    """
    # 提取数据
    num_total_layers = data['num_total_layers']
    num_heads = data['num_heads']
    global_min = data['global_min']
    global_max = data['global_max']
    heatmap_matrix = np.array([
        [v if v is not None else np.nan for v in row]
        for row in data['heatmap_matrix']
    ])
    split_line = data.get('split_line', None)

    # 检查是否有有效数据
    valid_data = heatmap_matrix[~np.isnan(heatmap_matrix)]
    if len(valid_data) == 0:
        print("  ✗ 没有有效数据，无法生成heatmap")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建heatmap
    fig, ax = plt.subplots(figsize=(12, 12))

    # 使用pcolormesh绘制heatmap
    heatmap_extended = np.full((num_total_layers + 1, num_heads + 1), np.nan)
    heatmap_extended[:num_total_layers, :num_heads] = heatmap_matrix

    X = np.arange(num_heads + 2)
    Y = np.arange(num_total_layers + 2)
    X_grid, Y_grid = np.meshgrid(X, Y)

    # 使用淡黄到深橙色的colormap
    colors = ['#FFF8DC', '#FFE4B5', '#FFD700', '#FFA500', '#FF8C00', '#FF7F50', '#FF6347', '#FF4500']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('yellow_to_orange', colors, N=n_bins)

    im = ax.pcolormesh(X_grid, Y_grid, heatmap_extended, cmap=cmap,
                       edgecolors='white', linewidths=0.5,
                       vmin=global_min, vmax=global_max,
                       shading='flat')

    # 设置坐标轴
    ax.set_xlabel('Head Index', fontsize=18, fontweight='bold')
    ax.set_ylabel('Layer Index', fontsize=18, fontweight='bold')

    # 标题
    word = data.get('word', 'Unknown')
    group_idx = data.get('group_idx')
    if group_idx is not None:
        title = f'Head-Layer Attention Heatmap - {word}, Group {group_idx+1}'
    else:
        title = f'Head-Layer Attention Heatmap - {word}'
    # ax.set_title(title, fontsize=18, fontweight='bold')

    # 设置x轴刻度（head索引）
    num_ticks_x = 8
    tick_indices_x = np.linspace(0, num_heads - 1, num_ticks_x, dtype=int)
    ax.set_xticks(tick_indices_x + 0.5)
    ax.set_xticklabels([f'H{i}' for i in tick_indices_x], fontsize=15, fontweight='bold')

    # 设置y轴刻度（层索引）
    num_ticks_y = 8
    tick_indices_y = np.linspace(0, num_total_layers - 1, num_ticks_y, dtype=int)
    ax.set_yticks(tick_indices_y + 0.5)
    ax.set_yticklabels([f'L{i}' for i in tick_indices_y], fontsize=15, fontweight='bold')

    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Concentration Index', fontsize=18, fontweight='bold')

    # 设置坐标轴范围
    ax.set_xlim(0, num_heads)
    ax.set_ylim(0, num_total_layers)

    # 绘制权重均分线（如果存在）
    if split_line and split_line.get('y_position') is not None:
        y_position = split_line['y_position']
        # ax.axhline(y=y_position, xmin=0, xmax=num_heads,
        #           color='green', linewidth=2.0, alpha=0.3, linestyle='-')
        split_layer = split_line.get('split_layer')
        split_position = split_line.get('split_position')
        if split_layer is not None and split_position is not None:
            print(f"    权重均分线位置: Layer {split_layer}, 层内位置 {split_position:.4f}, Y坐标 {y_position:.4f}")

    # 保存heatmap
    heatmap_file = os.path.join(output_dir, f"{output_filename_prefix}_heatmap.png")
    plt.savefig(heatmap_file, dpi=200, bbox_inches='tight')
    plt.close()

    # 打印统计信息
    valid_count = np.sum(~np.isnan(heatmap_matrix))
    total_count = num_total_layers * num_heads
    print(f"  ✓ Head-Layer Heatmap已重新生成: {os.path.basename(heatmap_file)}")
    print(f"    有效值数量: {valid_count}/{total_count}")
    print(f"    集中度范围: [{valid_data.min():.6f}, {valid_data.max():.6f}] (全局范围: [{global_min:.6f}, {global_max:.6f}])")

    # 打印 layer 21 的 32 个 head 的集中度值
    layer_21_idx = 21
    if layer_21_idx < num_total_layers:
        layer_21_concentrations = heatmap_matrix[layer_21_idx, :]
        if not np.all(np.isnan(layer_21_concentrations)):
            concentrations_str = ', '.join([f'{val:.6f}' if not np.isnan(val) else 'NaN' for val in layer_21_concentrations])
            print(f"    Layer 21 集中度值 (32个head): {concentrations_str}")
        else:
            print(f"    Layer 21: 无有效数据")


def generate_difference_heatmap(data1, data2, output_dir, output_filename_prefix):
    """
    计算两个 heatmap 的差值并生成新的 heatmap

    Args:
        data1: 第一个 JSON 数据字典
        data2: 第二个 JSON 数据字典
        output_dir: 输出目录
        output_filename_prefix: 输出文件名前缀（不含扩展名）
    """
    # 提取数据
    num_total_layers = data1['num_total_layers']
    num_heads = data1['num_heads']

    # 验证两个数据的维度是否一致
    if data1['num_total_layers'] != data2['num_total_layers'] or data1['num_heads'] != data2['num_heads']:
        print("  ✗ 错误: 两个 JSON 文件的维度不一致")
        print(f"    文件1: {data1['num_total_layers']} 层 × {data1['num_heads']} heads")
        print(f"    文件2: {data2['num_total_layers']} 层 × {data2['num_heads']} heads")
        return

    # 转换为 numpy 数组
    heatmap_matrix1 = np.array([
        [v if v is not None else np.nan for v in row]
        for row in data1['heatmap_matrix']
    ])
    heatmap_matrix2 = np.array([
        [v if v is not None else np.nan for v in row]
        for row in data2['heatmap_matrix']
    ])

    # 计算差值（data1 - data2）
    diff_matrix = heatmap_matrix1 - heatmap_matrix2

    # 检查是否有有效数据
    valid_data = diff_matrix[~np.isnan(diff_matrix)]
    if len(valid_data) == 0:
        print("  ✗ 没有有效数据，无法生成差值 heatmap")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建 heatmap
    fig, ax = plt.subplots(figsize=(12, 12))

    # 使用 pcolormesh 绘制 heatmap
    diff_extended = np.full((num_total_layers + 1, num_heads + 1), np.nan)
    diff_extended[:num_total_layers, :num_heads] = diff_matrix

    X = np.arange(num_heads + 2)
    Y = np.arange(num_total_layers + 2)
    X_grid, Y_grid = np.meshgrid(X, Y)

    # 创建分段的 colormap：负数（红色），0（白色），正数（绿色）
    # 从深红色 -> 浅红色 -> 白色 -> 浅绿色 -> 深绿色
    n_bins = 256
    # 创建完整的颜色列表：负数部分（深红到白）+ 正数部分（白到深绿）
    # 负数部分：深红色到白色（128个bins，索引0-127）
    colors_negative = ['#8B0000', '#DC143C', '#FF6347', '#FFB6C1', '#FFFFFF']  # 深红 -> 浅红 -> 白
    # 正数部分：白色到深绿色（128个bins，索引128-255）
    colors_positive = ['#FFFFFF', '#90EE90', '#32CD32', '#228B22', '#006400']  # 白 -> 浅绿 -> 深绿

    # 创建两个独立的 colormap
    cmap_negative = LinearSegmentedColormap.from_list('red_to_white', colors_negative, N=128)
    cmap_positive = LinearSegmentedColormap.from_list('white_to_green', colors_positive, N=128)

    # 合并两个 colormap：负数部分（0-127）+ 正数部分（128-255）
    colors_combined = []
    # 负数部分（从深红到白，索引0-127）
    for i in range(128):
        colors_combined.append(cmap_negative(i / 127.0))
    # 正数部分（从白到深绿，索引128-255）
    for i in range(128):
        colors_combined.append(cmap_positive(i / 127.0))

    cmap_diff = LinearSegmentedColormap.from_list('red_white_green', colors_combined, N=n_bins)

    # 计算 vmin 和 vmax（对称的，以 0 为中心）
    abs_max = max(abs(valid_data.min()), abs(valid_data.max()))
    vmin = -abs_max
    vmax = abs_max

    # 使用 TwoSlopeNorm 来确保 0 值对应白色
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    im = ax.pcolormesh(X_grid, Y_grid, diff_extended, cmap=cmap_diff,
                       norm=norm,
                       edgecolors='white', linewidths=0.5,
                       shading='flat')

    # 设置坐标轴
    ax.set_xlabel('Head Index', fontsize=18, fontweight='bold')
    ax.set_ylabel('Layer Index', fontsize=18, fontweight='bold')

    # 标题
    word1 = data1.get('word', 'Unknown')
    word2 = data2.get('word', 'Unknown')
    title = f'Head-Layer Attention Difference Heatmap - {word1} - {word2}'
    # ax.set_title(title, fontsize=18, fontweight='bold')

    # 设置 x 轴刻度（head 索引）
    num_ticks_x = 8
    tick_indices_x = np.linspace(0, num_heads - 1, num_ticks_x, dtype=int)
    ax.set_xticks(tick_indices_x + 0.5)
    ax.set_xticklabels([f'H{i}' for i in tick_indices_x], fontsize=15, fontweight='bold')

    # 设置 y 轴刻度（层索引）
    num_ticks_y = 8
    tick_indices_y = np.linspace(0, num_total_layers - 1, num_ticks_y, dtype=int)
    ax.set_yticks(tick_indices_y + 0.5)
    ax.set_yticklabels([f'L{i}' for i in tick_indices_y], fontsize=15, fontweight='bold')

    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Difference (Concentration Index)', fontsize=18, fontweight='bold')

    # 设置坐标轴范围
    ax.set_xlim(0, num_heads)
    ax.set_ylim(0, num_total_layers)

    # 保存 heatmap
    heatmap_file = os.path.join(output_dir, f"{output_filename_prefix}_difference_heatmap.png")
    plt.savefig(heatmap_file, dpi=200, bbox_inches='tight')
    plt.close()

    # 打印统计信息
    valid_count = np.sum(~np.isnan(diff_matrix))
    total_count = num_total_layers * num_heads
    positive_count = np.sum((diff_matrix > 0) & (~np.isnan(diff_matrix)))
    negative_count = np.sum((diff_matrix < 0) & (~np.isnan(diff_matrix)))
    zero_count = np.sum((diff_matrix == 0) & (~np.isnan(diff_matrix)))

    print(f"  ✓ 差值 Head-Layer Heatmap已生成: {os.path.basename(heatmap_file)}")
    print(f"    有效值数量: {valid_count}/{total_count}")
    print(f"    差值范围: [{valid_data.min():.6f}, {valid_data.max():.6f}]")
    print(f"    正值数量: {positive_count}, 负值数量: {negative_count}, 零值数量: {zero_count}")
    print(f"    颜色映射: 负数(红色) ← 0(白色) → 正数(绿色)")

    # 生成带网格线的原始 heatmap（基于 data1）
    print(f"\n正在生成带网格线的原始 heatmap...")

    # 找出正差值的前50%和负差值绝对值的前50%
    positive_diffs = diff_matrix[diff_matrix > 0]
    negative_diffs = diff_matrix[diff_matrix < 0]

    # 计算阈值
    positive_threshold = 0.0
    negative_threshold = 0.0

    if len(positive_diffs) > 0:
        # 正差值排序，取前50%的阈值
        positive_sorted = np.sort(positive_diffs)[::-1]  # 从大到小
        top_50_percent_idx = max(1, int(len(positive_sorted) * 0.5))
        positive_threshold = positive_sorted[top_50_percent_idx - 1]

    if len(negative_diffs) > 0:
        # 负差值取绝对值后排序，取前50%的阈值
        negative_abs_sorted = np.sort(np.abs(negative_diffs))[::-1]  # 从大到小
        top_50_percent_idx = max(1, int(len(negative_abs_sorted) * 0.5))
        negative_threshold = negative_abs_sorted[top_50_percent_idx - 1]

    # 创建原始 heatmap（使用 data1）
    fig2, ax2 = plt.subplots(figsize=(12, 12))

    # 使用 data1 的原始数据
    global_min1 = data1['global_min']
    global_max1 = data1['global_max']

    heatmap_extended1 = np.full((num_total_layers + 1, num_heads + 1), np.nan)
    heatmap_extended1[:num_total_layers, :num_heads] = heatmap_matrix1

    # 使用淡黄到深橙色的colormap（原始heatmap的colormap）
    colors = ['#FFF8DC', '#FFE4B5', '#FFD700', '#FFA500', '#FF8C00', '#FF7F50', '#FF6347', '#FF4500']
    cmap_original = LinearSegmentedColormap.from_list('yellow_to_orange', colors, N=256)

    im2 = ax2.pcolormesh(X_grid, Y_grid, heatmap_extended1, cmap=cmap_original,
                        edgecolors='white', linewidths=0.5,
                        vmin=global_min1, vmax=global_max1,
                        shading='flat')

    # 在正差值前50%的位置画绿色网格线
    if positive_threshold > 0:
        positive_grid_positions = []
        for layer_idx in range(num_total_layers):
            for head_idx in range(num_heads):
                if not np.isnan(diff_matrix[layer_idx, head_idx]) and diff_matrix[layer_idx, head_idx] >= positive_threshold:
                    positive_grid_positions.append((layer_idx, head_idx))

        # 批量绘制绿色网格线
        for layer_idx, head_idx in positive_grid_positions:
            # 画网格线：在单元格的边界上画线
            # 上边界
            ax2.plot([head_idx, head_idx + 1], [layer_idx + 1, layer_idx + 1],
                    color='green', linewidth=2.0, alpha=0.8)
            # 下边界
            ax2.plot([head_idx, head_idx + 1], [layer_idx, layer_idx],
                    color='green', linewidth=2.0, alpha=0.8)
            # 左边界
            ax2.plot([head_idx, head_idx], [layer_idx, layer_idx + 1],
                    color='green', linewidth=2.0, alpha=0.8)
            # 右边界
            ax2.plot([head_idx + 1, head_idx + 1], [layer_idx, layer_idx + 1],
                    color='green', linewidth=2.0, alpha=0.8)

    # 在负差值绝对值前50%的位置画红色网格线
    if negative_threshold > 0:
        negative_grid_positions = []
        for layer_idx in range(num_total_layers):
            for head_idx in range(num_heads):
                if not np.isnan(diff_matrix[layer_idx, head_idx]) and diff_matrix[layer_idx, head_idx] < 0:
                    abs_diff = abs(diff_matrix[layer_idx, head_idx])
                    if abs_diff >= negative_threshold:
                        negative_grid_positions.append((layer_idx, head_idx))

        # 批量绘制红色网格线
        for layer_idx, head_idx in negative_grid_positions:
            # 画网格线：在单元格的边界上画线
            # 上边界
            ax2.plot([head_idx, head_idx + 1], [layer_idx + 1, layer_idx + 1],
                    color='red', linewidth=2.0, alpha=0.8)
            # 下边界
            ax2.plot([head_idx, head_idx + 1], [layer_idx, layer_idx],
                    color='red', linewidth=2.0, alpha=0.8)
            # 左边界
            ax2.plot([head_idx, head_idx], [layer_idx, layer_idx + 1],
                    color='red', linewidth=2.0, alpha=0.8)
            # 右边界
            ax2.plot([head_idx + 1, head_idx + 1], [layer_idx, layer_idx + 1],
                    color='red', linewidth=2.0, alpha=0.8)

    # 设置坐标轴
    ax2.set_xlabel('Head Index', fontsize=18, fontweight='bold')
    ax2.set_ylabel('Layer Index', fontsize=18, fontweight='bold')

    # 标题
    word1 = data1.get('word', 'Unknown')
    word2 = data2.get('word', 'Unknown')
    title2 = f'Head-Layer Attention Heatmap with Difference Grid - {word1} vs {word2}'
    # ax2.set_title(title2, fontsize=18, fontweight='bold')

    # 设置 x 轴刻度（head 索引）
    ax2.set_xticks(tick_indices_x + 0.5)
    ax2.set_xticklabels([f'H{i}' for i in tick_indices_x], fontsize=15, fontweight='bold')

    # 设置 y 轴刻度（层索引）
    ax2.set_yticks(tick_indices_y + 0.5)
    ax2.set_yticklabels([f'L{i}' for i in tick_indices_y], fontsize=15, fontweight='bold')

    # 添加 colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Concentration Index', fontsize=18, fontweight='bold')

    # 设置坐标轴范围
    ax2.set_xlim(0, num_heads)
    ax2.set_ylim(0, num_total_layers)

    # 保存带网格线的 heatmap
    heatmap_with_grid_file = os.path.join(output_dir, f"{output_filename_prefix}_with_grid.png")
    plt.savefig(heatmap_with_grid_file, dpi=200, bbox_inches='tight')
    plt.close()

    # 打印网格线统计信息
    positive_grid_count = np.sum((diff_matrix >= positive_threshold) & (~np.isnan(diff_matrix))) if positive_threshold > 0 else 0
    negative_grid_count = np.sum((diff_matrix < 0) & (np.abs(diff_matrix) >= negative_threshold) & (~np.isnan(diff_matrix))) if negative_threshold > 0 else 0

    print(f"  ✓ 带网格线的原始 Heatmap已生成: {os.path.basename(heatmap_with_grid_file)}")
    print(f"    正差值阈值（前50%）: {positive_threshold:.6f}, 标记数量: {positive_grid_count} (绿色网格线)")
    print(f"    负差值阈值（前50%）: {negative_threshold:.6f}, 标记数量: {negative_grid_count} (红色网格线)")


def main():
    parser = argparse.ArgumentParser(
        description='从JSON文件重新生成Head-Layer Heatmap图例，或计算两个heatmap的差值',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 读取JSON并重新生成单个heatmap
  python regenerate_head_layer_heatmap.py data.json --output-dir output/

  # 使用默认输出目录（JSON文件所在目录）
  python regenerate_head_layer_heatmap.py data.json

  # 计算两个heatmap的差值
  python regenerate_head_layer_heatmap.py data1.json --diff data2.json --output-dir output/
        """
    )

    parser.add_argument('json_file', type=str, nargs='?', help='输入的JSON文件路径')
    parser.add_argument('--diff', type=str, default=None,
                       help='第二个JSON文件路径（如果提供，将计算两个heatmap的差值）')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录（默认：JSON文件所在目录）')

    args = parser.parse_args()

    # 如果提供了 --diff，计算差值
    if args.diff is not None:
        if args.json_file is None:
            print("错误: 使用 --diff 时必须提供第一个 JSON 文件")
            return

        # 检查两个 JSON 文件是否存在
        if not os.path.exists(args.json_file):
            print(f"错误: JSON文件不存在: {args.json_file}")
            return
        if not os.path.exists(args.diff):
            print(f"错误: JSON文件不存在: {args.diff}")
            return

        # 确定输出目录
        if args.output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(args.json_file))
        else:
            output_dir = args.output_dir

        # 生成输出文件名前缀
        json1_basename = os.path.splitext(os.path.basename(args.json_file))[0]
        json2_basename = os.path.splitext(os.path.basename(args.diff))[0]
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename_prefix = f"{json1_basename}_vs_{json2_basename}_new"

        # 加载两个数据文件
        print(f"正在加载第一个JSON文件: {args.json_file}")
        data1 = load_json_data(args.json_file)
        print(f"  词汇: {data1.get('word', 'Unknown')}")
        if data1.get('group_idx') is not None:
            print(f"  Group: {data1['group_idx'] + 1}")
        print(f"  数据维度: {data1['num_total_layers']} 层 × {data1['num_heads']} heads")
        print(f"  全局范围: [{data1['global_min']:.6f}, {data1['global_max']:.6f}]")

        print(f"\n正在加载第二个JSON文件: {args.diff}")
        data2 = load_json_data(args.diff)
        print(f"  词汇: {data2.get('word', 'Unknown')}")
        if data2.get('group_idx') is not None:
            print(f"  Group: {data2['group_idx'] + 1}")
        print(f"  数据维度: {data2['num_total_layers']} 层 × {data2['num_heads']} heads")
        print(f"  全局范围: [{data2['global_min']:.6f}, {data2['global_max']:.6f}]")

        # 生成差值 heatmap
        print(f"\n正在计算差值并生成heatmap...")
        generate_difference_heatmap(data1, data2, output_dir, output_filename_prefix)

        print(f"\n完成！输出目录: {output_dir}")
        print(f"输出文件前缀: {output_filename_prefix}")

    else:
        # 单个文件模式
        if args.json_file is None:
            parser.print_help()
            return

        # 检查JSON文件是否存在
        if not os.path.exists(args.json_file):
            print(f"错误: JSON文件不存在: {args.json_file}")
            return

        # 确定输出目录（如果没有指定，使用JSON文件所在目录）
        if args.output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(args.json_file))
        else:
            output_dir = args.output_dir

        # 生成输出文件名前缀（输入文件名+时间戳）
        json_basename = os.path.splitext(os.path.basename(args.json_file))[0]
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename_prefix = f"{json_basename}_new"

        # 加载数据
        print(f"正在加载JSON文件: {args.json_file}")
        data = load_json_data(args.json_file)
        print(f"  词汇: {data.get('word', 'Unknown')}")
        if data.get('group_idx') is not None:
            print(f"  Group: {data['group_idx'] + 1}")
        print(f"  数据维度: {data['num_total_layers']} 层 × {data['num_heads']} heads")
        print(f"  全局范围: [{data['global_min']:.6f}, {data['global_max']:.6f}]")

        # 重新生成heatmap
        print(f"\n正在重新生成heatmap...")
        regenerate_heatmap(data, output_dir, output_filename_prefix)

        print(f"\n完成！输出目录: {output_dir}")
        print(f"输出文件前缀: {output_filename_prefix}")


if __name__ == '__main__':
    main()
