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
from matplotlib.colors import LinearSegmentedColormap
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
    ax.set_title(title, fontsize=18, fontweight='bold')

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


def main():
    parser = argparse.ArgumentParser(
        description='从JSON文件重新生成Head-Layer Heatmap图例',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 读取JSON并重新生成（不过滤）
  python regenerate_head_layer_heatmap.py data.json --output-dir output/

  # 使用默认输出目录（JSON文件所在目录）
  python regenerate_head_layer_heatmap.py data.json
        """
    )

    parser.add_argument('json_file', type=str, help='输入的JSON文件路径')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录（默认：JSON文件所在目录）')

    args = parser.parse_args()

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename_prefix = f"{json_basename}_{timestamp}"

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
