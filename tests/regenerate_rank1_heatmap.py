#!/usr/bin/env python3
"""
从JSON文件重新生成Rank1 Probability Heatmap图例

用法:
    python regenerate_rank1_heatmap.py <json_file> [--output-dir <output_dir>] [--filter-words <word1,word2,...>] [--filter-steps <step1,step2,...>]

功能:
    1. 读取JSON文件（包含所有rank1概率数据）
    2. 可选：根据词汇或步骤筛选数据
    3. 重新生成完全相同的图例（heatmap和colorbar）
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import string
from datetime import datetime


def load_json_data(json_file):
    """加载JSON数据文件"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def build_matrices_from_rank1_data(all_rank1_data, num_total_layers):
    """
    从 all_rank1_data 构建所需的矩阵

    Args:
        all_rank1_data: rank1 数据列表
        num_total_layers: 总层数

    Returns:
        tuple: (all_probability_matrix, all_token_texts_matrix, all_token_labels, vmin, vmax)
    """
    num_all_tokens = len(all_rank1_data)

    all_probability_matrix = np.full((num_all_tokens, num_total_layers), np.nan)
    all_token_texts_matrix = [[''] * num_total_layers for _ in range(num_all_tokens)]
    all_token_labels = []
    all_valid_probs = []

    for row_idx, item in enumerate(all_rank1_data):
        probs = item['probs']
        texts = item['texts']
        step_idx = item['step_idx']

        for layer_idx in range(num_total_layers):
            if layer_idx < len(probs) and probs[layer_idx] is not None:
                prob_value = probs[layer_idx]
                all_probability_matrix[row_idx, layer_idx] = prob_value
                all_valid_probs.append(prob_value)
            if layer_idx < len(texts):
                all_token_texts_matrix[row_idx][layer_idx] = texts[layer_idx]

        all_token_labels.append(f"{step_idx}")

    # 计算 vmin 和 vmax
    if all_valid_probs:
        vmin = float(min(all_valid_probs))
        vmax = float(max(all_valid_probs))
    else:
        vmin = 0.0
        vmax = 1.0

    return all_probability_matrix, all_token_texts_matrix, all_token_labels, vmin, vmax


def filter_data(data, filter_words=None, filter_steps=None):
    """
    筛选数据

    Args:
        data: JSON数据字典
        filter_words: 要保留的词汇列表（None表示不过滤）
        filter_steps: 要保留的步骤索引列表（None表示不过滤）

    Returns:
        筛选后的数据字典
    """
    if filter_words is None and filter_steps is None:
        return data

    # 创建筛选后的数据
    filtered_rank1_data = []
    filter_words_set = set(filter_words) if filter_words else None
    filter_steps_set = set(filter_steps) if filter_steps else None

    for item in data['all_rank1_data']:
        word = item['word']
        step_idx = item['step_idx']

        # 检查是否应该保留
        if filter_words_set is not None and word not in filter_words_set:
            continue
        if filter_steps_set is not None and step_idx not in filter_steps_set:
            continue

        filtered_rank1_data.append(item)

    # 更新数据
    filtered_data = data.copy()
    filtered_data['all_rank1_data'] = filtered_rank1_data
    filtered_data['num_all_tokens'] = len(filtered_rank1_data)

    # 从 all_rank1_data 重新构建矩阵并计算 vmin/vmax
    _, _, _, vmin, vmax = build_matrices_from_rank1_data(
        filtered_rank1_data, data['num_total_layers']
    )
    filtered_data['vmin'] = vmin
    filtered_data['vmax'] = vmax

    return filtered_data


def regenerate_heatmap(data, output_dir, output_filename_prefix):
    """
    从数据重新生成heatmap图例

    Args:
        data: 数据字典（包含 all_rank1_data）
        output_dir: 输出目录
        output_filename_prefix: 输出文件名前缀（不含扩展名）
    """
    # 从 all_rank1_data 构建所需的矩阵
    num_total_layers = data['num_total_layers']
    all_rank1_data = data['all_rank1_data']

    all_probability_matrix, all_token_texts_matrix, all_token_labels, vmin, vmax = \
        build_matrices_from_rank1_data(all_rank1_data, num_total_layers)

    num_all_tokens = len(all_rank1_data)

    # 检查是否有有效数据
    all_valid_probabilities = all_probability_matrix[~np.isnan(all_probability_matrix)]
    if len(all_valid_probabilities) == 0:
        print("  ✗ 没有有效数据，无法生成heatmap")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 计算figsize（与原代码保持一致）
    base_width = 20 * 1.2
    base_height = base_width * (num_all_tokens / num_total_layers)
    fig, ax = plt.subplots(figsize=(base_width, base_height))

    # 设置aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # 使用pcolormesh
    probability_extended = np.full((num_all_tokens + 1, num_total_layers + 1), np.nan)
    probability_extended[:num_all_tokens, :num_total_layers] = all_probability_matrix

    X = np.arange(num_total_layers + 2)
    Y = np.arange(num_all_tokens + 2)
    X_grid, Y_grid = np.meshgrid(X, Y)

    # 原来的绿色colormap代码（已注释）：
    # # 创建自定义绿色colormap：类似原来的Greens，但最深的墨绿色改为中等绿色
    # # 原来的Greens colormap（最深墨绿色太深）已注释：
    # # cmap = 'Greens'
    # # 使用类似Greens的颜色序列，但最深色从墨绿色改为中等绿色
    # colors_green = ['#F7FCF5', '#E5F5E0', '#C7E9C0', '#A1D99B', '#74C476', '#41AB5D', '#238B45', '#228B22']
    # n_bins_green = 256
    # cmap = LinearSegmentedColormap.from_list('greens_light_to_medium', colors_green, N=n_bins_green)

    # 原来的白色到蓝色渐变（已注释，改为更优雅的蓝紫色渐变）：
    # colors_blue = ['#FFFFFF', '#E6F2FF', '#CCE5FF', '#99CCFF', '#66B2FF', '#3399FF', '#0080FF', '#0066CC']
    # n_bins_blue = 256
    # cmap = LinearSegmentedColormap.from_list('white_to_blue', colors_blue, N=n_bins_blue)

    # 创建从白色到中等深蓝色的渐变色colormap（基于用户提供的colorbar截图）
    # 从纯白色平滑过渡到中等深度、稍微柔和的蓝色，既专业又美观
    colors_white_to_blue = [
        '#FFFFFF',  # 纯白色
        '#F0F5FF',  # 极淡蓝色
        '#D6E5FF',  # 淡蓝色
        '#B8D4FF',  # 浅蓝色
        '#96C0FF',  # 中浅蓝色
        '#6FA8FF',  # 中蓝色
        '#4A8CFF',  # 中深蓝色
        '#2E6FCC'   # 中等深蓝色（稍微柔和，不刺眼）
    ]
    n_bins_white_to_blue = 256
    cmap = LinearSegmentedColormap.from_list('white_to_medium_blue', colors_white_to_blue, N=n_bins_white_to_blue)

    # 绘制heatmap
    im = ax.pcolormesh(X_grid, Y_grid, probability_extended, cmap=cmap,
                       edgecolors='white', linewidths=2.0,
                       vmin=vmin, vmax=vmax,
                       shading='flat')

    # 设置坐标轴
    ax.set_xlabel('Transformer Layers', fontsize=18, fontweight='bold')
    ax.set_ylabel('Token Index', fontsize=18, fontweight='bold')
    ax.set_title(f'Rank1 Probability Heatmap - All Instances ({num_all_tokens} tokens)',
                 fontsize=18, fontweight='bold', pad=10)

    # 设置x轴刻度（层索引）- 只显示6个刻度，索引从1开始（1到32）
    num_ticks = 6
    tick_indices = np.array([0, 7, 15, 23, 31])
    ax.set_xticks(tick_indices + 0.5)
    ax.set_xticklabels([f'L{i+1}' for i in tick_indices], fontsize=18, fontweight='bold')

    # 设置y轴刻度（token）
    ax.set_yticks(np.arange(num_all_tokens) + 0.5)
    ax.set_yticklabels(all_token_labels, fontsize=18, fontweight='bold')

    # 在每个单元格中心标注token文本（词汇）
    for row_idx in range(num_all_tokens):
        for layer_idx in range(num_total_layers):
            token_text = all_token_texts_matrix[row_idx][layer_idx]
            prob_value = all_probability_matrix[row_idx, layer_idx]

            # 只标注有效的token
            if token_text and not np.isnan(prob_value):
                # 清理token文本
                clean_text = token_text.replace('\n', ' ').replace('\r', ' ').strip()

                # 检查是否可以正常显示
                printable_chars = set(string.printable)
                if not all(c in printable_chars for c in clean_text) or not clean_text:
                    clean_text = " * "
                else:
                    # 限制长度
                    if len(clean_text) > 15:
                        clean_text = clean_text[:15] + '...'

                # 根据概率值选择文本颜色
                if vmax > vmin:
                    normalized = (prob_value - vmin) / (vmax - vmin)
                    text_color = 'white' if normalized > 0.5 else 'black'
                else:
                    text_color = 'black'

                # 在单元格中心标注文本
                ax.text(layer_idx + 0.5, row_idx + 0.5, clean_text,
                       ha='center', va='center',
                       fontsize=10, color=text_color, fontweight='bold',
                       rotation=45,
                       bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='white' if text_color == 'black' else 'black',
                                alpha=0.6, edgecolor='none'))

    # 设置坐标轴范围
    ax.set_xlim(0, num_total_layers)
    ax.set_ylim(0, num_all_tokens)

    # 保存heatmap
    combined_heatmap_file = os.path.join(output_dir, f"{output_filename_prefix}.png")
    plt.savefig(combined_heatmap_file, dpi=200, bbox_inches='tight')
    plt.close()

    # 单独保存colorbar和标签
    fig_cbar = plt.figure(figsize=(1.5, 8))
    ax_cbar = plt.subplot(1, 1, 1)
    ax_cbar.axis('off')
    # 创建colorbar，使用相同的cmap和norm确保颜色一致
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_cbar, orientation='vertical', fraction=1.0)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    cbar.set_label('Probability', fontsize=18, fontweight='bold')
    # 设置colorbar刻度标签字体大小
    cbar.ax.tick_params(labelsize=18)
    # 设置colorbar刻度标签加粗
    for label in cbar.ax.yaxis.get_majorticklabels():
        label.set_fontweight('bold')
    # 保存colorbar
    combined_colorbar_file = os.path.join(output_dir, f"{output_filename_prefix}_colorbar.png")
    plt.savefig(combined_colorbar_file, dpi=200, bbox_inches='tight')
    plt.close()

    # 打印统计信息
    valid_count = np.sum(~np.isnan(all_probability_matrix))
    total_count = num_all_tokens * num_total_layers
    print(f"  ✓ Rank1 Probability Heatmap已重新生成: {os.path.basename(combined_heatmap_file)}")
    print(f"  ✓ Colorbar和标签已重新生成: {os.path.basename(combined_colorbar_file)}")
    print(f"    有效值数量: {valid_count}/{total_count}")
    print(f"    概率值范围: [{all_valid_probabilities.min():.4f}, {all_valid_probabilities.max():.4f}]")


def main():
    parser = argparse.ArgumentParser(
        description='从JSON文件重新生成Rank1 Probability Heatmap图例',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 读取JSON并重新生成（不过滤）
  python regenerate_rank1_heatmap.py data.json --output-dir output/

  # 只保留特定词汇
  python regenerate_rank1_heatmap.py data.json --output-dir output/ --filter-words "cat,dog,bird"

  # 只保留特定步骤
  python regenerate_rank1_heatmap.py data.json --output-dir output/ --filter-steps "0,1,2,3"

  # 同时筛选词汇和步骤
  python regenerate_rank1_heatmap.py data.json --output-dir output/ --filter-words "cat,dog" --filter-steps "0,1,2"
        """
    )

    parser.add_argument('json_file', type=str, help='输入的JSON文件路径')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录（默认：JSON文件所在目录）')
    parser.add_argument('--filter-words', type=str, default=None,
                       help='要保留的词汇列表（逗号分隔），例如: "cat,dog,bird"')
    parser.add_argument('--filter-steps', type=str, default=None,
                       help='要保留的步骤索引列表（逗号分隔），例如: "0,1,2,3"')

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
    output_filename_prefix = f"{json_basename}_new"

    # 解析筛选条件
    filter_words = None
    if args.filter_words:
        filter_words = [w.strip() for w in args.filter_words.split(',')]

    filter_steps = None
    if args.filter_steps:
        filter_steps = [int(s.strip()) for s in args.filter_steps.split(',')]

    # 加载数据
    print(f"正在加载JSON文件: {args.json_file}")
    data = load_json_data(args.json_file)
    # 如果 JSON 中没有 num_all_tokens，从 all_rank1_data 计算
    if 'num_all_tokens' not in data:
        data['num_all_tokens'] = len(data.get('all_rank1_data', []))
    print(f"  原始数据: {data['num_all_tokens']} 个tokens, {data['num_total_layers']} 层")

    # 筛选数据（如果需要）
    if filter_words or filter_steps:
        print(f"正在筛选数据...")
        if filter_words:
            print(f"  保留词汇: {filter_words}")
        if filter_steps:
            print(f"  保留步骤: {filter_steps}")
        data = filter_data(data, filter_words=filter_words, filter_steps=filter_steps)
        print(f"  筛选后数据: {data['num_all_tokens']} 个tokens")

    # 重新生成heatmap
    print(f"\n正在重新生成heatmap...")
    regenerate_heatmap(data, output_dir, output_filename_prefix)

    print(f"\n完成！输出目录: {output_dir}")
    print(f"输出文件前缀: {output_filename_prefix}")


if __name__ == '__main__':
    main()
