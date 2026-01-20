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
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
import argparse


class CompressedCenterNorm(Normalize):
    """
    自定义归一化类，压缩中心区域 [-0.5, 0.5] 的色域，扩展两端区域的色域

    映射规则：
    - [-1.0, -0.5] -> [0.0, 0.4] (占用40%的色域)
    - [-0.5, 0.5] -> [0.4, 0.6] (占用20%的色域，压缩)
    - [0.5, 1.0] -> [0.6, 1.0] (占用40%的色域)

    这样使得 [-0.5, 0.5] 范围内的颜色变化更快，而两端的颜色变化更慢，
    从而在二值化热力图中，> 0.5 或 < -0.5 的像素点颜色差异更明显。
    """
    def __init__(self, vmin=-1.0, vmax=1.0, clip=False):
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        # 定义分段点
        self.threshold_low = -0.5
        self.threshold_high = 0.5

        # 定义颜色索引映射范围
        # [-1.0, -0.5] -> [0.0, 0.4]
        # [-0.5, 0.5] -> [0.4, 0.6]
        # [0.5, 1.0] -> [0.6, 1.0]
        self.idx_low_end = 0.4
        self.idx_high_start = 0.6

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        # 将输入值转换为numpy数组
        is_scalar = not isinstance(value, np.ndarray)
        if is_scalar:
            value = np.array([value])

        # 处理NaN值
        mask_valid = ~np.isnan(value)
        if not mask_valid.any():
            if is_scalar:
                return np.nan
            return np.full_like(value, np.nan, dtype=float)

        # 创建结果数组
        result = np.zeros_like(value, dtype=float)

        # 裁剪到有效范围
        if clip:
            value = np.clip(value, self.vmin, self.vmax)

        # 分段映射
        # 第一段: [-1.0, -0.5] -> [0.0, 0.4]
        mask1 = (value <= self.threshold_low) & mask_valid
        if mask1.any():
            # 线性映射: value从[-1.0, -0.5]映射到[0.0, 0.4]
            result[mask1] = 0.0 + (value[mask1] - self.vmin) / (self.threshold_low - self.vmin) * self.idx_low_end

        # 第二段: [-0.5, 0.5] -> [0.4, 0.6]
        mask2 = (value > self.threshold_low) & (value <= self.threshold_high) & mask_valid
        if mask2.any():
            # 线性映射: value从[-0.5, 0.5]映射到[0.4, 0.6]
            result[mask2] = self.idx_low_end + (value[mask2] - self.threshold_low) / (self.threshold_high - self.threshold_low) * (self.idx_high_start - self.idx_low_end)

        # 第三段: [0.5, 1.0] -> [0.6, 1.0]
        mask3 = (value > self.threshold_high) & mask_valid
        if mask3.any():
            # 线性映射: value从[0.5, 1.0]映射到[0.6, 1.0]
            result[mask3] = self.idx_high_start + (value[mask3] - self.threshold_high) / (self.vmax - self.threshold_high) * (1.0 - self.idx_high_start)

        # 处理NaN值
        result[~mask_valid] = np.nan

        # 确保结果在 [0, 1] 范围内
        result = np.clip(result, 0.0, 1.0)

        # 如果是标量输入，返回标量
        if is_scalar:
            return result[0]
        return result

    def inverse(self, value):
        """
        反向映射：从颜色索引 [0, 1] 映射回原始值 [-1.0, 1.0]
        """
        value = np.asarray(value)
        result = np.zeros_like(value, dtype=float)

        # 第一段: [0.0, 0.4] -> [-1.0, -0.5]
        mask1 = value <= self.idx_low_end
        if mask1.any():
            result[mask1] = self.vmin + (value[mask1] - 0.0) / self.idx_low_end * (self.threshold_low - self.vmin)

        # 第二段: [0.4, 0.6] -> [-0.5, 0.5]
        mask2 = (value > self.idx_low_end) & (value <= self.idx_high_start)
        if mask2.any():
            result[mask2] = self.threshold_low + (value[mask2] - self.idx_low_end) / (self.idx_high_start - self.idx_low_end) * (self.threshold_high - self.threshold_low)

        # 第三段: [0.6, 1.0] -> [0.5, 1.0]
        mask3 = value > self.idx_high_start
        if mask3.any():
            result[mask3] = self.threshold_high + (value[mask3] - self.idx_high_start) / (1.0 - self.idx_high_start) * (self.vmax - self.threshold_high)

        return result


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
    colors = ['#000080', '#4169E1', '#87CEEB', '#FFFFFF', '#90EE90', '#228B22', '#006400']
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('blue_white_green', colors, N=n_bins)

    # 使用 CompressedCenterNorm 压缩中心区域色域，扩展两端色域
    # 这样使得 [-0.5, 0.5] 范围内的颜色变化更快，而两端的颜色变化更慢
    norm = CompressedCenterNorm(vmin=-1.0, vmax=1.0)

    # 绘制热力图（反转y轴，使layer 1在底部，layer 32在顶部）
    # extent设置为[0.5, num_heads+0.5, 0.5, num_layers+0.5]，使得第(i,j)个像素的中心在(i+1, j+1)
    # 这样x=0和y=0是边界，第0列的中心在x=1.0，第1列的中心在x=2.0，以此类推
    im = ax.imshow(heatmap_data, cmap=cmap, norm=norm, aspect='equal',
                   interpolation='nearest', origin='lower',
                   extent=[0.5, num_heads + 0.5, 0.5, num_layers + 0.5])

    # 设置坐标轴标签
    ax.set_xlabel('Attention Heads', fontsize=40, fontweight='bold')
    ax.set_ylabel('Transformer Layers', fontsize=40, fontweight='bold')
    ax.set_title('Attn Head Suppression Score', fontsize=40, fontweight='bold', pad=30)

    # 设置坐标轴范围（包含边界，使得可以显示0刻度）
    ax.set_xlim(0, num_heads + 1.0)
    ax.set_ylim(0, num_layers + 1.0)

    # 设置刻度：原点显示 0，x 轴和 y 轴分别显示 16, 32
    # 由于像素中心在(i+1, j+1)，所以：
    # - 0在边界x=0
    # - 16对应第15列中心x=16
    # - 32对应第31列中心x=32
    major_ticks = [0, 16, 32]
    # x轴标签：0位置显示空字符串，避免与原点0重复
    x_labels = ['', '16', '32']
    # y轴标签：0位置也显示空字符串，原点0将手动添加在左下方
    y_labels = ['', '16', '32']

    ax.set_xticks(major_ticks)
    ax.set_xticklabels(x_labels, fontsize=40, fontweight='bold')
    ax.set_yticks(major_ticks)
    ax.set_yticklabels(y_labels, fontsize=40, fontweight='bold')

    # 设置刻度线加粗
    ax.tick_params(axis='x', which='major', width=2, length=6, labelsize=40)
    ax.tick_params(axis='y', which='major', width=2, length=6, labelsize=40)

    # 设置边框（spine）加粗并使用浅灰色
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color('#CCCCCC')  # 浅灰色

    # 在原点(0, 0)的左下方手动添加"0"标签
    ax.text(-1.0, -0.3, '0', fontsize=40, fontweight='bold',
            ha='left', va='top', transform=ax.transData)

    # 隐藏次要刻度标签
    ax.set_xticks(range(num_heads), minor=True)
    ax.set_yticks(range(num_layers), minor=True)

    # 调整布局
    plt.tight_layout()

    # 保存主图（不包含colorbar）
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 热力图已保存到: {output_path}")
    plt.close()

    # 单独保存colorbar
    colorbar_path = str(output_path).replace('.png', '_colorbar.png')
    fig_cbar = plt.figure(figsize=(1.5, 8))
    ax_cbar = plt.subplot(1, 1, 1)
    ax_cbar.axis('off')
    # 创建colorbar，使用相同的cmap和norm确保颜色一致
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_cbar, orientation='vertical', fraction=1.0)
    cbar.set_ticks([-1.0, 0.0, 1.0])
    cbar.ax.tick_params(labelsize=28, width=2)
    # 设置colorbar刻度标签加粗
    for label in cbar.ax.yaxis.get_majorticklabels():
        label.set_fontweight('bold')
    # 保存colorbar
    plt.savefig(colorbar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Colorbar已单独保存到: {colorbar_path}")


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

    # 使用 CompressedCenterNorm 压缩中心区域色域，扩展两端色域
    # 这样使得 [-0.5, 0.5] 范围内的颜色变化更快，而两端的颜色变化更慢
    # 在二值化热力图中，> 0.5 或 < -0.5 的像素点颜色差异会更明显
    norm = CompressedCenterNorm(vmin=-1.0, vmax=1.0)

    # 绘制二值化热力图
    # extent设置为[0.5, num_heads+0.5, 0.5, num_layers+0.5]，使得第(i,j)个像素的中心在(i+1, j+1)
    # 这样x=0和y=0是边界，第0列的中心在x=1.0，第1列的中心在x=2.0，以此类推
    im = ax.imshow(binarized_data, cmap=cmap, norm=norm, aspect='equal',
                   interpolation='nearest', origin='lower',
                   extent=[0.5, num_heads + 0.5, 0.5, num_layers + 0.5])

    # 设置坐标轴标签
    ax.set_xlabel('Attention Heads', fontsize=40, fontweight='bold')
    ax.set_ylabel('Transformer Layers', fontsize=40, fontweight='bold')
    ax.set_title(f'Critical Attn Head Suppression Score',
                 fontsize=40, fontweight='bold', pad=30)

    # 设置坐标轴范围（包含边界，使得可以显示0刻度）
    ax.set_xlim(0, num_heads + 1.0)
    ax.set_ylim(0, num_layers + 1.0)

    # 设置刻度：原点显示 0，x 轴和 y 轴分别显示 16, 32
    # 由于像素中心在(i+1, j+1)，所以：
    # - 0在边界x=0
    # - 16对应第15列中心x=16
    # - 32对应第31列中心x=32
    major_ticks = [0, 16, 32]
    # x轴标签：0位置显示空字符串，避免与原点0重复
    x_labels = ['', '16', '32']
    # y轴标签：0位置也显示空字符串，原点0将手动添加在左下方
    y_labels = ['', '16', '32']

    ax.set_xticks(major_ticks)
    ax.set_xticklabels(x_labels, fontsize=40, fontweight='bold')
    ax.set_yticks(major_ticks)
    ax.set_yticklabels(y_labels, fontsize=40, fontweight='bold')

    # 设置刻度线加粗
    ax.tick_params(axis='x', which='major', width=2, length=6, labelsize=40)
    ax.tick_params(axis='y', which='major', width=2, length=6, labelsize=40)

    # 设置边框（spine）加粗并使用浅灰色
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color('#CCCCCC')  # 浅灰色

    # 在原点(0, 0)的左下方手动添加"0"标签
    ax.text(-1.0, -0.3, '0', fontsize=40, fontweight='bold',
            ha='left', va='top', transform=ax.transData)

    # 隐藏次要刻度标签
    ax.set_xticks(range(num_heads), minor=True)
    ax.set_yticks(range(num_layers), minor=True)

    # 调整布局
    plt.tight_layout()

    # 保存主图（不包含colorbar）
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 二值化热力图已保存到: {output_path}")
    plt.close()

    # 单独保存colorbar
    colorbar_path = str(output_path).replace('.png', '_colorbar.png')
    fig_cbar = plt.figure(figsize=(1.5, 8))
    ax_cbar = plt.subplot(1, 1, 1)
    ax_cbar.axis('off')
    # 创建colorbar，使用相同的cmap和norm确保颜色一致
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_cbar, orientation='vertical', fraction=1.0)
    cbar.set_ticks([-1.0, 0.0, 1.0])
    cbar.ax.tick_params(labelsize=32, width=2)
    # 设置colorbar刻度标签加粗
    for label in cbar.ax.yaxis.get_majorticklabels():
        label.set_fontweight('bold')
    # 保存colorbar
    plt.savefig(colorbar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Colorbar已单独保存到: {colorbar_path}")

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
    print(f"  2. 原始热力图Colorbar: {heatmap_path.parent / f'{heatmap_path.stem}_colorbar.png'}")
    print(f"  3. 二值化热力图: {binarized_heatmap_path}")
    print(f"  4. 二值化热力图Colorbar: {binarized_heatmap_path.parent / f'{binarized_heatmap_path.stem}_colorbar.png'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
