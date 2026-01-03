"""
LLaVA v1.5 7B 模型架构分析和 Attention 可视化脚本

功能说明：
1. 输出 LLaVA 整个网络架构的不同层的基本信息(网络类型、输入输出 size、参数量等)
2. 输出文本或视觉的 hidden state，hidden layer 的 size，以及二者 attention 的信息
3. 输出特定层(如所有偶数层或奇数层)的 transformer 层的 Q、K 信息，生成 heatmap
4. 将 heatmap 映射到原图上的结果
5. 所有结果保存在 tests/output/ 路径中

功能说明
提取每个生成步骤的所有 32 层 attention
对每层取 attention 矩阵的最后一行(最后一个 hidden token 对序列中所有 token 的 attention)
从最后一行中提取对图像 token 的 attention(576 个值)
将 576 个值 reshape 到 24×24
映射回原图并生成多种可视化


"""

import argparse
import torch
import torch.nn as nn
import os
import json
import sys
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，不显示图片窗口
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LLaVA 模型架构分析和 Attention 可视化")

    # 模型参数
    parser.add_argument("--model-path", type=str,
                       default="/home/liying/Documents/llava-v1.5-7b",
                       help="模型路径")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备 (cuda/cpu)")
    parser.add_argument("--conv-mode", type=str, default="llava_v1",
                       help="对话模式")

    # 输入参数
    default_image_file = "/home/liying/Documents/dataset/coco/val2014/COCO_val2014_000000065883.jpg"
    default_prompt = "there is a boy, Yes or No?" # "there is a boy with blond hair and blue eyes, is this discription correct? Yes or No."
    # default_image_file = "./image.png"
    # default_prompt = "Please describe this image in detail."
    parser.add_argument("--image-file", type=str,
                       default=default_image_file,
                       help=f"图像文件路径(默认: {default_image_file})")
    parser.add_argument("--prompt", type=str, default=default_prompt,
                       help=f"提示词(默认: {default_prompt})")

    # 分析参数
    parser.add_argument("--target-layers", type=str, default="odd",
                       help="目标层索引，用逗号分隔，如 '0,2,4' 或 'even' 表示偶数层，'odd' 表示奇数层")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="输出目录(默认为 tests/output)")
    parser.add_argument("--max-new-tokens", type=int, default=10,
                       help="生成的最大token数量")
    parser.add_argument("--extract-generation-attention", type=bool, default=True,
                       help="是否提取生成过程中的attention map(True/False)")
    parser.add_argument("--save-attention-maps", type=bool, default=True,
                       help="是否保存attention map图片（默认False）")

    return parser.parse_args()


def load_image(image_file):
    """加载图像文件，支持本地文件和 URL"""
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"图像文件不存在: {image_file}")
        image = Image.open(image_file).convert("RGB")
    return image


def extract_hidden_states_and_attention(model, tokenizer, image_processor, image_file, prompt,
                                        conv_mode, device, output_dir, target_layers=None):
    """提取 hidden states 和 attention weights"""
    print("\n" + "=" * 80)
    print("提取 Hidden States 和 Attention")
    print("=" * 80)

    # 准备输入
    image = load_image(image_file)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    # 准备文本输入
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                     return_tensors='pt').unsqueeze(0).to(device)

    # 找到图像 token 占位符的位置(在 input_ids 中)
    image_token_placeholder_positions = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].cpu().numpy()
    print(f"\n原始 input_ids 信息:")
    print(f"  序列长度: {input_ids.shape[1]}")
    print(f"  图像 token 占位符数量: {len(image_token_placeholder_positions)}")
    if len(image_token_placeholder_positions) > 0:
        print(f"  图像 token 占位符位置: {image_token_placeholder_positions}")

    # 编码图像特征
    vision_hidden = None
    num_image_tokens = 0
    with torch.no_grad():
        vision_tower = model.get_vision_tower()
        if vision_tower is not None:
            try:
                image_features = vision_tower(image_tensor.unsqueeze(0).half().to(device))
                if hasattr(image_features, 'last_hidden_state'):
                    vision_hidden = image_features.last_hidden_state
                elif isinstance(image_features, tuple):
                    vision_hidden = image_features[0]
                elif isinstance(image_features, torch.Tensor):
                    vision_hidden = image_features
                else:
                    vision_hidden = None

                # 通过 projector
                if vision_hidden is not None and hasattr(model, 'mm_projector'):
                    vision_hidden = model.mm_projector(vision_hidden)

                if vision_hidden is not None:
                    # vision_hidden shape: [batch, num_patches, hidden_size]
                    # 展平后: [batch, num_patches, hidden_size] -> [batch, num_patches, hidden_size]
                    num_image_tokens = vision_hidden.shape[1]  # 获取图像 token 数量
                    print(f"Vision Hidden State Shape: {vision_hidden.shape}")
                    print(f"图像 token 数量: {num_image_tokens}")
            except Exception as e:
                print(f"⚠️  提取 vision hidden state 时出错: {e}")
                vision_hidden = None

    # Forward pass with output_attentions and output_hidden_states
    with torch.no_grad():
        # 准备 multimodal inputs
        (
            input_ids_processed,
            position_ids,
            attention_mask,
            _,
            inputs_embeds,
            _
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            None,
            None,
            None,
            image_tensor.unsqueeze(0).half().to(device)
        )

        # 计算处理后的序列长度
        print(f"\n[详细调试] 图像 token 位置识别:")
        print(f"  原始 input_ids 形状: {input_ids.shape}")
        print(f"  原始 input_ids 长度: {input_ids.shape[1]}")
        print(f"  图像 token 占位符位置: {image_token_placeholder_positions}")
        print(f"  图像 token 占位符数量: {len(image_token_placeholder_positions)}")
        print(f"  Vision hidden 形状: {vision_hidden.shape if vision_hidden is not None else 'None'}")
        print(f"  图像 token 数量 (从 vision_hidden): {num_image_tokens}")

        if inputs_embeds is not None:
            processed_seq_len = inputs_embeds.shape[1]
            print(f"\n  处理后的序列信息:")
            print(f"    inputs_embeds 形状: {inputs_embeds.shape}")
            print(f"    处理后序列长度: {processed_seq_len}")
            print(f"    序列长度变化: {processed_seq_len - input_ids.shape[1]} (增加了 {processed_seq_len - input_ids.shape[1]} 个 token)")

            # 计算图像 token 的实际位置
            # 在 prepare_inputs_labels_for_multimodal 中，图像特征会替换占位符
            # 如果原始序列中有一个占位符在位置 pos，那么图像 token 会从 pos 开始
            # 但实际序列长度会增加 (num_image_tokens - 1)
            if len(image_token_placeholder_positions) > 0 and num_image_tokens > 0:
                # 找到第一个图像 token 占位符的位置
                first_image_pos = int(image_token_placeholder_positions[0])
                print(f"    第一个图像 token 占位符位置: {first_image_pos}")

                # 图像 token 会替换这个占位符，所以实际位置从 first_image_pos 开始
                # 但需要考虑到序列长度的变化
                image_token_start = first_image_pos
                image_token_end = image_token_start + num_image_tokens
                actual_end = min(image_token_end, processed_seq_len)
                image_token_positions = np.arange(image_token_start, actual_end)

                print(f"    计算的图像 token 起始位置: {image_token_start}")
                print(f"    计算的图像 token 结束位置: {image_token_end} (限制到序列长度: {actual_end})")
                print(f"    图像 token 位置范围: [{image_token_start}, {actual_end-1}]")
                print(f"    图像 token 数量(计算): {len(image_token_positions)}")
                print(f"    图像 token 位置数组 (前10个): {image_token_positions[:10] if len(image_token_positions) > 10 else image_token_positions}")
                print(f"    图像 token 位置数组 (后10个): {image_token_positions[-10:] if len(image_token_positions) > 10 else image_token_positions}")
            else:
                print(f"    ⚠️  无法计算图像 token 位置:")
                print(f"      占位符数量: {len(image_token_placeholder_positions)}")
                print(f"      图像 token 数量: {num_image_tokens}")
                image_token_positions = np.array([])
        else:
            processed_seq_len = input_ids_processed.shape[1] if input_ids_processed is not None else input_ids.shape[1]
            print(f"\n  ⚠️  没有 inputs_embeds，使用原始方法")
            print(f"    处理后的序列长度: {processed_seq_len}")
            # 如果没有 inputs_embeds，说明没有图像，使用原始方法
            image_token_positions = image_token_placeholder_positions
            print(f"    使用占位符位置作为图像 token 位置: {image_token_positions}")

        # Forward through language model
        outputs = model.get_model().forward(
            input_ids=input_ids_processed if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )

    hidden_states = outputs.hidden_states
    attentions = outputs.attentions

    # 验证图像 token 位置
    if len(attentions) > 0 and attentions[0] is not None:
        first_attn = attentions[0]
        if isinstance(first_attn, torch.Tensor):
            actual_seq_len_from_attention = first_attn.shape[-1]  # attention shape: [batch, heads, seq, seq] 或 [heads, seq, seq]
            if len(first_attn.shape) == 4:
                actual_seq_len_from_attention = first_attn.shape[2]  # 使用 query 维度
            print(f"\n[验证] 从 attention weights 推断的序列长度: {actual_seq_len_from_attention}")
            if image_token_positions is not None and len(image_token_positions) > 0:
                max_image_pos = image_token_positions.max()
                print(f"  图像 token 最大位置: {max_image_pos}")
                if max_image_pos >= actual_seq_len_from_attention:
                    print(f"  ⚠️  警告: 图像 token 位置超出 attention 序列长度!")
                    print(f"     调整图像 token 位置到有效范围 [0, {actual_seq_len_from_attention-1}]")
                    image_token_positions = image_token_positions[image_token_positions < actual_seq_len_from_attention]
                    print(f"     调整后的图像 token 数量: {len(image_token_positions)}")

    # 输出中间层的详细信息
    print(f"\n中间层信息摘要:")
    if inputs_embeds is not None:
        actual_seq_len = inputs_embeds.shape[1]
    elif hidden_states is not None and len(hidden_states) > 0:
        actual_seq_len = hidden_states[0].shape[1]
    else:
        actual_seq_len = input_ids.shape[1]

    print(f"  原始 input_ids 长度: {input_ids.shape[1]}")
    print(f"  实际序列长度(处理后): {actual_seq_len}")

    if image_token_positions is not None and len(image_token_positions) > 0:
        print(f"  图像 token 数量: {len(image_token_positions)}")
        print(f"  图像 token 位置范围: [{image_token_positions.min()}, {image_token_positions.max()}]")
        text_token_count = actual_seq_len - len(image_token_positions)
        print(f"  文本 token 数量: {text_token_count}")
    else:
        text_token_count = actual_seq_len
        print(f"  图像 token 数量: 0")
        print(f"  文本 token 数量: {text_token_count}")

    # 保存 hidden states 和 attention
    states_file = os.path.join(output_dir, "hidden_states_info.json")
    states_info = {
        "num_layers": len(hidden_states),
        "hidden_states": [{"layer": i, "shape": list(h.shape)} for i, h in enumerate(hidden_states)],
        "attentions": [{"layer": i, "shape": list(attn.shape) if isinstance(attn, torch.Tensor) else None}
                      for i, attn in enumerate(attentions)],
        "vision_hidden_shape": list(vision_hidden.shape) if vision_hidden is not None else None,
        "image_token_positions": image_token_positions.tolist() if image_token_positions is not None and len(image_token_positions) > 0 else [],
        "input_sequence_length": int(input_ids.shape[1])
    }

    with open(states_file, 'w', encoding='utf-8') as f:
        json.dump(states_info, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Hidden states 信息已保存到: {states_file}")

    # 保存中间层详细信息
    intermediate_info = {
        'input_sequence_length': int(input_ids.shape[1]),
        'num_image_tokens': int(len(image_token_positions)) if image_token_positions is not None and len(image_token_positions) > 0 else 0,
        'num_text_tokens': int(text_token_count),
        'image_token_positions': image_token_positions.tolist() if image_token_positions is not None and len(image_token_positions) > 0 else [],
        'hidden_states_shapes': [list(h.shape) for h in hidden_states],
        'attention_shapes': [list(attn.shape) if isinstance(attn, torch.Tensor) else None for attn in attentions],
        'vision_hidden_shape': list(vision_hidden.shape) if vision_hidden is not None else None
    }
    intermediate_file = os.path.join(output_dir, "intermediate_layers_info.json")
    with open(intermediate_file, 'w', encoding='utf-8') as f:
        json.dump(intermediate_info, f, ensure_ascii=False, indent=2)
    print(f"✓ 中间层详细信息已保存到: {intermediate_file}")

    return hidden_states, attentions, vision_hidden, input_ids, image, image_token_positions


def visualize_attention_heatmap(attention_weights, layer_idx, output_dir, image=None,
                                input_ids=None, tokenizer=None, image_token_positions=None):
    """可视化 attention heatmap - 专门显示文本到图像的 attention"""
    if attention_weights is None:
        return None

    # attention_weights shape: [batch, num_heads, seq_len, seq_len]
    if isinstance(attention_weights, tuple):
        attention_weights = attention_weights[0]

    attention_weights = attention_weights.cpu().numpy()

    # 平均所有 head
    if len(attention_weights.shape) == 4:
        attention_weights = attention_weights[0]  # [num_heads, seq_len, seq_len]
        attention_weights = attention_weights.mean(axis=0)  # [seq_len, seq_len]
    elif len(attention_weights.shape) == 3:
        attention_weights = attention_weights[0].mean(axis=0)  # [seq_len, seq_len]

    seq_len = attention_weights.shape[0]

    # 提取文本到图像的 attention 子矩阵
    print(f"  [Debug Layer {layer_idx}] Attention weights 形状: {attention_weights.shape}")
    print(f"  [Debug Layer {layer_idx}] 序列长度 (seq_len): {seq_len}")
    print(f"  [Debug Layer {layer_idx}] 原始 image_token_positions: {image_token_positions}")
    print(f"  [Debug Layer {layer_idx}] image_token_positions 长度: {len(image_token_positions) if image_token_positions is not None else 0}")

    if image_token_positions is not None and len(image_token_positions) > 0:
        valid_positions = image_token_positions[image_token_positions < seq_len]
        print(f"  [Debug Layer {layer_idx}] 有效图像 token 位置 (在序列长度内): {valid_positions}")
        print(f"  [Debug Layer {layer_idx}] 有效图像 token 数量: {len(valid_positions)}")
        print(f"  [Debug Layer {layer_idx}] 有效图像 token 位置范围: [{valid_positions.min() if len(valid_positions) > 0 else 'N/A'}, {valid_positions.max() if len(valid_positions) > 0 else 'N/A'}]")

        all_positions = np.arange(seq_len)
        text_positions = np.setdiff1d(all_positions, valid_positions)
        print(f"  [Debug Layer {layer_idx}] 文本 token 位置数量: {len(text_positions)}")
        print(f"  [Debug Layer {layer_idx}] 文本 token 位置范围: [{text_positions.min() if len(text_positions) > 0 else 'N/A'}, {text_positions.max() if len(text_positions) > 0 else 'N/A'}]")

        if len(text_positions) > 0 and len(valid_positions) > 0:
            # 提取文本 token -> 图像 token 的 attention
            text_to_image_attn = attention_weights[np.ix_(text_positions, valid_positions)]  # [num_text, num_image]

            # 创建多个可视化
            fig = plt.figure(figsize=(20, 12))

            # 1. 文本->图像 attention heatmap (主要可视化)
            ax1 = plt.subplot(2, 3, 1)
            # 使用更明显的 colormap
            im1 = ax1.imshow(text_to_image_attn, cmap='hot', aspect='auto', interpolation='nearest')
            ax1.set_title(f'Text → Image Attention - Layer {layer_idx}\n({len(text_positions)} text × {len(valid_positions)} image)',
                         fontsize=12, fontweight='bold')
            ax1.set_xlabel('Image Token Position', fontsize=10)
            ax1.set_ylabel('Text Token Position', fontsize=10)
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            # 2. 每个文本 token 对图像的总 attention (行求和)
            ax2 = plt.subplot(2, 3, 2)
            text_attn_sums = text_to_image_attn.sum(axis=1)  # [num_text]
            ax2.barh(range(len(text_attn_sums)), text_attn_sums, color='coral')
            ax2.set_title('Total Attention per Text Token', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Sum of Attention to Image', fontsize=10)
            ax2.set_ylabel('Text Token Index', fontsize=10)
            ax2.grid(axis='x', alpha=0.3)

            # 3. 每个图像 token 接收的总 attention (列求和)
            ax3 = plt.subplot(2, 3, 3)
            image_attn_sums = text_to_image_attn.sum(axis=0)  # [num_image]
            ax3.bar(range(len(image_attn_sums)), image_attn_sums, color='steelblue')
            ax3.set_title('Total Attention per Image Token', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Image Token Index', fontsize=10)
            ax3.set_ylabel('Sum of Attention from Text', fontsize=10)
            ax3.grid(axis='y', alpha=0.3)

            # 4. 文本->图像 attention heatmap (使用 jet colormap，更明显)
            ax4 = plt.subplot(2, 3, 4)
            im4 = ax4.imshow(text_to_image_attn, cmap='jet', aspect='auto', interpolation='bilinear')
            ax4.set_title(f'Text → Image (Jet Colormap) - Layer {layer_idx}',
                         fontsize=12, fontweight='bold')
            ax4.set_xlabel('Image Token Position', fontsize=10)
            ax4.set_ylabel('Text Token Position', fontsize=10)
            plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

            # 5. 文本->图像 attention heatmap (使用 YlOrRd colormap)
            ax5 = plt.subplot(2, 3, 5)
            im5 = ax5.imshow(text_to_image_attn, cmap='YlOrRd', aspect='auto', interpolation='bilinear')
            ax5.set_title(f'Text → Image (YlOrRd) - Layer {layer_idx}',
                         fontsize=12, fontweight='bold')
            ax5.set_xlabel('Image Token Position', fontsize=10)
            ax5.set_ylabel('Text Token Position', fontsize=10)
            plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

            # 6. 归一化后的 attention (每行归一化，显示相对关注度)
            ax6 = plt.subplot(2, 3, 6)
            # 对每行进行归一化，使每个文本 token 的关注度分布更明显
            row_sums = text_to_image_attn.sum(axis=1, keepdims=True)
            # 避免除零和无效值
            row_sums = np.where(row_sums > 1e-10, row_sums, 1.0)  # 如果和为0，设为1避免除零
            text_to_image_norm = text_to_image_attn / row_sums
            # 处理可能的 NaN 和 Inf
            text_to_image_norm = np.nan_to_num(text_to_image_norm, nan=0.0, posinf=0.0, neginf=0.0)
            im6 = ax6.imshow(text_to_image_norm, cmap='hot', aspect='auto', interpolation='bilinear')
            ax6.set_title(f'Normalized Text → Image - Layer {layer_idx}\n(Row-normalized)',
                         fontsize=12, fontweight='bold')
            ax6.set_xlabel('Image Token Position', fontsize=10)
            ax6.set_ylabel('Text Token Position', fontsize=10)
            plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

            plt.tight_layout()
            heatmap_file = os.path.join(output_dir, f"attention_heatmap_layer_{layer_idx}.png")
            plt.savefig(heatmap_file, dpi=200, bbox_inches='tight')
            plt.close()

            print(f"  ✓ Layer {layer_idx} 文本->图像 attention heatmap 已保存: {heatmap_file}")
            print(f"    文本 token 数量: {len(text_positions)}, 图像 token 数量: {len(valid_positions)}")
            print(f"    Attention 值范围: [{text_to_image_attn.min():.4f}, {text_to_image_attn.max():.4f}]")

            return attention_weights
        else:
            print(f"  ⚠️  Layer {layer_idx}: 无法提取文本->图像 attention(缺少文本或图像 token)")

    # 如果没有图像 token 位置信息，显示完整矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(attention_weights, cmap='viridis', cbar=True,
                xticklabels=False, yticklabels=False, square=True)
    plt.title(f'Full Attention Matrix - Layer {layer_idx}\nShape: {attention_weights.shape}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')

    heatmap_file = os.path.join(output_dir, f"attention_heatmap_layer_{layer_idx}.png")
    plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Layer {layer_idx} 完整 attention heatmap 已保存: {heatmap_file}")

    return attention_weights


def analyze_text_tokens_attention(attention_weights, input_ids, tokenizer, image_token_positions, output_dir, layer_idx):
    """分析不同文本 token 对图像 token 的 attention，找出关键词"""
    try:
        seq_len = attention_weights.shape[0]
        valid_positions = image_token_positions[image_token_positions < seq_len]
        all_positions = np.arange(seq_len)
        text_positions = np.setdiff1d(all_positions, valid_positions)

        if len(text_positions) == 0:
            return None

        # 解码每个文本 token
        token_attentions = {}
        for text_idx in text_positions:
            # 确保索引在有效范围内
            if text_idx >= seq_len or text_idx < 0:
                continue
            # 确保 input_ids 索引有效
            if text_idx >= input_ids.shape[1]:
                continue

            try:
                token_id = input_ids[0][text_idx].item()
                token_text = tokenizer.decode([token_id])
                # 确保 valid_positions 不为空
                if len(valid_positions) == 0:
                    continue
                # 计算这个 token 对所有图像 token 的 attention 总和
                attn_to_image = attention_weights[text_idx, valid_positions].sum()
                token_attentions[token_text] = {
                    'position': int(text_idx),
                    'attention_sum': float(attn_to_image),
                    'attention_mean': float(attention_weights[text_idx, valid_positions].mean()),
                    'attention_max': float(attention_weights[text_idx, valid_positions].max())
                }
            except Exception as e:
                # 静默跳过错误，不打印每个错误
                continue

        # 按 attention 总和排序
        sorted_tokens = sorted(token_attentions.items(), key=lambda x: x[1]['attention_sum'], reverse=True)

        # 保存分析结果
        analysis_file = os.path.join(output_dir, f"token_attention_analysis_layer_{layer_idx}.json")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump({
                'layer': layer_idx,
                'top_tokens': sorted_tokens[:20],  # 前20个最关注的 token
                'all_tokens': token_attentions
            }, f, ensure_ascii=False, indent=2)

        print(f"  [Layer {layer_idx}] 文本 token attention 分析已保存")
        if sorted_tokens:
            print(f"    前5个最关注的 token: {[t[0] for t in sorted_tokens[:5]]}")

        return sorted_tokens
    except Exception as e:
        print(f"  ⚠️  Layer {layer_idx}: 分析文本 token attention 时出错: {e}")
        return None


def map_attention_to_image(attention_weights, image, image_token_positions, input_ids, tokenizer, output_dir, layer_idx):
    """将 attention weights 映射到原图上

    使用多种策略聚合 attention，生成更明显的可视化：
    1. 所有文本 token 对图像 token 的 attention 的最大值(突出最关注的区域)
    2. 所有文本 token 对图像 token 的 attention 的平均值
    3. 使用 top-k 文本 token 的 attention(关注最重要的 token)
    4. 图像 token 之间的 self-attention
    """
    if image is None or image_token_positions is None or len(image_token_positions) == 0:
        print(f"  ⚠️  Layer {layer_idx}: 无法映射 attention 到图像(缺少图像或图像 token 位置)")
        return None

    try:
        # 检查 attention_weights 的有效性
        if attention_weights is None or attention_weights.size == 0:
            print(f"  ⚠️  Layer {layer_idx}: attention_weights 为空")
            return None

        # 确保 image_token_positions 在有效范围内
        seq_len = attention_weights.shape[0]
        valid_positions = image_token_positions[image_token_positions < seq_len]

        if len(valid_positions) == 0:
            print(f"  ⚠️  Layer {layer_idx}: 没有有效的图像 token 位置")
            return None

        # 找到所有非图像 token 的位置
        all_positions = np.arange(seq_len)
        text_positions = np.setdiff1d(all_positions, valid_positions)

        # 策略 1: 使用最大值(突出最关注的区域)- 更明显
        if len(text_positions) > 0:
            # 对于每个图像 token，取所有文本 token 对它的 attention 的最大值
            image_attention_max = attention_weights[text_positions][:, valid_positions].max(axis=0)  # [num_image_tokens]
            # 也计算平均值作为参考
            image_attention_mean = attention_weights[text_positions][:, valid_positions].mean(axis=0)  # [num_image_tokens]

            # 策略 1.5: 使用加权平均，给高 attention 的 token 更高权重
            # 计算每个文本 token 对图像的总 attention
            text_to_image_sums = attention_weights[text_positions][:, valid_positions].sum(axis=1)  # [num_text_tokens]
            # 归一化权重
            text_weights = text_to_image_sums / (text_to_image_sums.sum() + 1e-10)
            # 加权平均
            image_attention_weighted = (attention_weights[text_positions][:, valid_positions] * text_weights[:, np.newaxis]).sum(axis=0)

            # 组合策略：使用最大值和加权平均的组合(突出高关注区域)
            image_attention_from_text = 0.6 * image_attention_max + 0.4 * image_attention_weighted

            print(f"  [Layer {layer_idx}] 策略1: 使用 {len(text_positions)} 个文本 token 的最大值+加权平均组合")
            print(f"    最大值范围: [{image_attention_max.min():.4f}, {image_attention_max.max():.4f}]")
            print(f"    平均值范围: [{image_attention_mean.min():.4f}, {image_attention_mean.max():.4f}]")
        else:
            image_attention_from_text = None

        # 策略 2: 使用图像 token 之间的 self-attention(如果策略1不可用)
        if image_attention_from_text is None or len(text_positions) == 0:
            image_to_image_attn = attention_weights[np.ix_(valid_positions, valid_positions)]
            image_attention_from_text = image_to_image_attn.max(axis=0)  # 使用最大值而不是平均值
            print(f"  [Layer {layer_idx}] 策略2: 使用图像 token 之间的 self-attention 最大值")

        # 如果值范围仍然太小，混合图像 self-attention
        attn_min = image_attention_from_text.min()
        attn_max = image_attention_from_text.max()
        attn_range = attn_max - attn_min

        if attn_range < 1e-5 and len(text_positions) > 0:
            image_to_image_attn = attention_weights[np.ix_(valid_positions, valid_positions)]
            image_self_attn = image_to_image_attn.max(axis=0)  # 使用最大值
            image_attention_from_text = 0.5 * image_attention_from_text + 0.5 * image_self_attn
            attn_min = image_attention_from_text.min()
            attn_max = image_attention_from_text.max()
            attn_range = attn_max - attn_min
            print(f"  [Layer {layer_idx}] 混合策略: 文本->图像 (50%) + 图像->图像最大值 (50%)")

        # 检查是否有有效值
        if np.isnan(image_attention_from_text).all() or np.isinf(image_attention_from_text).all():
            print(f"  ⚠️  Layer {layer_idx}: attention 值全为 NaN 或 Inf")
            return None

        # 移除 NaN 和 Inf
        image_attention = np.nan_to_num(image_attention_from_text, nan=0.0, posinf=0.0, neginf=0.0)

        # 使用 percentile 归一化，去除极值影响，使可视化更明显
        attn_min = np.percentile(image_attention, 5)  # 使用5%分位数而不是最小值
        attn_max = np.percentile(image_attention, 95)  # 使用95%分位数而不是最大值
        attn_range = attn_max - attn_min

        if attn_range < 1e-6:
            # 如果值范围仍然太小，使用 softmax 归一化
            image_attention_exp = np.exp((image_attention - image_attention.max()) * 10)  # 放大差异
            image_only_attention = image_attention_exp / image_attention_exp.sum()
            print(f"  [Layer {layer_idx}] 使用 softmax 归一化(值范围: {attn_range:.2e})")
        else:
            # 使用 percentile-based min-max 归一化，然后应用 gamma 校正增强对比度
            image_only_attention = np.clip((image_attention - attn_min) / attn_range, 0, 1)
            # Gamma 校正：增强高关注区域的对比度
            gamma = 0.5  # gamma < 1 会增强高值
            image_only_attention = np.power(image_only_attention, gamma)
            print(f"  [Layer {layer_idx}] 使用 percentile + gamma 归一化(范围: [{attn_min:.4f}, {attn_max:.4f}], gamma={gamma})")

        # 假设图像被分成 patches (CLIP 通常是 14x14 或 24x24)
        num_patches = len(image_only_attention)
        patch_size = int(np.sqrt(num_patches))

        # 如果不是完全平方数，调整 patch_size
        if patch_size * patch_size != num_patches:
            # 尝试找到最接近的完全平方数
            patch_size = int(np.sqrt(num_patches))
            if patch_size * patch_size < num_patches:
                patch_size += 1
            # 如果还是不够，填充或截断
            target_size = patch_size * patch_size
            if target_size > num_patches:
                # 填充
                padding = target_size - num_patches
                image_only_attention = np.pad(image_only_attention, (0, padding), mode='constant', constant_values=0)
            elif target_size < num_patches:
                # 截断
                image_only_attention = image_only_attention[:target_size]

        # 重塑为 2D
        attention_map = image_only_attention.reshape(patch_size, patch_size)

        # 上采样到图像大小
        try:
            from scipy.ndimage import zoom
            zoom_factors = (image.size[1] / patch_size, image.size[0] / patch_size)
            # 确保 zoom factors 是正数
            if zoom_factors[0] > 0 and zoom_factors[1] > 0:
                attention_map_upsampled = zoom(attention_map, zoom_factors, order=1)
            else:
                raise ValueError(f"Invalid zoom factors: {zoom_factors}")
        except (ImportError, ValueError, Exception) as e:
            # 如果没有 scipy 或出错，使用 torch 的插值
            import torch.nn.functional as F
            attention_tensor = torch.from_numpy(attention_map).unsqueeze(0).unsqueeze(0).float()
            attention_tensor = F.interpolate(attention_tensor,
                                            size=(image.size[1], image.size[0]),
                                            mode='bilinear',
                                            align_corners=False)
            attention_map_upsampled = attention_tensor.squeeze().numpy()

        # 确保 attention_map_upsampled 的形状正确
        if attention_map_upsampled.shape[0] != image.size[1] or attention_map_upsampled.shape[1] != image.size[0]:
            # 如果形状不匹配，重新调整
            import torch.nn.functional as F
            attention_tensor = torch.from_numpy(attention_map_upsampled).unsqueeze(0).unsqueeze(0).float()
            attention_tensor = F.interpolate(attention_tensor,
                                            size=(image.size[1], image.size[0]),
                                            mode='bilinear',
                                            align_corners=False)
            attention_map_upsampled = attention_tensor.squeeze().numpy()

        # 创建多种可视化方式 - 增强对比度和可见性
        # 先对 attention_map 进行增强处理
        attn_normalized = (attention_map_upsampled - attention_map_upsampled.min()) / (attention_map_upsampled.max() - attention_map_upsampled.min() + 1e-10)
        attn_enhanced = np.power(attn_normalized, 0.3)  # 更强的 gamma 校正，增强高值区域

        fig = plt.figure(figsize=(24, 16))

        # 1. 原图
        ax1 = plt.subplot(3, 3, 1)
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # 2. Attention heatmap (独立显示，增强对比度)
        ax2 = plt.subplot(3, 3, 2)
        im2 = ax2.imshow(attn_enhanced, cmap='hot', interpolation='bilinear', vmin=0, vmax=1)
        ax2.set_title(f'Attention Heatmap (Enhanced) - Layer {layer_idx}', fontsize=14, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # 3. Attention overlay (jet colormap, 高透明度，增强对比度)
        ax3 = plt.subplot(3, 3, 3)
        ax3.imshow(image)
        im3 = ax3.imshow(attn_enhanced, cmap='jet', alpha=0.7, interpolation='bilinear', vmin=0, vmax=1)
        ax3.set_title(f'Attention Overlay (Jet, Enhanced) - Layer {layer_idx}', fontsize=14, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # 4. Attention overlay (hot colormap, 增强对比度)
        ax4 = plt.subplot(3, 3, 4)
        ax4.imshow(image)
        im4 = ax4.imshow(attn_enhanced, cmap='hot', alpha=0.6, interpolation='bilinear', vmin=0, vmax=1)
        ax4.set_title(f'Attention Overlay (Hot, Enhanced) - Layer {layer_idx}', fontsize=14, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        # 5. Top 20% 高关注区域(更严格的阈值)
        ax5 = plt.subplot(3, 3, 5)
        ax5.imshow(image)
        threshold_80 = np.percentile(attention_map_upsampled, 80)
        attention_thresholded_80 = np.where(attention_map_upsampled >= threshold_80, attention_map_upsampled, 0)
        attn_thresh_norm = (attention_thresholded_80 - attention_thresholded_80.min()) / (attention_thresholded_80.max() - attention_thresholded_80.min() + 1e-10)
        im5 = ax5.imshow(attn_thresh_norm, cmap='Reds', alpha=0.8, interpolation='bilinear', vmin=0, vmax=1)
        ax5.set_title(f'Top 20% Attention - Layer {layer_idx}', fontsize=14, fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

        # 6. Top 10% 高关注区域(最严格的阈值)
        ax6 = plt.subplot(3, 3, 6)
        ax6.imshow(image)
        threshold_90 = np.percentile(attention_map_upsampled, 90)
        attention_thresholded_90 = np.where(attention_map_upsampled >= threshold_90, attention_map_upsampled, 0)
        attn_thresh_90_norm = (attention_thresholded_90 - attention_thresholded_90.min()) / (attention_thresholded_90.max() - attention_thresholded_90.min() + 1e-10)
        im6 = ax6.imshow(attn_thresh_90_norm, cmap='YlOrRd', alpha=0.9, interpolation='bilinear', vmin=0, vmax=1)
        ax6.set_title(f'Top 10% Attention - Layer {layer_idx}', fontsize=14, fontweight='bold')
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

        # 7. Attention contour (等高线图，增强可见性)
        ax7 = plt.subplot(3, 3, 7)
        ax7.imshow(image)
        y_coords = np.linspace(0, image.size[1]-1, attention_map_upsampled.shape[0])
        x_coords = np.linspace(0, image.size[0]-1, attention_map_upsampled.shape[1])
        X, Y = np.meshgrid(x_coords, y_coords)
        # 使用更少的等高线，但更明显
        contour = ax7.contour(X, Y, attention_map_upsampled, levels=5, colors='yellow', linewidths=3, alpha=0.9)
        # matplotlib 的 clabel 不支持 fontweight 参数，使用 fontsize 和 colors 来增强可见性
        ax7.clabel(contour, inline=True, fontsize=10, fmt='%.3f', colors='white')
        ax7.set_title(f'Attention Contour (5 levels) - Layer {layer_idx}', fontsize=14, fontweight='bold')
        ax7.axis('off')

        # 8. 使用 YlOrRd colormap (更明显的黄色到红色渐变)
        ax8 = plt.subplot(3, 3, 8)
        ax8.imshow(image)
        im8 = ax8.imshow(attn_enhanced, cmap='YlOrRd', alpha=0.65, interpolation='bilinear', vmin=0, vmax=1)
        ax8.set_title(f'Attention Overlay (YlOrRd) - Layer {layer_idx}', fontsize=14, fontweight='bold')
        ax8.axis('off')
        plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

        # 9. 使用 plasma colormap (紫色到黄色渐变，很醒目)
        ax9 = plt.subplot(3, 3, 9)
        ax9.imshow(image)
        im9 = ax9.imshow(attn_enhanced, cmap='plasma', alpha=0.7, interpolation='bilinear', vmin=0, vmax=1)
        ax9.set_title(f'Attention Overlay (Plasma) - Layer {layer_idx}', fontsize=14, fontweight='bold')
        ax9.axis('off')
        plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)

        plt.tight_layout()
        overlay_file = os.path.join(output_dir, f"attention_overlay_layer_{layer_idx}.png")
        plt.savefig(overlay_file, dpi=200, bbox_inches='tight')
        plt.close()

        # 保存统计信息
        stats = {
            'layer': layer_idx,
            'attention_stats': {
                'min': float(attention_map_upsampled.min()),
                'max': float(attention_map_upsampled.max()),
                'mean': float(attention_map_upsampled.mean()),
                'std': float(attention_map_upsampled.std()),
                'percentile_50': float(np.percentile(attention_map_upsampled, 50)),
                'percentile_75': float(np.percentile(attention_map_upsampled, 75)),
                'percentile_90': float(np.percentile(attention_map_upsampled, 90)),
                'percentile_95': float(np.percentile(attention_map_upsampled, 95))
            },
            'patch_size': patch_size,
            'num_patches': num_patches
        }
        stats_file = os.path.join(output_dir, f"attention_stats_layer_{layer_idx}.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"  ✓ Layer {layer_idx} attention overlay 已保存: {overlay_file}")
        print(f"  ✓ Layer {layer_idx} attention 统计信息已保存: {stats_file}")
        print(f"    Attention 范围: [{stats['attention_stats']['min']:.4f}, {stats['attention_stats']['max']:.4f}]")
        print(f"    Top 10% 区域阈值: {stats['attention_stats']['percentile_90']:.4f}")

        return attention_map_upsampled

    except Exception as e:
        print(f"  ⚠️  Layer {layer_idx}: 映射 attention 到图像时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_qk_information(model, tokenizer, image_processor, image_file, prompt,
                           conv_mode, device, output_dir, target_layers=None):
    """提取特定层的 Q、K 信息"""
    print("\n" + "=" * 80)
    print("提取 Q、K 信息")
    print("=" * 80)

    if target_layers is None:
        # 默认提取所有偶数层
        if hasattr(model, 'get_model') and hasattr(model.get_model(), 'layers'):
            num_layers = len(model.get_model().layers)
            target_layers = list(range(0, num_layers, 2))  # 偶数层
        else:
            target_layers = []

    print(f"目标层: {target_layers}")

    # 准备输入
    image = load_image(image_file)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                     return_tensors='pt').unsqueeze(0).to(device)

    # 找到图像 token 占位符的位置(在 input_ids 中)
    image_token_placeholder_positions = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].cpu().numpy()

    # 获取图像特征以确定图像 token 数量
    num_image_tokens = 0
    vision_hidden = None
    with torch.no_grad():
        vision_tower = model.get_vision_tower()
        if vision_tower is not None:
            try:
                image_features = vision_tower(image_tensor.unsqueeze(0).half().to(device))
                if hasattr(image_features, 'last_hidden_state'):
                    vision_hidden = image_features.last_hidden_state
                elif isinstance(image_features, tuple):
                    vision_hidden = image_features[0]
                elif isinstance(image_features, torch.Tensor):
                    vision_hidden = image_features
                else:
                    vision_hidden = None
                    print(f"  ⚠️  无法从 vision_tower 输出中提取 hidden state")

                if vision_hidden is not None and hasattr(model, 'mm_projector'):
                    vision_hidden = model.mm_projector(vision_hidden)
                    num_image_tokens = vision_hidden.shape[1]
                    print(f"  ✓ 成功获取图像特征: vision_hidden.shape={vision_hidden.shape}, num_image_tokens={num_image_tokens}")
                elif vision_hidden is not None:
                    print(f"  ⚠️  vision_hidden 存在但模型没有 mm_projector")
                    num_image_tokens = vision_hidden.shape[1] if len(vision_hidden.shape) >= 2 else 0
            except Exception as e:
                print(f"  ⚠️  提取 vision hidden state 时出错: {e}")
                import traceback
                traceback.print_exc()
                vision_hidden = None
        else:
            print(f"  ⚠️  模型没有 vision_tower")

    # 存储 Q、K 数据的字典
    qk_data = {}

    # 注册 hook 来提取 Q、K
    hooks = []
    lang_model = model.get_model()

    def make_q_hook(layer_idx):
        def q_hook(module, input, output):
            # q_proj 的输出就是 Q
            with torch.no_grad():
                if layer_idx not in qk_data:
                    qk_data[layer_idx] = {}
                qk_data[layer_idx]['q'] = output.cpu().numpy()
                qk_data[layer_idx]['q_shape'] = list(output.shape)
        return q_hook

    def make_k_hook(layer_idx):
        def k_hook(module, input, output):
            # k_proj 的输出就是 K
            with torch.no_grad():
                if layer_idx not in qk_data:
                    qk_data[layer_idx] = {}
                qk_data[layer_idx]['k'] = output.cpu().numpy()
                qk_data[layer_idx]['k_shape'] = list(output.shape)
        return k_hook

    # 为目标层注册 hooks(直接在 q_proj 和 k_proj 上注册)
    if hasattr(lang_model, 'layers'):
        for layer_idx in target_layers:
            if layer_idx < len(lang_model.layers):
                layer = lang_model.layers[layer_idx]
                if hasattr(layer, 'self_attn'):
                    attn = layer.self_attn
                    if hasattr(attn, 'q_proj'):
                        hook_q = attn.q_proj.register_forward_hook(make_q_hook(layer_idx))
                        hooks.append(hook_q)
                    if hasattr(attn, 'k_proj'):
                        hook_k = attn.k_proj.register_forward_hook(make_k_hook(layer_idx))
                        hooks.append(hook_k)

    # Forward pass
    with torch.no_grad():
        (
            input_ids_processed,
            position_ids,
            attention_mask,
            _,
            inputs_embeds,
            _
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            None,
            None,
            None,
            image_tensor.unsqueeze(0).half().to(device)
        )

        # 计算图像 token 的实际位置
        print(f"\n[详细调试 - extract_qk_information] 图像 token 位置识别:")
        print(f"  原始 input_ids 形状: {input_ids.shape}")
        print(f"  图像 token 占位符位置: {image_token_placeholder_positions}")
        print(f"  图像 token 数量 (从 vision_hidden): {num_image_tokens}")
        print(f"  Vision hidden 形状: {vision_hidden.shape if vision_hidden is not None else 'None'}")

        if inputs_embeds is not None:
            processed_seq_len = inputs_embeds.shape[1]
            print(f"  inputs_embeds 形状: {inputs_embeds.shape}")
            print(f"  处理后序列长度: {processed_seq_len}")
            print(f"  序列长度变化: {processed_seq_len - input_ids.shape[1]} (增加了 {processed_seq_len - input_ids.shape[1]} 个 token)")

            if len(image_token_placeholder_positions) > 0 and num_image_tokens > 0:
                first_image_pos = int(image_token_placeholder_positions[0])
                image_token_start = first_image_pos
                image_token_end = image_token_start + num_image_tokens
                actual_end = min(image_token_end, processed_seq_len)
                image_token_positions = np.arange(image_token_start, actual_end)

                print(f"  第一个图像 token 占位符位置: {first_image_pos}")
                print(f"  计算的图像 token 起始位置: {image_token_start}")
                print(f"  计算的图像 token 结束位置: {image_token_end} (限制到序列长度: {actual_end})")
                print(f"  图像 token 位置范围: [{image_token_start}, {actual_end-1}]")
                print(f"  图像 token 数量: {len(image_token_positions)}")
                print(f"  图像 token 位置数组 (前10个): {image_token_positions[:10] if len(image_token_positions) > 10 else image_token_positions}")
                print(f"  图像 token 位置数组 (后10个): {image_token_positions[-10:] if len(image_token_positions) > 10 else image_token_positions}")
            else:
                print(f"  ⚠️  无法计算图像 token 位置:")
                print(f"      占位符数量: {len(image_token_placeholder_positions)}")
                print(f"      图像 token 数量: {num_image_tokens}")
                image_token_positions = image_token_placeholder_positions
        else:
            print(f"  ⚠️  没有 inputs_embeds")
            image_token_positions = image_token_placeholder_positions

        outputs = lang_model.forward(
            input_ids=input_ids_processed if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )

    # 移除 hooks
    for hook in hooks:
        hook.remove()

    attentions = outputs.attentions

    # 保存 Q、K 数据
    if qk_data:
        qk_file = os.path.join(output_dir, "qk_information.json")
        # 转换为可序列化格式
        qk_info = {}
        for layer_idx, data in qk_data.items():
            try:
                q_data = data.get('q', None)
                k_data = data.get('k', None)

                layer_info = {
                    "q_shape": data.get('q_shape', 'N/A'),
                    "k_shape": data.get('k_shape', 'N/A')
                }

                if q_data is not None:
                    # 安全计算统计信息，避免 overflow
                    q_data = np.nan_to_num(q_data, nan=0.0, posinf=0.0, neginf=0.0)
                    layer_info.update({
                        "q_mean": float(np.mean(q_data)),
                        "q_std": float(np.std(q_data)),
                        "q_min": float(np.min(q_data)),
                        "q_max": float(np.max(q_data))
                    })

                if k_data is not None:
                    # 安全计算统计信息，避免 overflow
                    k_data = np.nan_to_num(k_data, nan=0.0, posinf=0.0, neginf=0.0)
                    layer_info.update({
                        "k_mean": float(np.mean(k_data)),
                        "k_std": float(np.std(k_data)),
                        "k_min": float(np.min(k_data)),
                        "k_max": float(np.max(k_data))
                    })

                qk_info[f"layer_{layer_idx}"] = layer_info
            except Exception as e:
                print(f"  ⚠️  处理 Layer {layer_idx} 的 Q、K 数据时出错: {e}")
                qk_info[f"layer_{layer_idx}"] = {"error": str(e)}

        with open(qk_file, 'w', encoding='utf-8') as f:
            json.dump(qk_info, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Q、K 信息已保存到: {qk_file}")

    # 处理目标层 - 可视化 attention
    for layer_idx in target_layers:
        if layer_idx < len(attentions) and attentions[layer_idx] is not None:
            print(f"\n{'='*60}")
            print(f"处理 Layer {layer_idx}...")
            print(f"{'='*60}")

            # 提取 attention weights
            attn = attentions[layer_idx]
            if isinstance(attn, tuple):
                attn = attn[0]
            attn_np = attn.cpu().numpy()

            # 平均所有 head
            if len(attn_np.shape) == 4:
                attn_np = attn_np[0].mean(axis=0)  # [seq_len, seq_len]
            elif len(attn_np.shape) == 3:
                attn_np = attn_np[0].mean(axis=0)

            # 分析文本 token 的 attention
            analyze_text_tokens_attention(
                attn_np, input_ids, tokenizer, image_token_positions, output_dir, layer_idx
            )

            # 可视化 attention heatmap
            attention_weights = visualize_attention_heatmap(
                attentions[layer_idx], layer_idx, output_dir,
                image, input_ids, tokenizer, image_token_positions
            )

            # 映射到图像
            if attention_weights is not None and image is not None and len(image_token_positions) > 0:
                map_attention_to_image(
                    attention_weights, image, image_token_positions, input_ids, tokenizer,
                    output_dir, layer_idx
                )

    print(f"\n✓ Q、K 信息提取完成")


def safe_decode_token_ids(tokenizer, token_ids, skip_special_tokens=False):
    """安全地解码token IDs，过滤掉无效的ID"""
    if not token_ids:
        return ""

    # 获取词汇表大小
    vocab_size = len(tokenizer) if hasattr(tokenizer, '__len__') else tokenizer.vocab_size

    # 过滤掉无效的token ID（超出词汇表范围或为负数）
    valid_token_ids = [tid for tid in token_ids if isinstance(tid, int) and 0 <= tid < vocab_size]

    if not valid_token_ids:
        return ""

    try:
        return tokenizer.batch_decode([valid_token_ids], skip_special_tokens=skip_special_tokens)[0]
    except Exception as e:
        # 如果批量解码失败，尝试逐个解码
        try:
            decoded_tokens = []
            for tid in valid_token_ids:
                try:
                    token_text = tokenizer.decode([tid])
                    decoded_tokens.append(token_text)
                except:
                    continue
            return "".join(decoded_tokens)
        except:
            return f"[解码失败: {str(e)}]"


def extract_attention_during_generation(model, tokenizer, image_processor, image_file, prompt,
                                       conv_mode, device, output_dir, target_layers, max_new_tokens=10, save_attention_maps=True):
    """在生成过程中提取每个预测词的attention map

    对于每个生成的token，提取所有32层的attention map，并可视化最后一行对图像token的attention

    Args:
        save_attention_maps: 是否保存attention map图片（True/False）
        target_layers: 目标层列表，None表示所有层，'even'表示偶数层，'odd'表示奇数层，或具体层索引列表
    """
    print("\n" + "=" * 80)
    print("在生成过程中提取 Attention Map")
    print("=" * 80)

    # 准备输入
    image = load_image(image_file)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    # 准备文本输入
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                     return_tensors='pt').unsqueeze(0).to(device)

    # 找到图像token占位符位置（在原始input_ids中）
    image_token_placeholder_positions = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].cpu().numpy()

    print(f"\n原始input_ids信息:")
    print(f"  序列长度: {input_ids.shape[1]}")
    print(f"  IMAGE_TOKEN_INDEX值: {IMAGE_TOKEN_INDEX}")
    print(f"  图像token占位符数量: {len(image_token_placeholder_positions)}")
    image_first_pos = 1
    if len(image_token_placeholder_positions) > 0:
        print(f"  图像token占位符位置: {image_token_placeholder_positions}")
        print(f"  第一个IMAGE_TOKEN_INDEX位置: {image_token_placeholder_positions[0]}")
        # 打印第一个IMAGE_TOKEN_INDEX前后的token信息
        image_first_pos = int(image_token_placeholder_positions[0])
        print(f"  第一个IMAGE_TOKEN_INDEX之前的token数量: {image_first_pos}")
        if image_first_pos > 0:
            tokens_before = input_ids[0, :image_first_pos].cpu().tolist()
            tokens_before_text = [tokenizer.decode([tid]) for tid in tokens_before]
            print(f"  之前的tokens({len(tokens_before_text)}): {tokens_before_text}")
        if image_first_pos < input_ids.shape[1] - 1:
            tokens_after = input_ids[0, image_first_pos+1:].cpu().tolist()
            tokens_after_text = [tokenizer.decode([tid]) for tid in tokens_after]
            print(f"  之后的tokens({len(tokens_after_text)}): {tokens_after_text}")
    else:
        print(f"  ⚠️  未找到IMAGE_TOKEN_INDEX占位符")

    # 获取图像特征以确定图像token数量
    num_image_tokens = 0
    vision_hidden = None
    with torch.no_grad():
        vision_tower = model.get_vision_tower()
        print(f"\n[调试] Vision Tower 检查:")
        print(f"  vision_tower is None: {vision_tower is None}")

        if vision_tower is not None:
            print(f"  vision_tower 类型: {type(vision_tower)}")
            try:
                print(f"  [调试] 开始提取图像特征...")
                print(f"  image_tensor shape: {image_tensor.shape}")
                print(f"  image_tensor.unsqueeze(0).half().to(device) shape: {image_tensor.unsqueeze(0).half().to(device).shape}")

                image_features = vision_tower(image_tensor.unsqueeze(0).half().to(device))
                print(f"  [调试] image_features 类型: {type(image_features)}")

                if hasattr(image_features, 'last_hidden_state'):
                    vision_hidden = image_features.last_hidden_state
                    print(f"  [调试] 使用 last_hidden_state, shape: {vision_hidden.shape}")
                elif isinstance(image_features, tuple):
                    vision_hidden = image_features[0]
                    print(f"  [调试] 使用 tuple[0], shape: {vision_hidden.shape}")
                elif isinstance(image_features, torch.Tensor):
                    vision_hidden = image_features
                    print(f"  [调试] 使用 Tensor, shape: {vision_hidden.shape}")
                else:
                    vision_hidden = None
                    print(f"  ⚠️  无法从 image_features 中提取 hidden state")
                    print(f"     image_features 类型: {type(image_features)}")
                    if hasattr(image_features, '__dict__'):
                        print(f"     image_features 属性: {list(image_features.__dict__.keys())}")

                if vision_hidden is not None:
                    print(f"  [调试] vision_hidden 提取成功, shape: {vision_hidden.shape}")
                    print(f"  [调试] 检查 mm_projector...")

                    # mm_projector 在 model.get_model() 中，不在 model 上
                    lang_model = model.get_model() if hasattr(model, 'get_model') else None
                    has_mm_projector = lang_model is not None and hasattr(lang_model, 'mm_projector')
                    print(f"    hasattr(model, 'get_model'): {hasattr(model, 'get_model')}")
                    print(f"    lang_model is not None: {lang_model is not None}")
                    print(f"    hasattr(lang_model, 'mm_projector'): {has_mm_projector}")

                    if has_mm_projector:
                        print(f"    mm_projector 类型: {type(lang_model.mm_projector)}")
                        vision_hidden = lang_model.mm_projector(vision_hidden)
                        num_image_tokens = vision_hidden.shape[1]
                        print(f"\n✓ 图像特征信息:")
                        print(f"  Vision hidden shape (经过 mm_projector): {vision_hidden.shape}")
                        print(f"  图像token数量: {num_image_tokens}")
                    else:
                        print(f"  ⚠️  模型没有 mm_projector（在 model.get_model() 中）")
                        # 如果没有 mm_projector，直接使用 vision_hidden
                        # 注意：vision_hidden 的维度是 [1, 576, 1024]，需要投影到 [1, 576, 4096]
                        # 但如果没有 projector，我们仍然可以使用 576 作为图像 token 数量
                        num_image_tokens = vision_hidden.shape[1] if len(vision_hidden.shape) >= 2 else 0
                        print(f"  使用 vision_hidden 的 shape[1] 作为 num_image_tokens: {num_image_tokens}")
                        print(f"  注意: vision_hidden 维度是 {vision_hidden.shape}，可能需要通过 mm_projector 投影")
                else:
                    print(f"  ⚠️  vision_hidden 为 None，无法继续处理")
            except Exception as e:
                print(f"  ⚠️  提取 vision hidden state 时出错: {e}")
                import traceback
                traceback.print_exc()
                vision_hidden = None
        else:
            print(f"  ⚠️  模型没有 vision_tower")

    # 准备multimodal inputs以确定图像token的实际位置
    with torch.no_grad():
        (
            input_ids_processed,
            position_ids,
            attention_mask,
            _,
            inputs_embeds,
            _
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            None,
            None,
            None,
            image_tensor.unsqueeze(0).half().to(device)
        )

        # 根据LLaVA源码逻辑计算图像token的实际位置
        # 在prepare_inputs_labels_for_multimodal中（llava_arch.py第171-203行）：
        # 1. image_token_indices = [-1] + [IMAGE_TOKEN_INDEX位置] + [end]
        # 2. 提取文本块：cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]]
        # 3. 交替拼接：文本emb[0] + 图像features + 文本emb[1] + ...
        #
        # 所以：图像token起始位置 = 第一个IMAGE_TOKEN_INDEX之前的所有token数量
        # （包括BOS token，如果有的话）

        if inputs_embeds is not None:
            processed_seq_len = inputs_embeds.shape[1]
            if len(image_token_placeholder_positions) > 0 and num_image_tokens > 0:
                # 找到第一个IMAGE_TOKEN_INDEX的位置（在原始input_ids中）
                first_image_placeholder_pos = int(image_token_placeholder_positions[0])

                # 根据源码逻辑：
                # - image_token_indices[0] = -1
                # - image_token_indices[1] = first_image_placeholder_pos
                # - 提取的文本块：cur_input_ids[-1+1:first_image_placeholder_pos] = cur_input_ids[0:first_image_placeholder_pos]
                # - 这个文本块会被embed，然后图像features会被插入
                # - 所以图像token的起始位置 = 第一个IMAGE_TOKEN_INDEX之前的所有token数量

                image_token_start = first_image_placeholder_pos
                image_token_end = image_token_start + num_image_tokens
                actual_end = min(image_token_end, processed_seq_len)
                image_token_positions = np.arange(image_token_start, actual_end)

                print(f"\n✓ 图像token位置计算:")
                print(f"  【原始input_ids】")
                print(f"    第一个IMAGE_TOKEN_INDEX位置: {first_image_placeholder_pos}")
                print(f"    该位置之前的token数量: {first_image_placeholder_pos}")
                if first_image_placeholder_pos > 0:
                    tokens_before = input_ids[0, :first_image_placeholder_pos].cpu().tolist()
                    tokens_before_text = [tokenizer.decode([tid]) for tid in tokens_before]
                    print(f"    之前的tokens: {tokens_before_text[:10]}...")
                print(f"  【处理后序列】")
                print(f"    图像token起始位置: {image_token_start} (等于原始input_ids中第一个IMAGE_TOKEN_INDEX的位置)")
                print(f"    图像token结束位置: {actual_end-1}")
                print(f"    图像token数量: {num_image_tokens}")
                print(f"    图像token位置范围: [{image_token_start}, {actual_end-1}]")
                print(f"    处理后序列总长度: {processed_seq_len}")
                print(f"    图像token之后的文本token数量: {processed_seq_len - actual_end}")
            else:
                image_token_positions = np.array([])
                image_token_start = None
                print(f"⚠️  无法确定图像token位置")
        else:
            image_token_positions = image_token_placeholder_positions
            if len(image_token_placeholder_positions) > 0:
                image_token_start = int(image_token_placeholder_positions[0])
            else:
                image_token_start = None

    # 使用generate并设置output_attentions=True
    print(f"\n开始生成，最多生成 {max_new_tokens} 个token...")
    print(f"\n【生成停止机制说明】")
    print(f"  1. EOS Token: 如果模型在某个步骤生成了 EOS token，生成会立即停止")
    print(f"     - EOS token 是模型预测生成的，不是预先设定的")
    print(f"     - 当模型认为应该结束时，会生成 EOS token")
    print(f"  2. 停止字符串: 如果设置了 stopping_criteria，会在每个步骤后检查是否包含停止字符串")
    print(f"     - 当前代码未设置 stopping_criteria，所以不依赖停止字符串")
    print(f"  3. max_new_tokens: 最多生成 {max_new_tokens} 个新 token")
    print(f"     - 如果达到这个限制，即使没有 EOS token 也会停止")

    with torch.no_grad():
        output_dict = model.generate(
            inputs=input_ids,
            images=image_tensor.unsqueeze(0).half().to(device),
            max_new_tokens=max_new_tokens,
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=False,  # 使用greedy decoding
            num_beams=1
            # 注意: 当前没有设置 stopping_criteria，所以只依赖 EOS token 和 max_new_tokens
        )

    generated_ids = output_dict.sequences
    all_attentions = output_dict.attentions  # 这是一个tuple，每个元素对应一个生成步骤

    # 详细分析 all_attentions 的结构
    print(f"\n{'='*80}")
    print(f"【Attention 结构分析】")
    print(f"{'='*80}")
    if all_attentions is not None:
        print(f"✓ all_attentions 类型: {type(all_attentions)}")
        print(f"✓ all_attentions 长度: {len(all_attentions)} (对应 {len(all_attentions)} 个生成步骤)")

        # 分析第一个步骤的 attention 结构
        if len(all_attentions) > 0:
            first_step_attentions = all_attentions[0]
            print(f"\n  [步骤 0] 的 attention 结构:")
            print(f"    类型: {type(first_step_attentions)}")
            if isinstance(first_step_attentions, (tuple, list)):
                print(f"    长度: {len(first_step_attentions)} (对应 {len(first_step_attentions)} 个 Transformer 层)")
                if len(first_step_attentions) > 0:
                    first_layer_attn = first_step_attentions[0]
                    if isinstance(first_layer_attn, torch.Tensor):
                        print(f"    第 0 层 attention 形状: {first_layer_attn.shape}")
                        if len(first_layer_attn.shape) == 4:
                            batch, num_heads, query_len, key_len = first_layer_attn.shape
                            print(f"      - batch_size: {batch}")
                            print(f"      - num_heads: {num_heads}")
                            print(f"      - query_len: {query_len} (只有最后一个新生成的 token)")
                            print(f"      - key_len: {key_len} (所有历史 token 的长度)")

        # 分析所有步骤
        print(f"\n  [所有步骤概览]:")
        for step_idx in range(len(all_attentions)):
            step_attn = all_attentions[step_idx]
            if isinstance(step_attn, (tuple, list)):
                print(f"    步骤 {step_idx}: {len(step_attn)} 层")
            else:
                print(f"    步骤 {step_idx}: {type(step_attn)}")
    else:
        print(f"⚠️  all_attentions 为 None")

    # 检查生成的序列
    input_len = input_ids.shape[1]
    output_len = generated_ids.shape[1]

    # generated_ids 只包含新生成的token
    generated_token_ids_raw = generated_ids[0].cpu().tolist()

    # 过滤掉 BOS token（如果存在），因为它不应该算作新生成的 token
    # BOS token 通常是输入的一部分，或者是在生成开始前自动添加的
    bos_token_id = tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None else None
    if bos_token_id is not None and len(generated_token_ids_raw) > 0 and generated_token_ids_raw[0] == bos_token_id:
        # 移除开头的 BOS token
        generated_token_ids = generated_token_ids_raw[1:]
        print(f"  [过滤] 移除了开头的 BOS token (token_id: {bos_token_id})")
    else:
        generated_token_ids = generated_token_ids_raw

    full_output_ids = input_ids[0].cpu().tolist() + generated_token_ids

    print(f"\n{'='*80}")
    print(f"【推理结果 - Output IDs】")
    print(f"{'='*80}")
    print(f"✓ 输入序列长度: {input_len}")
    print(f"✓ 生成后序列长度: {output_len}")
    print(f"✓ 完整Output IDs (包含input+output): {full_output_ids}")
    print(f"✓ 新生成的Token IDs: {generated_token_ids}")
    print(f"✓ 新生成的Token数量: {len(generated_token_ids)}")
    print(f"✓ Attention步骤数量: {len(all_attentions) if all_attentions is not None else 0}")

    # 详细分析 token 和 attention 的对应关系
    print(f"\n【Token 与 Attention 对应关系分析】")
    print(f"  generated_token_ids: {generated_token_ids}")

    # 检查每个 token 是否是特殊 token
    bos_token_id = tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None else None
    eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None else None
    pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else None

    print(f"  BOS token_id: {bos_token_id}")
    print(f"  EOS token_id: {eos_token_id}")
    print(f"  PAD token_id: {pad_token_id}")

    # 分析每个 token
    print(f"\n  每个 token 的详细信息:")
    for i, token_id in enumerate(generated_token_ids):
        token_type = "普通token"
        if bos_token_id is not None and token_id == bos_token_id:
            token_type = "BOS token (开始标记，通常不在生成步骤中)"
        elif eos_token_id is not None and token_id == eos_token_id:
            token_type = "EOS token (结束标记，生成后立即停止，可能没有对应的attention)"
        elif pad_token_id is not None and token_id == pad_token_id:
            token_type = "PAD token (填充标记)"

        try:
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        except:
            token_text = f"[无法解码: {token_id}]"

        has_attention = i < len(all_attentions) if all_attentions is not None else False
        print(f"    Token {i}: ID={token_id}, 文本='{token_text}', 类型={token_type}, 有attention={has_attention}")

    # 过滤策略：根据 all_attentions 的长度来过滤 generated_token_ids
    # 1. 如果 BOS token 在开头，过滤掉它（即使它有 attention，也不应该算作新生成的 token）
    # 2. 过滤掉没有对应 attention 的 token（通常是 EOS token）
    # 3. 确保过滤后的 token 数量与 all_attentions 的长度一致
    num_attention_steps = len(all_attentions) if all_attentions is not None else 0

    if num_attention_steps == 0:
        print(f"\n  ⚠️  警告: all_attentions 为空，无法进行过滤")
    else:
        # 第一步：过滤掉开头的 BOS token（如果存在）
        filtered_generated_token_ids = generated_token_ids.copy()
        if bos_token_id is not None and len(filtered_generated_token_ids) > 0 and filtered_generated_token_ids[0] == bos_token_id:
            # 移除开头的 BOS token
            filtered_generated_token_ids = filtered_generated_token_ids[1:]
            print(f"\n  [过滤步骤1] 移除了开头的 BOS token (token_id: {bos_token_id})")
            print(f"    - 过滤前数量: {len(generated_token_ids)}")
            print(f"    - 过滤后数量: {len(filtered_generated_token_ids)}")

        # 第二步：根据 all_attentions 的长度，只保留对应数量的 token
        # 如果过滤后仍有超出，说明有 token 没有对应的 attention（通常是 EOS token）
        if len(filtered_generated_token_ids) > num_attention_steps:
            filtered_out_tokens = filtered_generated_token_ids[num_attention_steps:]
            filtered_generated_token_ids = filtered_generated_token_ids[:num_attention_steps]

            print(f"\n  [过滤步骤2] 根据 all_attentions 长度过滤多余的 token:")
            print(f"    - 过滤前数量: {len(filtered_generated_token_ids) + len(filtered_out_tokens)}")
            print(f"    - all_attentions 数量: {num_attention_steps}")
            print(f"    - 过滤后数量: {len(filtered_generated_token_ids)}")
            if filtered_out_tokens:
                print(f"    - 被过滤掉的 token IDs: {filtered_out_tokens}")
                for token_id in filtered_out_tokens:
                    try:
                        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                        token_type = "普通token"
                        if eos_token_id is not None and token_id == eos_token_id:
                            token_type = "EOS token (生成后立即停止，没有对应的attention)"
                        print(f"      - Token ID {token_id}: '{token_text}' ({token_type})")
                    except:
                        print(f"      - Token ID {token_id}: [无法解码]")

        # 更新 generated_token_ids 为过滤后的版本
        if len(filtered_generated_token_ids) != len(generated_token_ids):
            generated_token_ids = filtered_generated_token_ids
            # 同时更新 full_output_ids（移除被过滤的 token）
            full_output_ids = input_ids[0].cpu().tolist() + generated_token_ids
            print(f"\n  ✓ 已更新 generated_token_ids 和 full_output_ids")
            print(f"    - 最终 generated_token_ids 数量: {len(generated_token_ids)}")
            print(f"    - all_attentions 数量: {num_attention_steps}")
            if len(generated_token_ids) == num_attention_steps:
                print(f"    ✓ 现在数量一致！")
            else:
                print(f"    ⚠️  数量仍不一致，可能需要进一步检查")

    # 检查停止原因
    print(f"\n【生成停止原因分析】")
    if len(generated_token_ids) > 0:
        last_token_id = generated_token_ids[-1]
        eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None else None

        if eos_token_id is not None and last_token_id == eos_token_id:
            print(f"  ✓ 停止原因: 模型生成了 EOS token (token_id: {eos_token_id})")
            print(f"    - EOS token 是在步骤 {len(generated_token_ids)-1} 生成的")
            print(f"    - 说明: EOS token 是模型在推理过程中预测生成的，不是预先设定的")
        elif len(generated_token_ids) >= max_new_tokens:
            print(f"  ✓ 停止原因: 达到了 max_new_tokens 限制 ({max_new_tokens})")
            print(f"    - 说明: 即使没有 EOS token，达到最大 token 数也会停止")
        else:
            print(f"  ✓ 停止原因: 可能是其他停止条件")
            print(f"    - 最后生成的 token_id: {last_token_id}")
            if eos_token_id is not None:
                print(f"    - EOS token_id: {eos_token_id} (不匹配)")
            # 检查是否可能是停止字符串（虽然当前代码没有设置 stopping_criteria）
            print(f"    - 注意: 当前代码未设置 stopping_criteria，所以不依赖停止字符串")
    else:
        print(f"  ⚠️  没有生成任何 token")

    # 检查并过滤无效的token ID
    vocab_size = len(tokenizer) if hasattr(tokenizer, '__len__') else getattr(tokenizer, 'vocab_size', 32000)
    valid_full_output_ids = [tid for tid in full_output_ids if isinstance(tid, int) and 0 <= tid < vocab_size]
    valid_generated_token_ids = [tid for tid in generated_token_ids if isinstance(tid, int) and 0 <= tid < vocab_size]

    if len(valid_full_output_ids) != len(full_output_ids):
        print(f"⚠️  过滤掉 {len(full_output_ids) - len(valid_full_output_ids)} 个无效的token ID")
    if len(valid_generated_token_ids) != len(generated_token_ids):
        print(f"⚠️  过滤掉 {len(generated_token_ids) - len(valid_generated_token_ids)} 个无效的生成token ID")

    # 解码完整的序列（包含input和output）- 使用安全解码
    full_text_with_special = safe_decode_token_ids(tokenizer, valid_full_output_ids, skip_special_tokens=False)
    full_text = safe_decode_token_ids(tokenizer, valid_full_output_ids, skip_special_tokens=True)

    # 解码只包含新生成的部分 - 使用安全解码
    generated_text_with_special = safe_decode_token_ids(tokenizer, valid_generated_token_ids, skip_special_tokens=False)
    generated_text = safe_decode_token_ids(tokenizer, valid_generated_token_ids, skip_special_tokens=True)

    print(f"\n{'='*80}")
    print(f"【文本语义信息】")
    print(f"{'='*80}")
    print(f"✓ 完整序列文本(含特殊token): {full_text_with_special}")
    print(f"✓ 完整序列文本(不含特殊token): {full_text}")
    print(f"✓ 新生成部分文本(含特殊token): {generated_text_with_special}")
    print(f"✓ 新生成部分文本(不含特殊token): {generated_text}")

    # 解码每个生成的token(不跳过特殊token，以便看到实际内容) - 使用安全解码
    generated_tokens = []
    for tid in valid_generated_token_ids:
        try:
            token_text = tokenizer.decode([tid])
            generated_tokens.append(token_text)
        except Exception as e:
            generated_tokens.append(f"[无效token: {tid}]")

    # 过滤掉空token和只有空格的token
    valid_tokens = [(i, t) for i, t in enumerate(generated_tokens) if t.strip() or t in ['<s>', '</s>', '<unk>']]
    print(f"\n✓ 生成的tokens数量: {len(generated_tokens)}")
    print(f"✓ 有效tokens数量: {len(valid_tokens)}")
    if len(valid_tokens) > 0:
        print(f"✓ 所有生成的tokens: {valid_tokens}")

    # 检查attention数量和生成的token数量是否一致
    num_attention_steps = len(all_attentions) if all_attentions is not None else 0
    num_generated_tokens = len(generated_token_ids)

    if num_attention_steps != num_generated_tokens:
        print(f"\n⚠️  警告: attention数量({num_attention_steps})与生成的token数量({num_generated_tokens})不一致")
        # 如果token数量为0但attention数量>0，说明提取有问题，使用attention数量
        if num_generated_tokens == 0 and num_attention_steps > 0:
            print(f"  检测到token数量为0但attention数量为{num_attention_steps}，可能是提取逻辑问题")
            print(f"  尝试从attention中推断生成的token...")
            # 无法从attention中直接获取token，但可以处理attention步骤
            # 这种情况下，我们只能处理attention，无法显示对应的token
            print(f"  将处理 {num_attention_steps} 个attention步骤，但无法显示对应的token文本")
        else:
            print(f"  将只处理前 {min(num_attention_steps, num_generated_tokens)} 个步骤")

    print(f"{'='*80}\n")

    # 如果生成的tokens为空，使用步骤索引作为标识
    if len(valid_tokens) == 0:
        print(f"⚠️  警告: 没有有效的生成tokens，将使用步骤索引作为标识")

    # 为每个生成步骤创建输出目录
    generation_output_dir = os.path.join(output_dir, "generation_attention")
    os.makedirs(generation_output_dir, exist_ok=True)

    # 处理每个生成步骤的attention
    if all_attentions is None or len(all_attentions) == 0:
        print(f"⚠️  没有attention信息")
        return generation_output_dir

    # 累积生成的文本，用于显示每次循环后的完整文本
    accumulated_text = ""

    # 确定要处理的步骤数 - 使用过滤后的有效token IDs
    # 优先使用实际的token数量，如果没有token但有attention，使用attention数量
    if len(valid_generated_token_ids) > 0:
        num_steps_to_process = min(len(all_attentions) if all_attentions is not None else 0, len(valid_generated_token_ids))
    elif all_attentions is not None and len(all_attentions) > 0:
        num_steps_to_process = len(all_attentions)
        print(f"⚠️  无法获取生成的token，但检测到 {num_steps_to_process} 个attention步骤，将处理这些步骤")
    else:
        num_steps_to_process = 0
        print(f"⚠️  没有可处理的步骤")

    for step_idx in range(num_steps_to_process):
        if generated_token_ids[step_idx] == 0 or generated_token_ids[step_idx] == 1 or generated_token_ids[step_idx] == 2:
            continue
        step_attentions = all_attentions[step_idx] if all_attentions is not None and step_idx < len(all_attentions) else None

        # step_attentions是一个tuple，包含所有层的attention
        # 每个元素是一个tensor，shape是 [batch, num_heads, seq_len, seq_len]
        if step_attentions is None or len(step_attentions) == 0:
            print(f"  ⚠️  步骤 {step_idx} 没有attention信息")
            continue

        # 获取当前步骤的token信息 - 使用过滤后的有效token IDs
        if step_idx < len(valid_generated_token_ids) and len(valid_generated_token_ids) > 0:
            token_id = valid_generated_token_ids[step_idx]
            try:
                current_token = tokenizer.decode([token_id])
            except Exception as e:
                current_token = f"[无效token: {token_id}]"
        else:
            # 如果无法获取token，尝试从完整序列中推断
            # 在生成过程中，第step_idx个token应该在位置 input_len + step_idx
            if step_idx < len(valid_full_output_ids) - input_len:
                token_id = valid_full_output_ids[input_len + step_idx]
                try:
                    current_token = tokenizer.decode([token_id])
                    print(f"  ✓ 从完整序列中提取步骤 {step_idx} 的token: '{current_token}' (ID: {token_id})")
                except Exception as e:
                    current_token = f"[无效token: {token_id}]"
                    print(f"  ⚠️  步骤 {step_idx} 的token ID {token_id} 无效")
            else:
                # 如果仍然无法获取，使用占位符
                token_id = None
                current_token = f"[Token_{step_idx}_Unknown]"
                print(f"  ⚠️  步骤 {step_idx} 无法获取对应的token，使用占位符")

        # 累积文本（用于显示每次循环后的完整生成文本）
        accumulated_text += current_token

        # 解码当前步骤之前的所有token（包括input和已生成的output）
        # 在生成过程中，每一步的序列长度是 input_len + step_idx + 1
        current_sequence_ids = valid_full_output_ids[:input_len + step_idx + 1]
        current_sequence_text = safe_decode_token_ids(tokenizer, current_sequence_ids, skip_special_tokens=True)

        # 清理token文本用于文件名（移除特殊字符，限制长度）
        token_name = current_token.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('<', '').replace('>', '').replace('"', '').replace("'", '').replace('\n', '_').replace('\r', '_')
        # 限制文件名长度，避免过长
        if len(token_name) > 50:
            token_name = token_name[:50]
        if not token_name or token_name.strip() == '':
            token_name = f"token_{token_id}"

        print(f"\n{'='*60}")
        print(f"【生成步骤 {step_idx+1}/{num_steps_to_process} - LLaVA生成词汇】")
        print(f"  当前步骤预测的新词: '{current_token}'")
        print(f"  当前步骤Token ID: {token_id}")
        print(f"  累积生成的文本(仅新生成部分): '{accumulated_text}'")
        print(f"  完整序列文本(包含input+已生成的output): '{current_sequence_text}'")
        print(f"  文件名标识: {token_name}")
        print(f"{'='*60}")

        num_layers = len(step_attentions)
        print(f"  层数: {num_layers}")

        # 为这个步骤创建目录
        step_dir = os.path.join(generation_output_dir, f"step_{step_idx}_{token_name}")
        os.makedirs(step_dir, exist_ok=True)

        # 处理每一层的attention
        for layer_idx, layer_attn in enumerate(step_attentions):
            if layer_attn is None:
                continue

            # 检查是否是目标层
            if target_layers is not None and layer_idx not in target_layers:
                continue

            # 处理attention tensor的形状
            if isinstance(layer_attn, tuple):
                layer_attn = layer_attn[0]

            # 确保是tensor
            if not isinstance(layer_attn, torch.Tensor):
                print(f"  ⚠️  Layer {layer_idx}: attention不是tensor，类型: {type(layer_attn)}")
                continue

            layer_attn_np = layer_attn.cpu().numpy()

            # 处理attention tensor的形状
            # 在生成过程中，attention的形状可能是:
            # - [batch, num_heads, 1, seq_len] (只有最后一个token的attention)
            # - [num_heads, 1, seq_len]
            # - [1, seq_len] (已经平均过head)
            # - [batch, num_heads, seq_len, seq_len] (完整的attention矩阵)
            print(f"  [Layer {layer_idx}] 原始attention形状: {layer_attn_np.shape}")

            if len(layer_attn_np.shape) == 4:
                # [batch, num_heads, query_len, key_len]
                batch_size, num_heads, query_len, key_len = layer_attn_np.shape
                if query_len == 1:
                    # 只有最后一个token的attention: [batch, num_heads, 1, key_len]
                    last_row_attention = layer_attn_np[0].mean(axis=0).squeeze()  # [key_len]
                    seq_len = key_len
                else:
                    # 完整的attention矩阵: [batch, num_heads, seq_len, seq_len]
                    layer_attn_np = layer_attn_np[0].mean(axis=0)  # [seq_len, seq_len]
                    seq_len = layer_attn_np.shape[0]
                    last_row_attention = layer_attn_np[-1, :]  # [seq_len]
            elif len(layer_attn_np.shape) == 3:
                # [num_heads, query_len, key_len]
                num_heads, query_len, key_len = layer_attn_np.shape
                if query_len == 1:
                    # 只有最后一个token的attention: [num_heads, 1, key_len]
                    last_row_attention = layer_attn_np.mean(axis=0).squeeze()  # [key_len]
                    seq_len = key_len
                else:
                    # 完整的attention矩阵: [num_heads, seq_len, seq_len]
                    layer_attn_np = layer_attn_np.mean(axis=0)  # [seq_len, seq_len]
                    seq_len = layer_attn_np.shape[0]
                    last_row_attention = layer_attn_np[-1, :]  # [seq_len]
            elif len(layer_attn_np.shape) == 2:
                # [query_len, key_len] 或 [1, key_len]
                query_len, key_len = layer_attn_np.shape
                if query_len == 1:
                    # 只有最后一个token的attention: [1, key_len]
                    last_row_attention = layer_attn_np[0, :]  # [key_len]
                    seq_len = key_len
                else:
                    # 完整的attention矩阵: [seq_len, seq_len]
                    seq_len = query_len
                    last_row_attention = layer_attn_np[-1, :]  # [seq_len]
            elif len(layer_attn_np.shape) == 1:
                # 已经是 [seq_len]，直接使用
                last_row_attention = layer_attn_np
                seq_len = len(layer_attn_np)
            else:
                print(f"  ⚠️  Layer {layer_idx}: 意外的attention形状 {layer_attn_np.shape}")
                continue

            print(f"  [Layer {layer_idx}] 序列长度: {seq_len}, 最后一行attention形状: {last_row_attention.shape}, 范围: [{last_row_attention.min():.4f}, {last_row_attention.max():.4f}], 和: {last_row_attention.sum():.4f}")

            # 确定图像token数量（优先使用实际值，如果没有则使用默认576）
            actual_num_image_tokens = num_image_tokens if num_image_tokens > 0 else 576
            image_token_start = image_first_pos  # 跳过BOS token（位置0）
            image_token_end = image_token_start + actual_num_image_tokens

            # 确保不超过序列长度
            if image_token_end > seq_len:
                print(f"  ⚠️  Layer {layer_idx}: 图像token范围超出序列长度，调整: [{image_token_start}, {seq_len-1}]")
                image_token_end = seq_len
                actual_num_image_tokens = image_token_end - image_token_start

            # 提取图像token位置
            valid_image_positions = np.arange(image_token_start, image_token_end)
            print(f"  [Layer {layer_idx}] 图像token位置范围: [{image_token_start}, {image_token_end-1}], 数量: {len(valid_image_positions)}")

            # 提取对图像token的attention值
            image_attention = last_row_attention[valid_image_positions]  # [len(valid_image_positions)]
            # print(f"  [Layer {layer_idx}] 提取的图像attention形状: {image_attention.shape}, 范围: [{image_attention.min():.4f}, {image_attention.max():.4f}]")

            # 确保有576个值（如果不足则填充，如果过多则截断）
            if len(image_attention) < 576:
                # 填充到576
                padding = 576 - len(image_attention)
                image_attention = np.pad(image_attention, (0, padding), mode='constant', constant_values=0)
                print(f"  [Layer {layer_idx}] 图像attention数量不足，已填充: {len(image_attention) - padding} -> 576")
            elif len(image_attention) > 576:
                # 截断到576
                image_attention = image_attention[:576]
                print(f"  [Layer {layer_idx}] 图像attention数量过多，已截断到576")

            # Reshape到24×24（576 = 24×24）
            patch_size = 24
            if len(image_attention) != 576:
                print(f"  ⚠️  Layer {layer_idx}: 图像attention数量不是576，当前: {len(image_attention)}")
                # 确保是576
                if len(image_attention) < 576:
                    image_attention = np.pad(image_attention, (0, 576 - len(image_attention)), mode='constant', constant_values=0)
                else:
                    image_attention = image_attention[:576]

            # Reshape到24×24
            attention_map = image_attention.reshape(patch_size, patch_size)
            # print(f"  [Layer {layer_idx}] Reshape后的attention map形状: {attention_map.shape}")

            # 可视化并映射到原图（如果启用）
            if save_attention_maps:
                visualize_step_attention_map(
                    attention_map, image, layer_idx, step_idx, current_token, token_name,
                    patch_size, step_dir
                )
            else:
                print(f"  [Layer {layer_idx}] Attention map已提取（未保存图片）")

        print(f"  ✓ 步骤 {step_idx+1} 的所有层attention已保存到: {step_dir}")

    print(f"\n✓ 所有生成步骤的attention map已保存到: {generation_output_dir}")
    return generation_output_dir


import numpy as np
import matplotlib.pyplot as plt

def plot_attention_pixel_grid(attn_matrix, layer_idx, ax=None, vmax=0.7):
    """
    清晰地绘制注意力矩阵的像素网格

    参数:
    - attn_matrix: 注意力矩阵 (24, 24)
    - layer_idx: 层索引
    - ax: matplotlib轴对象，如果为None则创建新图
    - vmax: 颜色最大值，控制颜色范围

    返回:
    - 绘图轴对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # 确保输入是24x24
    assert attn_matrix.shape == (24, 24), f"矩阵形状应为(24,24)，实际是{attn_matrix.shape}"

    # 使用imshow，但调整参数确保像素清晰
    im = ax.imshow(attn_matrix,
                  cmap='jet',
                  alpha=0.7,
                  interpolation='nearest',  # 关键：最近邻插值
                  aspect='equal',          # 保持纵横比
                  vmin=0, vmax=vmax,
                  extent=[0, 24, 24, 0])  # 定义显示范围，y轴从上到下

    # # 添加网格线，使像素边界清晰
    # ax.set_xticks(np.arange(0, 25, 1))
    # ax.set_yticks(np.arange(0, 25, 1))
    # ax.grid(True, color='black', linewidth=0.5, alpha=0.3)

    # 设置标题
    ax.set_title(f'Layer {layer_idx}: Attention Pixel Grid',
                fontsize=14, fontweight='bold', pad=20)

    # 添加坐标轴标签（可选）
    ax.set_xlabel('X Position', fontsize=10)
    ax.set_ylabel('Y Position', fontsize=10)

    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Strength', rotation=270, labelpad=15)

    return ax


def visualize_step_attention_map(attention_map, image, layer_idx, step_idx, token_text, token_name,
                                patch_size, output_dir):
    """可视化单个步骤、单层的attention map并映射到原图

    Args:
        attention_map: 24x24的attention map
        image: 原始图像
        layer_idx: 层索引
        step_idx: 生成步骤索引
        token_text: token的文本内容（用于显示）
        token_name: token的清理后的名称（用于文件名）
        patch_size: patch大小（24）
        output_dir: 输出目录
    """
    try:
        # 归一化attention map
        attn_min = attention_map.min()
        attn_max = attention_map.max()
        attn_range = attn_max - attn_min

        if attn_range < 1e-6:
            # 如果值范围太小，使用softmax归一化
            attn_exp = np.exp((attention_map - attention_map.max()) * 10)
            attn_normalized = attn_exp / attn_exp.sum()
        else:
            # 使用percentile归一化
            attn_min = np.percentile(attention_map, 5)
            attn_max = np.percentile(attention_map, 95)
            attn_range = attn_max - attn_min
            if attn_range < 1e-6:
                attn_normalized = np.ones_like(attention_map) / attention_map.size
            else:
                attn_normalized = np.clip((attention_map - attn_min) / attn_range, 0, 1)
                # Gamma校正增强对比度
                attn_normalized = np.power(attn_normalized, 0.3)

        # 上采样到图像大小
        try:
            from scipy.ndimage import zoom
            zoom_factors = (image.size[1] / patch_size, image.size[0] / patch_size)
            if zoom_factors[0] > 0 and zoom_factors[1] > 0:
                attention_map_upsampled = zoom(attn_normalized, zoom_factors, order=1)
            else:
                raise ValueError(f"Invalid zoom factors: {zoom_factors}")
        except (ImportError, ValueError, Exception):
            import torch.nn.functional as F
            attention_tensor = torch.from_numpy(attn_normalized).unsqueeze(0).unsqueeze(0).float()
            attention_tensor = F.interpolate(
                attention_tensor,
                size=(image.size[1], image.size[0]),
                mode='bilinear',
                align_corners=False
            )
            attention_map_upsampled = attention_tensor.squeeze().numpy()

        # 确保形状正确
        if attention_map_upsampled.shape[0] != image.size[1] or attention_map_upsampled.shape[1] != image.size[0]:
            import torch.nn.functional as F
            attention_tensor = torch.from_numpy(attention_map_upsampled).unsqueeze(0).unsqueeze(0).float()
            attention_tensor = F.interpolate(
                attention_tensor,
                size=(image.size[1], image.size[0]),
                mode='bilinear',
                align_corners=False
            )
            attention_map_upsampled = attention_tensor.squeeze().numpy()

        # 创建多种可视化
        fig = plt.figure(figsize=(24, 6))

        # 1. 原图
        ax1 = plt.subplot(1, 4, 1)
        ax1.imshow(image)
        ax1.set_title(f'Original Image, Step {step_idx+1}, Layer {layer_idx}, Token: "{token_text}"',
                     fontsize=12, fontweight='bold')
        ax1.axis('off')

        # 2. Attention heatmap (独立)
        ax2 = plt.subplot(1, 4, 2)
        im2 = ax2.imshow(attention_map_upsampled, cmap='hot', interpolation='bilinear', vmin=0, vmax=1)
        ax2.set_title(f'Attention Heatmap, Layer {layer_idx}', fontsize=12, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # 3. Jet overlay
        ax3 = plt.subplot(1, 4, 3)
        ax3.imshow(image)
        im3 = ax3.imshow(attention_map_upsampled, cmap='jet', alpha=0.7, interpolation='bilinear', vmin=0, vmax=1)
        ax3.set_title(f'Jet Overlay, Layer {layer_idx}', fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # # 4. Jet
        # ax4 = plt.subplot(1, 4, 4)
        # im4 = ax4.imshow(attn_normalized, cmap='jet', alpha=0.7, interpolation='bilinear', vmin=0, vmax=1)
        # ax4.set_title(f'Normalized Attention\nLayer {layer_idx}', fontsize=12, fontweight='bold')
        # ax4.axis('off')
        # plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        # 4. Jet overlay
        ax4 = plt.subplot(1, 4, 4)

        # 方法1: 截取jet颜色映射的前70%（深蓝到淡黄）
        jet_cmap = cm.get_cmap('jet')
        # 截取0-0.7的颜色范围
        colors = jet_cmap(np.linspace(0, 0.7, 256))
        new_cmap = mcolors.LinearSegmentedColormap.from_list('truncated_jet', colors)

        im4 = ax4.imshow(attn_normalized, cmap=new_cmap, alpha=0.7,
                        interpolation='bilinear', vmin=0, vmax=1)
        ax4.set_title(f'Normalized Attention, Layer {layer_idx}',
                    fontsize=12, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        # # 4. Jet overlay
        # ax4 = plt.subplot(1, 4, 4)
        # plot_attention_pixel_grid(attn_normalized, layer_idx, ax=ax4, vmax=0.8)


        # # 4. Hot overlay
        # ax4 = plt.subplot(3, 3, 4)
        # ax4.imshow(image)
        # im4 = ax4.imshow(attention_map_upsampled, cmap='hot', alpha=0.6, interpolation='bilinear', vmin=0, vmax=1)
        # ax4.set_title(f'Hot Overlay\nLayer {layer_idx}', fontsize=12, fontweight='bold')
        # ax4.axis('off')
        # plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        # # 5. Top 20% attention
        # ax5 = plt.subplot(3, 3, 5)
        # ax5.imshow(image)
        # threshold_80 = np.percentile(attention_map_upsampled, 80)
        # attn_thresh_80 = np.where(attention_map_upsampled >= threshold_80, attention_map_upsampled, 0)
        # attn_thresh_80_norm = (attn_thresh_80 - attn_thresh_80.min()) / (attn_thresh_80.max() - attn_thresh_80.min() + 1e-10)
        # im5 = ax5.imshow(attn_thresh_80_norm, cmap='Reds', alpha=0.8, interpolation='bilinear', vmin=0, vmax=1)
        # ax5.set_title(f'Top 20% Attention\nLayer {layer_idx}', fontsize=12, fontweight='bold')
        # ax5.axis('off')
        # plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

        # # 6. Top 10% attention
        # ax6 = plt.subplot(3, 3, 6)
        # ax6.imshow(image)
        # threshold_90 = np.percentile(attention_map_upsampled, 90)
        # attn_thresh_90 = np.where(attention_map_upsampled >= threshold_90, attention_map_upsampled, 0)
        # attn_thresh_90_norm = (attn_thresh_90 - attn_thresh_90.min()) / (attn_thresh_90.max() - attn_thresh_90.min() + 1e-10)
        # im6 = ax6.imshow(attn_thresh_90_norm, cmap='YlOrRd', alpha=0.9, interpolation='bilinear', vmin=0, vmax=1)
        # ax6.set_title(f'Top 10% Attention\nLayer {layer_idx}', fontsize=12, fontweight='bold')
        # ax6.axis('off')
        # plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

        # # 7. Contour
        # ax7 = plt.subplot(3, 3, 7)
        # ax7.imshow(image)
        # y_coords = np.linspace(0, image.size[1]-1, attention_map_upsampled.shape[0])
        # x_coords = np.linspace(0, image.size[0]-1, attention_map_upsampled.shape[1])
        # X, Y = np.meshgrid(x_coords, y_coords)
        # contour = ax7.contour(X, Y, attention_map_upsampled, levels=5, colors='yellow', linewidths=3, alpha=0.9)
        # ax7.clabel(contour, inline=True, fontsize=10, fmt='%.3f', colors='white')
        # ax7.set_title(f'Contour Map\nLayer {layer_idx}', fontsize=12, fontweight='bold')
        # ax7.axis('off')

        # # 8. YlOrRd overlay
        # ax8 = plt.subplot(3, 3, 8)
        # ax8.imshow(image)
        # im8 = ax8.imshow(attention_map_upsampled, cmap='YlOrRd', alpha=0.65, interpolation='bilinear', vmin=0, vmax=1)
        # ax8.set_title(f'YlOrRd Overlay\nLayer {layer_idx}', fontsize=12, fontweight='bold')
        # ax8.axis('off')
        # plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

        # # 9. Plasma overlay
        # ax9 = plt.subplot(3, 3, 9)
        # ax9.imshow(image)
        # im9 = ax9.imshow(attention_map_upsampled, cmap='plasma', alpha=0.7, interpolation='bilinear', vmin=0, vmax=1)
        # ax9.set_title(f'Plasma Overlay\nLayer {layer_idx}', fontsize=12, fontweight='bold')
        # ax9.axis('off')
        # plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)

        plt.tight_layout()
        # 在文件名中包含token信息
        output_file = os.path.join(output_dir, f"layer_{layer_idx}_token_{token_name}_attention.png")
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Layer {layer_idx} attention map已保存: {os.path.basename(output_file)}")

    except Exception as e:
        print(f"  ⚠️  Layer {layer_idx}: 可视化attention map时出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    args = parse_args()

    # 设置输出目录
    if args.output_dir is None:
        output_dir = os.path.join(current_dir, "output", f"attention_analysis")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("LLaVA 模型架构分析和 Attention 可视化")
    print("=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"设备: {args.device}")
    print(f"图像: {args.image_file}")
    print(f"提示词: {args.prompt}")
    print(f"输出目录: {output_dir}")
    print("=" * 80)

    # 加载模型
    print("\n[1/4] 正在加载模型...")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=model_name,
        device=args.device,
        device_map=args.device
    )
    print(f"✓ 模型加载完成: {model_name}")

    # 提取 hidden states 和 attention
    print("\n[3/4] 提取 Hidden States 和 Attention...")
    hidden_states, attentions, vision_hidden, input_ids, image, image_token_positions = extract_hidden_states_and_attention(
        model, tokenizer, image_processor, args.image_file, args.prompt,
        args.conv_mode, args.device, output_dir
    )

    # 解析目标层
    target_layers = None
    if args.target_layers:
        if args.target_layers.lower() == 'even':
            if hasattr(model, 'get_model') and hasattr(model.get_model(), 'layers'):
                num_layers = len(model.get_model().layers)
                target_layers = list(range(0, num_layers, 2))
        elif args.target_layers.lower() == 'odd':
            if hasattr(model, 'get_model') and hasattr(model.get_model(), 'layers'):
                num_layers = len(model.get_model().layers)
                target_layers = list(range(1, num_layers, 2))
        else:
            target_layers = [int(x.strip()) for x in args.target_layers.split(',')]
    else:
        # 默认偶数层
        if hasattr(model, 'get_model') and hasattr(model.get_model(), 'layers'):
            num_layers = len(model.get_model().layers)
            target_layers = list(range(0, num_layers, 2))

    # 提取 Q、K 信息并可视化
    if not args.extract_generation_attention:
        print("\n[4/4] 提取 Q、K 信息并可视化...")
        extract_qk_information(
            model, tokenizer, image_processor, args.image_file, args.prompt,
            args.conv_mode, args.device, output_dir, target_layers
        )
    else:
        # 提取生成过程中的attention map
        print("\n[4/4] 提取生成过程中的 Attention Map...")
        if not args.save_attention_maps:
            print("  ⚠️  注意: 已禁用保存attention map图片，只会提取数据")
        extract_attention_during_generation(
            model, tokenizer, image_processor, args.image_file, args.prompt,
            args.conv_mode, args.device, output_dir, target_layers,
            args.max_new_tokens, save_attention_maps=args.save_attention_maps
        )

    print("\n" + "=" * 80)
    print("✓ 分析完成！")
    print(f"所有结果已保存到: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
