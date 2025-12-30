"""
LLaVA v1.5 7B 模型架构分析和 Attention 可视化脚本

功能说明：
1. 输出 LLaVA 整个网络架构的不同层的基本信息（网络类型、输入输出 size、参数量等）
2. 输出文本或视觉的 hidden state，hidden layer 的 size，以及二者 attention 的信息
3. 输出特定层（如所有偶数层或奇数层）的 transformer 层的 Q、K 信息，生成 heatmap
4. 将 heatmap 映射到原图上的结果
5. 所有结果保存在 tests/output/ 路径中
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


def load_image(image_file):
    """加载图像文件，支持本地文件和 URL"""
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def analyze_model_architecture(model, tokenizer, output_dir):
    """分析模型架构，输出各层信息"""
    print("\n" + "=" * 80)
    print("模型架构分析")
    print("=" * 80)

    arch_info = {
        "model_type": type(model).__name__,
        "total_parameters": 0,
        "trainable_parameters": 0,
        "components": {}
    }

    # 计算总参数量
    total_params, trainable_params = count_parameters(model)
    arch_info["total_parameters"] = total_params
    arch_info["trainable_parameters"] = trainable_params

    print(f"模型类型: {arch_info['model_type']}")
    print(f"总参数量: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"可训练参数量: {trainable_params:,} ({trainable_params / 1e9:.2f}B)")

    # 分析 Vision Tower
    if hasattr(model, 'get_vision_tower'):
        vision_tower = model.get_vision_tower()
        if vision_tower is not None:
            vision_params, _ = count_parameters(vision_tower)
            arch_info["components"]["vision_tower"] = {
                "type": type(vision_tower).__name__,
                "parameters": vision_params,
                "hidden_size": getattr(vision_tower, 'hidden_size', 'N/A') if hasattr(vision_tower, 'hidden_size') else 'N/A'
            }
            print(f"\nVision Tower:")
            print(f"  类型: {type(vision_tower).__name__}")
            print(f"  参数量: {vision_params:,}")

    # 分析 MM Projector
    if hasattr(model, 'mm_projector'):
        projector = model.mm_projector
        if projector is not None:
            projector_params, _ = count_parameters(projector)
            arch_info["components"]["mm_projector"] = {
                "type": type(projector).__name__,
                "parameters": projector_params
            }
            print(f"\nMM Projector:")
            print(f"  类型: {type(projector).__name__}")
            print(f"  参数量: {projector_params:,}")

    # 分析 Language Model
    if hasattr(model, 'get_model'):
        lang_model = model.get_model()
        if lang_model is not None:
            lang_params, _ = count_parameters(lang_model)
            arch_info["components"]["language_model"] = {
                "type": type(lang_model).__name__,
                "parameters": lang_params
            }
            print(f"\nLanguage Model:")
            print(f"  类型: {type(lang_model).__name__}")
            print(f"  参数量: {lang_params:,}")

            # 分析 Transformer Layers
            if hasattr(lang_model, 'layers'):
                layers = lang_model.layers
                arch_info["components"]["transformer_layers"] = {
                    "num_layers": len(layers),
                    "layers": []
                }
                print(f"\nTransformer Layers: {len(layers)} 层")

                for i, layer in enumerate(layers):
                    layer_params, _ = count_parameters(layer)
                    layer_info = {
                        "layer_index": i,
                        "type": type(layer).__name__,
                        "parameters": layer_params
                    }

                    # 获取层配置信息
                    if hasattr(layer, 'self_attn'):
                        attn = layer.self_attn
                        if hasattr(attn, 'num_heads'):
                            layer_info["num_heads"] = attn.num_heads
                        if hasattr(attn, 'hidden_size'):
                            layer_info["hidden_size"] = attn.hidden_size
                        if hasattr(attn, 'head_dim'):
                            layer_info["head_dim"] = attn.head_dim

                    arch_info["components"]["transformer_layers"]["layers"].append(layer_info)
                    print(f"  Layer {i}: {type(layer).__name__}, 参数量: {layer_params:,}")

    # 分析 LM Head
    if hasattr(model, 'lm_head'):
        lm_head = model.lm_head
        if lm_head is not None:
            head_params, _ = count_parameters(lm_head)
            arch_info["components"]["lm_head"] = {
                "type": type(lm_head).__name__,
                "parameters": head_params,
                "in_features": lm_head.in_features if hasattr(lm_head, 'in_features') else 'N/A',
                "out_features": lm_head.out_features if hasattr(lm_head, 'out_features') else 'N/A'
            }
            print(f"\nLM Head:")
            print(f"  类型: {type(lm_head).__name__}")
            print(f"  参数量: {head_params:,}")

    # 保存架构信息
    arch_file = os.path.join(output_dir, "model_architecture.json")
    with open(arch_file, 'w', encoding='utf-8') as f:
        json.dump(arch_info, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 架构信息已保存到: {arch_file}")

    return arch_info


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

    # 找到图像 token 占位符的位置（在 input_ids 中）
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
                print(f"    图像 token 数量（计算）: {len(image_token_positions)}")
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

    print(f"\nHidden States:")
    print(f"  层数: {len(hidden_states)}")
    for i, h in enumerate(hidden_states):
        print(f"  Layer {i}: {h.shape}")

    print(f"\nAttention Weights:")
    print(f"  层数: {len(attentions)}")
    for i, attn in enumerate(attentions):
        if attn is not None:
            print(f"  Layer {i}: {attn.shape if isinstance(attn, torch.Tensor) else 'N/A'}")

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
    print(f"  实际序列长度（处理后）: {actual_seq_len}")

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
            print(f"  ⚠️  Layer {layer_idx}: 无法提取文本->图像 attention（缺少文本或图像 token）")

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
    1. 所有文本 token 对图像 token 的 attention 的最大值（突出最关注的区域）
    2. 所有文本 token 对图像 token 的 attention 的平均值
    3. 使用 top-k 文本 token 的 attention（关注最重要的 token）
    4. 图像 token 之间的 self-attention
    """
    if image is None or image_token_positions is None or len(image_token_positions) == 0:
        print(f"  ⚠️  Layer {layer_idx}: 无法映射 attention 到图像（缺少图像或图像 token 位置）")
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

        # 策略 1: 使用最大值（突出最关注的区域）- 更明显
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

            # 组合策略：使用最大值和加权平均的组合（突出高关注区域）
            image_attention_from_text = 0.6 * image_attention_max + 0.4 * image_attention_weighted

            print(f"  [Layer {layer_idx}] 策略1: 使用 {len(text_positions)} 个文本 token 的最大值+加权平均组合")
            print(f"    最大值范围: [{image_attention_max.min():.4f}, {image_attention_max.max():.4f}]")
            print(f"    平均值范围: [{image_attention_mean.min():.4f}, {image_attention_mean.max():.4f}]")
        else:
            image_attention_from_text = None

        # 策略 2: 使用图像 token 之间的 self-attention（如果策略1不可用）
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
            print(f"  [Layer {layer_idx}] 使用 softmax 归一化（值范围: {attn_range:.2e}）")
        else:
            # 使用 percentile-based min-max 归一化，然后应用 gamma 校正增强对比度
            image_only_attention = np.clip((image_attention - attn_min) / attn_range, 0, 1)
            # Gamma 校正：增强高关注区域的对比度
            gamma = 0.5  # gamma < 1 会增强高值
            image_only_attention = np.power(image_only_attention, gamma)
            print(f"  [Layer {layer_idx}] 使用 percentile + gamma 归一化（范围: [{attn_min:.4f}, {attn_max:.4f}], gamma={gamma}）")

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

        # 5. Top 20% 高关注区域（更严格的阈值）
        ax5 = plt.subplot(3, 3, 5)
        ax5.imshow(image)
        threshold_80 = np.percentile(attention_map_upsampled, 80)
        attention_thresholded_80 = np.where(attention_map_upsampled >= threshold_80, attention_map_upsampled, 0)
        attn_thresh_norm = (attention_thresholded_80 - attention_thresholded_80.min()) / (attention_thresholded_80.max() - attention_thresholded_80.min() + 1e-10)
        im5 = ax5.imshow(attn_thresh_norm, cmap='Reds', alpha=0.8, interpolation='bilinear', vmin=0, vmax=1)
        ax5.set_title(f'Top 20% Attention - Layer {layer_idx}', fontsize=14, fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

        # 6. Top 10% 高关注区域（最严格的阈值）
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

    # 找到图像 token 占位符的位置（在 input_ids 中）
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

    # 为目标层注册 hooks（直接在 q_proj 和 k_proj 上注册）
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
    parser.add_argument("--image-file", type=str,
                       default="/home/liying/Documents/dataset/coco/val2014/COCO_val2014_000000065883.jpg",
                       help="图像文件路径")
    parser.add_argument("--prompt", type=str, default="there is a boy with blond hair and blue eyes, is this discription correct? Yes or No.",
                       help="提示词")

    # 分析参数
    parser.add_argument("--target-layers", type=str, default=None,
                       help="目标层索引，用逗号分隔，如 '0,2,4' 或 'even' 表示偶数层，'odd' 表示奇数层")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="输出目录（默认为 tests/output）")

    return parser.parse_args()


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

    # 分析模型架构
    print("\n[2/4] 分析模型架构...")
    arch_info = analyze_model_architecture(model, tokenizer, output_dir)

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
    print("\n[4/4] 提取 Q、K 信息并可视化...")
    extract_qk_information(
        model, tokenizer, image_processor, args.image_file, args.prompt,
        args.conv_mode, args.device, output_dir, target_layers
    )

    print("\n" + "=" * 80)
    print("✓ 分析完成！")
    print(f"所有结果已保存到: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
