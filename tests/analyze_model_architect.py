"""
模型架构分析和推理维度追踪脚本

功能：
1. 详细分析 LLaVA 模型的架构，包括各组件参数、配置信息
2. 追踪推理过程中数据维度的变化（从输入到输出的完整流程）
3. 输出详细的 JSON 报告
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
from llava.utils import disable_torch_init

import project

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




def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def analyze_model_architecture(model, tokenizer, output_dir):
    """详细分析模型架构，输出各层信息"""
    print("\n" + "=" * 80)
    print("模型架构详细分析")
    print("=" * 80)

    arch_info = {
        "model_type": type(model).__name__,
        "model_config": {},
        "total_parameters": 0,
        "trainable_parameters": 0,
        "components": {}
    }

    # 获取模型配置
    if hasattr(model, 'config'):
        config = model.config
        arch_info["model_config"] = {
            "vocab_size": getattr(config, 'vocab_size', 'N/A'),
            "hidden_size": getattr(config, 'hidden_size', 'N/A'),
            "intermediate_size": getattr(config, 'intermediate_size', 'N/A'),
            "num_hidden_layers": getattr(config, 'num_hidden_layers', 'N/A'),
            "num_attention_heads": getattr(config, 'num_attention_heads', 'N/A'),
            "max_position_embeddings": getattr(config, 'max_position_embeddings', 'N/A'),
            "mm_use_im_start_end": getattr(config, 'mm_use_im_start_end', False),
            "mm_vision_tower": getattr(config, 'mm_vision_tower', 'N/A'),
            "mm_projector_type": getattr(config, 'mm_projector_type', 'N/A'),
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
            vision_info = {
                "type": type(vision_tower).__name__,
                "parameters": vision_params,
                "hidden_size": getattr(vision_tower, 'hidden_size', 'N/A') if hasattr(vision_tower, 'hidden_size') else 'N/A'
            }

            # 获取 Vision Tower 的详细配置
            if hasattr(vision_tower, 'config'):
                v_config = vision_tower.config
                vision_info["config"] = {
                    "image_size": getattr(v_config, 'image_size', 'N/A'),
                    "patch_size": getattr(v_config, 'patch_size', 'N/A'),
                    "num_patches": getattr(v_config, 'num_patches', 'N/A'),
                    "hidden_size": getattr(v_config, 'hidden_size', 'N/A'),
                    "num_hidden_layers": getattr(v_config, 'num_hidden_layers', 'N/A'),
                    "num_attention_heads": getattr(v_config, 'num_attention_heads', 'N/A'),
                }

            arch_info["components"]["vision_tower"] = vision_info
            print(f"\nVision Tower:")
            print(f"  类型: {type(vision_tower).__name__}")
            print(f"  参数量: {vision_params:,}")
            if "config" in vision_info:
                print(f"  图像尺寸: {vision_info['config'].get('image_size', 'N/A')}")
                print(f"  Patch大小: {vision_info['config'].get('patch_size', 'N/A')}")
                print(f"  Patch数量: {vision_info['config'].get('num_patches', 'N/A')}")

    # 分析 MM Projector
    lang_model = model.get_model() if hasattr(model, 'get_model') else model
    if hasattr(lang_model, 'mm_projector'):
        projector = lang_model.mm_projector
        if projector is not None:
            projector_params, _ = count_parameters(projector)
            projector_info = {
                "type": type(projector).__name__,
                "parameters": projector_params,
                "layers": []
            }

            # 分析 projector 的每一层
            if isinstance(projector, nn.Sequential):
                for i, layer in enumerate(projector):
                    layer_params, _ = count_parameters(layer)
                    layer_info = {
                        "layer_index": i,
                        "type": type(layer).__name__,
                        "parameters": layer_params
                    }
                    if isinstance(layer, nn.Linear):
                        layer_info["in_features"] = layer.in_features
                        layer_info["out_features"] = layer.out_features
                    projector_info["layers"].append(layer_info)

            arch_info["components"]["mm_projector"] = projector_info
            print(f"\nMM Projector:")
            print(f"  类型: {type(projector).__name__}")
            print(f"  参数量: {projector_params:,}")
            if projector_info["layers"]:
                print(f"  层数: {len(projector_info['layers'])}")
                for layer in projector_info["layers"]:
                    if "in_features" in layer:
                        print(f"    Layer {layer['layer_index']}: {layer['type']} ({layer['in_features']} -> {layer['out_features']})")

    # 分析 Language Model
    if hasattr(model, 'get_model'):
        lang_model = model.get_model()
        if lang_model is not None:
            lang_params, _ = count_parameters(lang_model)
            lang_info = {
                "type": type(lang_model).__name__,
                "parameters": lang_params
            }

            # 获取 embedding 信息
            if hasattr(lang_model, 'embed_tokens'):
                embed = lang_model.embed_tokens
                embed_params, _ = count_parameters(embed)
                lang_info["embed_tokens"] = {
                    "type": type(embed).__name__,
                    "parameters": embed_params,
                    "num_embeddings": embed.num_embeddings if hasattr(embed, 'num_embeddings') else 'N/A',
                    "embedding_dim": embed.embedding_dim if hasattr(embed, 'embedding_dim') else 'N/A'
                }

            # 获取 norm 信息
            if hasattr(lang_model, 'norm'):
                norm = lang_model.norm
                norm_params, _ = count_parameters(norm)
                lang_info["norm"] = {
                    "type": type(norm).__name__,
                    "parameters": norm_params,
                    "normalized_shape": getattr(norm, 'normalized_shape', 'N/A') if hasattr(norm, 'normalized_shape') else 'N/A'
                }

            arch_info["components"]["language_model"] = lang_info
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
                        "parameters": layer_params,
                        "submodules": {}
                    }

                    # 获取 Self-Attention 信息
                    if hasattr(layer, 'self_attn'):
                        attn = layer.self_attn
                        attn_params, _ = count_parameters(attn)
                        attn_info = {
                            "type": type(attn).__name__,
                            "parameters": attn_params
                        }
                        if hasattr(attn, 'num_heads'):
                            attn_info["num_heads"] = attn.num_heads
                        if hasattr(attn, 'hidden_size'):
                            attn_info["hidden_size"] = attn.hidden_size
                        if hasattr(attn, 'head_dim'):
                            attn_info["head_dim"] = attn.head_dim
                        if hasattr(attn, 'q_proj'):
                            q_params, _ = count_parameters(attn.q_proj)
                            attn_info["q_proj"] = {
                                "parameters": q_params,
                                "in_features": attn.q_proj.in_features if hasattr(attn.q_proj, 'in_features') else 'N/A',
                                "out_features": attn.q_proj.out_features if hasattr(attn.q_proj, 'out_features') else 'N/A'
                            }
                        layer_info["submodules"]["self_attn"] = attn_info

                    # 获取 MLP 信息
                    if hasattr(layer, 'mlp'):
                        mlp = layer.mlp
                        mlp_params, _ = count_parameters(mlp)
                        mlp_info = {
                            "type": type(mlp).__name__,
                            "parameters": mlp_params
                        }
                        if hasattr(mlp, 'gate_proj'):
                            gate_params, _ = count_parameters(mlp.gate_proj)
                            mlp_info["gate_proj"] = {
                                "parameters": gate_params,
                                "in_features": mlp.gate_proj.in_features if hasattr(mlp.gate_proj, 'in_features') else 'N/A',
                                "out_features": mlp.gate_proj.out_features if hasattr(mlp.gate_proj, 'out_features') else 'N/A'
                            }
                        if hasattr(mlp, 'up_proj'):
                            up_params, _ = count_parameters(mlp.up_proj)
                            mlp_info["up_proj"] = {
                                "parameters": up_params,
                                "in_features": mlp.up_proj.in_features if hasattr(mlp.up_proj, 'in_features') else 'N/A',
                                "out_features": mlp.up_proj.out_features if hasattr(mlp.up_proj, 'out_features') else 'N/A'
                            }
                        if hasattr(mlp, 'down_proj'):
                            down_params, _ = count_parameters(mlp.down_proj)
                            mlp_info["down_proj"] = {
                                "parameters": down_params,
                                "in_features": mlp.down_proj.in_features if hasattr(mlp.down_proj, 'in_features') else 'N/A',
                                "out_features": mlp.down_proj.out_features if hasattr(mlp.down_proj, 'out_features') else 'N/A'
                            }
                        layer_info["submodules"]["mlp"] = mlp_info

                    # 获取 Norm 信息
                    if hasattr(layer, 'input_layernorm'):
                        input_norm = layer.input_layernorm
                        input_norm_params, _ = count_parameters(input_norm)
                        layer_info["submodules"]["input_layernorm"] = {
                            "type": type(input_norm).__name__,
                            "parameters": input_norm_params
                        }

                    if hasattr(layer, 'post_attention_layernorm'):
                        post_norm = layer.post_attention_layernorm
                        post_norm_params, _ = count_parameters(post_norm)
                        layer_info["submodules"]["post_attention_layernorm"] = {
                            "type": type(post_norm).__name__,
                            "parameters": post_norm_params
                        }

                    arch_info["components"]["transformer_layers"]["layers"].append(layer_info)
                    print(f"  Layer {i}: {type(layer).__name__}, 参数量: {layer_params:,}")
                    if "self_attn" in layer_info["submodules"]:
                        attn_info = layer_info["submodules"]["self_attn"]
                        if "num_heads" in attn_info:
                            print(f"    - Self-Attention: {attn_info['num_heads']} heads, hidden_size={attn_info.get('hidden_size', 'N/A')}")

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
            if "in_features" in arch_info["components"]["lm_head"]:
                print(f"  输入维度: {arch_info['components']['lm_head']['in_features']}")
                print(f"  输出维度: {arch_info['components']['lm_head']['out_features']}")

    # 保存架构信息
    arch_file = os.path.join(output_dir, "model_architecture.json")
    with open(arch_file, 'w', encoding='utf-8') as f:
        json.dump(arch_info, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 架构信息已保存到: {arch_file}")

    return arch_info


def extract_and_visualize_causal_mask(model, tokenizer, image_processor, image_file, prompt,
                                      conv_mode, device, output_dir):
    """提取并可视化 causal mask，特别标注视觉 token 区域"""
    print("\n" + "=" * 80)
    print("Causal Mask 提取和可视化")
    print("=" * 80)

    # 1. 准备输入
    print("\n[步骤 1] 准备输入")
    print("-" * 80)

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

    print(f"  输入文本长度: {input_ids.shape[1]} tokens")

    # 2. 提取图像特征并准备多模态输入
    print("\n[步骤 2] 准备多模态输入")
    print("-" * 80)

    num_image_tokens = 0
    with torch.no_grad():
        # 获取图像特征
        vision_tower = model.get_vision_tower()
        if vision_tower is not None:
            image_features = vision_tower(image_tensor.unsqueeze(0).half().to(device))
            if hasattr(image_features, 'last_hidden_state'):
                vision_hidden = image_features.last_hidden_state
            elif isinstance(image_features, tuple):
                vision_hidden = image_features[0]
            elif isinstance(image_features, torch.Tensor):
                vision_hidden = image_features
            else:
                vision_hidden = None

            if vision_hidden is not None:
                vision_hidden = model.get_model().mm_projector(vision_hidden)
                num_image_tokens = vision_hidden.shape[1]
                print(f"  图像 token 数量: {num_image_tokens}")
            else:
                print("  警告: 无法提取图像特征")
        else:
            print("  警告: Vision tower 不存在")

        # 准备多模态输入
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

        seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids_processed.shape[1]
        print(f"  处理后的序列长度: {seq_len}")

    # 3. 确定图像 token 的位置
    print("\n[步骤 3] 确定图像 token 位置")
    print("-" * 80)

    # 找到原始 input_ids 中 IMAGE_TOKEN_INDEX 的位置
    image_token_positions = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()
    print(f"  原始 input_ids 中 IMAGE_TOKEN_INDEX 位置: {image_token_positions}")

    # 根据 prepare_inputs_labels_for_multimodal 的逻辑计算图像 token 的实际位置
    # 图像 token 会被插入到 IMAGE_TOKEN_INDEX 的位置
    # 第一个 IMAGE_TOKEN_INDEX 之前的所有 token 数量就是图像 token 的起始位置
    if len(image_token_positions) > 0:
        first_image_pos = image_token_positions[0]
        # 在 prepare_inputs_labels_for_multimodal 中，图像特征会替换 IMAGE_TOKEN_INDEX
        # 所以图像 token 的起始位置 = first_image_pos（因为 IMAGE_TOKEN_INDEX 被替换了）
        image_token_start = first_image_pos
        image_token_end = image_token_start + num_image_tokens
        print(f"  图像 token 位置范围: [{image_token_start}, {image_token_end})")
        print(f"  图像 token 数量: {num_image_tokens}")
    else:
        image_token_start = None
        image_token_end = None
        print("  未找到图像 token")

    # 4. 创建 causal mask
    print("\n[步骤 4] 创建 Causal Mask")
    print("-" * 80)

    # 创建标准的 causal mask（下三角矩阵）
    # causal_mask[i, j] = True 表示位置 i 可以 attend 到位置 j
    # 对于 causal mask，i >= j 的位置应该是 True（可以 attend）
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    print(f"  Causal mask 形状: {causal_mask.shape}")
    print(f"  Causal mask 类型: {causal_mask.dtype}")

    # 5. 可视化 causal mask
    print("\n[步骤 5] 可视化 Causal Mask")
    print("-" * 80)

    # 转换为 numpy 用于可视化
    mask_np = causal_mask.cpu().numpy().astype(float)

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # 左图：完整的 causal mask
    ax1 = axes[0]
    im1 = ax1.imshow(mask_np, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax1.set_title('Causal Mask Matrix (Full View)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Key Position (Attendable Positions)', fontsize=12)
    ax1.set_ylabel('Query Position (Current Token Position)', fontsize=12)

    # 添加网格线以便更好地查看
    ax1.set_xticks(range(0, seq_len, max(1, seq_len // 20)))
    ax1.set_yticks(range(0, seq_len, max(1, seq_len // 20)))
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # 标注图像 token 区域
    if image_token_start is not None and image_token_end is not None:
        # 在图像 token 区域添加矩形框
        rect = patches.Rectangle(
            (image_token_start - 0.5, image_token_start - 0.5),
            num_image_tokens,
            num_image_tokens,
            linewidth=3,
            edgecolor='blue',
            facecolor='none',
            label='Vision Token Region'
        )
        ax1.add_patch(rect)

        # 添加文本标注
        ax1.text(
            image_token_start + num_image_tokens / 2,
            image_token_start - 5,
            f'Vision Tokens\n[{image_token_start}:{image_token_end}]',
            ha='center',
            va='bottom',
            fontsize=10,
            color='blue',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='blue')
        )

        # 添加垂直线和水平线标记图像 token 区域
        ax1.axvline(x=image_token_start - 0.5, color='blue', linestyle='--', linewidth=2, alpha=0.7)
        ax1.axvline(x=image_token_end - 0.5, color='blue', linestyle='--', linewidth=2, alpha=0.7)
        ax1.axhline(y=image_token_start - 0.5, color='blue', linestyle='--', linewidth=2, alpha=0.7)
        ax1.axhline(y=image_token_end - 0.5, color='blue', linestyle='--', linewidth=2, alpha=0.7)

    ax1.legend(loc='upper right')
    plt.colorbar(im1, ax=ax1, label='Can Attend (1=Yes, 0=No)')

    # 右图：放大的图像 token 区域（如果存在）
    ax2 = axes[1]
    if image_token_start is not None and image_token_end is not None:
        # 显示图像 token 区域及其周围区域
        margin = min(50, seq_len // 4)
        start_idx = max(0, image_token_start - margin)
        end_idx = min(seq_len, image_token_end + margin)

        mask_zoom = mask_np[start_idx:end_idx, start_idx:end_idx]
        im2 = ax2.imshow(mask_zoom, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax2.set_title(f'Causal Mask (Zoomed: Vision Token Region)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Key Position', fontsize=12)
        ax2.set_ylabel('Query Position', fontsize=12)

        # 标注图像 token 区域
        vis_start_rel = image_token_start - start_idx
        vis_end_rel = image_token_end - start_idx
        rect2 = patches.Rectangle(
            (vis_start_rel - 0.5, vis_start_rel - 0.5),
            num_image_tokens,
            num_image_tokens,
            linewidth=3,
            edgecolor='red',
            facecolor='none',
            label='Vision Token Region'
        )
        ax2.add_patch(rect2)

        # 添加网格和标注
        ax2.set_xticks(range(0, mask_zoom.shape[1], max(1, mask_zoom.shape[1] // 10)))
        ax2.set_yticks(range(0, mask_zoom.shape[0], max(1, mask_zoom.shape[0] // 10)))
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # 标注关键信息
        ax2.text(
            vis_start_rel + num_image_tokens / 2,
            vis_start_rel - 2,
            f'Vision Tokens\n[{image_token_start}:{image_token_end}]\n{num_image_tokens} tokens',
            ha='center',
            va='bottom',
            fontsize=10,
            color='red',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='red')
        )

        # 分析视觉 token 的因果限制
        vision_mask_region = mask_np[image_token_start:image_token_end, image_token_start:image_token_end]
        vision_self_attention = vision_mask_region.sum() / (num_image_tokens * num_image_tokens)

        # 视觉 token 对之前文本 token 的 attention
        if image_token_start > 0:
            vision_to_text = mask_np[image_token_start:image_token_end, :image_token_start].sum() / (num_image_tokens * image_token_start)
        else:
            vision_to_text = 0.0

        # 后续文本 token 对视觉 token 的 attention
        if image_token_end < seq_len:
            text_to_vision = mask_np[image_token_end:, image_token_start:image_token_end].sum() / ((seq_len - image_token_end) * num_image_tokens)
        else:
            text_to_vision = 0.0

        # 计算视觉 token 之间的因果限制详情
        # 在下三角矩阵中，对于 n 个 token，可 attend 的位置数是 n*(n+1)/2，比例约为 50%
        expected_causal_ratio = 0.5  # 下三角矩阵的理论比例
        is_causal = abs(vision_self_attention - expected_causal_ratio) < 0.1

        info_text = (
            f"Causal Mask Analysis:\n"
            f"Vision self-attention: {vision_self_attention:.2%}\n"
            f"  (Expected ~50% for causal mask)\n"
            f"Vision → Text (before): {vision_to_text:.2%}\n"
            f"Text (after) → Vision: {text_to_vision:.2%}\n"
            f"\nNote: Vision tokens follow\n"
            f"causal constraints (later tokens\n"
            f"cannot attend to earlier ones)"
        )
        ax2.text(
            0.02, 0.98,
            info_text,
            transform=ax2.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        ax2.legend(loc='lower right')
    else:
        # 如果没有图像 token，显示完整的 mask
        im2 = ax2.imshow(mask_np, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax2.set_title('Causal Mask Matrix (No Vision Tokens)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Key Position', fontsize=12)
        ax2.set_ylabel('Query Position', fontsize=12)

    plt.colorbar(im2, ax=ax2, label='Can Attend (1=Yes, 0=No)')

    plt.tight_layout()

    # 保存图片
    mask_image_file = os.path.join(output_dir, "causal_mask_visualization.png")
    plt.savefig(mask_image_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Causal mask 可视化已保存到: {mask_image_file}")

    plt.close()

    # 6. 保存详细信息到 JSON
    mask_info = {
        "sequence_length": int(seq_len),
        "num_image_tokens": int(num_image_tokens) if image_token_start is not None else 0,
        "image_token_start": int(image_token_start) if image_token_start is not None else None,
        "image_token_end": int(image_token_end) if image_token_end is not None else None,
        "causal_mask_shape": list(causal_mask.shape),
        "causal_mask_dtype": str(causal_mask.dtype),
        "mask_statistics": {
            "total_attendable_pairs": int(causal_mask.sum().item()),
            "total_pairs": int(causal_mask.numel()),
            "attendable_ratio": float(causal_mask.sum().item() / causal_mask.numel())
        }
    }

    if image_token_start is not None and image_token_end is not None:
        vision_mask_region = mask_np[image_token_start:image_token_end, image_token_start:image_token_end]
        vision_self_attention = vision_mask_region.sum() / (num_image_tokens * num_image_tokens)

        if image_token_start > 0:
            vision_to_text = mask_np[image_token_start:image_token_end, :image_token_start].sum() / (num_image_tokens * image_token_start)
        else:
            vision_to_text = 0.0

        if image_token_end < seq_len:
            text_to_vision = mask_np[image_token_end:, image_token_start:image_token_end].sum() / ((seq_len - image_token_end) * num_image_tokens)
        else:
            text_to_vision = 0.0

        # 详细分析视觉 token 区域内的因果模式
        # 检查是否是标准的下三角矩阵（因果掩码）
        vision_mask_region_bool = vision_mask_region.astype(bool)
        is_lower_triangular = np.allclose(vision_mask_region_bool, np.tril(np.ones_like(vision_mask_region_bool)))

        mask_info["vision_token_analysis"] = {
            "vision_self_attention_ratio": float(vision_self_attention),
            "vision_to_text_ratio": float(vision_to_text),
            "text_to_vision_ratio": float(text_to_vision),
            "follows_causal_constraint": bool(is_lower_triangular),
            "explanation": {
                "vision_self_attention": f"Vision tokens follow causal masking: ratio={vision_self_attention:.2%} (~50% for lower triangular mask). This means later vision tokens cannot attend to earlier vision tokens, even though they represent the same image.",
                "vision_to_text": "Vision tokens can attend to previous text tokens (causal constraint allows)",
                "text_to_vision": "Subsequent text tokens can attend to vision tokens (causal constraint allows)",
                "causal_constraint_note": "In LLaVA, vision tokens are inserted sequentially in the sequence, so they follow the same causal masking as text tokens. Position i can only attend to positions j where j <= i."
            }
        }

    mask_info_file = os.path.join(output_dir, "causal_mask_info.json")
    with open(mask_info_file, 'w', encoding='utf-8') as f:
        json.dump(mask_info, f, ensure_ascii=False, indent=2)
    print(f"✓ Causal mask 详细信息已保存到: {mask_info_file}")

    return mask_info


def trace_inference_dimensions(model, tokenizer, image_processor, image_file, prompt,
                               conv_mode, device, output_dir):
    """追踪推理过程中的维度变化"""
    print("\n" + "=" * 80)
    print("推理维度追踪")
    print("=" * 80)

    dimension_trace = {
        "input_info": {},
        "stages": []
    }

    # 1. 输入准备阶段
    print("\n[阶段 1] 输入准备")
    print("-" * 80)

    # 加载图像
    image = load_image(image_file)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    dimension_trace["input_info"] = {
        "image_file": image_file,
        "prompt": prompt,
        "image_size": list(image.size),
        "image_tensor_shape": list(image_tensor.shape),
        "image_tensor_dtype": str(image_tensor.dtype)
    }

    print(f"  图像尺寸: {image.size}")
    print(f"  图像张量形状: {image_tensor.shape}")

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

    dimension_trace["input_info"]["full_prompt"] = full_prompt
    dimension_trace["input_info"]["input_ids_shape"] = list(input_ids.shape)
    dimension_trace["input_info"]["input_ids_length"] = int(input_ids.shape[1])

    print(f"  输入文本长度: {len(full_prompt)} 字符")
    print(f"  input_ids 形状: {input_ids.shape}")
    print(f"  input_ids 长度: {input_ids.shape[1]} tokens")

    # 2. Vision Tower 阶段
    print("\n[阶段 2] Vision Tower")
    print("-" * 80)

    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        with torch.no_grad():
            image_features = vision_tower(image_tensor.unsqueeze(0).half().to(device))

            # 处理不同的输出格式
            if hasattr(image_features, 'last_hidden_state'):
                vision_hidden = image_features.last_hidden_state
            elif isinstance(image_features, tuple):
                vision_hidden = image_features[0]
            elif isinstance(image_features, torch.Tensor):
                vision_hidden = image_features
            else:
                vision_hidden = None

            if vision_hidden is not None:
                vision_stage = {
                    "stage_name": "vision_tower",
                    "input_shape": list(image_tensor.unsqueeze(0).shape),
                    "output_shape": list(vision_hidden.shape),
                    "output_dtype": str(vision_hidden.dtype),
                    "num_image_tokens": int(vision_hidden.shape[1])
                }
                dimension_trace["stages"].append(vision_stage)

                print(f"  输入形状: {image_tensor.unsqueeze(0).shape}")
                print(f"  输出形状: {vision_hidden.shape}")
                print(f"  图像token数量: {vision_hidden.shape[1]}")

    # 3. MM Projector 阶段
    print("\n[阶段 3] MM Projector")
    print("-" * 80)

    lang_model = model.get_model() if hasattr(model, 'get_model') else model
    if hasattr(lang_model, 'mm_projector') and vision_hidden is not None:
        with torch.no_grad():
            projected_features = lang_model.mm_projector(vision_hidden)

            projector_stage = {
                "stage_name": "mm_projector",
                "input_shape": list(vision_hidden.shape),
                "output_shape": list(projected_features.shape),
                "output_dtype": str(projected_features.dtype),
                "num_image_tokens": int(projected_features.shape[1])
            }
            dimension_trace["stages"].append(projector_stage)

            print(f"  输入形状: {vision_hidden.shape}")
            print(f"  输出形状: {projected_features.shape}")
            print(f"  投影后图像token数量: {projected_features.shape[1]}")

    # 4. Prepare Inputs for Multimodal
    print("\n[阶段 4] Prepare Multimodal Inputs")
    print("-" * 80)

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

        prepare_stage = {
            "stage_name": "prepare_multimodal_inputs",
            "input_ids_shape": list(input_ids.shape),
            "input_ids_processed_shape": list(input_ids_processed.shape) if input_ids_processed is not None else None,
            "position_ids_shape": list(position_ids.shape) if position_ids is not None else None,
            "attention_mask_shape": list(attention_mask.shape) if attention_mask is not None else None,
            "inputs_embeds_shape": list(inputs_embeds.shape) if inputs_embeds is not None else None,
            "sequence_length": int(inputs_embeds.shape[1]) if inputs_embeds is not None else int(input_ids_processed.shape[1]) if input_ids_processed is not None else None
        }
        dimension_trace["stages"].append(prepare_stage)

        print(f"  原始 input_ids 形状: {input_ids.shape}")
        if input_ids_processed is not None:
            print(f"  处理后的 input_ids 形状: {input_ids_processed.shape}")
        if inputs_embeds is not None:
            print(f"  inputs_embeds 形状: {inputs_embeds.shape}")
            print(f"  序列长度: {inputs_embeds.shape[1]}")
        if position_ids is not None:
            print(f"  position_ids 形状: {position_ids.shape}")
        if attention_mask is not None:
            print(f"  attention_mask 形状: {attention_mask.shape}")

    # 5. Language Model Forward
    print("\n[阶段 5] Language Model Forward")
    print("-" * 80)

    lang_model = model.get_model()
    if lang_model is not None:
        with torch.no_grad():
            outputs = lang_model.forward(
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

            # 分析每一层的 hidden states 和 attentions
            transformer_stages = []
            for i, (hidden, attn) in enumerate(zip(hidden_states, attentions)):
                layer_stage = {
                    "layer_index": i,
                    "stage_name": f"transformer_layer_{i}",
                    "hidden_state_shape": list(hidden.shape),
                    "hidden_state_dtype": str(hidden.dtype)
                }

                if attn is not None:
                    if isinstance(attn, tuple):
                        attn = attn[0]
                    layer_stage["attention_shape"] = list(attn.shape) if isinstance(attn, torch.Tensor) else None
                    layer_stage["attention_dtype"] = str(attn.dtype) if isinstance(attn, torch.Tensor) else None

                    # 解析 attention 形状
                    if isinstance(attn, torch.Tensor) and len(attn.shape) == 4:
                        batch, num_heads, query_len, key_len = attn.shape
                        layer_stage["attention_info"] = {
                            "batch_size": int(batch),
                            "num_heads": int(num_heads),
                            "query_len": int(query_len),
                            "key_len": int(key_len),
                            "head_dim": int(hidden.shape[-1] / num_heads) if hidden.shape[-1] % num_heads == 0 else 'N/A'
                        }

                transformer_stages.append(layer_stage)

                if i < 3 or i >= len(hidden_states) - 1:  # 只打印前3层和最后1层
                    print(f"  Layer {i}:")
                    print(f"    Hidden state 形状: {hidden.shape}")
                    if attn is not None and isinstance(attn, torch.Tensor):
                        print(f"    Attention 形状: {attn.shape}")
                        if "attention_info" in layer_stage:
                            info = layer_stage["attention_info"]
                            print(f"      - Heads: {info.get('num_heads', 'N/A')}, Query: {info.get('query_len', 'N/A')}, Key: {info.get('key_len', 'N/A')}")

            dimension_trace["stages"].extend(transformer_stages)

            # 最后一层的 hidden state
            last_hidden = hidden_states[-1]
            dimension_trace["final_hidden_state"] = {
                "shape": list(last_hidden.shape),
                "dtype": str(last_hidden.dtype)
            }

    # 6. LM Head 阶段
    print("\n[阶段 6] LM Head")
    print("-" * 80)

    if hasattr(model, 'lm_head') and last_hidden is not None:
        with torch.no_grad():
            # 应用 norm（如果有）
            if hasattr(lang_model, 'norm'):
                normalized_hidden = lang_model.norm(last_hidden)
                norm_stage = {
                    "stage_name": "final_norm",
                    "input_shape": list(last_hidden.shape),
                    "output_shape": list(normalized_hidden.shape)
                }
                dimension_trace["stages"].append(norm_stage)
                print(f"  Norm 输入形状: {last_hidden.shape}")
                print(f"  Norm 输出形状: {normalized_hidden.shape}")
                last_hidden = normalized_hidden

            # 通过 lm_head
            logits = model.lm_head(last_hidden)

            lm_head_stage = {
                "stage_name": "lm_head",
                "input_shape": list(last_hidden.shape),
                "output_shape": list(logits.shape),
                "vocab_size": int(logits.shape[-1])
            }
            dimension_trace["stages"].append(lm_head_stage)

            print(f"  输入形状: {last_hidden.shape}")
            print(f"  输出形状: {logits.shape}")
            print(f"  词汇表大小: {logits.shape[-1]}")

    # 保存维度追踪信息
    trace_file = os.path.join(output_dir, "inference_dimension_trace.json")
    with open(trace_file, 'w', encoding='utf-8') as f:
        json.dump(dimension_trace, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 维度追踪信息已保存到: {trace_file}")

    return dimension_trace


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LLaVA 模型架构分析和推理维度追踪")

    # 模型参数
    parser.add_argument("--model-path", type=str, default="/home/liying/Documents/llava-v1.5-7b", help="模型路径")
    parser.add_argument("--model-base", type=str, default=None, help="基础模型路径")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")

    # 输入参数
    parser.add_argument("--image-file", type=str,
                       default="/home/liying/Documents/dataset/coco/val2014/COCO_val2014_000000065883.jpg",  # "there is a boy with blond hair and blue eyes, is this discription correct? Yes or No."
                       help="图像文件路径")
    parser.add_argument("--prompt", type=str,
                       default="there is a boy with blond hair and blue eyes, is this discription correct? Yes or No.",  # "Please describe this image in detail."
                       help="提示词")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="对话模式")

    # 输出参数
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")

    args = parser.parse_args()

    # 设置输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(current_dir, "results", "model_analysis", timestamp)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("LLaVA 模型架构分析和推理维度追踪")
    print("=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"设备: {args.device}")
    print(f"图像文件: {args.image_file}")
    print(f"提示词: {args.prompt}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 80)

    # 加载模型
    print("\n[1/3] 正在加载模型...")
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    device = args.device

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, device=device
    )
    print(f"✓ 模型加载完成: {model_name}")

    # 分析模型架构
    print("\n[2/3] 正在分析模型架构...")
    arch_info = analyze_model_architecture(model, tokenizer, args.output_dir)

    # 追踪推理维度
    print("\n[3/4] 正在追踪推理维度...")
    dimension_trace = trace_inference_dimensions(
        model, tokenizer, image_processor, args.image_file, args.prompt,
        args.conv_mode, device, args.output_dir
    )

    # 提取和可视化 causal mask
    print("\n[4/4] 正在提取和可视化 Causal Mask...")
    mask_info = extract_and_visualize_causal_mask(
        model, tokenizer, image_processor, args.image_file, args.prompt,
        args.conv_mode, device, args.output_dir
    )

    print("\n" + "=" * 80)
    print("✓ 分析完成！")
    print("=" * 80)
    print(f"架构信息: {os.path.join(args.output_dir, 'model_architecture.json')}")
    print(f"维度追踪: {os.path.join(args.output_dir, 'inference_dimension_trace.json')}")
    print(f"Causal Mask 可视化: {os.path.join(args.output_dir, 'causal_mask_visualization.png')}")
    print(f"Causal Mask 信息: {os.path.join(args.output_dir, 'causal_mask_info.json')}")


if __name__ == "__main__":
    main()