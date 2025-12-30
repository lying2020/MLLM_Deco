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

    # 编码图像特征
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

                # 通过 projector
                if vision_hidden is not None and hasattr(model, 'mm_projector'):
                    vision_hidden = model.mm_projector(vision_hidden)

                if vision_hidden is not None:
                    print(f"Vision Hidden State Shape: {vision_hidden.shape}")
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

    # 保存 hidden states 和 attention
    states_file = os.path.join(output_dir, "hidden_states_info.json")
    states_info = {
        "num_layers": len(hidden_states),
        "hidden_states": [{"layer": i, "shape": list(h.shape)} for i, h in enumerate(hidden_states)],
        "attentions": [{"layer": i, "shape": list(attn.shape) if isinstance(attn, torch.Tensor) else None}
                      for i, attn in enumerate(attentions)],
        "vision_hidden_shape": list(vision_hidden.shape) if vision_hidden is not None else None
    }

    with open(states_file, 'w', encoding='utf-8') as f:
        json.dump(states_info, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Hidden states 信息已保存到: {states_file}")

    return hidden_states, attentions, vision_hidden, input_ids, image


def visualize_attention_heatmap(attention_weights, layer_idx, output_dir, image=None,
                                input_ids=None, tokenizer=None, image_token_positions=None):
    """可视化 attention heatmap"""
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

    # 创建 heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(attention_weights, cmap='viridis', cbar=True,
                xticklabels=False, yticklabels=False, square=True)
    plt.title(f'Attention Heatmap - Layer {layer_idx}\nShape: {attention_weights.shape}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')

    heatmap_file = os.path.join(output_dir, f"attention_heatmap_layer_{layer_idx}.png")
    plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Layer {layer_idx} attention heatmap 已保存: {heatmap_file}")

    return attention_weights


def map_attention_to_image(attention_weights, image, image_token_positions, output_dir, layer_idx):
    """将 attention weights 映射到原图上"""
    if image is None or image_token_positions is None:
        return None

    # 获取图像 token 的 attention
    image_attention = attention_weights[:, image_token_positions].mean(axis=1)  # [seq_len]

    # 只取图像 token 部分的 attention
    image_only_attention = image_attention[image_token_positions]  # [num_image_tokens]

    # 归一化
    image_only_attention = (image_only_attention - image_only_attention.min()) / (image_only_attention.max() - image_only_attention.min() + 1e-8)

    # 假设图像被分成 patches (CLIP 通常是 14x14 或 24x24)
    # 这里需要根据实际的 vision encoder 来确定
    num_patches = len(image_only_attention)
    patch_size = int(np.sqrt(num_patches))

    if patch_size * patch_size != num_patches:
        # 如果不是完全平方数，尝试其他方式
        patch_size = int(np.sqrt(num_patches))
        if patch_size * patch_size < num_patches:
            patch_size += 1

    # 重塑为 2D
    attention_map = image_only_attention[:patch_size * patch_size].reshape(patch_size, patch_size)

    # 上采样到图像大小（使用 PIL 或 torch）
    try:
        from scipy.ndimage import zoom
        attention_map_upsampled = zoom(attention_map,
                                       (image.size[1] / patch_size, image.size[0] / patch_size),
                                       order=1)
    except ImportError:
        # 如果没有 scipy，使用 torch 的插值
        import torch.nn.functional as F
        attention_tensor = torch.from_numpy(attention_map).unsqueeze(0).unsqueeze(0).float()
        attention_tensor = F.interpolate(attention_tensor,
                                        size=(image.size[1], image.size[0]),
                                        mode='bilinear',
                                        align_corners=False)
        attention_map_upsampled = attention_tensor.squeeze().numpy()

    # 创建可视化
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 原图
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Attention overlay
    axes[1].imshow(image)
    im = axes[1].imshow(attention_map_upsampled, cmap='jet', alpha=0.5, interpolation='bilinear')
    axes[1].set_title(f'Attention Overlay - Layer {layer_idx}')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])

    overlay_file = os.path.join(output_dir, f"attention_overlay_layer_{layer_idx}.png")
    plt.savefig(overlay_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Layer {layer_idx} attention overlay 已保存: {overlay_file}")

    return attention_map_upsampled


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

    # 找到图像 token 的位置
    image_token_positions = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].cpu().numpy()

    # 存储 Q、K 数据的字典
    qk_data = {}

    # 注册 hook 来提取 Q、K
    hooks = []
    lang_model = model.get_model()

    def make_hook(layer_idx):
        def attention_hook(module, input, output):
            # 尝试从 attention 模块提取 Q、K
            if hasattr(module, 'q_proj') and hasattr(module, 'k_proj'):
                hidden_states = input[0] if isinstance(input, tuple) else input
                with torch.no_grad():
                    q = module.q_proj(hidden_states)
                    k = module.k_proj(hidden_states)
                    qk_data[layer_idx] = {
                        'q': q.cpu().numpy(),
                        'k': k.cpu().numpy(),
                        'q_shape': list(q.shape),
                        'k_shape': list(k.shape)
                    }
        return attention_hook

    # 为目标层注册 hooks
    if hasattr(lang_model, 'layers'):
        for layer_idx in target_layers:
            if layer_idx < len(lang_model.layers):
                layer = lang_model.layers[layer_idx]
                if hasattr(layer, 'self_attn'):
                    hook = layer.self_attn.register_forward_hook(make_hook(layer_idx))
                    hooks.append(hook)

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
            qk_info[f"layer_{layer_idx}"] = {
                "q_shape": data['q_shape'],
                "k_shape": data['k_shape'],
                "q_mean": float(data['q'].mean()),
                "q_std": float(data['q'].std()),
                "k_mean": float(data['k'].mean()),
                "k_std": float(data['k'].std()),
                "q_min": float(data['q'].min()),
                "q_max": float(data['q'].max()),
                "k_min": float(data['k'].min()),
                "k_max": float(data['k'].max())
            }
        with open(qk_file, 'w', encoding='utf-8') as f:
            json.dump(qk_info, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Q、K 信息已保存到: {qk_file}")

    # 处理目标层 - 可视化 attention
    for layer_idx in target_layers:
        if layer_idx < len(attentions) and attentions[layer_idx] is not None:
            print(f"\n处理 Layer {layer_idx}...")

            # 可视化 attention heatmap
            attention_weights = visualize_attention_heatmap(
                attentions[layer_idx], layer_idx, output_dir,
                image, input_ids, tokenizer, image_token_positions
            )

            # 映射到图像
            if attention_weights is not None and image is not None and len(image_token_positions) > 0:
                map_attention_to_image(
                    attention_weights, image, image_token_positions,
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
    parser.add_argument("--prompt", type=str, default="Is there a car in the image?",
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
    hidden_states, attentions, vision_hidden, input_ids, image = extract_hidden_states_and_attention(
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
