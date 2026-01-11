#!/usr/bin/env python3
"""
POPE 评估脚本 - 直接运行版本
基于 test_llava_v15_7b.py 的实现，针对 POPE benchmark 优化
自动检测数据集和模型，使用默认参数，无需输入参数即可运行
"""

import argparse
from jax import default_device
import torch
import os
import json
from tqdm import tqdm
import requests
from io import BytesIO
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import warnings

# 抑制常见的无害警告
warnings.filterwarnings('ignore', message='.*You are using a model of type llava to instantiate a model of type llava_llama.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub')

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) != 'MLLM_Deco' else current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from project import llava_v15_7b_path
from eval_tool.eval_pope import evaluate_pope

from PIL import Image
import re
from transformers import set_seed
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from torchvision import transforms


def recorder(out):
    """将输出转换为 Yes/No"""
    if not out or not out.strip():
        # 如果输出为空，返回 "No"（但应该记录警告）
        return "No"

    out_lower = out.lower().strip()
    word_list = re.split(r'[^\w]+', out_lower)

    # 检查是否包含 "yes"
    if "yes" in word_list:
        return "Yes"
    # 检查是否包含 "no"
    elif "no" in word_list:
        return "No"
    else:
        # 如果既没有 "yes" 也没有 "no"，默认返回 "No"
        # 但这种情况应该被记录
        return "No"


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


def _tensor_to_pil_image(image_tensor):
    """
    将图像tensor转换为PIL Image

    Args:
        image_tensor: 形状为 [C, H, W] 的tensor，值范围在 [0, 1]

    Returns:
        PIL Image对象
    """
    # 确保tensor在CPU上
    if image_tensor.is_cuda:
        image_tensor = image_tensor.cpu()

    # 将值限制在[0, 1]范围内
    image_tensor = torch.clamp(image_tensor, 0.0, 1.0)

    # 转换为numpy数组 [C, H, W] -> [H, W, C]
    image_np = image_tensor.permute(1, 2, 0).numpy()

    # 转换为[0, 255]范围的uint8
    image_np = (image_np * 255).astype(np.uint8)

    # 转换为PIL Image
    image_pil = Image.fromarray(image_np)

    return image_pil


def add_gaussian_noise_ddpm(image_tensor, timestep, num_timesteps=1000, device='cuda:0', verbose=False):
    """
    DDPM前向扩散过程：对图像添加高斯噪声

    重要说明：加噪声的时机和累积方式
    - 输入：image_tensor 是 LLaVA 的 image_processor 输出
    - LLaVA 使用 CLIPImageProcessor，它会对图像进行标准化：(image - mean) / std
    - 标准化后的值范围不是 [0, 1]，而是大约在 [-2, 2] 左右
    - 步骤1：检测并处理输入图像的值范围
    - 步骤2：将图像转换到 [-1, 1] 范围（DDPM标准做法）
    - 步骤3：在归一化后的图像上添加高斯噪声
    - 步骤4：将加噪后的图像转换回原始范围（LLaVA期望的输入范围）

    噪声累积方式：
    - 使用DDPM的标准公式：x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
    - 这个公式允许从原始图像 x_0 直接计算到任意时间步 t 的噪声图像 x_t
    - 当 timestep 接近 num_timesteps-1 时，alpha_cumprod_t 应该接近 0，使得图像接近纯噪声
    - 为了确保最后一步是纯噪声，我们调整 beta_end 使其在最后一步时 alpha_cumprod 足够小

    Args:
        image_tensor: 原始图像tensor，形状为 [C, H, W]
            - 如果是 CLIPImageProcessor 输出，值范围大约在 [-2, 2]（标准化后的值）
            - 如果是简单的归一化，值范围在 [0, 1]
        timestep: 当前时间步（0到num_timesteps-1）
        num_timesteps: 总时间步数（默认1000）
        device: 设备
        verbose: 是否输出详细信息

    Returns:
        noisy_image: 加噪后的图像tensor，值范围与输入相同（保持原始范围）
    """
    # 确保image_tensor在正确的设备上
    image_tensor = image_tensor.to(device)

    # 检测输入图像的值范围（用于确定是否需要转换）
    if verbose and timestep == 0:
        min_val = image_tensor.min().item()
        max_val = image_tensor.max().item()
        mean_val = image_tensor.mean().item()
        print(f"  [图像范围检测] min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}")

    # 判断输入图像的值范围
    # CLIPImageProcessor 标准化后的值通常在 [-2, 2] 左右
    # 简单的 [0, 1] 归一化值在 [0, 1]
    min_val = image_tensor.min().item()
    max_val = image_tensor.max().item()

    # 如果值范围在 [-3, 3] 左右，认为是标准化后的值（CLIPImageProcessor输出）
    # 如果值范围在 [0, 1] 左右，认为是简单的归一化
    is_normalized = (min_val >= -3.0 and max_val <= 3.0 and (min_val < 0 or max_val > 1))

    if is_normalized:
        # 输入是标准化后的值（CLIPImageProcessor输出），需要先转换到 [0, 1]
        # 使用经验值：CLIP 标准化后的值大约在 [-2, 2]，我们将其映射到 [0, 1]
        # 更安全的方法：使用 min-max 归一化
        # 但为了保持一致性，我们假设标准化后的值大约在 [-2.5, 2.5] 范围
        # 将其线性映射到 [0, 1]：x_norm = (x + 2.5) / 5.0
        # 或者更简单：假设值在 [-3, 3] 范围，映射到 [0, 1]
        image_min = -3.0
        image_max = 3.0
        image_range = image_max - image_min
        image_tensor_01 = (image_tensor - image_min) / image_range  # 转换到 [0, 1]

        if verbose and timestep == 0:
            print(f"  [图像转换] 检测到标准化后的值，转换到 [0, 1] 范围")
    else:
        # 输入已经是 [0, 1] 范围
        image_tensor_01 = image_tensor
        if verbose and timestep == 0:
            print(f"  [图像转换] 检测到 [0, 1] 范围的值，无需转换")

    # 计算噪声调度
    # 对于较少的步数（如10步），需要调整beta_end以确保最后一步接近纯噪声
    # 标准DDPM使用 beta_start=0.0001, beta_end=0.02 (对于1000步)
    # 对于更少的步数，我们需要更大的beta_end来确保累积噪声足够大

    # 目标：在最后一步时，alpha_cumprod 应该接近 0.01 左右（即图像几乎全是噪声）
    # 使用更精确的方法：根据目标 alpha_cumprod 反推 beta_end
    beta_start = 0.0001
    target_final_alpha_cumprod = 0.01  # 最后一步时，图像应该只有1%的原始信息，99%是噪声

    # 对于线性调度，近似计算：如果所有步的alpha都相同，那么 alpha_cumprod = alpha^num_timesteps
    # 因此：alpha ≈ (target_final_alpha_cumprod)^(1/num_timesteps)
    # beta = 1 - alpha
    if num_timesteps <= 10:
        # 对于10步或更少，使用较大的beta_end确保最后一步接近纯噪声
        # 计算：如果最后一步 alpha_cumprod = 0.01，那么平均 alpha ≈ 0.01^(1/10) ≈ 0.63
        # 所以 beta_end ≈ 1 - 0.63 = 0.37，但考虑到线性调度，我们使用稍小的值
        beta_end = 0.4
    elif num_timesteps <= 50:
        # 对于50步：alpha ≈ 0.01^(1/50) ≈ 0.91，beta_end ≈ 0.09
        beta_end = 0.12
    elif num_timesteps <= 100:
        # 对于100步：alpha ≈ 0.01^(1/100) ≈ 0.95，beta_end ≈ 0.05
        beta_end = 0.06
    else:
        # 对于1000步，使用标准值（标准DDPM的beta_end=0.02）
        beta_end = 0.02

    betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # 验证：最后一步的 alpha_cumprod 应该足够小
    if verbose and timestep == 0:  # 只在第一次调用时打印
        final_alpha_cumprod = alphas_cumprod[-1].item()
        print(f"  [噪声调度] num_timesteps={num_timesteps}, beta_end={beta_end:.4f}, "
              f"final_alpha_cumprod={final_alpha_cumprod:.6f} "
              f"(目标: <0.01，当前: {'✓' if final_alpha_cumprod < 0.01 else '✗'})")

    # 获取当前时间步的alpha_cumprod
    # 注意：timestep 从 0 开始，所以最后一步是 num_timesteps - 1
    alpha_cumprod_t = alphas_cumprod[timestep]

    # 生成随机噪声（与图像形状相同）
    # 为了确保可重复性，可以使用固定种子，但这里使用随机噪声
    noise = torch.randn_like(image_tensor, device=device)

    # 计算加噪后的图像
    # x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
    # 当 alpha_cumprod_t 接近 0 时，图像接近纯噪声
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod_t)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod_t)

    # 步骤1：将图像从 [0, 1] 归一化到 [-1, 1] 进行扩散
    # 这是DDPM的标准做法，在归一化后的空间中进行扩散
    image_normalized = image_tensor_01 * 2.0 - 1.0  # [0, 1] -> [-1, 1]

    # 步骤2：在归一化后的图像上添加高斯噪声
    # 当 timestep 接近 num_timesteps-1 时，sqrt_alpha_cumprod 接近 0，sqrt_one_minus_alpha_cumprod 接近 1
    # 这意味着图像几乎完全是噪声
    noisy_image = sqrt_alpha_cumprod * image_normalized + sqrt_one_minus_alpha_cumprod * noise

    # 将值限制在[-1, 1]范围内
    noisy_image = torch.clamp(noisy_image, -1.0, 1.0)

    # 步骤3：转换回 [0, 1] 范围
    noisy_image_01 = (noisy_image + 1.0) / 2.0

    # 步骤4：如果输入是标准化后的值，需要转换回原始范围
    if is_normalized:
        # 转换回标准化后的值范围
        noisy_image = noisy_image_01 * image_range + image_min
    else:
        # 保持 [0, 1] 范围
        noisy_image = noisy_image_01

    return noisy_image


def extract_attention_from_output(output_dict, image_token_start=35, num_image_tokens=576, num_total_layers=32, num_heads=32):
    """
    从模型输出中提取attention信息，计算att_visual/att_all

    Args:
        output_dict: 模型生成的输出字典，包含attentions
        image_token_start: 图像token的起始位置
        num_image_tokens: 图像token数量（默认576）
        num_total_layers: 总层数（默认32）
        num_heads: 每层的head数（默认32）

    Returns:
        tuple: (total_ratio, layer_ratios)
            - total_ratio: 所有层的 att_visual / att_all 的比值（如果提取失败返回None）
            - layer_ratios: 每层的 att_visual / att_all 的比值列表 [32]，如果提取失败返回None
    """
    if not hasattr(output_dict, 'attentions') or output_dict.attentions is None:
        return None, None

    # 获取最后一个生成步骤的attention（因为只生成一个token）
    if len(output_dict.attentions) == 0:
        return None, None

    # 最后一个步骤的attention（所有层的attention）
    last_step_attentions = output_dict.attentions[-1]

    if last_step_attentions is None or len(last_step_attentions) == 0:
        return None, None

    # 收集所有层的att_visual和att_all
    total_att_visual = 0.0
    total_att_all = 0.0
    layer_ratios = np.zeros(num_total_layers)  # [32]

    for layer_idx in range(min(num_total_layers, len(last_step_attentions))):
        layer_attn = last_step_attentions[layer_idx]
        if layer_attn is None:
            continue

        # 处理attention tensor，提取最后一行的attention
        if isinstance(layer_attn, tuple):
            layer_attn = layer_attn[0]

        if not isinstance(layer_attn, torch.Tensor):
            continue

        layer_attn_np = layer_attn.cpu().numpy()

        # 提取最后一行的attention（对所有head求和）
        # 处理不同形状的attention tensor
        if len(layer_attn_np.shape) == 4:
            # [batch, num_heads, seq_len, seq_len]
            batch_size, num_heads_actual, seq_len, _ = layer_attn_np.shape
            if seq_len == 1:
                last_row_attention = np.sum(layer_attn_np[0, :, 0, :], axis=0)  # [seq_len]
            else:
                last_row_attention = np.sum(layer_attn_np[0, :, -1, :], axis=0)  # [seq_len]
        elif len(layer_attn_np.shape) == 3:
            # [num_heads, seq_len, seq_len]
            num_heads_actual, seq_len, _ = layer_attn_np.shape
            if seq_len == 1:
                last_row_attention = np.sum(layer_attn_np[:, 0, :], axis=0)  # [seq_len]
            else:
                last_row_attention = np.sum(layer_attn_np[:, -1, :], axis=0)  # [seq_len]
        else:
            continue

        # 计算att_all：整行所有attention值的和
        att_all = np.sum(last_row_attention)

        # 计算att_visual：576个visual attention值的和
        seq_len_actual = len(last_row_attention)
        image_token_end_actual = min(image_token_start + num_image_tokens, seq_len_actual)
        valid_image_positions = np.arange(image_token_start, image_token_end_actual)

        if len(valid_image_positions) > 0:
            image_attention = last_row_attention[valid_image_positions]
            att_visual = np.sum(image_attention)
        else:
            att_visual = 0.0

        total_att_visual += att_visual
        total_att_all += att_all

        # 计算该层的比值
        if att_all > 0:
            layer_ratios[layer_idx] = att_visual / att_all

    # 计算总比值
    if total_att_all > 0:
        total_ratio = total_att_visual / total_att_all
        return float(total_ratio), layer_ratios
    else:
        return None, None


def analyze_diffusion_attention(model, tokenizer, image_processor, image_file, prompt, conv_mode, device,
                                num_diffusion_steps=1000, image_token_start=35, num_image_tokens=576,
                                num_total_layers=32, num_heads=32, max_new_tokens=15, temperature=0,
                                output_dir=None, question_id=None, verbose=False, target_layers=None):
    """
    对图像进行DDPM前向扩散，分析每一步的attention信息

    Args:
        model: LLaVA模型
        tokenizer: tokenizer
        image_processor: 图像处理器
        image_file: 图像文件路径
        prompt: 提示词
        conv_mode: 对话模式
        device: 设备
        num_diffusion_steps: 扩散步数（默认1000）
        image_token_start: 图像token起始位置
        num_image_tokens: 图像token数量
        num_total_layers: 总层数
        num_heads: 每层head数
        max_new_tokens: 最大生成token数
        temperature: 生成温度
        output_dir: 输出目录
        question_id: 问题ID（用于保存文件）
        verbose: 是否输出详细信息
        target_layers: 目标层列表（可选，用于生成每层折线图）

    Returns:
        ratios: 每一步的att_visual/att_all比值列表（所有层的总和）
    """
    if verbose:
        print(f"\n  [扩散分析] 开始分析 {num_diffusion_steps} 步扩散过程...")

    # 加载原始图像
    image = load_image(image_file)
    original_image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    original_image_tensor = original_image_tensor.to(device)

    # 准备输入（使用文件顶部已导入的模块，不重复导入）
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

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # 存储每一步的比值
    ratios = []  # 所有层的总和
    layer_ratios_per_step = []  # 每层每步的比值 [num_diffusion_steps, num_total_layers]

    # 保存原始图像（用于对比）
    if output_dir:
        # 将原始图像tensor转换为PIL Image并保存
        original_image_pil = _tensor_to_pil_image(original_image_tensor)
        original_image_path = os.path.join(output_dir, "original_image.png")
        original_image_pil.save(original_image_path)
        if verbose:
            print(f"  ✓ 原始图像已保存: {os.path.basename(original_image_path)}")

    # 对每一步进行扩散和推理
    for step in tqdm(range(num_diffusion_steps), desc="扩散步骤", disable=not verbose):
        # 对图像加噪
        noisy_image_tensor = add_gaussian_noise_ddpm(
            original_image_tensor, step, num_timesteps=num_diffusion_steps, device=device, verbose=verbose
        )

        # 保存加噪后的图像（每隔一定步数保存，避免文件过多）
        if output_dir and (step == 0 or step == num_diffusion_steps - 1 or (step + 1) % max(1, num_diffusion_steps // 10) == 0):
            noisy_image_pil = _tensor_to_pil_image(noisy_image_tensor)
            noisy_image_path = os.path.join(output_dir, f"noisy_image_step_{step:04d}.png")
            noisy_image_pil.save(noisy_image_path)
            if verbose and step < 5:  # 只对前几步输出详细信息
                print(f"  ✓ 加噪图像已保存 (step {step}): {os.path.basename(noisy_image_path)}")

        # 准备生成参数
        generate_kwargs = {
            "inputs": input_ids,
            "images": noisy_image_tensor.unsqueeze(0).half().to(device),
            "do_sample": False,  # 使用greedy decoding
            "temperature": None,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
            "return_dict": True,
            "return_dict_in_generate": True,
            "output_attentions": True,  # 输出attention
            "output_hidden_states": False,
            "stopping_criteria": [stopping_criteria]
        }

        # 生成回答
        with torch.inference_mode():
            with torch.no_grad():
                output_dict = model.generate(**generate_kwargs)

        # 提取attention信息
        total_ratio, layer_ratios = extract_attention_from_output(
            output_dict, image_token_start, num_image_tokens, num_total_layers, num_heads
        )

        if total_ratio is not None:
            ratios.append(total_ratio)
        else:
            ratios.append(0.0)  # 如果提取失败，使用0.0

        if layer_ratios is not None:
            layer_ratios_per_step.append(layer_ratios.copy())
        else:
            layer_ratios_per_step.append(np.zeros(num_total_layers))

    # 绘制图表
    if output_dir and len(ratios) > 0:
        os.makedirs(output_dir, exist_ok=True)

        # 创建图表 - 学术论文风格
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 0,  # 移除标题
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 10,
            'figure.titlesize': 0,
            'axes.linewidth': 1.0,
            'grid.linewidth': 0.5,
            'lines.linewidth': 2.0,
            'lines.markersize': 5,
        })

        fig, ax = plt.subplots(figsize=(5.5, 4.0))  # 更紧凑的尺寸，适合论文
        steps = np.arange(1, len(ratios) + 1)

        # 使用更专业的配色（深蓝色，适合学术论文）
        # 折线加粗到1.2倍：2.0 * 1.2 = 2.4
        ax.plot(steps, ratios, color='#1f77b4', linewidth=2.4, alpha=0.9)

        # 设置标签（更简洁，加粗并放大1.2倍）
        label_fontsize = int(12 * 1.2)  # 14.4 -> 14
        ax.set_xlabel('Diffusion Step', fontsize=label_fontsize, fontweight='bold')
        ax.set_ylabel('Attention Ratio', fontsize=label_fontsize, fontweight='bold')

        # 网格样式（更subtle）
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # 显示所有边框，使用浅蓝色
        light_blue = '#ADD8E6'
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(light_blue)
            spine.set_linewidth(1.0)

        # 设置刻度样式（加粗并放大1.2倍）
        tick_fontsize = int(11 * 1.2)  # 13.2 -> 13
        ax.tick_params(direction='in', length=4, width=0.8, labelsize=tick_fontsize)
        # 设置刻度标签加粗
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        # 保存图表（更高DPI，适合论文）
        if question_id is not None:
            filename = f"diffusion_attention_ratio_q{question_id}.png"
        else:
            filename = "diffusion_attention_ratio.png"

        output_file = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        if verbose:
            print(f"  ✓ 扩散attention分析图表已保存: {os.path.basename(output_file)}")
            print(f"    平均比值: {np.mean(ratios):.4f}")
            print(f"    最小比值: {np.min(ratios):.4f}")
            print(f"    最大比值: {np.max(ratios):.4f}")

    # 为每个目标层生成折线图（显示该层在不同扩散step的比值变化）
    if target_layers is not None and len(target_layers) > 0 and len(layer_ratios_per_step) > 0:
        layer_ratios_array = np.array(layer_ratios_per_step)  # [num_diffusion_steps, num_total_layers]

        if verbose:
            print(f"\n  [生成每层折线图] 为 {len(target_layers)} 个目标层生成折线图...")

        for layer_idx in target_layers:
            if layer_idx >= num_total_layers:
                continue

            # 提取该层在所有扩散步的比值
            layer_ratios = layer_ratios_array[:, layer_idx]  # [num_diffusion_steps]

            # 创建折线图 - 学术论文风格
            plt.rcParams.update({
                'font.family': 'serif',
                'font.serif': ['Times New Roman', 'DejaVu Serif'],
                'font.size': 12,
                'axes.labelsize': 12,
                'axes.titlesize': 0,  # 移除标题
                'xtick.labelsize': 11,
                'ytick.labelsize': 11,
                'legend.fontsize': 10,
                'figure.titlesize': 0,
                'axes.linewidth': 1.0,
                'grid.linewidth': 0.5,
                'lines.linewidth': 2.0,
                'lines.markersize': 5,
            })

            fig_layer, ax_layer = plt.subplots(figsize=(5.5, 4.0))  # 更紧凑的尺寸
            steps = np.arange(1, num_diffusion_steps + 1)  # 从1开始编号

            # 绘制折线图 - 使用更专业的配色
            # 折线加粗到1.2倍：2.0 * 1.2 = 2.4
            ax_layer.plot(steps, layer_ratios, color='#1f77b4', linewidth=2.4, alpha=0.9, marker='o', markersize=4, markevery=max(1, num_diffusion_steps//20))

            # 设置标签（更简洁，加粗并放大1.2倍）
            label_fontsize = int(12 * 1.2)  # 14.4 -> 14
            ax_layer.set_xlabel('Diffusion Step', fontsize=label_fontsize, fontweight='bold')
            ax_layer.set_ylabel('Attention Ratio', fontsize=label_fontsize, fontweight='bold')

            # 网格样式（更subtle）
            ax_layer.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax_layer.set_xlim(0.5, num_diffusion_steps + 0.5)

            # 显示所有边框，使用浅蓝色
            light_blue = '#ADD8E6'
            for spine in ax_layer.spines.values():
                spine.set_visible(True)
                spine.set_color(light_blue)
                spine.set_linewidth(1.0)

            # 设置刻度样式（加粗并放大1.2倍）
            tick_fontsize = int(11 * 1.2)  # 13.2 -> 13
            ax_layer.tick_params(direction='in', length=4, width=0.8, labelsize=tick_fontsize)
            # 设置刻度标签加粗
            for label in ax_layer.get_xticklabels():
                label.set_fontweight('bold')
            for label in ax_layer.get_yticklabels():
                label.set_fontweight('bold')

            plt.tight_layout()

            # 保存该层的折线图（更高DPI，适合论文）
            if question_id is not None:
                layer_fig_file = os.path.join(output_dir, f"diffusion_attention_ratio_q{question_id}_layer_{layer_idx}.png")
            else:
                layer_fig_file = os.path.join(output_dir, f"diffusion_attention_ratio_layer_{layer_idx}.png")
            plt.savefig(layer_fig_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()

            if verbose:
                print(f"    ✓ Layer {layer_idx} 折线图已保存: {os.path.basename(layer_fig_file)}")

    return ratios


def auto_detect_questions(coco_baseline_dir="pope_coco", split="adversarial", coco_root="/home/liying/Documents/dataset/coco"):
    """
    从 pope_coco 目录读取问题文件

    Args:
        coco_baseline_dir: pope_coco 目录路径
        split: split 名称 (adversarial, popular, random)
        coco_root: COCO 数据集根目录
    """
    coco_baseline_dir = Path(coco_baseline_dir)
    coco_root = Path(coco_root)

    # 构建问题文件路径
    question_file = coco_baseline_dir / f"coco_pope_{split}.json"

    if not question_file.exists():
        raise FileNotFoundError(f"找不到问题文件: {question_file}")

    all_questions = []

    # 读取 JSONL 格式的文件（每行一个 JSON）
    with open(question_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            # 构建完整的图像路径
            # sample['image'] 格式: "val2014/COCO_val2014_000000031041.jpg"
            image_relative_path = sample['image']
            image_path = coco_root / image_relative_path
            image_path = image_path.resolve()

            # 检查图像文件是否存在
            if not image_path.exists():
                print(f"⚠️  警告: 图像文件不存在: {image_path}")
                continue

            all_questions.append({
                "question_id": sample['question_id'],
                "image": str(image_path),
                "text": sample['text']
            })

    # 按 question_id 排序
    all_questions.sort(key=lambda x: x['question_id'])

    return all_questions


def auto_generate_gt_file(coco_baseline_dir="pope_coco", split="adversarial", coco_root="/home/liying/Documents/dataset/coco", output_file=None, results_dir=None):
    """
    从 pope_coco 目录读取真值（Ground Truth）文件

    Args:
        coco_baseline_dir: pope_coco 目录路径
        split: split 名称 (adversarial, popular, random)
        coco_root: COCO 数据集根目录
        output_file: 输出文件路径（如果不指定，将自动生成）
        results_dir: 结果目录（如果不指定，将使用默认路径）

    Returns:
        真值文件路径
    """
    coco_baseline_dir = Path(coco_baseline_dir)
    coco_root = Path(coco_root)

    # 构建问题文件路径（GT 数据也在同一个文件中）
    gt_file = coco_baseline_dir / f"coco_pope_{split}.json"

    if not gt_file.exists():
        raise FileNotFoundError(f"找不到真值文件: {gt_file}")

    if output_file is None:
        # 如果未指定输出文件，保存到 results 目录
        if results_dir is None:
            project_root = Path(__file__).parent
            results_dir = os.path.join(project_root, "results", "pope")
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, f"pope_gt_{split}.json")
    else:
        # 如果指定了输出文件，确保目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir:  # 如果 output_dir 不为空字符串，创建目录
            os.makedirs(output_dir, exist_ok=True)
        # 如果 output_dir 为空字符串，说明文件在当前目录，不需要创建目录

    all_gt_data = []

    # 读取 JSONL 格式的文件（每行一个 JSON）
    with open(gt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            # 构建完整的图像路径
            # sample['image'] 格式: "val2014/COCO_val2014_000000031041.jpg"
            image_relative_path = sample['image']
            image_path = coco_root / image_relative_path
            image_path = image_path.resolve()

            all_gt_data.append({
                "question_id": sample['question_id'],
                "image": str(image_path),
                "text": sample['text'],
                "label": sample['label'].lower().strip()  # 确保 label 是小写
            })

    # 按 question_id 排序
    all_gt_data.sort(key=lambda x: x['question_id'])

    # 保存真值文件（JSONL 格式，每行一个 JSON）
    with open(output_file, 'w', encoding='utf-8') as f:
        for gt_item in all_gt_data:
            f.write(json.dumps(gt_item, ensure_ascii=False) + '\n')

    print(f"✓ 已生成真值文件: {output_file} ({len(all_gt_data)} 个样本)")
    return output_file


def auto_generate_question_file(coco_baseline_dir="pope_coco", split="adversarial", coco_root="/home/liying/Documents/dataset/coco", output_file=None, results_dir=None):
    """
    自动生成问题文件

    Returns:
        问题文件路径
    """
    if output_file is None:
        # 如果未指定输出文件，保存到 results 目录
        if results_dir is None:
            project_root = Path(__file__).parent
            results_dir = os.path.join(project_root, "results", "pope")
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, f"pope_questions_{split}.jsonl")
    else:
        # 如果指定了输出文件，确保目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir:  # 如果 output_dir 不为空字符串，创建目录
            os.makedirs(output_dir, exist_ok=True)
        # 如果 output_dir 为空字符串，说明文件在当前目录，不需要创建目录

    questions = auto_detect_questions(coco_baseline_dir, split, coco_root)

    # 保存问题文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')

    print(f"✓ 已生成问题文件: {output_file} ({len(questions)} 个问题)")
    return output_file


def prepare_inputs(model, tokenizer, image_processor, image_file: str, prompt: str, conv_mode: str, device: str, verbose: bool = False):
    """
    准备模型输入（参考 test_llava_v15_7b.py）

    Returns:
        input_ids, image_tensor, stopping_criteria
    """
    # 加载图像
    image = load_image(image_file)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    if verbose:
        print(f"\n  [输入准备] 图像信息:")
        print(f"    - 图像路径: {image_file}")
        print(f"    - 图像尺寸: {image.size}")
        print(f"    - 图像张量形状: {image_tensor.shape}")

    # 准备文本输入
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt

    # 对于 Yes/No 问题，添加明确的输出格式说明（参考 pope_llava.py）
    # 这有助于模型生成更简洁的回答
    qs = qs + " Please answer with Yes or No only."

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    if verbose:
        print(f"  [输入准备] 文本信息:")
        print(f"    - 原始提示词: {prompt}")
        print(f"    - 完整提示词长度: {len(full_prompt)} 字符")
        print(f"    - 完整提示词预览: {full_prompt[:200]}...")

    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                     return_tensors='pt').unsqueeze(0).to(device)

    if verbose:
        print(f"  [输入准备] Token 信息:")
        print(f"    - input_ids 形状: {input_ids.shape}")
        print(f"    - input_ids 长度: {input_ids.shape[1]} tokens")
        # 解码前几个 token 看看
        decoded_input = tokenizer.decode(input_ids[0, :20], skip_special_tokens=False)
        print(f"    - 前20个 tokens 解码: {decoded_input}")

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    if verbose:
        print(f"  [输入准备] 停止条件:")
        print(f"    - 停止字符串: '{stop_str}'")

    return input_ids, image_tensor, stopping_criteria, stop_str


def generate_response(model, tokenizer, input_ids, image_tensor, stopping_criteria,
                     temperature, top_p, max_new_tokens, device,
                     use_deco=False, alpha=None, threshold_top_p=None,
                     threshold_top_k=None, early_exit_layers=None, num_beams=1, verbose: bool = False):
    """
    生成回答（参考 test_llava_v15_7b.py）

    Returns:
        outputs: 生成的文本
        output_token_len: 生成的 token 长度
        input_token_len: 输入的 token 长度
    """
    do_sample = True if temperature > 0 else False

    if verbose:
        print(f"\n  [生成参数] 配置信息:")
        print(f"    - use_deco: {use_deco}")
        print(f"    - do_sample: {do_sample}")
        print(f"    - temperature: {temperature if temperature > 0 else 'None (greedy)'}")
        print(f"    - top_p: {top_p}")
        print(f"    - num_beams: {num_beams}")
        print(f"    - max_new_tokens: {max_new_tokens}")
        if use_deco:
            print(f"    - alpha: {alpha}")
            print(f"    - threshold_top_p: {threshold_top_p}")
            print(f"    - threshold_top_k: {threshold_top_k}")
            print(f"    - early_exit_layers: {early_exit_layers}")

    # 准备生成参数（完全参考 test_llava_v15_7b.py 的实现）
    # LLaVA 的 generate 方法使用 inputs 作为关键字参数
    generate_kwargs = {
        "inputs": input_ids,  # 注意：LLaVA 使用 inputs 参数名（与 test_llava_v15_7b.py 保持一致）
        "images": image_tensor.unsqueeze(0).half().to(device),
        "do_sample": do_sample,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p,
        "num_beams": num_beams,  # 添加 num_beams 参数（重要！）
        "max_new_tokens": max_new_tokens,
        "return_dict": True,
        "return_dict_in_generate": True,
        "output_hidden_states": True,
        "stopping_criteria": [stopping_criteria]
    }

    if use_deco:
        generate_kwargs.update({
            "use_deco": True,
            "alpha": alpha,
            "threshold_top_p": threshold_top_p,
            "threshold_top_k": threshold_top_k,
            "early_exit_layers": early_exit_layers,
        })

    if verbose:
        print(f"  [生成过程] 开始生成...")
        print(f"    - input_ids 形状: {input_ids.shape}")
        print(f"    - images 形状: {image_tensor.unsqueeze(0).half().to(device).shape}")

    with torch.inference_mode():
        with torch.no_grad():
            # 使用 **generate_kwargs 展开所有参数（与 test_llava_v15_7b.py 保持一致）
            output_dict = model.generate(**generate_kwargs)

    # 解码输出（完全参考 test_llava_v15_7b.py 的实现）
    output_ids = output_dict.sequences
    input_token_len = input_ids.shape[1]

    # 检查 output_ids 是否包含 input_ids（与 test_llava_v15_7b.py 保持一致）
    if verbose:
        print(f"\n  [生成过程] 检查 output_ids 内容:")
        print(f"    - output_ids 形状: {output_ids.shape}")
        print(f"    - input_ids 形状: {input_ids.shape}")
        print(f"    - output_ids 前几个 token: {output_ids[0, :min(5, output_ids.shape[1])].tolist()}")
        print(f"    - input_ids 前几个 token: {input_ids[0, :min(5, input_ids.shape[1])].tolist()}")

        # 检查 output_ids 是否以 input_ids 开头
        if output_ids.shape[1] >= input_token_len:
            prefix_match = (input_ids[0] == output_ids[0, :input_token_len]).all().item()
            print(f"    - output_ids 前 {input_token_len} 个 token 是否与 input_ids 匹配: {prefix_match}")
        else:
            print(f"    - ⚠️  警告: output_ids 长度 ({output_ids.shape[1]}) < input_ids 长度 ({input_token_len})")
            print(f"    - 这说明 output_ids 可能只包含新生成的 token，而不是完整序列")
            print(f"    - 需要手动拼接 input_ids 和 output_ids")

    # 如果 output_ids 不包含 input_ids，手动拼接（与 test_llava_v15_7b.py 保持一致）
    if output_ids.shape[1] < input_token_len:
        # output_ids 只包含新生成的 token，需要拼接 input_ids
        if verbose:
            print(f"\n  [修复] 手动拼接 input_ids 和 output_ids:")
            print(f"    - input_ids: {input_ids.shape}")
            print(f"    - output_ids (仅新生成的): {output_ids.shape}")
        output_ids = torch.cat([input_ids, output_ids], dim=1)
        if verbose:
            print(f"    - 拼接后的 output_ids: {output_ids.shape}")
    elif output_ids.shape[1] >= input_token_len:
        # 检查前 input_token_len 个 token 是否与 input_ids 匹配
        prefix_match = (input_ids[0] == output_ids[0, :input_token_len]).all().item()
        if not prefix_match:
            if verbose:
                print(f"\n  [修复] output_ids 前缀与 input_ids 不匹配，使用 input_ids 替换前缀")
            # 替换前缀为 input_ids
            output_ids = torch.cat([input_ids, output_ids[:, input_token_len:]], dim=1)

    output_token_len = output_ids.shape[1] - input_token_len

    if verbose:
        print(f"\n  [生成结果] Token 信息:")
        print(f"    - input_ids length: {input_token_len}")
        print(f"    - output_ids 形状: {output_ids.shape}")
        print(f"    - output_ids length: {output_ids.shape[1]}")
        print(f"    - new generated tokens length: {output_token_len}")

        # 显示新生成的 token IDs
        if output_token_len > 0:
            generated_ids = output_ids[:, input_token_len:]
            print(f"    - 生成的 token IDs 形状: {generated_ids.shape}")
            print(f"    - 生成的 token IDs: {generated_ids[0].tolist()}")
            generated_decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
            print(f"    - 生成的 token 解码（带特殊token）: {repr(generated_decoded)}")
        else:
            print(f"    - ⚠️  警告: 没有生成新的 token！")

    # 获取新生成的 token（跳过可能的 BOS token，与 test_llava_v15_7b.py 保持一致）
    if output_token_len > 0:
        generated_ids = output_ids[:, input_token_len:]
        # 如果新生成的 token 以 BOS token 开头，跳过它
        bos_token_id = tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None else None
        if bos_token_id is not None and generated_ids.shape[1] > 0 and generated_ids[0, 0].item() == bos_token_id:
            if verbose:
                print(f"\n  [生成结果] 检测到新生成的 token 以 BOS token ({bos_token_id}) 开头，跳过它")
            generated_ids = generated_ids[:, 1:]  # 跳过第一个 BOS token
            if generated_ids.shape[1] > 0:
                outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            else:
                outputs = ""
        else:
            outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    else:
        outputs = ""

    if verbose:
        print(f"\n  [生成结果] 最终输出:")
        raw_output = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=False)[0] if output_token_len > 0 else '(empty)'
        print(f"    - 原始输出 (带特殊token): {raw_output}")
        print(f"    - 最终输出 (去特殊token): {repr(outputs)}")
        print(f"    - 输出长度: {len(outputs)} 字符")

    return outputs, output_token_len, input_token_len


def compare_deco_vs_vanilla(deco_results, vanilla_results, deco_answers_file, vanilla_answers_file,
                            gt_file, output_file):
    """
    对比 Deco 和 Vanilla 的结果，生成对比表格和不一致 case 的 JSON 文件

    Args:
        deco_results: Deco 版本的评估结果
        vanilla_results: Vanilla 版本的评估结果
        deco_answers_file: Deco 版本的答案文件路径
        vanilla_answers_file: Vanilla 版本的答案文件路径
        gt_file: 真值文件路径
        output_file: 输出 JSON 文件路径
    """
    # 加载答案文件
    deco_answers = {item["question_id"]: item for item in [json.loads(line) for line in open(deco_answers_file, 'r', encoding='utf-8')]}
    vanilla_answers = {item["question_id"]: item for item in [json.loads(line) for line in open(vanilla_answers_file, 'r', encoding='utf-8')]}
    gt_data = {item["question_id"]: item for item in [json.loads(line) for line in open(gt_file, 'r', encoding='utf-8')]}

    # 找到结果不一致的 case
    inconsistent_cases = []
    common_question_ids = set(deco_answers.keys()) & set(vanilla_answers.keys())

    for qid in common_question_ids:
        deco_answer = deco_answers[qid].get("text", "").strip()
        vanilla_answer = vanilla_answers[qid].get("text", "").strip()

        if deco_answer != vanilla_answer:
            # 获取图片文件名（不包含路径）
            image_path = deco_answers[qid].get("image", "")
            image_filename = os.path.basename(image_path) if image_path else ""

            # 获取 GT 答案
            gt_answer = gt_data.get(qid, {}).get("label", "").strip().lower()

            case_info = {
                "question_id": qid,
                "question": deco_answers[qid].get("prompt", ""),
                "image": image_filename,  # 只保存文件名
                "gt_answer": gt_answer,
                "vanilla_answer": vanilla_answer,
                "deco_answer": deco_answer,
                "vanilla_correct": vanilla_answer.lower() == gt_answer,
                "deco_correct": deco_answer.lower() == gt_answer,
                "vanilla_raw_output": vanilla_answers[qid].get("metadata", {}).get("raw_output", ""),
                "deco_raw_output": deco_answers[qid].get("metadata", {}).get("raw_output", "")
            }
            inconsistent_cases.append(case_info)

    # 保存不一致的 case 到 JSON 文件
    comparison_result = {
        "summary": {
            "total_cases": len(common_question_ids),
            "inconsistent_cases": len(inconsistent_cases),
            "consistent_cases": len(common_question_ids) - len(inconsistent_cases),
            "inconsistency_rate": len(inconsistent_cases) / len(common_question_ids) if len(common_question_ids) > 0 else 0
        },
        "metrics_comparison": {
            "vanilla": {
                "accuracy": vanilla_results.get('metrics', {}).get('accuracy', 0),
                "precision": vanilla_results.get('metrics', {}).get('precision', 0),
                "recall": vanilla_results.get('metrics', {}).get('recall', 0),
                "f1": vanilla_results.get('metrics', {}).get('f1', 0)
            },
            "deco": {
                "accuracy": deco_results.get('metrics', {}).get('accuracy', 0),
                "precision": deco_results.get('metrics', {}).get('precision', 0),
                "recall": deco_results.get('metrics', {}).get('recall', 0),
                "f1": deco_results.get('metrics', {}).get('f1', 0)
            },
            "difference": {
                "accuracy": deco_results.get('metrics', {}).get('accuracy', 0) - vanilla_results.get('metrics', {}).get('accuracy', 0),
                "precision": deco_results.get('metrics', {}).get('precision', 0) - vanilla_results.get('metrics', {}).get('precision', 0),
                "recall": deco_results.get('metrics', {}).get('recall', 0) - vanilla_results.get('metrics', {}).get('recall', 0),
                "f1": deco_results.get('metrics', {}).get('f1', 0) - vanilla_results.get('metrics', {}).get('f1', 0)
            }
        },
        "inconsistent_cases": inconsistent_cases
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_result, f, ensure_ascii=False, indent=2)

    return comparison_result


def print_comparison_table(deco_results, vanilla_results, split_name=""):
    """
    打印 Deco vs Vanilla 的对比表格

    Args:
        deco_results: Deco 版本的评估结果
        vanilla_results: Vanilla 版本的评估结果
        split_name: Split 名称（可选）
    """
    deco_metrics = deco_results.get('metrics', {})
    vanilla_metrics = vanilla_results.get('metrics', {})

    title = f"Deco vs Vanilla 对比{' - ' + split_name if split_name else ''}"
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"{'指标':<15} {'Vanilla':<12} {'Deco':<12} {'差异':<12} {'变化':<10}")
    print("-" * 80)

    metrics_list = [
        ('Accuracy', 'accuracy'),
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('F1 Score', 'f1')
    ]

    for metric_name, metric_key in metrics_list:
        vanilla_val = vanilla_metrics.get(metric_key, 0)
        deco_val = deco_metrics.get(metric_key, 0)
        diff = deco_val - vanilla_val
        change = f"{diff:+.4f}" if diff != 0 else "0.0000"
        change_symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="

        print(f"{metric_name:<15} {vanilla_val:<12.4f} {deco_val:<12.4f} {diff:<12.4f} {change_symbol} {change}")

    print("=" * 80)


def save_summary_to_file(summary_file, args, gt_file, question_file, answers_file, errors_file,
                         results=None, model_name=None, error=None):
    """
    保存评估结果总结到txt文件

    Args:
        summary_file: 总结文件路径
        args: 命令行参数
        gt_file: 真值文件路径
        question_file: 问题文件路径
        answers_file: 答案文件路径
        errors_file: 错误样本文件路径
        results: 评估结果字典（如果评估成功）
        model_name: 模型名称
        error: 错误信息（如果评估失败）
    """
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("POPE 评估结果总结\n")
        f.write("=" * 80 + "\n\n")

        # 基本信息
        f.write("【基本信息】\n")
        f.write("-" * 80 + "\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据集 Split: {args.split}\n")
        f.write(f"模型路径: {args.model_path}\n")
        if model_name:
            f.write(f"模型名称: {model_name}\n")
        f.write(f"设备: {args.device}\n")
        f.write(f"评测样本数: {args.num_samples if args.num_samples > 0 else '全部'}\n")
        f.write("\n")

        # Deco配置
        f.write("【Deco 配置】\n")
        f.write("-" * 80 + "\n")
        f.write(f"使用 Deco: {'是' if args.use_deco else '否'}\n")
        if args.use_deco:
            f.write(f"  - Alpha: {args.alpha}\n")
            f.write(f"  - Threshold Top-p: {args.threshold_top_p}\n")
            f.write(f"  - Threshold Top-k: {args.threshold_top_k}\n")
            f.write(f"  - Early Exit Layers: {args.start_layer}-{args.end_layer}\n")
        f.write("\n")

        # 生成参数
        f.write("【生成参数】\n")
        f.write("-" * 80 + "\n")
        f.write(f"Temperature: {args.temperature if args.temperature > 0 else 'None (greedy)'}\n")
        f.write(f"Top-p: {args.top_p if args.top_p else 'None'}\n")
        f.write(f"Max New Tokens: {args.max_new_tokens}\n")
        f.write(f"Random Seed: {args.seed}\n")
        f.write("\n")

        # 文件路径
        f.write("【文件路径】\n")
        f.write("-" * 80 + "\n")
        f.write(f"真值文件 (GT): {gt_file}\n")
        f.write(f"问题文件: {question_file}\n")
        f.write(f"答案文件: {answers_file}\n")
        f.write(f"错误样本文件: {errors_file}\n")
        f.write(f"总结文件: {summary_file}\n")
        f.write("\n")

        # 评估结果
        f.write("【评估结果】\n")
        f.write("-" * 80 + "\n")
        if results is not None:
            metrics = results.get('metrics', {})
            f.write(f"Accuracy:  {metrics.get('accuracy', 0):.4f}\n")
            f.write(f"Precision: {metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall:    {metrics.get('recall', 0):.4f}\n")
            f.write(f"F1 Score:  {metrics.get('f1', 0):.4f}\n")

            # 如果有错误样本信息
            if 'error_samples' in results:
                error_count = len(results['error_samples'])
                total_count = results.get('total_count', 0)
                f.write(f"\n错误样本数: {error_count} / {total_count}\n")
        elif error:
            f.write(f"评估失败: {error}\n")
        else:
            f.write("评估结果未生成\n")
        f.write("\n")

        # 分隔线
        f.write("=" * 80 + "\n")
        f.write("总结文件生成时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("=" * 80 + "\n")


def eval_model(args):
    """评估模型"""
    print("=" * 80)
    print("POPE 数据集评估")
    print("=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"设备: {args.device}")
    print(f"问题文件: {args.question_file}")
    print(f"答案文件: {args.answers_file}")
    if args.use_deco:
        print(f"Deco 参数: use_deco={args.use_deco}, alpha={args.alpha}, layers={args.start_layer}-{args.end_layer}")
    else:
        print(f"使用原生 LLaVA 模型（Deco 已禁用）")
    print("=" * 80)

    # 加载模型
    print("\n[1/3] 正在加载模型...")
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    device = args.device if isinstance(args.device, str) else f"cuda:{args.device}" if args.device >= 0 else "cpu"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, device=device
    )
    print(f"✓ 模型加载完成: {model_name}")

    # 确定对话模式
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    # 加载问题
    print(f"\n[2/3] 正在加载问题文件: {args.question_file}")
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    total_questions = len(questions)

    # 如果指定了评测数量，则只使用前 N 个
    if args.num_samples > 0:
        questions = questions[:args.num_samples]
        print(f"✓ 加载了 {total_questions} 个问题，将评测前 {len(questions)} 个")
    else:
        print(f"✓ 加载了 {len(questions)} 个问题，将评测全部")

    # 如果启用了扩散分析，先快速评估所有 case，过滤出回答 "Yes" 的 case
    if hasattr(args, 'enable_diffusion_analysis') and args.enable_diffusion_analysis:
        print(f"\n[预评估] 扩散分析已启用，先快速评估所有 case 以过滤出回答 'Yes' 的 case...")
        yes_questions = []
        yes_question_ids = set()

        # 准备 Deco 参数（用于预评估）
        early_exit_layers = None
        if args.use_deco:
            early_exit_layers = [i for i in range(args.start_layer, args.end_layer)]

        for sample_idx, line in enumerate(tqdm(questions, desc="预评估进度")):
            idx = line["question_id"]
            image_file = line["image"]
            qs = line["text"]

            # 准备输入
            input_ids, image_tensor, stopping_criteria, stop_str = prepare_inputs(
                model, tokenizer, image_processor, image_file, qs, conv_mode, device, verbose=False
            )

            # 生成回答
            outputs, _, _ = generate_response(
                model, tokenizer, input_ids, image_tensor, stopping_criteria,
                args.temperature, args.top_p, args.max_new_tokens, device,
                use_deco=args.use_deco,
                alpha=args.alpha,
                threshold_top_p=args.threshold_top_p,
                threshold_top_k=args.threshold_top_k,
                early_exit_layers=early_exit_layers,
                num_beams=1,
                verbose=False
            )

            # 移除停止字符串
            if outputs and outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            # 转换为 Yes/No
            answer = recorder(outputs)

            # 只保留回答 "Yes" 的 case
            if answer == "Yes":
                yes_questions.append(line)
                yes_question_ids.add(idx)

        print(f"✓ 预评估完成: 共 {len(questions)} 个 case，其中 {len(yes_questions)} 个回答 'Yes'")
        print(f"  将只对这 {len(yes_questions)} 个 'Yes' case 进行后续处理（包括扩散分析）")

        # 更新 questions 列表，只保留回答 "Yes" 的 case
        questions = yes_questions
        if len(questions) == 0:
            print("⚠️  警告: 没有找到回答 'Yes' 的 case，将跳过所有处理")
            return

    # 准备输出文件
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file) if os.path.dirname(answers_file) else ".", exist_ok=True)
    ans_file = open(answers_file, "w")

    # 准备 Deco 参数
    early_exit_layers = None
    if args.use_deco:
        early_exit_layers = [i for i in range(args.start_layer, args.end_layer)]

    # 处理每个问题
    print(f"\n[3/3] 开始评估...")

    # 计算需要输出详细信息的样本索引（最多10个，均匀分布）
    total_samples = len(questions)
    max_debug_samples = min(5, total_samples)
    if total_samples > 0:
        debug_indices = set()
        if total_samples <= max_debug_samples:
            # 如果样本数少于等于5个，全部输出详细信息
            debug_indices = set(range(total_samples))
        else:
            # 均匀分布选择样本
            step = total_samples / max_debug_samples
            for i in range(max_debug_samples):
                idx = int(i * step)
                debug_indices.add(idx)

        print(f"将输出 {len(debug_indices)} 个样本的详细信息用于调试（样本索引: {sorted(debug_indices)}）")

    for sample_idx, line in enumerate(tqdm(questions, desc="处理进度")):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs

        # 判断是否需要输出详细信息
        verbose = sample_idx in debug_indices

        if verbose:
            print("\n" + "=" * 80)
            print(f"[样本 {sample_idx + 1}/{total_samples}] Question ID: {idx}")
            print("=" * 80)
            print(f"问题: {qs}")
            print(f"图像: {image_file}")

        # 准备输入
        if verbose:
            print("\n" + "-" * 80)
            print("[准备输入]")
            print("-" * 80)
        input_ids, image_tensor, stopping_criteria, stop_str = prepare_inputs(
            model, tokenizer, image_processor, image_file, qs, conv_mode, device, verbose=verbose
        )

        # 生成回答
        if verbose:
            print("\n" + "-" * 80)
            print("[生成回答]")
            print("-" * 80)
        outputs, output_token_len, input_token_len = generate_response(
            model, tokenizer, input_ids, image_tensor, stopping_criteria,
            args.temperature, args.top_p, args.max_new_tokens, device,
            use_deco=args.use_deco,
            alpha=args.alpha,
            threshold_top_p=args.threshold_top_p,
            threshold_top_k=args.threshold_top_k,
            early_exit_layers=early_exit_layers,
            num_beams=1,  # 添加 num_beams 参数，默认值为 1（贪婪搜索）
            verbose=verbose
        )

        # 移除停止字符串
        if outputs and outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        # 如果输出为空，记录警告
        if not outputs:
            if verbose:
                print(f"\n  [Warning] 问题 {idx} 生成结果为空，output_token_len={output_token_len}")
            else:
                print(f"  [Warning] 问题 {idx} 生成结果为空，output_token_len={output_token_len}")

        # 转换为 Yes/No
        answer = recorder(outputs)

        if verbose:
            print(f"\n  [后处理] 结果转换:")
            print(f"    - 原始输出: '{outputs}'")
            print(f"    - 转换后答案: '{answer}'")
            print("=" * 80)

        # 如果启用了扩散分析，进行扩散分析（此时所有 case 都应该是 "Yes"）
        if hasattr(args, 'enable_diffusion_analysis') and args.enable_diffusion_analysis:
            if answer == "Yes":
                if verbose:
                    print("\n" + "-" * 80)
                    print("[扩散分析] 开始扩散过程分析...")
                    print("-" * 80)
                else:
                    print(f"  [扩散分析] Question ID {idx}: 开始扩散分析...")

                # 创建输出目录（为每个case创建单独的子文件夹）
                diffusion_analysis_base_dir = os.path.join(os.path.dirname(answers_file), "diffusion_analysis")
                os.makedirs(diffusion_analysis_base_dir, exist_ok=True)
                diffusion_output_dir = os.path.join(diffusion_analysis_base_dir, f"q{idx}")
                os.makedirs(diffusion_output_dir, exist_ok=True)

                # 进行扩散分析
                ratios = analyze_diffusion_attention(
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    image_file=image_file,
                    prompt=qs,
                    conv_mode=conv_mode,
                    device=device,
                    num_diffusion_steps=getattr(args, 'num_diffusion_steps', 1000),
                    image_token_start=35,
                    num_image_tokens=576,
                    num_total_layers=32,
                    num_heads=32,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    output_dir=diffusion_output_dir,
                    question_id=idx,
                    verbose=verbose,
                    target_layers=getattr(args, 'target_layers', None)
                )

                if verbose:
                    print(f"  ✓ 扩散分析完成，共 {len(ratios)} 步")
            else:
                # 这种情况不应该发生，因为已经在预评估阶段过滤掉了
                print(f"  ⚠️  警告: Question ID {idx} 的答案为 '{answer}'，但应该在预评估阶段已被过滤")

        # 保存结果
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": answer,
            "model_id": model_name,
            "image": image_file,
            "metadata": {
                "output_token_len": output_token_len,
                "input_token_len": input_token_len,
                "raw_output": outputs
            }
        }, ensure_ascii=False) + "\n")
        ans_file.flush()

    ans_file.close()
    print(f"\n✓ 评估完成！结果已保存到: {answers_file}")


def main():
    """主函数 - 自动检测并使用默认配置"""
    # 项目根目录
    project_root = Path(__file__).parent

    # 自动检测可用 GPU
    if torch.cuda.is_available():
        device = 0
        device_str = "cuda:0"
    else:
        device = -1
        device_str = "cpu"
        print("⚠ 未检测到 CUDA，将使用 CPU（速度较慢）")

    # 默认配置
    default_config = {
        "model_path": llava_v15_7b_path,
        "device": device_str,
        "coco_baseline_dir": str(project_root / "pope_coco"),
        "coco_root": "/home/liying/Documents/dataset/coco",
        "split": ["adversarial", "popular", "random"],  # 默认评估 adversarial split
        "use_deco": False,
        "alpha": 0.8,
        "threshold_top_p": 0.9,
        "threshold_top_k": 20,
        "start_layer": 20,
        "end_layer": 29,
        "temperature": 0,
        "top_p": None,
        "max_new_tokens": 15,  # POPE 只需要 Yes/No，但给一些缓冲
        "num_samples": 10,
        "seed": 42,
        "target_layers": [0, 5, 7, 9, 11, 13, 15, 17, 21, 25, 29, 31],
        "enable_diffusion_analysis": True,
        "num_diffusion_steps": 40
    }

    # 解析参数（所有参数都有默认值）
    parser = argparse.ArgumentParser(description="POPE 评估 - 直接运行版本（所有参数可选）")

    # 数据集参数
    # 注意：如果默认值是列表，argparse 需要特殊处理
    default_split = default_config["split"]
    if isinstance(default_split, list):
        # 如果默认值是列表，转换为逗号分隔的字符串
        default_split_str = ','.join(default_split)
    else:
        default_split_str = str(default_split)

    parser.add_argument("--split", type=str, default=default_split_str,
                       help="数据集 split，可以是单个值或逗号分隔的多个值（例如: adversarial,popular,random）")
    parser.add_argument("--coco-baseline-dir", type=str, default=default_config["coco_baseline_dir"],
                       help="coco_baseline_dir 目录路径")
    parser.add_argument("--coco-root", type=str, default=default_config["coco_root"],
                       help="COCO 数据集根目录路径")
    parser.add_argument("--question-file", type=str, default=None,
                       help="问题文件路径（如果不指定，将自动生成）")

    # 模型参数
    parser.add_argument("--model-path", type=str, default=default_config["model_path"],
                       help="模型路径")
    parser.add_argument("--model-base", type=str, default=None, help="基础模型路径")
    parser.add_argument("--device", type=str, default=default_config["device"],
                       help="设备 (cuda:0/cpu)")

    # 输出参数
    parser.add_argument("--answers-file", type=str, default=None,
                       help="输出答案文件路径（如果不指定，将自动生成）")

    # 生成参数
    parser.add_argument("--temperature", type=float, default=default_config["temperature"],
                       help="生成温度（-1表示贪婪生成）")
    parser.add_argument("--top-p", type=float, default=default_config["top_p"], help="Top-p采样")
    parser.add_argument("--max-new-tokens", type=int, default=default_config["max_new_tokens"],
                       help="最大生成 token 数")
    parser.add_argument("--num-samples", type=int, default=default_config["num_samples"],
                       help="评测数量（0表示评测所有case，非零表示只评测前N个）")

    # Deco 参数（默认不使用 Deco，只使用原生 LLaVA 模型）
    parser.add_argument("--use-deco", type=bool, default=default_config["use_deco"],
                       help="启用 Deco 早退机制（默认：False，使用原生 LLaVA 模型）")
    parser.add_argument("--alpha", type=float, default=default_config["alpha"],
                       help="Deco 置信度阈值参数")
    parser.add_argument("--threshold-top-p", type=float, default=default_config["threshold_top_p"],
                       help="早退判断的 top-p 阈值")
    parser.add_argument("--threshold-top-k", type=int, default=default_config["threshold_top_k"],
                       help="早退判断的 top-k 阈值")
    parser.add_argument("--start-layer", type=int, default=default_config["start_layer"],
                       help="允许早退的起始层索引")
    parser.add_argument("--end-layer", type=int, default=default_config["end_layer"],
                       help="允许早退的结束层索引")

    # 其他参数
    parser.add_argument("--seed", type=int, default=default_config["seed"], help="随机种子")

    # 扩散分析参数
    parser.add_argument("--enable-diffusion-analysis", type=bool, default=default_config["enable_diffusion_analysis"],
                       help="启用DDPM前向扩散分析（默认：True）")
    parser.add_argument("--num-diffusion-steps", type=int, default=default_config["num_diffusion_steps"],
                       help="扩散步数（默认：1000）")
    parser.add_argument("--target-layers", type=str, default=None,
                       help="目标层列表，逗号分隔（例如: 0,3,5,7,9,11,13,15,17,21,23,25,29,31）。如果不指定，将使用默认值")

    args = parser.parse_args()
    set_seed(args.seed)

    # 解析 target_layers 参数
    if args.target_layers is None:
        # 使用默认值
        args.target_layers = default_config.get("target_layers", [0, 3, 5, 7, 9, 11, 13, 15, 17, 21, 23, 25, 29, 31])
    else:
        # 解析逗号分隔的字符串
        try:
            args.target_layers = [int(x.strip()) for x in args.target_layers.split(',') if x.strip()]
        except ValueError:
            raise ValueError(f"无效的 target_layers 值: {args.target_layers}。应该是逗号分隔的整数列表（例如: 0,3,5,7）")

    # 解析 split 参数（支持单个值、逗号分隔的多个值，或列表）
    if isinstance(args.split, list):
        # 如果已经是列表，直接使用
        splits = [s.strip() if isinstance(s, str) else str(s) for s in args.split]
    elif isinstance(args.split, str):
        # 如果是字符串，检查是否包含逗号
        split_input = args.split.strip()
        if ',' in split_input:
            splits = [s.strip() for s in split_input.split(',')]
        else:
            splits = [split_input]
    else:
        # 其他类型，转换为字符串列表
        splits = [str(args.split)]

    # 验证 split 值
    valid_splits = ["adversarial", "popular", "random"]
    for split in splits:
        if split not in valid_splits:
            raise ValueError(f"无效的 split 值: {split}。有效值: {valid_splits}")

    print("=" * 80)
    print(f"将处理 {len(splits)} 个 split: {', '.join(splits)}")
    print("=" * 80)

    # 准备 results 目录
    results_dir = os.path.join(project_root, "results", "pope")

    # 存储所有结果
    all_results = []

    # 循环处理每个 split
    for split_idx, current_split in enumerate(splits, 1):
        print("\n" + "=" * 80)
        print(f"[{split_idx}/{len(splits)}] 处理 split: {current_split}")
        print("=" * 80)

        # 为当前 split 创建 args 副本
        split_args = argparse.Namespace(**vars(args))
        split_args.split = current_split

        # 自动生成真值文件（GT 文件）
        print("\n" + "-" * 80)
        print(f"自动生成真值文件（Ground Truth）- {current_split}")
        print("-" * 80)
        gt_file = auto_generate_gt_file(
            coco_baseline_dir=split_args.coco_baseline_dir,
            split=current_split,
            coco_root=split_args.coco_root,
            output_file=None,  # 使用默认路径（results/pope/pope_gt_{split}.json）
            results_dir=results_dir
        )

        # 自动生成问题文件（如果未指定）
        if split_args.question_file is None:
            print("\n" + "-" * 80)
            print(f"自动生成问题文件 - {current_split}")
            print("-" * 80)
            question_file = auto_generate_question_file(
                coco_baseline_dir=split_args.coco_baseline_dir,
                split=current_split,
                coco_root=split_args.coco_root,
                output_file=None,  # 使用默认路径（results/pope/pope_questions_{split}.jsonl）
                results_dir=results_dir
            )
            split_args.question_file = question_file
        else:
            # 如果指定了问题文件，只对第一个 split 使用，其他 split 会报错
            if split_idx > 1:
                raise ValueError(f"当处理多个 split 时，不能指定 --question-file。请移除该参数以自动生成问题文件。")
            # 检查问题文件是否存在
            if not os.path.exists(split_args.question_file):
                raise FileNotFoundError(f"问题文件不存在: {split_args.question_file}")

        # 自动生成答案文件路径（如果未指定）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        answers_dir = os.path.join(project_root, "results", "pope")
        os.makedirs(answers_dir, exist_ok=True)

        # 如果使用 Deco，需要同时运行 vanilla 版本进行对比
        vanilla_answers_file = None
        if split_args.use_deco:
            print("\n" + "=" * 80)
            print(f"检测到使用 Deco，将同时运行 Vanilla 版本进行对比")
            print("=" * 80)

            # 先运行 Vanilla 版本
            print("\n" + "-" * 80)
            print(f"[1/2] 运行 Vanilla 版本 - {current_split}")
            print("-" * 80)
            vanilla_args = argparse.Namespace(**vars(split_args))
            vanilla_args.use_deco = False
            vanilla_args.answers_file = os.path.join(answers_dir, f"pope_{current_split}_vanilla_{timestamp}.jsonl")

            eval_model(vanilla_args)
            vanilla_answers_file = vanilla_args.answers_file

            # 然后运行 Deco 版本
            print("\n" + "-" * 80)
            print(f"[2/2] 运行 Deco 版本 - {current_split}")
            print("-" * 80)
            if split_args.answers_file is None:
                split_args.answers_file = os.path.join(answers_dir, f"pope_{current_split}_deco_{timestamp}.jsonl")

            eval_model(split_args)
        else:
            # 不使用 Deco，正常处理
            if split_args.answers_file is None:
                split_args.answers_file = os.path.join(answers_dir, f"pope_{current_split}_vanilla_{timestamp}.jsonl")
            else:
                # 如果指定了答案文件，只对第一个 split 使用，其他 split 会报错
                if split_idx > 1:
                    raise ValueError(f"当处理多个 split 时，不能指定 --answers-file。请移除该参数以自动生成答案文件。")

            # 运行评估
            eval_model(split_args)

        print("\n" + "=" * 80)
        print(f"模型评估完成 - {current_split}")
        print("=" * 80)
        print(f"真值文件（GT）: {gt_file}")
        print(f"问题文件: {split_args.question_file}")
        print(f"答案文件: {split_args.answers_file}")

        # 自动执行评估
        print("\n" + "=" * 80)
        print(f"自动执行结果评估 - {current_split}")
        print("=" * 80)

        # 生成错误文件路径（保持与答案文件相同的命名规则）
        errors_file = split_args.answers_file.replace('.jsonl', '_errors.json')
        # 生成总结文件路径
        summary_file = split_args.answers_file.replace('.jsonl', '_summary.txt')

        # 直接调用评估函数
        results = evaluate_pope(
            gt_files_path=gt_file,
            gen_files_path=split_args.answers_file,
            output_errors_path=errors_file,
            verbose=True
        )

        print("\n" + "=" * 80)
        print(f"✓ 结果评估完成 - {current_split}")
        print("=" * 80)
        print(f"错误样本文件: {errors_file}")
        print(f"\n关键指标:")
        print(f"  - Accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"  - Precision: {results['metrics']['precision']:.4f}")
        print(f"  - Recall: {results['metrics']['recall']:.4f}")
        print(f"  - F1: {results['metrics']['f1']:.4f}")

        # 保存总结到txt文件
        save_summary_to_file(
            summary_file=summary_file,
            args=split_args,
            gt_file=gt_file,
            question_file=split_args.question_file,
            answers_file=split_args.answers_file,
            errors_file=errors_file,
            results=results,
            model_name=get_model_name_from_path(split_args.model_path)
        )
        print(f"\n✓ 结果总结已保存到: {summary_file}")

        # 如果使用 Deco，进行对比
        if split_args.use_deco and vanilla_answers_file:
            print("\n" + "=" * 80)
            print(f"对比 Deco vs Vanilla - {current_split}")
            print("=" * 80)

            # 评估 Vanilla 版本
            vanilla_errors_file = vanilla_answers_file.replace('.jsonl', '_errors.json')
            vanilla_results = evaluate_pope(
                gt_files_path=gt_file,
                gen_files_path=vanilla_answers_file,
                output_errors_path=vanilla_errors_file,
                verbose=False  # 不重复打印详细信息
            )

            # 生成对比 JSON 文件
            comparison_file = split_args.answers_file.replace('.jsonl', '_comparison.json')
            comparison_result = compare_deco_vs_vanilla(
                deco_results=results,
                vanilla_results=vanilla_results,
                deco_answers_file=split_args.answers_file,
                vanilla_answers_file=vanilla_answers_file,
                gt_file=gt_file,
                output_file=comparison_file
            )

            # 打印对比表格
            print_comparison_table(deco_results=results, vanilla_results=vanilla_results, split_name=current_split)

            print(f"\n✓ 对比结果已保存到: {comparison_file}")
            print(f"  - 总样本数: {comparison_result['summary']['total_cases']}")
            print(f"  - 不一致样本数: {comparison_result['summary']['inconsistent_cases']}")
            print(f"  - 不一致率: {comparison_result['summary']['inconsistency_rate']:.2%}")

            # 保存结果到列表（包含对比信息）
            all_results.append({
                'split': current_split,
                'gt_file': gt_file,
                'question_file': split_args.question_file,
                'answers_file': split_args.answers_file,
                'vanilla_answers_file': vanilla_answers_file,
                'errors_file': errors_file,
                'summary_file': summary_file,
                'comparison_file': comparison_file,
                'metrics': results['metrics'],
                'vanilla_metrics': vanilla_results['metrics'],
                'comparison': comparison_result
            })
        else:
            # 不使用 Deco，只保存当前结果
            all_results.append({
                'split': current_split,
                'gt_file': gt_file,
                'question_file': split_args.question_file,
                'answers_file': split_args.answers_file,
                'errors_file': errors_file,
                'summary_file': summary_file,
                'metrics': results['metrics']
            })

        print("=" * 80)

    # 打印所有结果的总结
    if len(splits) > 1:
        print("\n" + "=" * 80)
        print("所有 Split 评估总结")
        print("=" * 80)
        print(f"{'Split':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
        print("-" * 80)
        for result in all_results:
            if 'metrics' in result:
                metrics = result['metrics']
                print(f"{result['split']:<15} {metrics.get('accuracy', 0):<12.4f} {metrics.get('precision', 0):<12.4f} {metrics.get('recall', 0):<12.4f} {metrics.get('f1', 0):<12.4f}")
            else:
                print(f"{result['split']:<15} {'ERROR':<12}")
        print("=" * 80)

        # 计算平均指标（如果所有都成功）
        successful_results = [r for r in all_results if 'metrics' in r]
        if successful_results:
            avg_metrics = {
                'accuracy': sum(r['metrics']['accuracy'] for r in successful_results) / len(successful_results),
                'precision': sum(r['metrics']['precision'] for r in successful_results) / len(successful_results),
                'recall': sum(r['metrics']['recall'] for r in successful_results) / len(successful_results),
                'f1': sum(r['metrics']['f1'] for r in successful_results) / len(successful_results),
            }
            print(f"\n平均指标（{len(successful_results)} 个 split）:")
            print(f"  - Accuracy:  {avg_metrics['accuracy']:.4f}")
            print(f"  - Precision: {avg_metrics['precision']:.4f}")
            print(f"  - Recall:    {avg_metrics['recall']:.4f}")
            print(f"  - F1 Score:  {avg_metrics['f1']:.4f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
