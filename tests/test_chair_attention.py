#!/usr/bin/env python3
"""
CHAIR 评估脚本 - 生成图像描述并保存为 JSONL 格式
参考 run_pope_eval.py 的实现, 针对 CHAIR benchmark 优化
自动检测数据集和模型, 使用默认参数, 无需输入参数即可运行

CHAIR 评估需要:
1. COCO 2014 val2014 图像目录
2. 生成的描述文件(JSONL 格式): {"image_id": int, "caption": str}
3. COCO annotations 目录(用于后续的 chair.py 评估)

使用步骤:
1. 运行此脚本生成描述文件:
   python run_chair_eval.py --coco-root /path/to/coco --output-file results/chair/captions.jsonl

2. 使用 chair.py 计算 CHAIR 指标:
   python chair.py --cap_file results/chair/captions.jsonl --image_id_key image_id --caption_key caption \
                   --coco_path /path/to/coco/annotations_trainval2014/annotations/
"""

import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
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

import project as project
from PIL import Image
import requests
from io import BytesIO
from transformers import set_seed
from eval_tool.chair import evaluate_chair, CHAIR
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def load_image(image_file):
    """加载图像文件, 支持本地文件和 URL"""
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"图像文件不存在: {image_file}")
        image = Image.open(image_file).convert("RGB")
    return image


def get_coco_val2014_images(coco_root: str, image_id_list: Optional[List[int]] = None, max_images: int = 0):
    """
    获取 COCO val2014 图像列表

    Args:
        coco_root: COCO 数据集根目录(包含 val2014 子目录)
        image_id_list: 可选的图像 ID 列表, 如果提供则只返回这些图像
        max_images: 最大图像数量(0 表示全部)

    Returns:
        List[Dict]: 包含 image_id 和 image_path 的字典列表
    """
    coco_root = Path(coco_root)
    val2014_dir = coco_root / "val2014"

    if not val2014_dir.exists():
        raise FileNotFoundError(f"COCO val2014 目录不存在: {val2014_dir}")

    images = []

    if image_id_list is not None:
        # 如果提供了图像 ID 列表, 只处理这些图像
        for image_id in image_id_list:
            image_filename = f"COCO_val2014_{str(image_id).zfill(12)}.jpg"
            image_path = val2014_dir / image_filename
            if image_path.exists():
                images.append({
                    "image_id": image_id,
                    "image_path": str(image_path)
                })
            else:
                print(f"⚠️  警告: 图像文件不存在: {image_path}")
    else:
        # 扫描 val2014 目录中的所有图像
        image_files = sorted(val2014_dir.glob("COCO_val2014_*.jpg"))
        for image_file in image_files:
            # 从文件名提取 image_id
            # 格式: COCO_val2014_000000123456.jpg
            filename = image_file.stem  # 去掉 .jpg
            image_id = int(filename.split("_")[-1])
            images.append({
                "image_id": image_id,
                "image_path": str(image_file)
            })

    # 限制图像数量
    if max_images > 0:
        images = images[:max_images]

    # 按 image_id 排序
    images.sort(key=lambda x: x['image_id'])

    return images


def prepare_inputs(model, tokenizer, image_processor, image_file: str, prompt: str, conv_mode: str, device: str, verbose: bool = False):
    """
    准备模型输入

    Returns:
        input_ids, image_tensor, stopping_criteria, stop_str
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

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    if verbose:
        print(f"  [输入准备] 文本信息:")
        print(f"    - 原始提示词: {prompt}")
        print(f"    - 完整提示词长度: {len(full_prompt)} 字符")

    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                     return_tensors='pt').unsqueeze(0).to(device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    return input_ids, image_tensor, stopping_criteria, stop_str


def generate_response(model, tokenizer, input_ids, image_tensor, stopping_criteria,
                     temperature, top_p, max_new_tokens, device,
                     use_deco=False, alpha=None, threshold_top_p=None,
                     threshold_top_k=None, early_exit_layers=None, num_beams=1, verbose: bool = False):
    """
    生成回答

    Returns:
        outputs: 生成的文本
        output_token_len: 生成的 token 长度
        input_token_len: 输入的 token 长度
    """
    do_sample = True if temperature > 0 else False

    # 准备生成参数
    generate_kwargs = {
        "inputs": input_ids,
        "images": image_tensor.unsqueeze(0).half().to(device),
        "do_sample": do_sample,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p,
        "num_beams": num_beams,
        "max_new_tokens": max_new_tokens,
        "return_dict": True,
        "return_dict_in_generate": True,
        "output_attentions": True,  # 添加 attention 输出
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

    with torch.inference_mode():
        with torch.no_grad():
            output_dict = model.generate(**generate_kwargs)

    # 解码输出
    output_ids = output_dict.sequences
    input_token_len = input_ids.shape[1]

    # 如果 output_ids 不包含 input_ids, 手动拼接
    if output_ids.shape[1] < input_token_len:
        output_ids = torch.cat([input_ids, output_ids], dim=1)
    elif output_ids.shape[1] >= input_token_len:
        # 检查前 input_token_len 个 token 是否与 input_ids 匹配
        prefix_match = (input_ids[0] == output_ids[0, :input_token_len]).all().item()
        if not prefix_match:
            output_ids = torch.cat([input_ids, output_ids[:, input_token_len:]], dim=1)

    output_token_len = output_ids.shape[1] - input_token_len

    # 获取新生成的 token
    if output_token_len > 0:
        generated_ids = output_ids[:, input_token_len:]
        # 如果新生成的 token 以 BOS token 开头, 跳过它
        bos_token_id = tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None else None
        if bos_token_id is not None and generated_ids.shape[1] > 0 and generated_ids[0, 0].item() == bos_token_id:
            generated_ids = generated_ids[:, 1:]
            if generated_ids.shape[1] > 0:
                outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            else:
                outputs = ""
        else:
            outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    else:
        outputs = ""

    # 返回额外的信息用于 attention 分析
    all_attentions = output_dict.attentions if hasattr(output_dict, 'attentions') else None
    all_hidden_states = output_dict.hidden_states if hasattr(output_dict, 'hidden_states') else None

    return outputs, output_token_len, input_token_len, output_ids, all_attentions, all_hidden_states


def identify_object_tokens_in_caption(caption, tokenizer, output_ids, input_token_len, chair_evaluator=None):
    """
    识别描述中的名词/物体，并找到它们在生成序列中的 token 位置

    Args:
        caption: 生成的描述文本
        tokenizer: tokenizer对象
        output_ids: 完整的输出序列（包含input+output）
        input_token_len: 输入序列长度
        chair_evaluator: CHAIR评估器对象（可选，如果提供则使用CHAIR的方法）

    Returns:
        List[Dict]: 包含物体信息的列表，每个元素包含：
            - object_word: 物体词汇
            - node_word: 规范化后的物体词汇
            - token_positions: 在生成序列中的token位置列表
            - token_texts: 对应的token文本列表
    """
    object_tokens_info = []

    if not caption or not caption.strip():
        return object_tokens_info

    # 如果没有提供chair_evaluator，使用简单的NLTK方法识别名词
    if chair_evaluator is None:
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer
            from nltk.corpus import wordnet

            # 确保NLTK数据已下载
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)

            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                nltk.download('averaged_perceptron_tagger', quiet=True)

            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)

            # 使用NLTK识别名词
            words = nltk.word_tokenize(caption.lower())
            tagged_sent = nltk.pos_tag(words)
            wnl = WordNetLemmatizer()

            nouns = []
            for word, pos in tagged_sent:
                if pos.startswith('NN'):  # 名词
                    lemma = wnl.lemmatize(word, pos=wordnet.NOUN)
                    nouns.append((word, lemma))
        except Exception as e:
            print(f"  ⚠️  使用NLTK识别名词时出错: {e}")
            nouns = []
    else:
        # 使用CHAIR的方法识别物体
        try:
            words, node_words, idxs, double_words = chair_evaluator.caption_to_words(caption)
            nouns = [(w, nw) for w, nw in zip(words, node_words)]
        except Exception as e:
            print(f"  ⚠️  使用CHAIR识别物体时出错: {e}")
            nouns = []

    if not nouns:
        return object_tokens_info

    # 解码完整的输出序列，找到每个名词对应的token位置
    try:
        # 获取生成的token部分（跳过input部分）
        generated_ids = output_ids[0, input_token_len:].cpu().tolist()

        # 解码整个生成序列，找到每个token的文本
        full_generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        # 对于每个名词，找到它在序列中的位置
        for word, node_word in nouns:
            # 尝试找到这个词汇在生成序列中的位置
            # 由于tokenization可能将单词分割成多个token，需要找到所有相关的token
            word_lower = word.lower()
            node_word_lower = node_word.lower()

            # 方法1: 直接搜索词汇（可能跨多个token）
            # 方法2: 逐个token解码，找到包含该词汇的token
            token_positions = []
            token_texts = []

            # 从input_token_len开始搜索（只搜索生成的token）
            current_text = ""
            for token_idx, token_id in enumerate(generated_ids):
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                current_text += token_text

                # 检查当前累积的文本是否包含目标词汇
                if word_lower in current_text.lower() or node_word_lower in current_text.lower():
                    # 找到包含该词汇的token
                    if token_idx not in token_positions:
                        token_positions.append(input_token_len + token_idx)
                        token_texts.append(token_text)

            # 如果没找到，尝试更宽松的匹配
            if not token_positions:
                # 检查每个token的文本是否包含词汇的一部分
                for token_idx, token_id in enumerate(generated_ids):
                    token_text = tokenizer.decode([token_id], skip_special_tokens=False).lower()
                    if word_lower in token_text or token_text in word_lower or \
                       node_word_lower in token_text or token_text in node_word_lower:
                        if input_token_len + token_idx not in token_positions:
                            token_positions.append(input_token_len + token_idx)
                            token_texts.append(tokenizer.decode([token_id], skip_special_tokens=False))

            if token_positions:
                object_tokens_info.append({
                    'object_word': word,
                    'node_word': node_word,
                    'token_positions': token_positions,
                    'token_texts': token_texts
                })
    except Exception as e:
        print(f"  ⚠️  识别物体token位置时出错: {e}")

    return object_tokens_info


def visualize_object_attention_map(attention_map, image, layer_idx, step_idx, token_text, token_name,
                                patch_size, output_dir):
    """可视化物体token的attention map

    Args:
        attention_map: 24x24的attention map（原始logits值，不归一化）
        image: 原始图像
        layer_idx: 层索引
        step_idx: 生成步骤索引
        token_text: token的文本内容（用于显示）
        token_name: token的清理后的名称（用于文件名）
        patch_size: patch大小（24）
        output_dir: 输出目录
    """
    # 不归一化，保持原始logits值
    attn_values = attention_map.copy()

    # 获取值的范围用于colorbar
    attn_min = attn_values.min()
    attn_max = attn_values.max()

    # 创建自定义colormap，只使用jet的前一半（从深蓝到淡黄，不包含深红）
    try:
        jet_full = plt.colormaps['jet']
    except (AttributeError, KeyError):
        jet_full = plt.cm.get_cmap('jet')
    colors_half = jet_full(np.linspace(0, 0.75, 256))
    jet_half = LinearSegmentedColormap.from_list('jet_half', colors_half)

    # 创建1×3的可视化布局
    fig = plt.figure(figsize=(18, 6))

    # 1. 原图
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(image)
    ax1.set_title(f'Original Image\nStep {step_idx+1}, Layer {layer_idx}, Token: "{token_text}"',
                 fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. Jet colormap（不叠加原图，使用一半颜色域）
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(attn_values, cmap=jet_half, interpolation='bilinear', vmin=attn_min, vmax=attn_max)
    ax2.set_title(f'Jet Colormap (24×24)\nLayer {layer_idx}', fontsize=12, fontweight='bold')
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.4e}'))
    cbar2.set_label('Logit Value', fontsize=10, fontweight='bold')

    # 3. Jet colormap和原图叠加（原图显示更明显）
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(image)
    im3 = ax3.imshow(attn_values, cmap='jet', alpha=0.4, interpolation='bilinear', vmin=attn_min, vmax=attn_max)
    ax3.set_title(f'Jet Overlay\nLayer {layer_idx}', fontsize=12, fontweight='bold')
    ax3.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.4e}'))
    cbar3.set_label('Logit Value', fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_file = os.path.join(output_dir, f"layer_{layer_idx}_token_{token_name}_attention.png")
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Layer {layer_idx} attention map已保存: {os.path.basename(output_file)}")
    print(f"      Logits范围: [{attn_min:.4e}, {attn_max:.4e}]")


def extract_object_attention_maps(model, tokenizer, image_processor, image_file, prompt, conv_mode, device,
                                  output_ids, input_token_len, all_attentions, object_tokens_info,
                                  image_tensor, output_dir, target_layers=None):
    """
    提取名词/物体token的attention map

    Args:
        model: 模型对象
        tokenizer: tokenizer对象
        image_processor: 图像处理器
        image_file: 图像文件路径
        prompt: 提示词
        conv_mode: 对话模式
        device: 设备
        output_ids: 完整的输出序列
        input_token_len: 输入序列长度
        all_attentions: 所有生成步骤的attention（tuple，每个元素对应一个步骤）
        object_tokens_info: 物体token信息列表
        image_tensor: 图像tensor
        output_dir: 输出目录
        target_layers: 目标层列表（None表示所有层，'even'表示偶数层，'odd'表示奇数层）
    """
    if not object_tokens_info or all_attentions is None:
        return

    # 加载图像
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")

    # 确定目标层
    lang_model = model.get_model()
    num_total_layers = len(lang_model.layers) if hasattr(lang_model, 'layers') else 32

    if target_layers is None:
        selected_layers = list(range(num_total_layers))
    elif isinstance(target_layers, str):
        if target_layers.lower() == 'even':
            selected_layers = list(range(0, num_total_layers, 2))
        elif target_layers.lower() == 'odd':
            selected_layers = list(range(1, num_total_layers, 2))
        else:
            try:
                selected_layers = [int(x.strip()) for x in target_layers.split(',')]
            except:
                selected_layers = list(range(num_total_layers))
    elif isinstance(target_layers, list):
        selected_layers = target_layers
    else:
        selected_layers = list(range(num_total_layers))

    # 获取图像token位置信息（参考test_llava_v15_7b_attention.py的逻辑）
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                     return_tensors='pt').unsqueeze(0).to(device)

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

    # 计算图像token位置
    vision_tower = model.get_vision_tower()
    num_image_tokens = 0
    if vision_tower is not None:
        with torch.no_grad():
            image_features = vision_tower(image_tensor.unsqueeze(0).half().to(device))
            if hasattr(image_features, 'last_hidden_state'):
                vision_hidden = image_features.last_hidden_state
            elif isinstance(image_features, tuple):
                vision_hidden = image_features[0]
            elif isinstance(image_features, torch.Tensor):
                vision_hidden = image_features
            else:
                vision_hidden = None

            if vision_hidden is not None and hasattr(model, 'mm_projector'):
                vision_hidden = model.mm_projector(vision_hidden)
                num_image_tokens = vision_hidden.shape[1]

    image_token_start = 1  # 跳过BOS token
    image_token_end = image_token_start + (num_image_tokens if num_image_tokens > 0 else 576)

    # 为每个物体提取attention map
    for obj_info in object_tokens_info:
        object_word = obj_info['object_word']
        node_word = obj_info['node_word']
        token_positions = obj_info['token_positions']

        if not token_positions:
            continue

        # 为这个物体创建目录
        obj_dir = os.path.join(output_dir, f"object_{node_word.replace(' ', '_')}")
        os.makedirs(obj_dir, exist_ok=True)

        # 对于每个token位置，提取attention map
        for token_pos in token_positions:
            # 找到这个token是在哪个生成步骤产生的
            # token_pos是绝对位置，需要转换为生成步骤索引
            if token_pos < input_token_len:
                continue  # 跳过input部分的token

            step_idx = token_pos - input_token_len

            if step_idx >= len(all_attentions) or all_attentions[step_idx] is None:
                continue

            step_attentions = all_attentions[step_idx]

            # 处理每一层的attention（只处理选定的层）
            for layer_idx, layer_attn in enumerate(step_attentions):
                if layer_idx not in selected_layers:
                    continue

                if layer_attn is None:
                    continue

                # 处理attention tensor的形状
                if isinstance(layer_attn, tuple):
                    layer_attn = layer_attn[0]

                if not isinstance(layer_attn, torch.Tensor):
                    continue

                layer_attn_np = layer_attn.cpu().numpy()

                # 处理attention tensor的形状
                if len(layer_attn_np.shape) == 4:
                    batch_size, num_heads, query_len, key_len = layer_attn_np.shape
                    if query_len == 1:
                        last_row_attention = layer_attn_np[0].mean(axis=0).squeeze()
                        seq_len = key_len
                    else:
                        layer_attn_np = layer_attn_np[0].mean(axis=0)
                        seq_len = layer_attn_np.shape[0]
                        last_row_attention = layer_attn_np[-1, :]
                elif len(layer_attn_np.shape) == 3:
                    num_heads, query_len, key_len = layer_attn_np.shape
                    if query_len == 1:
                        last_row_attention = layer_attn_np.mean(axis=0).squeeze()
                        seq_len = key_len
                    else:
                        layer_attn_np = layer_attn_np.mean(axis=0)
                        seq_len = layer_attn_np.shape[0]
                        last_row_attention = layer_attn_np[-1, :]
                elif len(layer_attn_np.shape) == 2:
                    query_len, key_len = layer_attn_np.shape
                    if query_len == 1:
                        last_row_attention = layer_attn_np[0, :]
                        seq_len = key_len
                    else:
                        seq_len = query_len
                        last_row_attention = layer_attn_np[-1, :]
                elif len(layer_attn_np.shape) == 1:
                    last_row_attention = layer_attn_np
                    seq_len = len(layer_attn_np)
                else:
                    continue

                # 提取对图像token的attention
                actual_num_image_tokens = num_image_tokens if num_image_tokens > 0 else 576
                image_token_end_actual = min(image_token_start + actual_num_image_tokens, seq_len)
                valid_image_positions = np.arange(image_token_start, image_token_end_actual)

                if len(valid_image_positions) == 0:
                    continue

                image_attention = last_row_attention[valid_image_positions]

                # 确保有576个值
                if len(image_attention) < 576:
                    padding = 576 - len(image_attention)
                    image_attention = np.pad(image_attention, (0, padding), mode='constant', constant_values=0)
                elif len(image_attention) > 576:
                    image_attention = image_attention[:576]

                # Reshape到24×24，保持原始logits值，不归一化
                patch_size = 24
                attention_map = image_attention.reshape(patch_size, patch_size)

                # 可视化attention map
                token_text = obj_info['token_texts'][token_positions.index(token_pos)] if token_pos in token_positions else object_word
                token_name = node_word.replace(' ', '_').replace('/', '_').replace('\\', '_')

                visualize_object_attention_map(
                    attention_map, image, layer_idx, step_idx, token_text, token_name,
                    patch_size, obj_dir
                )


def compare_deco_vs_vanilla(deco_results, vanilla_results, deco_captions_file, vanilla_captions_file,
                            output_file):
    """
    对比 Deco 和 Vanilla 的结果, 生成对比表格和不一致 case 的 JSON 文件

    Args:
        deco_results: Deco 版本的评估结果
        vanilla_results: Vanilla 版本的评估结果
        deco_captions_file: Deco 版本的描述文件路径
        vanilla_captions_file: Vanilla 版本的描述文件路径
        output_file: 输出 JSON 文件路径
    """
    # 加载描述文件
    deco_captions = {}
    with open(deco_captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            image_id = item.get("image_id")
            if image_id is not None:
                deco_captions[image_id] = item

    vanilla_captions = {}
    with open(vanilla_captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            image_id = item.get("image_id")
            if image_id is not None:
                vanilla_captions[image_id] = item

    # 找到描述不一致的 case
    inconsistent_cases = []
    common_image_ids = set(deco_captions.keys()) & set(vanilla_captions.keys())

    for image_id in common_image_ids:
        deco_caption = deco_captions[image_id].get("caption", "").strip()
        vanilla_caption = vanilla_captions[image_id].get("caption", "").strip()

        # 如果描述不同, 记录这个 case
        if deco_caption != vanilla_caption:
            # 获取图片文件名(不包含路径)
            # image_id 是数字, 需要构造文件名
            image_filename = f"COCO_val2014_{str(image_id).zfill(12)}.jpg"

            # 获取两个版本的 CHAIR 指标(如果可用)
            deco_sentence_metrics = None
            vanilla_sentence_metrics = None

            if deco_results and 'sentences' in deco_results:
                for s in deco_results['sentences']:
                    if s.get('image_id') == image_id:
                        deco_sentence_metrics = s.get('metrics', {})
                        break

            if vanilla_results and 'sentences' in vanilla_results:
                for s in vanilla_results['sentences']:
                    if s.get('image_id') == image_id:
                        vanilla_sentence_metrics = s.get('metrics', {})
                        break

            case_info = {
                "image_id": image_id,
                "image": image_filename,  # 只保存文件名
                "vanilla_caption": vanilla_caption,
                "deco_caption": deco_caption,
                "vanilla_metrics": vanilla_sentence_metrics,
                "deco_metrics": deco_sentence_metrics
            }
            inconsistent_cases.append(case_info)

    # 保存不一致的 case 到 JSON 文件
    comparison_result = {
        "summary": {
            "total_cases": len(common_image_ids),
            "inconsistent_cases": len(inconsistent_cases),
            "consistent_cases": len(common_image_ids) - len(inconsistent_cases),
            "inconsistency_rate": len(inconsistent_cases) / len(common_image_ids) if len(common_image_ids) > 0 else 0
        },
        "metrics_comparison": {
            "vanilla": {
                "CHAIRs": vanilla_results.get('overall_metrics', {}).get('CHAIRs', 0) if vanilla_results else 0,
                "CHAIRi": vanilla_results.get('overall_metrics', {}).get('CHAIRi', 0) if vanilla_results else 0,
                "Recall": vanilla_results.get('overall_metrics', {}).get('Recall', 0) if vanilla_results else 0,
                "Len": vanilla_results.get('overall_metrics', {}).get('Len', 0) if vanilla_results else 0
            },
            "deco": {
                "CHAIRs": deco_results.get('overall_metrics', {}).get('CHAIRs', 0) if deco_results else 0,
                "CHAIRi": deco_results.get('overall_metrics', {}).get('CHAIRi', 0) if deco_results else 0,
                "Recall": deco_results.get('overall_metrics', {}).get('Recall', 0) if deco_results else 0,
                "Len": deco_results.get('overall_metrics', {}).get('Len', 0) if deco_results else 0
            },
            "difference": {
                "CHAIRs": (deco_results.get('overall_metrics', {}).get('CHAIRs', 0) if deco_results else 0) -
                          (vanilla_results.get('overall_metrics', {}).get('CHAIRs', 0) if vanilla_results else 0),
                "CHAIRi": (deco_results.get('overall_metrics', {}).get('CHAIRi', 0) if deco_results else 0) -
                          (vanilla_results.get('overall_metrics', {}).get('CHAIRi', 0) if vanilla_results else 0),
                "Recall": (deco_results.get('overall_metrics', {}).get('Recall', 0) if deco_results else 0) -
                          (vanilla_results.get('overall_metrics', {}).get('Recall', 0) if vanilla_results else 0),
                "Len": (deco_results.get('overall_metrics', {}).get('Len', 0) if deco_results else 0) -
                       (vanilla_results.get('overall_metrics', {}).get('Len', 0) if vanilla_results else 0)
            }
        },
        "inconsistent_cases": inconsistent_cases
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_result, f, ensure_ascii=False, indent=2)

    return comparison_result


def print_comparison_table(deco_results, vanilla_results):
    """
    打印 Deco vs Vanilla 的对比表格

    Args:
        deco_results: Deco 版本的评估结果
        vanilla_results: Vanilla 版本的评估结果
    """
    deco_metrics = deco_results.get('overall_metrics', {}) if deco_results else {}
    vanilla_metrics = vanilla_results.get('overall_metrics', {}) if vanilla_results else {}

    print("\n" + "=" * 80)
    print("Deco vs Vanilla 对比")
    print("=" * 80)
    print(f"{'指标':<15} {'Vanilla':<12} {'Deco':<12} {'差异':<12} {'变化':<10}")
    print("-" * 80)

    metrics_list = [
        ('CHAIRs', 'CHAIRs'),
        ('CHAIRi', 'CHAIRi'),
        ('Recall', 'Recall'),
        ('Len', 'Len')
    ]

    for metric_name, metric_key in metrics_list:
        vanilla_val = vanilla_metrics.get(metric_key, 0)
        deco_val = deco_metrics.get(metric_key, 0)
        diff = deco_val - vanilla_val
        change = f"{diff:+.4f}" if diff != 0 else "0.0000"
        change_symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="

        # 对于 CHAIRs 和 CHAIRi, 越小越好, 所以符号相反
        if metric_key in ['CHAIRs', 'CHAIRi']:
            change_symbol = "↓" if diff > 0 else "↑" if diff < 0 else "="

        print(f"{metric_name:<15} {vanilla_val:<12.4f} {deco_val:<12.4f} {diff:<12.4f} {change_symbol} {change}")

    print("=" * 80)


def save_summary_to_file(summary_file, args, output_file, chair_results_file=None,
                         chair_errors_file=None, results=None, model_name=None, error=None):
    """
    保存 CHAIR 评估结果总结到txt文件

    Args:
        summary_file: 总结文件路径
        args: 命令行参数
        output_file: 输出描述文件路径
        chair_results_file: CHAIR 详细结果文件路径(可选)
        chair_errors_file: CHAIR 错误样本文件路径(可选)
        results: 评估结果字典(如果评估成功)
        model_name: 模型名称
        error: 错误信息(如果评估失败)
    """
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CHAIR 评估结果总结\n")
        f.write("=" * 80 + "\n\n")

        # 基本信息
        f.write("【基本信息】\n")
        f.write("-" * 80 + "\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型路径: {args.model_path}\n")
        if model_name:
            f.write(f"模型名称: {model_name}\n")
        f.write(f"设备: {args.device}\n")
        f.write(f"COCO 根目录: {args.coco_root}\n")
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
        f.write(f"Num Beams: {args.num_beams}\n")
        f.write(f"Random Seed: {args.seed}\n")
        f.write("\n")

        # 文件路径
        f.write("【文件路径】\n")
        f.write("-" * 80 + "\n")
        f.write(f"输出描述文件: {output_file}\n")
        if chair_results_file:
            f.write(f"CHAIR 详细结果文件: {chair_results_file}\n")
        if chair_errors_file:
            f.write(f"CHAIR 错误样本文件: {chair_errors_file}\n")
        f.write(f"总结文件: {summary_file}\n")
        f.write("\n")

        # 评估结果
        f.write("【评估结果】\n")
        f.write("-" * 80 + "\n")
        if results is not None:
            if 'overall_metrics' in results:
                metrics = results['overall_metrics']
                f.write(f"CHAIRs (句子级别): {metrics.get('CHAIRs', 0):.4f}\n")
                f.write(f"CHAIRi (实例级别): {metrics.get('CHAIRi', 0):.4f}\n")
                f.write(f"Recall (召回率):   {metrics.get('Recall', 0):.4f}\n")
                f.write(f"Len (平均长度):    {metrics.get('Len', 0):.4f}\n")

            # 统计错误样本
            if 'sentences' in results:
                total_samples = len(results['sentences'])
                error_samples = [
                    s for s in results['sentences']
                    if s.get('metrics', {}).get('CHAIRs', 0) > 0
                ]
                error_count = len(error_samples)
                f.write(f"\n总样本数: {total_samples}\n")
                f.write(f"包含幻觉的样本数: {error_count}\n")
                if total_samples > 0:
                    f.write(f"幻觉样本比例: {error_count / total_samples * 100:.2f}%\n")
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
    """评估模型, 生成图像描述"""
    print("=" * 80)
    print("CHAIR 评估 - 生成图像描述")
    print("=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"设备: {args.device}")
    print(f"COCO 根目录: {args.coco_root}")
    print(f"输出文件: {args.output_file}")
    if args.use_deco:
        print(f"Deco 参数: use_deco={args.use_deco}, alpha={args.alpha}, layers={args.start_layer}-{args.end_layer}")
    else:
        print(f"使用原生 LLaVA 模型(Deco 已禁用)")
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

    # 获取图像列表
    print(f"\n[2/3] 正在获取图像列表...")

    # 如果提供了图像 ID 列表文件, 读取它
    image_id_list = None
    if args.image_id_list_file:
        image_id_list_file = os.path.expanduser(args.image_id_list_file)
        if not os.path.exists(image_id_list_file):
            raise FileNotFoundError(f"图像 ID 列表文件不存在: {image_id_list_file}")

        # 检测文件格式: JSON 数组或文本文件(每行一个 ID)
        with open(image_id_list_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # 尝试解析为 JSON(支持 JSON 数组格式, 如 ["COCO_val2014_000000001171.jpg", ...])
        if content.startswith('[') and content.endswith(']'):
            # JSON 数组格式
            image_names = json.loads(content)
            image_id_list = []
            for name in image_names:
                if isinstance(name, str):
                    # 从文件名提取 image_id
                    # 格式: "COCO_val2014_000000001171.jpg" 或 "COCO_val2014_000000001171"
                    if name.endswith('.jpg'):
                        name = name[:-4]  # 移除 .jpg 后缀
                    # 提取最后的数字部分
                    parts = name.split('_')
                    if len(parts) > 0:
                        try:
                            image_id = int(parts[-1])
                            image_id_list.append(image_id)
                        except ValueError:
                            print(f"⚠️  警告: 无法从文件名提取 image_id: {name}")
                elif isinstance(name, int):
                    # 直接是数字 ID
                    image_id_list.append(name)
                else:
                    print(f"⚠️  警告: 无法处理的数据类型: {type(name)}, 值: {name}")
            print(f"✓ 从 JSON 文件读取了 {len(image_id_list)} 个图像 ID")
        else:
            # 文本文件格式(每行一个 ID)
            with open(image_id_list_file, 'r', encoding='utf-8') as f:
                image_id_list = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 尝试解析为整数
                    try:
                        image_id = int(line)
                        image_id_list.append(image_id)
                    except ValueError:
                        # 如果不是数字, 尝试从文件名格式提取
                        if 'COCO_val2014_' in line:
                            if line.endswith('.jpg'):
                                line = line[:-4]
                            parts = line.split('_')
                            if len(parts) > 0:
                                try:
                                    image_id = int(parts[-1])
                                    image_id_list.append(image_id)
                                except ValueError:
                                    print(f"⚠️  警告: 无法从行提取 image_id: {line}")
                        else:
                            print(f"⚠️  警告: 无法解析行: {line}")
            print(f"✓ 从文本文件读取了 {len(image_id_list)} 个图像 ID")

    images = get_coco_val2014_images(
        coco_root=args.coco_root,
        image_id_list=image_id_list,
        max_images=args.num_samples if args.num_samples > 0 else 0
    )
    print(f"✓ 找到 {len(images)} 个图像")

    # 准备输出文件
    output_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    output_f = open(output_file, "w", encoding="utf-8")

    # 准备 Deco 参数
    early_exit_layers = None
    if args.use_deco:
        early_exit_layers = [i for i in range(args.start_layer, args.end_layer)]

    # 处理每个图像
    print(f"\n[3/3] 开始生成描述...")
    # prompt = "Please describe this image in detail."
    prompt = "Please help me describe the image in detail."


    # 计算需要输出详细信息的样本索引(如果启用debug模式)
    debug_mode = getattr(args, 'debug', False)
    total_samples = len(images)
    debug_indices = set()

    if debug_mode:
        # Debug模式: 输出所有样本的详细信息(因为只有10个样本)
        if total_samples > 0:
            debug_indices = set(range(total_samples))
            print(f"Debug模式: 将输出所有 {len(debug_indices)} 个样本的详细信息")
    else:
        # 非Debug模式: 最多输出10个样本的详细信息(均匀分布)
        max_debug_samples = min(10, total_samples)
        if total_samples > 0:
            if total_samples <= max_debug_samples:
                debug_indices = set(range(total_samples))
            else:
                step = total_samples / max_debug_samples
                for i in range(max_debug_samples):
                    idx = int(i * step)
                    debug_indices.add(idx)
            if len(debug_indices) > 0:
                print(f"将输出 {len(debug_indices)} 个样本的详细信息用于调试(样本索引: {sorted(debug_indices)})")

    for sample_idx, image_info in enumerate(tqdm(images, desc="处理进度")):
        image_id = image_info["image_id"]
        image_file = image_info["image_path"]

        # 判断是否需要输出详细信息(debug模式或选中的样本)
        verbose = sample_idx in debug_indices

        if verbose:
            print("\n" + "=" * 80)
            print(f"[样本 {sample_idx + 1}/{total_samples}] Image ID: {image_id}")
            print("=" * 80)
            print(f"图像: {image_file}")

        # 准备输入
        input_ids, image_tensor, stopping_criteria, stop_str = prepare_inputs(
            model, tokenizer, image_processor, image_file, prompt, conv_mode, device, verbose=verbose
        )

        # 生成回答
        outputs, output_token_len, input_token_len, output_ids, all_attentions, all_hidden_states = generate_response(
            model, tokenizer, input_ids, image_tensor, stopping_criteria,
            args.temperature, args.top_p, args.max_new_tokens, device,
            use_deco=args.use_deco,
            alpha=args.alpha,
            threshold_top_p=args.threshold_top_p,
            threshold_top_k=args.threshold_top_k,
            early_exit_layers=early_exit_layers,
            num_beams=args.num_beams,
            verbose=verbose
        )

        # 移除停止字符串
        if outputs and outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        # 如果输出为空, 记录警告
        if not outputs:
            if verbose:
                print(f"\n  [Warning] 图像 {image_id} 生成结果为空, output_token_len={output_token_len}")
            else:
                print(f"  [Warning] 图像 {image_id} 生成结果为空, output_token_len={output_token_len}")

        if verbose:
            print(f"\n  [生成结果] 描述:")
            print(f"    - 输出长度: {len(outputs)} 字符")
            print(f"    - 描述预览: {outputs[:200]}...")
            print("=" * 80)

        # 保存结果(CHAIR 格式: image_id 和 caption)
        result = {
            "image_id": image_id,
            "caption": outputs
        }
        output_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        output_f.flush()

        # 如果启用了提取物体attention map，则处理
        extract_object_attention = getattr(args, 'extract_object_attention', False)
        if extract_object_attention and all_attentions is not None and outputs:
            try:
                # 初始化CHAIR评估器（用于识别物体）
                coco_annotations_path = os.path.join(args.coco_root, "annotations_trainval2014", "annotations")
                if not os.path.exists(coco_annotations_path):
                    if hasattr(project, 'coco_annotations_path'):
                        coco_annotations_path = project.coco_annotations_path

                if os.path.exists(coco_annotations_path):
                    chair_evaluator = CHAIR(coco_annotations_path)
                else:
                    chair_evaluator = None
                    print(f"  ⚠️  无法找到COCO annotations路径，将使用简单的NLTK方法识别名词")

                # 识别描述中的物体
                object_tokens_info = identify_object_tokens_in_caption(
                    outputs, tokenizer, output_ids, input_token_len, chair_evaluator
                )

                if object_tokens_info:
                    if verbose:
                        print(f"\n  [物体识别] 找到 {len(object_tokens_info)} 个物体:")
                        for obj_info in object_tokens_info:
                            print(f"    - {obj_info['object_word']} (规范化: {obj_info['node_word']})")
                            print(f"      Token位置: {obj_info['token_positions']}")

                    # 为这个图像创建输出目录
                    image_output_dir = os.path.join(os.path.dirname(output_file), "object_attention_maps", f"image_{image_id}")
                    os.makedirs(image_output_dir, exist_ok=True)

                    # 提取物体attention map
                    target_layers = getattr(args, 'target_layers', 'even')
                    extract_object_attention_maps(
                        model, tokenizer, image_processor, image_file, prompt, conv_mode, device,
                        output_ids, input_token_len, all_attentions, object_tokens_info,
                        image_tensor, image_output_dir, target_layers=target_layers
                    )

                    if verbose:
                        print(f"  ✓ 物体attention map已保存到: {image_output_dir}")
            except Exception as e:
                print(f"  ⚠️  提取物体attention map时出错: {e}")
                import traceback
                if verbose:
                    traceback.print_exc()

    output_f.close()
    print(f"\n✓ 描述生成完成！结果已保存到: {output_file}")

    # 自动计算 CHAIR 指标(默认启用)
    auto_evaluate = getattr(args, 'auto_evaluate', True)  # 默认为 True
    if auto_evaluate:
        # 构建 annotations 路径
        coco_annotations_path = os.path.join(args.coco_root, "annotations_trainval2014", "annotations")
        if not os.path.exists(coco_annotations_path):
            # 尝试使用 project 中的路径
            if hasattr(project, 'coco_annotations_path'):
                coco_annotations_path = project.coco_annotations_path
            else:
                raise FileNotFoundError(f"找不到 COCO annotations 目录: {coco_annotations_path}")

        print("\n" + "=" * 80)
        print("自动计算 CHAIR 指标...")
        print("=" * 80)

        # 生成结果文件路径(参考run_pope_eval.py的路径结构)
        results_dir = os.path.dirname(output_file)
        # 保存详细结果(包含所有中间信息)
        chair_results_file = output_file.replace('.jsonl', '_chair_results.json')
        # 保存错误样本(如果有)
        chair_errors_file = output_file.replace('.jsonl', '_chair_errors.json')
        # 生成总结文件路径
        summary_file = output_file.replace('.jsonl', '_summary.txt')

        # 计算需要输出详细信息的样本索引(如果启用debug模式)
        debug_indices = None
        if getattr(args, 'debug', False):
            # 读取生成的描述文件, 确定样本数量
            with open(output_file, 'r', encoding='utf-8') as f:
                total_samples = sum(1 for _ in f)

            if total_samples > 0:
                # 如果样本数少于等于10个, 全部输出详细信息
                if total_samples <= 10:
                    debug_indices = set(range(total_samples))
                else:
                    # 均匀分布选择样本(最多10个)
                    max_debug_samples = min(10, total_samples)
                    step = total_samples / max_debug_samples
                    debug_indices = set()
                    for i in range(max_debug_samples):
                        idx = int(i * step)
                        debug_indices.add(idx)

                print(f"Debug模式: 将输出 {len(debug_indices)} 个样本的详细信息(样本索引: {sorted(debug_indices)})")

        # 调用 evaluate_chair 函数
        results = evaluate_chair(
            cap_file=output_file,
            coco_path=coco_annotations_path,
            image_id_key="image_id",
            caption_key="caption",
            cache_file=os.path.join(results_dir, "chair_evaluator.pkl"),
            use_cache=True,
            save_path=chair_results_file,
            verbose=True,
            debug=getattr(args, 'debug', False),
            debug_indices=debug_indices
        )

        print("\n" + "=" * 80)
        print("✓ CHAIR 指标计算完成！")
        print("=" * 80)
        print(f"详细结果文件: {chair_results_file}")
        print(f"输出文件: {output_file}")

        # 保存错误样本(包含幻觉的样本)
        error_count = 0
        if results and 'sentences' in results:
            error_samples = [
                s for s in results['sentences']
                if s.get('metrics', {}).get('CHAIRs', 0) > 0
            ]
            error_count = len(error_samples)
            if error_samples:
                with open(chair_errors_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'error_count': len(error_samples),
                        'total_samples': len(results['sentences']),
                        'error_samples': error_samples
                    }, f, indent=2, ensure_ascii=False)
                print(f"错误样本文件: {chair_errors_file} ({len(error_samples)} 个包含幻觉的样本)")

        # 保存总结到txt文件
        model_name = get_model_name_from_path(args.model_path)
        save_summary_to_file(
            summary_file=summary_file,
            args=args,
            output_file=output_file,
            chair_results_file=chair_results_file,
            chair_errors_file=chair_errors_file if error_count > 0 else None,
            results=results,
            model_name=model_name
        )
        print(f"\n✓ 结果总结已保存到: {summary_file}")
    else:
        print(f"\n下一步: 使用 chair.py 计算 CHAIR 指标")
        print(f"  python chair.py --cap_file {output_file} --image_id_key image_id --caption_key caption \\")
        print(f"                  --coco_path {args.coco_root}/annotations_trainval2014/annotations/")


def main():
    """主函数 - 自动检测并使用默认配置"""
    # 项目根目录
    project_root = Path(__file__).parent

    # 自动检测可用 GPU
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
        print("⚠ 未检测到 CUDA, 将使用 CPU(速度较慢)")

    # 默认配置
    default_config = {
        "model_path": project.llava_v15_7b_path,
        "device": device,
        "coco_root": project.coco_data_path,  # 需要根据实际情况修改
        "image_id_list_file": "pope_coco/coco_baseline_500.json",
        "use_deco": True,
        "alpha": 0.6,
        "threshold_top_p": 0.9,
        "threshold_top_k": 20,
        "start_layer": 20,
        "end_layer": 29,
        "temperature": -1,
        "top_p": None,
        "max_new_tokens": 512,  # CHAIR 需要详细描述
        "num_beams": 10,
        "num_samples": 0,  # 0 表示处理所有图像
        "seed": 42
    }

    # 解析参数(所有参数都有默认值)
    parser = argparse.ArgumentParser(description="CHAIR 评估 - 生成图像描述(所有参数可选)")

    # 数据集参数
    parser.add_argument("--coco-root", type=str, default=default_config["coco_root"],
                       help="COCO 数据集根目录路径(包含 val2014 子目录)")
    parser.add_argument("--image_id_list_file", type=str, default=default_config["image_id_list_file"],
                       help="图像 ID 列表文件, 支持两种格式: 1) JSON 数组格式(如 [\"COCO_val2014_000000001171.jpg\", ...]);2) 文本文件(每行一个 image_id 或图像文件名)。如果提供则只处理这些图像")
    parser.add_argument("--num-samples", type=int, default=default_config["num_samples"],
                       help="处理图像数量(0表示处理所有图像, 非零表示只处理前N个)")

    # 模型参数
    parser.add_argument("--model-path", type=str, default=default_config["model_path"],
                       help="模型路径")
    parser.add_argument("--model-base", type=str, default=None, help="基础模型路径")
    parser.add_argument("--device", type=str, default=default_config["device"],
                       help="设备 (cuda:0/cpu)")

    # 输出参数
    parser.add_argument("--output-file", type=str, default=None,
                       help="输出描述文件路径(JSONL 格式, 如果不指定, 将自动生成)")

    # 生成参数
    parser.add_argument("--temperature", type=float, default=default_config["temperature"],
                       help="生成温度(-1表示贪婪生成)")
    parser.add_argument("--top-p", type=float, default=default_config["top_p"], help="Top-p采样")
    parser.add_argument("--max-new-tokens", type=int, default=default_config["max_new_tokens"],
                       help="最大生成 token 数")
    parser.add_argument("--num-beams", type=int, default=default_config["num_beams"],
                       help="Beam search 的 beam 数量")

    # Deco 参数(默认不使用 Deco, 只使用原生 LLaVA 模型)
    parser.add_argument("--use-deco", action="store_true", default=default_config["use_deco"],
                       help="启用 Deco 早退机制(默认: False, 使用原生 LLaVA 模型)")
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
    parser.add_argument("--no-auto-evaluate", action="store_true", default=False,
                       help="禁用自动计算 CHAIR 指标(默认会自动计算)")
    parser.add_argument("--debug", action="store_true", default=False,
                       help="启用debug模式, 输出每个样本的详细处理过程")

    args = parser.parse_args()
    set_seed(args.seed)

    # 设置 auto_evaluate 参数(默认启用, 除非指定 --no-auto-evaluate)
    args.auto_evaluate = not args.no_auto_evaluate

    # 自动生成输出文件路径(如果未指定)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "results", "chair")
    os.makedirs(output_dir, exist_ok=True)

    # 如果使用 Deco, 需要同时运行 vanilla 版本进行对比
    vanilla_output_file = None
    vanilla_results = None

    if args.use_deco:
        print("\n" + "=" * 80)
        print("检测到使用 Deco, 将同时运行 Vanilla 版本进行对比")
        print("=" * 80)

        # 先运行 Vanilla 版本
        print("\n" + "-" * 80)
        print("[1/2] 运行 Vanilla 版本")
        print("-" * 80)
        vanilla_args = argparse.Namespace(**vars(args))
        vanilla_args.use_deco = False
        vanilla_args.output_file = os.path.join(output_dir, f"chair_captions_vanilla_{timestamp}.jsonl")
        vanilla_args.auto_evaluate = args.auto_evaluate  # 保持相同的 auto_evaluate 设置

        eval_model(vanilla_args)
        vanilla_output_file = vanilla_args.output_file

        # 然后运行 Deco 版本
        print("\n" + "-" * 80)
        print("[2/2] 运行 Deco 版本")
        print("-" * 80)
        if args.output_file is None:
            args.output_file = os.path.join(output_dir, f"chair_captions_deco_{timestamp}.jsonl")

        eval_model(args)

        # 如果两个版本都完成了评估, 进行对比
        if vanilla_args.auto_evaluate and args.auto_evaluate:
            print("\n" + "=" * 80)
            print("对比 Deco vs Vanilla")
            print("=" * 80)

            # 加载两个版本的结果
            vanilla_results_file = vanilla_output_file.replace('.jsonl', '_chair_results.json')
            deco_results_file = args.output_file.replace('.jsonl', '_chair_results.json')

            if os.path.exists(vanilla_results_file) and os.path.exists(deco_results_file):
                with open(vanilla_results_file, 'r', encoding='utf-8') as f:
                    vanilla_results = json.load(f)
                with open(deco_results_file, 'r', encoding='utf-8') as f:
                    deco_results = json.load(f)

                # 生成对比 JSON 文件
                comparison_file = args.output_file.replace('.jsonl', '_comparison.json')
                comparison_result = compare_deco_vs_vanilla(
                    deco_results=deco_results,
                    vanilla_results=vanilla_results,
                    deco_captions_file=args.output_file,
                    vanilla_captions_file=vanilla_output_file,
                    output_file=comparison_file
                )

                # 打印对比表格
                print_comparison_table(deco_results=deco_results, vanilla_results=vanilla_results)

                print(f"\n✓ 对比结果已保存到: {comparison_file}")
                print(f"  - 总样本数: {comparison_result['summary']['total_cases']}")
                print(f"  - 描述不一致样本数: {comparison_result['summary']['inconsistent_cases']}")
                print(f"  - 不一致率: {comparison_result['summary']['inconsistency_rate']:.2%}")
            else:
                print("⚠️  无法找到评估结果文件, 跳过对比")
    else:
        # 不使用 Deco, 正常处理
        if args.output_file is None:
            args.output_file = os.path.join(output_dir, f"chair_captions_vanilla_{timestamp}.jsonl")

        # 运行评估
        eval_model(args)

    print("\n" + "=" * 80)
    print("✓ 所有评估完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
