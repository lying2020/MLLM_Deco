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
import string
from tqdm import tqdm
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import warnings

# 抑制常见的无害警告
warnings.filterwarnings('ignore', message='.*You are using a model of type llava to instantiate a model of type llava_llama.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub')
# 忽略字体警告
warnings.filterwarnings('ignore', category=UserWarning, message='Glyph.*missing from font')
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
from eval_tool.chair import evaluate_chair, CHAIR, get_chair_evaluator
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，不显示图片窗口
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from test_llava_v15_7b_attention import visualize_step_attention_map, plot_attention_pixel_grid, load_image


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
    # 为了确保原始输出的token_id是logits最高的，强制使用greedy decoding
    # 如果temperature <= 0，使用greedy decoding（argmax）
    # 如果temperature > 0，仍然使用采样，但会在debug信息中说明
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

    if verbose and temperature > 0:
        print(f"\n  [警告] 当前使用采样模式 (temperature={temperature})，原始输出的token可能不是logits最高的")
        print(f"  建议: 设置 temperature=0 或 temperature=-1 使用greedy decoding以确保一致性")

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

    # LLaVA 可能已经修改了 transformer 模块，output_ids 中不包含输入信息
    # 直接使用完整的 output_ids 作为生成的 token
    generated_ids = output_ids
    output_token_len = generated_ids.shape[1]

    # 处理 BOS token 和最终解码
    if output_token_len > 0:
        # 如果新生成的 token 以 BOS token 开头, 跳过它
        bos_token_id = tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None else None
        if bos_token_id is not None and generated_ids.shape[1] > 0:
            first_token = generated_ids[0, 0].item()
            if first_token == bos_token_id:
                generated_ids = generated_ids[:, 1:]
                output_token_len = generated_ids.shape[1]

        if generated_ids.shape[1] > 0:
            # 使用 batch_decode 和 skip_special_tokens=True 来安全解码
            outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        else:
            outputs = ""
    else:
        outputs = ""

    # 返回额外的信息用于 attention 分析
    all_attentions = output_dict.attentions if hasattr(output_dict, 'attentions') else None
    all_hidden_states_raw = output_dict.hidden_states if hasattr(output_dict, 'hidden_states') else None

    # 过滤掉embedding层（第一个隐藏层），只保留transformer层的隐藏状态
    # all_hidden_states_raw 的结构：每个步骤包含33个元素（索引0是embedding层，索引1-32是transformer层）
    # 过滤后：每个步骤只包含32个transformer层的隐藏状态
    all_hidden_states = None
    if all_hidden_states_raw is not None:
        if isinstance(all_hidden_states_raw, (tuple, list)):
            # 对每个步骤的hidden_states，去掉第一个元素（embedding层）
            all_hidden_states = []
            for step_hidden_states in all_hidden_states_raw:
                if step_hidden_states is None:
                    all_hidden_states.append(None)
                elif isinstance(step_hidden_states, (tuple, list)) and len(step_hidden_states) > 0:
                    # 跳过第一个元素（embedding层），只保留transformer层（索引1-32）
                    transformer_hidden_states = step_hidden_states[1:] if len(step_hidden_states) > 1 else step_hidden_states
                    all_hidden_states.append(transformer_hidden_states)
                else:
                    all_hidden_states.append(step_hidden_states)
            all_hidden_states = tuple(all_hidden_states) if isinstance(all_hidden_states_raw, tuple) else all_hidden_states
        else:
            all_hidden_states = all_hidden_states_raw

    # 返回原始的 output_ids（从模型生成的完整序列，包含input+output），用于 attention 分析
    return outputs, output_token_len, input_token_len, output_ids, all_attentions, all_hidden_states


def _is_singular_plural_match(word1, word2):
    """
    检查两个单词是否是单复数关系

    Args:
        word1: 第一个单词（小写）
        word2: 第二个单词（小写）

    Returns:
        bool: 如果是单复数关系返回True
    """
    if word1 == word2:
        return True

    # 检查 word2 是否是 word1 的复数形式
    if word2 == word1 + "s" or word2 == word1 + "es":
        return True

    # 检查 word1 是否是 word2 的复数形式
    if word1 == word2 + "s" or word1 == word2 + "es":
        return True

    # 处理一些特殊情况（如 y -> ies）
    if word1.endswith("y") and word2 == word1[:-1] + "ies":
        return True
    if word2.endswith("y") and word1 == word2[:-1] + "ies":
        return True

    # 处理 f -> ves 的情况（如 leaf -> leaves）
    if word1.endswith("f") and word2 == word1[:-1] + "ves":
        return True
    if word2.endswith("f") and word1 == word2[:-1] + "ves":
        return True

    return False


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
    # 使用字典结构，以 node_word 作为 key，避免重复
    # object_tokens_info: {object_word: {object_word, token_positions, token_texts, matched_tokens_detail}}
    object_tokens_info = {}

    if not caption or not caption.strip():
        return object_tokens_info, {
            'full_generated_text': '',
            'all_tokens_detail': [],
            'total_tokens': 0
        }

    # 如果没有提供chair_evaluator，使用简单的NLTK方法识别名词
    if chair_evaluator is None:
        raise ValueError("chair_evaluator is required")

    # 使用CHAIR的方法识别物体
    words, node_words, idxs, double_words = chair_evaluator.caption_to_words(caption)
    nouns = [(w, nw) for w, nw in zip(words, node_words)]

    if not nouns:
        # 即使没有找到物体，也返回token详细信息
        # LLaVA 的 output_ids 可能不包含输入信息，直接使用完整的 output_ids
        generated_ids = output_ids[0].cpu().tolist()
        full_generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        all_tokens_detail = []
        char_pos = 0
        for token_idx, token_id in enumerate(generated_ids):
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            token_start = char_pos
            token_end = char_pos + len(token_text)

            all_tokens_detail.append({
                'token_idx': token_idx,
                'absolute_position': token_idx,  # LLaVA 的 output_ids 不包含 input，所以绝对位置就是 token_idx
                'token_id': int(token_id),
                'token_text': token_text,
                'token_text_stripped': token_text.strip(),
                'char_start': token_start,
                'char_end': token_end,
                'char_length': len(token_text)
            })

            char_pos = token_end

        return object_tokens_info, {
            'full_generated_text': full_generated_text,
            'all_tokens_detail': all_tokens_detail,
            'total_tokens': len(generated_ids)
        }

    # 解码完整的输出序列，找到每个名词对应的token位置
    # LLaVA 的 output_ids 可能不包含输入信息，直接使用完整的 output_ids
    generated_ids = output_ids[0].cpu().tolist()

    # 解码整个生成序列
    full_generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    full_generated_text_lower = full_generated_text.lower()

    # 收集所有token的详细信息（用于JSON输出）
    all_tokens_detail = []
    char_pos = 0
    for token_idx, token_id in enumerate(generated_ids):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        token_start = char_pos
        token_end = char_pos + len(token_text)

        all_tokens_detail.append({
            'token_idx': token_idx,
            'absolute_position': token_idx,  # LLaVA 的 output_ids 不包含 input，所以绝对位置就是 token_idx
            'token_id': int(token_id),
            'token_text': token_text,
            'token_text_stripped': token_text.strip(),
            'char_start': token_start,
            'char_end': token_end,
            'char_length': len(token_text)
        })

        char_pos = token_end

    # 对于每个名词，找到它在序列中的位置
    for word, node_word in nouns:
        word_lower = word.lower()

        token_positions = []
        token_texts = []
        token_groups = []  # 存储token组，每个组是一个连续的token范围 [(start_idx, end_idx), ...]

        # 方法：精确匹配，只匹配真正组成目标词汇的token （如 "the"）
        # 1. 首先检查单个token是否精确等于目标词汇（去除前后空格）
        # 2. 如果单个token匹配失败，使用滑动窗口匹配多个token的组合（处理被分解的单词, 如 "chair" → "ch" + "air"）
        # 3. 对于多词短语，也使用滑动窗口匹配（如 "traffic light"）
        search_word = word_lower

        if not search_word:
            continue

        # 1. 检查单个token精确匹配（快速路径，支持单复数匹配）
        single_token_matched = False
        for token_idx, token_id in enumerate(generated_ids):
            token_text = tokenizer.decode([token_id], skip_special_tokens=False).strip().lower()
            # 检查精确匹配或单复数匹配
            if _is_singular_plural_match(token_text, search_word):
                if token_idx not in token_positions:
                    token_positions.append(token_idx)
                    token_texts.append(tokenizer.decode([token_id], skip_special_tokens=False))
                    # 单个token作为一个组
                    token_groups.append((token_idx, token_idx))
                single_token_matched = True

        # 2. 如果单个token匹配失败，或者目标词汇是多词（如 "traffic light"），使用滑动窗口匹配
        # 对于单个单词，也尝试多token组合匹配（处理被tokenizer分解的情况）
        if not single_token_matched or ' ' in search_word:
            import re

            # 确定滑动窗口的最大大小
            # 对于单个单词，尝试2-5个token的组合（通常一个单词最多被分解成2-3个token）
            # 对于多词短语，使用词数+2作为最大窗口
            if ' ' in search_word:
                words_in_phrase = search_word.split()
                max_window_size = min(len(words_in_phrase) + 2, len(generated_ids))
                min_window_size = len(words_in_phrase)
            else:
                # 单个单词：尝试1-5个token的组合（1已经在上面检查过了，这里从2开始）
                max_window_size = min(5, len(generated_ids))
                min_window_size = 2

            # 记录已匹配的token范围，避免重复匹配
            # matched_ranges: [(start_token_idx, end_token_idx), ...]
            matched_ranges = []

            # 使用滑动窗口匹配
            for window_size in range(min_window_size, max_window_size + 1):
                for start_idx in range(len(generated_ids) - window_size + 1):
                    # 检查这个窗口是否与已匹配的范围有重叠
                    end_idx = start_idx + window_size - 1
                    is_overlapping = False
                    for matched_start, matched_end in matched_ranges:
                        # 如果当前窗口与已匹配的范围有重叠，跳过
                        if not (end_idx < matched_start or start_idx > matched_end):
                            is_overlapping = True
                            break

                    if is_overlapping:
                        continue

                    # 获取窗口内的token序列
                    window_tokens = generated_ids[start_idx:start_idx + window_size]
                    window_text = tokenizer.decode(window_tokens, skip_special_tokens=False).strip().lower()

                    # 关键检查：只有当窗口解码后的文本正好等于目标词汇或单复数匹配时，才认为是有效匹配
                    # 这样可以确保窗口内的所有token都用于匹配目标词汇，避免部分匹配
                    # 例如：如果窗口是 [7, 8]，解码后是 "airplane"，正好等于目标词汇，匹配成功
                    # 如果窗口是 [7, 8, 9]，解码后是 "airplanepark"，不等于目标词汇，不应该匹配
                    # 支持单复数匹配：如果目标词汇是 "book"，窗口文本是 "books"，也应该匹配
                    if _is_singular_plural_match(window_text, search_word):
                        # 精确匹配成功，记录这个匹配范围
                        matched_start_token = start_idx
                        matched_end_token = start_idx + window_size - 1
                        matched_ranges.append((matched_start_token, matched_end_token))

                        # 添加窗口内的所有token
                        for i in range(window_size):
                            token_idx = start_idx + i
                            if token_idx not in token_positions:
                                token_positions.append(token_idx)
                                token_texts.append(tokenizer.decode([generated_ids[token_idx]], skip_special_tokens=False))

                        token_groups.append((matched_start_token, matched_end_token))

        if not token_positions:
            continue

        # 获取匹配的token的详细信息
        matched_tokens_detail = []
        for abs_pos in token_positions:
            token_idx = abs_pos  # LLaVA 的 output_ids 不包含 input，所以 abs_pos 就是 token_idx
            if 0 <= token_idx < len(all_tokens_detail):
                matched_tokens_detail.append(all_tokens_detail[token_idx])

        # 如果该 word 已存在，合并 token_positions
        if word in object_tokens_info:
            # 合并 token_positions（去重并排序）
            existing_positions = set(object_tokens_info[word]['token_positions'])
            new_positions = set(token_positions)
            merged_positions = sorted(list(existing_positions | new_positions))

            # 合并 token_texts 和 matched_tokens_detail
            existing_texts = object_tokens_info[word]['token_texts']
            existing_details = object_tokens_info[word]['matched_tokens_detail']

            # 添加新的 token_texts（去重）
            for token_text in token_texts:
                if token_text not in existing_texts:
                    existing_texts.append(token_text)

            # 添加新的 matched_tokens_detail（基于 position 去重）
            existing_detail_positions = {detail['absolute_position'] for detail in existing_details}
            for detail in matched_tokens_detail:
                if detail['absolute_position'] not in existing_detail_positions:
                    existing_details.append(detail)
                    existing_detail_positions.add(detail['absolute_position'])

            # 合并 token_groups
            existing_groups = object_tokens_info[word].get('token_groups', [])
            # 合并并去重token组
            all_groups = existing_groups + token_groups
            # 去重：如果两个组的范围相同，只保留一个
            unique_groups = []
            seen_groups = set()
            for group in all_groups:
                if group not in seen_groups:
                    unique_groups.append(group)
                    seen_groups.add(group)
            # 按起始位置排序
            unique_groups.sort(key=lambda x: x[0])

            # 更新信息
            object_tokens_info[word]['token_positions'] = merged_positions
            object_tokens_info[word]['token_texts'] = existing_texts
            object_tokens_info[word]['matched_tokens_detail'] = existing_details
            object_tokens_info[word]['token_groups'] = unique_groups
        else:
            # 首次出现，创建新条目
            object_tokens_info[word] = {
                'object_word': word,
                'node_word': node_word,
                'token_positions': token_positions,
                'token_texts': token_texts,
                'matched_tokens_detail': matched_tokens_detail,
                'token_groups': token_groups  # 存储token组信息
            }

    # 返回结果，包含所有token的详细信息
    return object_tokens_info, {
        'full_generated_text': full_generated_text,
        'all_tokens_detail': all_tokens_detail,
        'total_tokens': len(generated_ids)
    }


def extract_top_p_tokens(logits, tokenizer, threshold_top_p=0.9):
    """从logits中提取top-p tokens，输出所有threshold内的logits、token和词汇

    Args:
        logits: [vocab_size] 的logits tensor
        tokenizer: tokenizer对象
        threshold_top_p: top-p阈值（默认0.9）

    Returns:
        dict: 包含top tokens信息的字典，包括所有threshold内的logits
    """
    # 确保logits是tensor
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)

    # 计算softmax概率
    probs = torch.softmax(logits, dim=-1)

    # 按概率排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # 计算累积概率
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

    # 找到不超过threshold_top_p的tokens
    top_p_mask = cumsum_probs <= threshold_top_p
    top_p_indices = sorted_indices[top_p_mask]
    top_p_probs = sorted_probs[top_p_mask]

    # 如果第一个token的概率已经超过threshold，至少包含第一个
    if len(top_p_indices) == 0:
        top_p_indices = sorted_indices[:1]
        top_p_probs = sorted_probs[:1]

    # 获取最大logit的token
    max_logit_idx = torch.argmax(logits).item()
    max_logit_value = logits[max_logit_idx].item()
    max_logit_prob = probs[max_logit_idx].item()

    # 解码tokens - 输出所有threshold内的logits、token和词汇
    top_p_tokens = []
    for idx, prob in zip(top_p_indices, top_p_probs):
        token_id = idx.item() if isinstance(idx, torch.Tensor) else idx
        token_text = tokenizer.decode([token_id])
        logit_value = logits[token_id].item() if isinstance(logits[token_id], torch.Tensor) else logits[token_id]
        top_p_tokens.append({
            'token_id': int(token_id),
            'token_text': token_text,
            'probability': float(prob.item() if isinstance(prob, torch.Tensor) else prob),
            'logit': float(logit_value)
        })

    # 获取前5个最高logits的tokens
    top_5_indices = torch.topk(logits, k=min(5, len(logits))).indices
    top_5_tokens = []
    for idx in top_5_indices:
        token_id = idx.item()
        token_text = tokenizer.decode([token_id])
        logit_value = logits[token_id].item()
        prob_value = probs[token_id].item()
        top_5_tokens.append({
            'token_id': int(token_id),
            'token_text': token_text,
            'logit': float(logit_value),
            'probability': float(prob_value)
        })

    result = {
        'max_logit_token': {
            'token_id': int(max_logit_idx),
            'token_text': tokenizer.decode([max_logit_idx]),
            'logit': float(max_logit_value),
            'probability': float(max_logit_prob)
        },
        'top_p_tokens': top_p_tokens,  # 所有threshold内的tokens
        'top_5_tokens': top_5_tokens,  # 前5个最高logits的tokens
        'top_p_threshold': threshold_top_p,
        'total_top_p_tokens': len(top_p_tokens)
    }

    return result


def visualize_top5_logits_heatmap(layer_lm_head_outputs, output_dir, step_idx, num_layers=32):
    """生成5×32的heatmap，显示每层前5个最高概率的token

    使用绿色渐变colormap显示概率值
    像素点之间有间隔，并在每个像素点上标注词汇（旋转45度）

    Args:
        layer_lm_head_outputs: 字典，包含每层的lm_head输出信息
        output_dir: 输出目录
        step_idx: 生成步骤索引
        num_layers: 总层数（默认32）
    """
    # 收集所有层的前top_tokens_num个tokens
    top_tokens_num = 5
    # 创建一个5×32的矩阵，存储softmax后的概率值
    probability_matrix = np.full((top_tokens_num, num_layers), np.nan)  # 使用NaN表示无效值
    token_texts_matrix = [[''] * num_layers for _ in range(top_tokens_num)]  # 存储token文本

    # 遍历所有层
    for layer_key, top_p_info in layer_lm_head_outputs.items():
        # 跳过final_layer，只处理数字层
        if layer_key == 'final_layer':
            continue

        layer_idx = int(layer_key)
        if layer_idx >= num_layers:
            continue

        # 获取前top_tokens_num个tokens
        top_5_tokens = top_p_info.get('top_5_tokens', [])
        rank = 0
        for token_info in top_5_tokens:
            if rank < top_tokens_num:  # 最多top_tokens_num个
                probability_matrix[rank, layer_idx] = token_info['probability']
                token_texts_matrix[rank][layer_idx] = token_info['token_text']
                rank += 1
            else:
                break

    # 只处理非NaN的值
    valid_probabilities = probability_matrix[~np.isnan(probability_matrix)]
    if len(valid_probabilities) == 0:
        print(f"  ⚠️  步骤 {step_idx+1}: 没有有效的概率值，跳过heatmap生成")
        return

    # 计算figsize，使得每个像素点的高宽相等
    # 5行32列，所以高度应该是宽度的 5/32
    # 设置一个合适的宽度，然后根据比例计算高度
    base_width = 20
    base_height = base_width * (top_tokens_num / num_layers)
    fig, ax = plt.subplots(figsize=(base_width, base_height))

    # 设置aspect ratio，确保每个像素点的高宽相等
    # 由于数据是 5行32列，我们需要让 x 和 y 的单位长度相等
    ax.set_aspect('equal', adjustable='box')

    # 使用pcolormesh而不是imshow，这样可以控制像素块之间的间隔
    # 需要扩展矩阵以匹配pcolormesh的要求（需要多一行一列）
    probability_extended = np.full((top_tokens_num + 1, num_layers + 1), np.nan)
    probability_extended[:top_tokens_num, :num_layers] = probability_matrix

    # 创建坐标网格（pcolormesh需要比数据多一个点的网格）
    # shading='flat' 要求 C 的维度是 (Y-1, X-1)，所以网格需要比数据多1个点
    X = np.arange(num_layers + 2)  # 增加1个点
    Y = np.arange(top_tokens_num + 2)  # 增加1个点
    X_grid, Y_grid = np.meshgrid(X, Y)

    # 绘制heatmap，使用绿色渐变colormap，设置edgecolors来创建间隔效果
    im = ax.pcolormesh(X_grid, Y_grid, probability_extended, cmap='Greens',
                       edgecolors='white', linewidths=2.0,
                       vmin=np.nanmin(probability_matrix), vmax=np.nanmax(probability_matrix),
                       shading='flat')

    # 设置坐标轴
    ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top 5 Rank', fontsize=12, fontweight='bold')
    ax.set_title(f'Top 5 Probability Heatmap - Step {step_idx+1}',
                 fontsize=14, fontweight='bold')

    # 设置x轴刻度（层索引）- 只显示6个刻度，并加粗
    num_ticks = 6
    tick_indices = np.linspace(0, num_layers - 1, num_ticks, dtype=int)
    ax.set_xticks(tick_indices + 0.5)
    ax.set_xticklabels([f'L{i}' for i in tick_indices], rotation=45, ha='right', fontsize=8, fontweight='bold')

    # 设置y轴刻度（排名）- 放在单元格中心
    ax.set_yticks(np.arange(top_tokens_num) + 0.5)
    ax.set_yticklabels([f'Rank {i+1}' for i in range(top_tokens_num)], fontsize=10)

    # 在每个像素块中心标注token文本（词汇），旋转45度
    for rank in range(top_tokens_num):
        for layer_idx in range(num_layers):
            token_text = token_texts_matrix[rank][layer_idx]
            prob_value = probability_matrix[rank, layer_idx]

            # 只标注有效的token
            if token_text and not np.isnan(prob_value):
                # 清理token文本，移除换行符和特殊字符，限制长度
                clean_text = token_text.replace('\n', ' ').replace('\r', ' ').strip()

                # 检查是否可以正常显示（只包含可打印字符）
                printable_chars = set(string.printable)
                if not all(c in printable_chars for c in clean_text) or not clean_text:
                    clean_text = " * "
                else:
                    # 限制长度，避免文本过长
                    if len(clean_text) > 12:
                        clean_text = clean_text[:12] + '...'

                # 根据概率值选择文本颜色（深色或浅色）
                max_prob = np.nanmax(probability_matrix)
                min_prob = np.nanmin(probability_matrix)
                if max_prob > min_prob:
                    normalized = (prob_value - min_prob) / (max_prob - min_prob)
                    text_color = 'white' if normalized > 0.5 else 'black'
                else:
                    text_color = 'black'

                # 在单元格中心标注文本（词汇），旋转45度
                # 使用半透明背景以提高可读性
                ax.text(layer_idx + 0.5, rank + 0.5, clean_text,
                       ha='center', va='center',
                       fontsize=9, color=text_color, fontweight='bold',
                       rotation=45,  # 旋转45度
                       bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='white' if text_color == 'black' else 'black',
                                alpha=0.6, edgecolor='none'))

    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Probability', fontsize=10, fontweight='bold')

    # 设置坐标轴范围，确保显示所有单元格
    ax.set_xlim(0, num_layers)
    ax.set_ylim(0, top_tokens_num)

    plt.tight_layout()

    # 保存图片
    heatmap_file = os.path.join(output_dir, f"top5_logits_heatmap_step_{step_idx+1}.png")
    plt.savefig(heatmap_file, dpi=200, bbox_inches='tight')
    plt.close()

    # 统计信息
    valid_count = np.sum(~np.isnan(probability_matrix))
    total_count = top_tokens_num * num_layers
    print(f"  ✓ 步骤 {step_idx+1} 的Top 5 Probability Heatmap已保存: {os.path.basename(heatmap_file)}")
    print(f"    有效token数量: {valid_count}/{total_count}")
    if len(valid_probabilities) > 0:
        print(f"    概率值范围: [{valid_probabilities.min():.4f}, {valid_probabilities.max():.4f}]")


def enhance_attention_map(attention_map, method='min_max_normalize'):
    """
    增强或归一化attention map，使像素点差异更明显

    Args:
        attention_map: 原始attention map (24x24)
        method: 增强方法 ('min_max_normalize', 'z_score_normalize', 'power_law', 'sigmoid')

    Returns:
        增强后的attention map
    """
    attn = attention_map.copy()

    if method == 'min_max_normalize':
        # Min-Max归一化到[0, 1]
        attn_min = attn.min()
        attn_max = attn.max()
        if attn_max > attn_min:
            attn = (attn - attn_min) / (attn_max - attn_min)
    elif method == 'z_score_normalize':
        # Z-score归一化
        attn_mean = attn.mean()
        attn_std = attn.std()
        if attn_std > 0:
            attn = (attn - attn_mean) / attn_std
            # 然后映射到[0, 1]
            attn = (attn - attn.min()) / (attn.max() - attn.min()) if attn.max() > attn.min() else attn
    elif method == 'power_law':
        # 幂律增强 (gamma correction)
        attn_min = attn.min()
        attn_max = attn.max()
        if attn_max > attn_min:
            attn = (attn - attn_min) / (attn_max - attn_min)
            gamma = 0.5  # 增强对比度
            attn = np.power(attn, gamma)
    elif method == 'sigmoid':
        # Sigmoid增强
        attn_mean = attn.mean()
        attn_std = attn.std()
        if attn_std > 0:
            attn = 1 / (1 + np.exp(-(attn - attn_mean) / (attn_std * 2)))

    return attn


def visualize_object_attention_map(attention_map, image, layer_idx, step_idx, token_text, token_name,
                                patch_size, output_dir, predicted_token_word=None):
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
        predicted_token_word: 该层预测的token词汇（用于文件名）
    """
    # 增强attention map，使像素点差异更明显
    attn_values_enhanced = enhance_attention_map(attention_map.copy(), method='min_max_normalize')

    # 保持原始值用于显示
    attn_values = attention_map.copy()

    # 获取值的范围用于colorbar
    attn_min = attn_values.min()
    attn_max = attn_values.max()
    attn_enhanced_min = attn_values_enhanced.min()
    attn_enhanced_max = attn_values_enhanced.max()

    # 将原图resize成正方形，和attention map的尺寸一致
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    else:
        image_pil = image

    # 直接resize到方形尺寸（可能会压缩）
    display_size = patch_size * 20  # 480x480
    image_resized = image_pil.resize((display_size, display_size), Image.Resampling.LANCZOS)
    image_array = np.array(image_resized)

    # 统一所有子图的显示范围（都是正方形）
    unified_extent = [0, patch_size, 0, patch_size]

    # 构建文件名基础部分
    filename_base = [f"layer_{layer_idx}", f"step_{step_idx}", f"token_{token_name}"]

    # 3. 单独保存增强后的attention map（图3）
    fig3 = plt.figure(figsize=(8, 8))
    ax3 = plt.subplot(1, 1, 1)
    im3 = ax3.imshow(attn_values_enhanced, cmap='jet', interpolation='nearest',
                     vmin=attn_enhanced_min, vmax=attn_enhanced_max, extent=unified_extent, aspect='equal')
    ax3.set_title(f'Attention Map, Token: "{token_text}"', fontsize=14, fontweight='bold')  # 12 * 1.2 = 14.4，约14
    ax3.axis('off')
    # 保存图3
    filename_parts_3 = filename_base + ["attention_map.png"]
    output_file_3 = os.path.join(output_dir, "_".join(filename_parts_3))
    plt.savefig(output_file_3, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    ✓ 图3 (Attention Map) 已保存: {os.path.basename(output_file_3)}")

    # 单独保存图3的colorbar
    fig_cbar3 = plt.figure(figsize=(1, 6))
    ax_cbar3 = plt.subplot(1, 1, 1)
    ax_cbar3.axis('off')
    # 创建colorbar
    sm3 = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=attn_enhanced_min, vmax=attn_enhanced_max))
    sm3.set_array([])
    cbar3 = plt.colorbar(sm3, ax=ax_cbar3, orientation='vertical', fraction=1.0)
    cbar3.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
    filename_parts_cbar3 = filename_base + ["attention_map_colorbar.png"]
    output_file_cbar3 = os.path.join(output_dir, "_".join(filename_parts_cbar3))
    plt.savefig(output_file_cbar3, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    ✓ 图3 Colorbar 已保存: {os.path.basename(output_file_cbar3)}")

    # 4. 单独保存增强后的attention map和原图叠加（图4）
    fig4 = plt.figure(figsize=(8, 8))
    ax4 = plt.subplot(1, 1, 1)
    # 先显示原图
    ax4.imshow(image_array, extent=unified_extent, aspect='equal')
    # 将attention map上采样到和原图相同的尺寸
    scale_factor = display_size // patch_size
    attn_upsampled = np.repeat(np.repeat(attn_values_enhanced, scale_factor, axis=0), scale_factor, axis=1)
    # 使用较低的alpha值，让原图更可见
    im4 = ax4.imshow(attn_upsampled, cmap='jet', alpha=0.4, interpolation='bilinear',
                     vmin=attn_enhanced_min, vmax=attn_enhanced_max, extent=unified_extent, aspect='equal')
    ax4.set_title(f'Attention Map Overlay, Token: "{token_text}"', fontsize=14, fontweight='bold')  # 12 * 1.2 = 14.4，约14
    ax4.axis('off')
    # 保存图4（不使用前缀）
    output_file_4 = os.path.join(output_dir, "attention_map_overlay.png")
    plt.savefig(output_file_4, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    ✓ 图4 (Attention Map Overlay) 已保存: {os.path.basename(output_file_4)}")

    # 单独保存图4的colorbar（不使用前缀，因为不同层的colorbar都一样）
    output_file_cbar4 = os.path.join(output_dir, "attention_map_overlay_colorbar.png")
    # 检查是否已经保存过colorbar，避免重复保存
    if not os.path.exists(output_file_cbar4):
        fig_cbar4 = plt.figure(figsize=(1, 6))
        ax_cbar4 = plt.subplot(1, 1, 1)
        ax_cbar4.axis('off')
        # 创建colorbar
        sm4 = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=attn_enhanced_min, vmax=attn_enhanced_max))
        sm4.set_array([])
        cbar4 = plt.colorbar(sm4, ax=ax_cbar4, orientation='vertical', fraction=1.0)
        cbar4.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
        cbar4.set_label('Normalized Value', fontsize=12, fontweight='bold')  # 10 * 1.2 = 12
        plt.savefig(output_file_cbar4, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"    ✓ 图4 Colorbar 已保存: {os.path.basename(output_file_cbar4)}")

    print(f"    ✓ Layer {layer_idx} attention maps已保存")
    print(f"      原始Logits范围: [{attn_min:.4e}, {attn_max:.4e}]")
    print(f"      增强后范围: [{attn_enhanced_min:.4f}, {attn_enhanced_max:.4f}]")


def _parse_step_generated_words(output_ids, tokenizer, outputs_text):
    """解析每个步骤生成的token对应的词汇"""
    step_generated_words = {}
    if outputs_text and output_ids is not None:
        generated_ids = output_ids[0].cpu().tolist()
        for step_idx, token_id in enumerate(generated_ids):
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            token_text = token_text.strip().replace('\n', ' ').replace('\t', ' ')
            if token_text:
                step_generated_words[step_idx] = token_text
    return step_generated_words


def _load_image(image_file):
    """加载图像"""
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def _determine_target_layers(model, target_layers):
    """确定目标层列表"""
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
            selected_layers = [int(x.strip()) for x in target_layers.split(',')]
    elif isinstance(target_layers, list):
        selected_layers = target_layers
    else:
        selected_layers = list(range(num_total_layers))

    return selected_layers, num_total_layers


def _get_image_token_info(model, tokenizer, prompt, image_tensor, device):
    """获取图像token位置信息"""
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                     return_tensors='pt').unsqueeze(0).to(device)

    with torch.no_grad():
        model.prepare_inputs_labels_for_multimodal(
            input_ids, None, None, None, None,
            image_tensor.unsqueeze(0).half().to(device)
        )

    # 编码图像特征
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

                if vision_hidden is not None and hasattr(model, 'mm_projector'):
                    vision_hidden = model.mm_projector(vision_hidden)

                if vision_hidden is not None:
                    num_image_tokens = vision_hidden.shape[1]
                    print(f"Vision Hidden State Shape: {vision_hidden.shape}")
                    print(f"图像 token 数量: {num_image_tokens}")
            except Exception as e:
                print(f"⚠️  提取 vision hidden state 时出错: {e}")

    image_token_start = 35  # 跳过BOS token
    return image_token_start, num_image_tokens


def _collect_target_token_positions(object_tokens_info):
    """收集所有目标token位置"""
    target_token_positions = set()
    for object_word, obj_info in object_tokens_info.items():
        token_groups = obj_info.get('token_groups', [])
        token_positions = obj_info.get('token_positions', [])

        if not token_groups and not token_positions:
            continue

        if len(token_groups)>0:
            for token_idx in range(token_groups[0][0], token_groups[0][1] + 1):
                target_token_positions.add(token_idx)
            # if len(token_groups) > 1:
            #     for token_idx in range(token_groups[-1][0], token_groups[-1][1] + 1):
            #         target_token_positions.add(token_idx)
            #     print(f"  ⚠️  物体 {object_word} 有 {len(token_groups)} 个token组: {token_groups}")
        else:
            if token_positions:
                target_token_positions.add(token_positions[0])
                if len(token_positions) > 1:
                    target_token_positions.add(token_positions[-1])
                    print(f"  ⚠️  物体 {object_word} 在 {token_positions} 位置出现多次，取第一次和最后一次的位置")

    return target_token_positions


def _extract_layer_logits_for_tokens(model, tokenizer, output_ids, all_hidden_states,
                                     target_token_positions, device):
    """为目标token位置提取各层的logits信息"""
    step_lm_head_outputs = {}

    if all_hidden_states is None or not hasattr(model, 'lm_head') or not target_token_positions:
        return step_lm_head_outputs

    print(f"\n  [提取Logits] 为目标 token 位置提取各层的logits信息...")
    print(f"    目标 token 位置: {sorted(target_token_positions)} (共 {len(target_token_positions)} 个)")

    lang_model = model.get_model()
    norm_layer = lang_model.norm if hasattr(lang_model, 'norm') else None

    for step_idx in sorted(target_token_positions):
        if step_idx >= len(all_hidden_states):
            continue

        step_hidden_states = all_hidden_states[step_idx-1]
        if step_hidden_states is None:
            continue

        step_lm_head_outputs[step_idx] = {}

        original_token_id = None
        original_token_text = None
        if output_ids is not None and step_idx < output_ids.shape[1]:
            original_token_id = output_ids[0, step_idx].item()
            original_token_text = tokenizer.decode([original_token_id], skip_special_tokens=True)

            # print(f"\n  [Debug] Step {step_idx} - output_ids检查:")
            # print(f"    - output_ids.shape: {output_ids.shape}")
            # print(f"    - output_ids[0, :step_idx+3]: {output_ids[0, :step_idx+3].tolist() if step_idx+3 <= output_ids.shape[1] else 'N/A'}")
            # print(f"    - 当前step_idx: {step_idx}, 对应token位置: {step_idx}")

        if isinstance(step_hidden_states, (tuple, list)):
            for layer_idx_all in range(len(step_hidden_states)):
                layer_hidden = step_hidden_states[layer_idx_all]
                if isinstance(layer_hidden, torch.Tensor):
                    if len(layer_hidden.shape) == 3:
                        last_hidden = layer_hidden[0, -1, :]
                    elif len(layer_hidden.shape) == 2:
                        last_hidden = layer_hidden[-1, :]
                    else:
                        last_hidden = None

                    if last_hidden is not None:
                        with torch.no_grad():
                            if norm_layer is not None:
                                last_hidden_normalized = norm_layer(last_hidden.unsqueeze(0).to(device))
                                last_hidden_normalized = last_hidden_normalized.squeeze(0)
                            else:
                                last_hidden_normalized = last_hidden.to(device)

                            layer_logits = model.lm_head(last_hidden_normalized)
                            transformer_layer_idx = layer_idx_all

                            if layer_idx_all == len(step_hidden_states) - 1:
                                predicted_token_id = layer_logits.argmax().item()
                                predicted_token_text = tokenizer.decode([predicted_token_id], skip_special_tokens=True)
                                top5_logits, top5_indices = torch.topk(layer_logits, 5)
                                top5_tokens = [tokenizer.decode([idx.item()], skip_special_tokens=True) for idx in top5_indices]

                                original_logit = layer_logits[original_token_id].item() if original_token_id is not None else None
                                original_rank = None
                                if original_logit is not None:
                                    original_rank = torch.sum(layer_logits > original_logit).item() + 1

                                print(f"\n  [Debug Layer 32] Step {step_idx+1}:")
                                print(f"    - 原始输出 token_id: {original_token_id}")
                                print(f"    - 原始输出 token_text: '{original_token_text}'")
                                print(f"    - 第32层预测 token_id (argmax): {predicted_token_id}")
                                print(f"    - 第32层预测 token_text (argmax): '{predicted_token_text}'")
                                print(f"    - 是否一致: {original_token_id == predicted_token_id}")
                                if original_token_id != predicted_token_id:
                                    predicted_logit = layer_logits[predicted_token_id].item()
                                    print(f"    - ⚠️  不一致！")
                                    if original_logit is not None:
                                        print(f"    - 原始token logit: {original_logit:.4f} (排名: {original_rank})")
                                    print(f"    - 预测token logit: {predicted_logit:.4f} (排名: 1)")
                                    print(f"    - Top-5 tokens: {list(zip(top5_tokens, [f'{logit:.4f}' for logit in top5_logits.tolist()]))}")
                                    print(f"    - 可能原因: 使用了采样（temperature > 0），实际生成的token不是argmax")
                                    print(f"    - 或者: hidden state的索引对应关系可能有问题")

                            top_p_info = extract_top_p_tokens(layer_logits, tokenizer, threshold_top_p=0.9)
                            step_lm_head_outputs[step_idx][transformer_layer_idx] = top_p_info

    print(f"  ✓ 已为 {len(step_lm_head_outputs)} 个目标 token 位置提取logits信息")
    return step_lm_head_outputs


def _organize_target_words_by_steps(object_tokens_info):
    """按步骤组织目标词汇，对于出现多次的词汇，只保留第一次和最后一次的group"""
    step_target_words = {}
    for word, obj_info in object_tokens_info.items():
        token_positions = obj_info.get('token_positions', [])
        token_groups = obj_info.get('token_groups', [])

        if not token_groups and not token_positions:
            continue

        # 处理token_groups：如果有多个，只保留第一个和最后一个
        filtered_token_groups = []
        filtered_token_positions = []
        if token_groups:
            if len(token_groups) == 1:
                filtered_token_groups = token_groups
            else:
                # 只保留第一个
                filtered_token_groups = [token_groups[0]] # [token_groups[0], token_groups[-1]]
                # print(f"  ⚠️  词汇 '{word}' 有 {len(token_groups)} 个token组: {token_groups}，只保留第一个和最后一个")
            # 从保留的groups中提取步骤和位置
            word_steps = []
            for group in filtered_token_groups:
                group_positions = list(range(group[0], group[1] + 1))
                word_steps.extend(group_positions)
                filtered_token_positions.extend(group_positions)
            word_steps = sorted(list(set(word_steps)))
            filtered_token_positions = sorted(list(set(filtered_token_positions)))
        else:
            # 如果没有token_groups，使用token_positions
            if token_positions:
                if len(token_positions) == 1:
                    word_steps = [token_positions[0]]
                    filtered_token_positions = [token_positions[0]]
                else:
                    # 只保留第一个
                    word_steps = [token_positions[0]] # [token_positions[0], token_positions[-1]]
                    filtered_token_positions = [token_positions[0]] # [token_positions[0], token_positions[-1]]
                    # print(f"  ⚠️  词汇 '{word}' 在 {token_positions} 位置出现多次，只保留第一次和最后一次的位置")
            else:
                continue

        step_target_words[word] = {
            'steps': word_steps,
            'token_groups': filtered_token_groups,
            'token_positions': filtered_token_positions
        }

    return step_target_words


def _process_attention_tensor(layer_attn):
    """处理attention tensor，提取last_row_attention"""
    if isinstance(layer_attn, tuple):
        layer_attn = layer_attn[0]

    if not isinstance(layer_attn, torch.Tensor):
        return None

    layer_attn_np = layer_attn.cpu().numpy()

    # 处理不同形状的attention tensor
    if len(layer_attn_np.shape) == 4:
        if layer_attn_np.shape[2] == 1:  # query_len == 1
            last_row_attention = layer_attn_np[0].mean(axis=0).squeeze()
        else:
            layer_attn_np = layer_attn_np[0].mean(axis=0)
            last_row_attention = layer_attn_np[-1, :]
    elif len(layer_attn_np.shape) == 3:
        if layer_attn_np.shape[1] == 1:  # query_len == 1
            last_row_attention = layer_attn_np.mean(axis=0).squeeze()
        else:
            layer_attn_np = layer_attn_np.mean(axis=0)
            last_row_attention = layer_attn_np[-1, :]
    elif len(layer_attn_np.shape) == 2:
        if layer_attn_np.shape[0] == 1:  # query_len == 1
            last_row_attention = layer_attn_np[0, :]
        else:
            last_row_attention = layer_attn_np[-1, :]
    elif len(layer_attn_np.shape) == 1:
        last_row_attention = layer_attn_np
    else:
        return None

    return last_row_attention


def _extract_image_attention_map(last_row_attention, image_token_start, num_image_tokens):
    """从last_row_attention中提取图像attention map"""
    actual_num_image_tokens = num_image_tokens if num_image_tokens > 0 else 576
    seq_len = len(last_row_attention)
    image_token_end_actual = min(image_token_start + actual_num_image_tokens, seq_len)
    valid_image_positions = np.arange(image_token_start, image_token_end_actual)

    if len(valid_image_positions) == 0:
        return None

    image_attention = last_row_attention[valid_image_positions]

    # 确保有576个值
    if len(image_attention) < 576:
        image_attention = np.pad(image_attention, (0, 576 - len(image_attention)), mode='constant', constant_values=0)
    elif len(image_attention) > 576:
        image_attention = image_attention[:576]

    # Reshape到24×24
    patch_size = 24
    attention_map = image_attention.reshape(patch_size, patch_size)

    return attention_map


def _get_predicted_token_for_layer(model, tokenizer, step_hidden_states, layer_idx, device):
    """获取指定层预测的token和概率

    Returns:
        tuple: (predicted_token_word, probability) 或 (None, None)
    """
    predicted_token_word = None
    probability = None
    if step_hidden_states is not None and hasattr(model, 'lm_head'):
        lang_model = model.get_model()
        norm_layer = lang_model.norm if hasattr(lang_model, 'norm') else None

        hidden_state_idx = layer_idx
        if isinstance(step_hidden_states, (tuple, list)) and hidden_state_idx < len(step_hidden_states):
            layer_hidden = step_hidden_states[hidden_state_idx]
            if isinstance(layer_hidden, torch.Tensor):
                if len(layer_hidden.shape) == 3:
                    last_hidden = layer_hidden[0, -1, :]
                elif len(layer_hidden.shape) == 2:
                    last_hidden = layer_hidden[-1, :]
                else:
                    last_hidden = None

                if last_hidden is not None:
                    with torch.no_grad():
                        if norm_layer is not None:
                            last_hidden_normalized = norm_layer(last_hidden.unsqueeze(0).to(device))
                            last_hidden_normalized = last_hidden_normalized.squeeze(0)
                        else:
                            last_hidden_normalized = last_hidden.to(device)

                        layer_logits = model.lm_head(last_hidden_normalized)
                        # 计算softmax概率
                        probs = torch.softmax(layer_logits, dim=-1)
                        predicted_token_id = layer_logits.argmax().item()
                        predicted_token_word = tokenizer.decode([predicted_token_id], skip_special_tokens=True)
                        probability = probs[predicted_token_id].item()

    return predicted_token_word, probability


def _extract_attention_maps_for_word(model, tokenizer, word, word_info, all_attentions,
                                    all_hidden_states, selected_layers, image_token_start,
                                    num_image_tokens, device):
    """为单个词汇提取attention map，按token_groups组织数据"""
    word_steps = word_info['steps']
    token_groups = word_info.get('token_groups', [])

    if not word_steps:
        return {}

    # 按token_groups组织数据：{group_idx: {layer_idx: [attention_map1, attention_map2, ...]}}
    group_attention_maps = {}  # {group_idx: {layer_idx: [attention_map1, ...]}}
    group_predicted_tokens = {}  # {group_idx: {layer_idx: [token1, ...]}}
    group_token_probabilities = {}  # {group_idx: {layer_idx: [prob1, ...]}}

    # 如果没有token_groups，使用所有步骤作为一个组
    if not token_groups:
        token_groups = [(word_steps[0], word_steps[-1])] if word_steps else []

    # 为每个token_group收集attention maps
    for group_idx, (group_start, group_end) in enumerate(token_groups):
        group_attention_maps[group_idx] = {}
        group_predicted_tokens[group_idx] = {}
        group_token_probabilities[group_idx] = {}

        # 遍历该组内的所有步骤
        for step_idx in range(group_start, group_end + 1):
            # step_idx 是从 1 开始的生成步骤索引，all_attentions 是从 0 开始的数组
            attn_idx = step_idx - 1
            if attn_idx < 0 or attn_idx >= len(all_attentions) or all_attentions[attn_idx] is None:
                continue

            step_attentions = all_attentions[attn_idx]
            step_hidden_states = None
            if all_hidden_states is not None and attn_idx < len(all_hidden_states):
                step_hidden_states = all_hidden_states[attn_idx]

            for layer_idx, layer_attn in enumerate(step_attentions):
                if layer_idx not in selected_layers or layer_attn is None:
                    continue

                predicted_token_word, probability = _get_predicted_token_for_layer(
                    model, tokenizer, step_hidden_states, layer_idx, device
                )

                last_row_attention = _process_attention_tensor(layer_attn)

                if last_row_attention is None:
                    continue

                attention_map = _extract_image_attention_map(
                    last_row_attention, image_token_start, num_image_tokens
                )

                if attention_map is None:
                    continue

                if layer_idx not in group_attention_maps[group_idx]:
                    group_attention_maps[group_idx][layer_idx] = []
                    group_predicted_tokens[group_idx][layer_idx] = []
                    group_token_probabilities[group_idx][layer_idx] = []

                group_attention_maps[group_idx][layer_idx].append(attention_map)
                group_predicted_tokens[group_idx][layer_idx].append(predicted_token_word)
                group_token_probabilities[group_idx][layer_idx].append(probability if probability is not None else 0.0)

    return {
        'group_attention_maps': group_attention_maps,
        'group_predicted_tokens': group_predicted_tokens,
        'group_token_probabilities': group_token_probabilities,
        'token_groups': token_groups
    }


def _combine_and_visualize_attention_maps(word, word_info, attention_data, image,
                                         selected_layers, output_dir, patch_size=24):
    """按token_groups组合attention maps并可视化"""
    word_steps = word_info['steps']
    token_groups = attention_data.get('token_groups', word_info.get('token_groups', []))
    group_attention_maps = attention_data['group_attention_maps']
    group_predicted_tokens = attention_data['group_predicted_tokens']
    group_token_probabilities = attention_data.get('group_token_probabilities', {})

    safe_word = word.replace(' ', '_').replace('/', '_').replace('\\', '_')
    if len(word_steps) == 1:
        step_label = f"step_{word_steps[0]}"
    else:
        step_label = f"step_{'_'.join(map(str, word_steps))}"

    step_dir = os.path.join(output_dir, f"{step_label}_{safe_word}")
    os.makedirs(step_dir, exist_ok=True)

    print(f"\n  [词汇 '{word}'] 步骤: {word_steps}, Token组: {token_groups}")
    print(f"    输出目录: {os.path.basename(step_dir)}")

    # 为每个token_group生成attention map
    for group_idx, (group_start, group_end) in enumerate(token_groups):
        if group_idx not in group_attention_maps:
            continue

        group_maps = group_attention_maps[group_idx]
        group_tokens = group_predicted_tokens[group_idx]
        group_probs = group_token_probabilities.get(group_idx, {})

        # 为每个层叠加该组内的所有步骤
        for layer_idx in sorted(group_maps.keys()):
            if layer_idx not in selected_layers:
                continue

            attention_maps = group_maps[layer_idx]
            predicted_tokens = group_tokens[layer_idx]
            probabilities = group_probs.get(layer_idx, [])

            if not attention_maps:
                continue

            # 叠加该组内的所有步骤
            if len(attention_maps) == 1:
                combined_attention_map = attention_maps[0]
                combined_predicted_token = predicted_tokens[0] if predicted_tokens[0] else None
            else:
                # 使用概率作为权重进行加权平均
                if probabilities and len(probabilities) == len(attention_maps) and any(p > 0 for p in probabilities):
                    # 归一化概率作为权重
                    weights = np.array(probabilities)
                    weights = weights / weights.sum()  # 归一化
                    # 加权平均
                    combined_attention_map = np.average(attention_maps, axis=0, weights=weights)
                    combined_predicted_token = '_'.join([t for t in predicted_tokens if t]) if any(predicted_tokens) else None
                else:
                    # 如果没有概率信息，使用简单平均
                    combined_attention_map = np.mean(attention_maps, axis=0)
                    combined_predicted_token = '_'.join([t for t in predicted_tokens if t]) if any(predicted_tokens) else None

            # 打印统计信息
            attn_min = float(combined_attention_map.min())
            attn_max = float(combined_attention_map.max())
            group_steps = list(range(group_start, group_end + 1))
            print(f"\n    [Debug] 词汇 '{word}', Layer {layer_idx}, Token组 {group_idx+1}: {group_steps}")
            print(f"      - 叠加的步骤数: {len(attention_maps)}")
            if probabilities and len(probabilities) == len(attention_maps) and any(p > 0 for p in probabilities):
                weights = np.array(probabilities)
                weights = weights / weights.sum()
                print(f"      - 使用概率加权平均 (权重: {[f'{w:.3f}' for w in weights]})")
            else:
                print(f"      - 使用简单平均 (无概率信息)")
            print(f"      - 最终24×24 attention map: min={attn_min:.4e}, max={attn_max:.4e}")
            if combined_predicted_token:
                print(f"      - 组合的预测tokens: '{combined_predicted_token}'")

            token_text = word
            token_name = safe_word
            step_idx_for_filename = str(group_steps[0]) + '_' + str(group_steps[-1])

            # 如果多个组，在文件名中包含组信息
            if len(token_groups) > 1:
                token_name = f"{token_name}_group{group_idx+1}"

            visualize_object_attention_map(
                combined_attention_map, image, layer_idx, step_idx_for_filename, token_text, token_name,
                patch_size, step_dir, predicted_token_word=combined_predicted_token
            )


def _generate_heatmaps_for_words(step_target_words, step_lm_head_outputs, output_dir, num_total_layers):
    """为所有目标词汇生成5×32 heatmap（每个单步单独生成，不叠加）"""
    if not step_lm_head_outputs:
        return

    print(f"\n  [生成5×32 Heatmap] 为目标词汇生成heatmap...")


    for word, word_info in step_target_words.items():
        word_steps = word_info['steps']
        token_groups = word_info.get('token_groups', [])
        if not word_steps:
            continue

        safe_word = word.replace(' ', '_').replace('/', '_').replace('\\', '_')
        if len(word_steps) == 1:
            step_label = f"step_{word_steps[0]}"
        else:
            step_label = f"step_{'_'.join(map(str, word_steps))}"

        step_output_dir = os.path.join(output_dir, f"{step_label}_{safe_word}")
        os.makedirs(step_output_dir, exist_ok=True)

        # 收集所有需要生成heatmap的步骤（从token_groups中提取，如果没有则使用word_steps）
        steps_to_visualize = set()
        if token_groups:
            for group_start, group_end in token_groups:
                for step_idx in range(group_start, group_end + 1):
                    steps_to_visualize.add(step_idx)
        else:
            steps_to_visualize = set(word_steps)

        # 为每个单步生成heatmap
        for step_idx in sorted(steps_to_visualize):
            if step_idx not in step_lm_head_outputs:
                continue

            layer_outputs = step_lm_head_outputs[step_idx]
            if not layer_outputs:
                continue

            # 创建输出目录
            # step_output_dir = os.path.join(output_dir, f"step_{step_idx}_{safe_word}")
            # os.makedirs(step_output_dir, exist_ok=True)

            # 直接使用该步骤的layer_outputs，不叠加
            visualize_top5_logits_heatmap(
                layer_outputs, step_output_dir, step_idx, num_total_layers
            )

            # 保存词汇语义信息到JSON文件
            lm_head_data = {}
            for layer_key, top_p_info in layer_outputs.items():
                top_p_info_filtered = {k: v for k, v in top_p_info.items() if k != 'top_p_tokens'}
                lm_head_data[str(layer_key)] = top_p_info_filtered

            json_file = os.path.join(step_output_dir, f"step_{step_idx}_top5_tokens.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(lm_head_data, f, ensure_ascii=False, indent=2)
            print(f"  ✓ 词汇 '{word}' 步骤 {step_idx} 的词汇语义信息已保存: {os.path.basename(json_file)}")


def _generate_rank1_heatmaps_for_words(step_target_words, step_lm_head_outputs, output_dir, num_total_layers):
    """为每个词汇生成rank1概率heatmap（提取每个token的rank1行）

    如果词汇有多个token_groups，为每个token_group生成一张独立的rank1 heatmap。
    在像素点上显示预测的文本词汇信息，而不是概率值（颜色用概率值表示）。
    最后生成一张整合所有词汇的rank1 heatmap，按照token顺序排列。
    """
    if not step_lm_head_outputs:
        return

    print(f"\n  [生成Rank1 Heatmap] 为目标词汇生成rank1概率heatmap...")

    # 收集所有词汇的rank1数据，用于最后生成整合图
    all_rank1_data = []  # [(word, step_idx, [layer0_prob, ...], [layer0_text, ...]), ...]

    for word, word_info in step_target_words.items():
        word_steps = word_info['steps']
        token_groups = word_info.get('token_groups', [])
        if not word_steps:
            continue

        safe_word = word.replace(' ', '_').replace('/', '_').replace('\\', '_')
        if len(word_steps) == 1:
            step_label = f"step_{word_steps[0]}"
        else:
            step_label = f"step_{'_'.join(map(str, word_steps))}"

        step_output_dir = os.path.join(output_dir, f"{step_label}_{safe_word}")
        os.makedirs(step_output_dir, exist_ok=True)

        # 确定要处理的token组列表
        # 如果有token_groups，为每个group生成一张图；否则使用所有word_steps作为一个组
        groups_to_process = []
        if token_groups:
            # 为每个token_group生成一张图
            for group_idx, (group_start, group_end) in enumerate(token_groups):
                group_steps = [s for s in range(group_start, group_end + 1)
                              if s in step_lm_head_outputs]
                if group_steps:
                    groups_to_process.append((group_idx, group_steps, (group_start, group_end)))
        else:
            # 没有token_groups，使用所有word_steps作为一个组
            group_steps = [s for s in word_steps if s in step_lm_head_outputs]
            if group_steps:
                groups_to_process.append((None, group_steps, None))

        if not groups_to_process:
            print(f"  ⚠️  词汇 '{word}': 没有在step_lm_head_outputs中找到对应的token步骤，跳过rank1 heatmap生成")
            continue

        # 为每个token_group生成一张独立的rank1 heatmap
        for group_idx, token_steps, group_range in groups_to_process:
            # 收集该group内每个token的rank1数据（概率和文本）
            rank1_data = []  # [(step_idx, [layer0_prob, layer1_prob, ...]), ...]
            rank1_texts = []  # [[layer0_text, layer1_text, ...], ...] 每个token对应的所有层的rank1文本
            token_labels = []  # 用于y轴标签

            for step_idx in sorted(token_steps):
                # 这里step_idx肯定在step_lm_head_outputs中，因为上面已经过滤了
                layer_outputs = step_lm_head_outputs[step_idx]
                if not layer_outputs:
                    continue

                # 提取该token在所有层的rank1概率和文本（从5×32 heatmap的最底下一行提取）
                rank1_probs = []
                rank1_token_texts = []
                for layer_idx in range(num_total_layers):
                    # layer_outputs 的键是整数（transformer_layer_idx），从0开始
                    # 但需要检查是否存在该层的数据
                    top_p_info = None
                    if layer_idx in layer_outputs:
                        top_p_info = layer_outputs[layer_idx]
                    elif str(layer_idx) in layer_outputs:
                        top_p_info = layer_outputs[str(layer_idx)]

                    if top_p_info is not None:
                        top_5_tokens = top_p_info.get('top_5_tokens', [])
                        if len(top_5_tokens) > 0:
                            # rank1是第一个（索引0），即5×32 heatmap的最底下一行
                            rank1_probs.append(top_5_tokens[0].get('probability', np.nan))
                            rank1_token_texts.append(top_5_tokens[0].get('token_text', ''))
                        else:
                            rank1_probs.append(np.nan)
                            rank1_token_texts.append('')
                    else:
                        rank1_probs.append(np.nan)
                        rank1_token_texts.append('')

                # 只有当至少有一层有有效数据时才添加
                if any(not np.isnan(p) for p in rank1_probs):
                    rank1_data.append((step_idx, rank1_probs))
                    rank1_texts.append(rank1_token_texts)
                    token_labels.append(f"Token {step_idx}")
                    # 同时收集到all_rank1_data中，用于生成整合图
                    all_rank1_data.append((word, step_idx, rank1_probs, rank1_token_texts))

            if not rank1_data:
                continue

            # 创建heatmap矩阵：行数=token数量，列数=32层
            num_tokens = len(rank1_data)
            probability_matrix = np.full((num_tokens, num_total_layers), np.nan)
            token_texts_matrix = [[''] * num_total_layers for _ in range(num_tokens)]

            for row_idx, (step_idx, probs) in enumerate(rank1_data):
                for layer_idx in range(num_total_layers):
                    if layer_idx < len(probs):
                        probability_matrix[row_idx, layer_idx] = probs[layer_idx]
                    if row_idx < len(rank1_texts) and layer_idx < len(rank1_texts[row_idx]):
                        token_texts_matrix[row_idx][layer_idx] = rank1_texts[row_idx][layer_idx]

            # 只处理非NaN的值
            valid_probabilities = probability_matrix[~np.isnan(probability_matrix)]
            if len(valid_probabilities) == 0:
                group_info = f"group {group_idx+1}" if group_idx is not None else "all tokens"
                print(f"  ⚠️  词汇 '{word}' {group_info}: 没有有效的rank1概率值，跳过heatmap生成")
                continue

            # 计算figsize（增大1.2倍，使像素点更大）
            base_width = 20 * 1.2
            base_height = base_width * (num_tokens / num_total_layers)
            fig, ax = plt.subplots(figsize=(base_width, base_height))

            # 设置aspect ratio
            ax.set_aspect('equal', adjustable='box')

            # 使用pcolormesh
            probability_extended = np.full((num_tokens + 1, num_total_layers + 1), np.nan)
            probability_extended[:num_tokens, :num_total_layers] = probability_matrix

            X = np.arange(num_total_layers + 2)
            Y = np.arange(num_tokens + 2)
            X_grid, Y_grid = np.meshgrid(X, Y)

            # 绘制heatmap
            im = ax.pcolormesh(X_grid, Y_grid, probability_extended, cmap='Greens',
                               edgecolors='white', linewidths=2.0,
                               vmin=np.nanmin(probability_matrix), vmax=np.nanmax(probability_matrix),
                               shading='flat')

            # 设置坐标轴
            ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
            ax.set_ylabel('Token', fontsize=12, fontweight='bold')

            # 标题：如果有多个group，在标题中标注group信息
            if group_idx is not None and len(groups_to_process) > 1:
                title = f'Rank1 Probability Heatmap - {word}, Group {group_idx+1} ({num_tokens} token{"s" if num_tokens > 1 else ""})'
            else:
                title = f'Rank1 Probability Heatmap - {word}, ({num_tokens} token{"s" if num_tokens > 1 else ""})'
            ax.set_title(title, fontsize=14, fontweight='bold')

            # 设置x轴刻度（层索引）- 只显示6个刻度，并加粗
            num_ticks = 6
            tick_indices = np.linspace(0, num_total_layers - 1, num_ticks, dtype=int)
            ax.set_xticks(tick_indices + 0.5)
            ax.set_xticklabels([f'L{i}' for i in tick_indices], rotation=45, ha='right', fontsize=8, fontweight='bold')

            # 设置y轴刻度（token）
            ax.set_yticks(np.arange(num_tokens) + 0.5)
            ax.set_yticklabels(token_labels, fontsize=10)

            # 在每个单元格中心标注token文本（词汇），而不是概率值
            for row_idx in range(num_tokens):
                for layer_idx in range(num_total_layers):
                    token_text = token_texts_matrix[row_idx][layer_idx]
                    prob_value = probability_matrix[row_idx, layer_idx]

                    # 只标注有效的token
                    if token_text and not np.isnan(prob_value):
                        # 清理token文本，移除换行符和特殊字符，限制长度
                        clean_text = token_text.replace('\n', ' ').replace('\r', ' ').strip()

                        # 检查是否可以正常显示（只包含可打印字符）
                        printable_chars = set(string.printable)
                        if not all(c in printable_chars for c in clean_text) or not clean_text:
                            clean_text = " * "
                        else:
                            # 限制长度，避免文本过长
                            if len(clean_text) > 12:
                                clean_text = clean_text[:12] + '...'

                        # 根据概率值选择文本颜色（深色或浅色）
                        max_prob = np.nanmax(probability_matrix)
                        min_prob = np.nanmin(probability_matrix)
                        if max_prob > min_prob:
                            normalized = (prob_value - min_prob) / (max_prob - min_prob)
                            text_color = 'white' if normalized > 0.5 else 'black'
                        else:
                            text_color = 'black'

                        # 在单元格中心标注文本（词汇），旋转45度
                        # 使用半透明背景以提高可读性
                        ax.text(layer_idx + 0.5, row_idx + 0.5, clean_text,
                               ha='center', va='center',
                               fontsize=9, color=text_color, fontweight='bold',
                               rotation=45,  # 旋转45度
                               bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='white' if text_color == 'black' else 'black',
                                        alpha=0.6, edgecolor='none'))

            # 添加colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Probability', fontsize=10, fontweight='bold')

            # 设置坐标轴范围
            ax.set_xlim(0, num_total_layers)
            ax.set_ylim(0, num_tokens)

            plt.tight_layout()

            # 保存图片：如果有多个group，在文件名中包含group信息
            if group_idx is not None and len(groups_to_process) > 1:
                heatmap_file = os.path.join(step_output_dir, f"rank1_probability_heatmap_{safe_word}_group{group_idx+1}.png")
            else:
                heatmap_file = os.path.join(step_output_dir, f"rank1_probability_heatmap_{safe_word}.png")
            plt.savefig(heatmap_file, dpi=200, bbox_inches='tight')
            plt.close()

            # 统计信息
            valid_count = np.sum(~np.isnan(probability_matrix))
            total_count = num_tokens * num_total_layers
            group_info = f"Group {group_idx+1}" if group_idx is not None and len(groups_to_process) > 1 else ""
            print(f"  ✓ 词汇 '{word}' {group_info} 的Rank1 Probability Heatmap已保存: {os.path.basename(heatmap_file)}")
            print(f"    有效值数量: {valid_count}/{total_count}")
            if len(valid_probabilities) > 0:
                print(f"    概率值范围: [{valid_probabilities.min():.4f}, {valid_probabilities.max():.4f}]")

    # 生成整合所有词汇的rank1 heatmap
    if all_rank1_data:
        print(f"\n  [生成整合Rank1 Heatmap] 整合所有词汇的rank1结果...")

        # 按照token顺序（step_idx）排序
        all_rank1_data.sort(key=lambda x: x[1])  # 按step_idx排序

        # 创建整合的heatmap矩阵
        num_all_tokens = len(all_rank1_data)
        all_probability_matrix = np.full((num_all_tokens, num_total_layers), np.nan)
        all_token_texts_matrix = [[''] * num_total_layers for _ in range(num_all_tokens)]
        all_token_labels = []

        for row_idx, (word, step_idx, probs, texts) in enumerate(all_rank1_data):
            for layer_idx in range(num_total_layers):
                if layer_idx < len(probs):
                    all_probability_matrix[row_idx, layer_idx] = probs[layer_idx]
                if layer_idx < len(texts):
                    all_token_texts_matrix[row_idx][layer_idx] = texts[layer_idx]
            # 标签包含词汇名和token索引
            all_token_labels.append(f"Token {step_idx}")

        # 只处理非NaN的值
        all_valid_probabilities = all_probability_matrix[~np.isnan(all_probability_matrix)]
        if len(all_valid_probabilities) > 0:
            # 计算figsize（增大1.2倍，使像素点更大）
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

            # 绘制heatmap
            im = ax.pcolormesh(X_grid, Y_grid, probability_extended, cmap='Greens',
                               edgecolors='white', linewidths=2.0,
                               vmin=np.nanmin(all_probability_matrix), vmax=np.nanmax(all_probability_matrix),
                               shading='flat')

            # 设置坐标轴（增大到1.5倍）
            # ax.set_xlabel('Layer Index', fontsize=18, fontweight='bold')  # 12 * 1.5 = 18
            # ax.set_ylabel('Token', fontsize=18, fontweight='bold')  # 12 * 1.5 = 18
            ax.set_title(f'Rank1 Probability Heatmap - All Words ({num_all_tokens} tokens)',
                         fontsize=14, fontweight='bold')

            # 设置x轴刻度（层索引）- 只显示6个刻度，并加粗（增大到1.5倍）
            num_ticks = 6
            tick_indices = np.linspace(0, num_total_layers - 1, num_ticks, dtype=int)
            ax.set_xticks(tick_indices + 0.5)
            ax.set_xticklabels([f'L{i}' for i in tick_indices], rotation=45, ha='right', fontsize=12, fontweight='bold')  # 8 * 1.5 = 12

            # 设置y轴刻度（token）- 加粗并增大到1.5倍
            ax.set_yticks(np.arange(num_all_tokens) + 0.5)
            ax.set_yticklabels(all_token_labels, fontsize=14, fontweight='bold')  # 9 * 1.5 = 13.5，约14

            # 在每个单元格中心标注token文本（词汇），而不是概率值
            for row_idx in range(num_all_tokens):
                for layer_idx in range(num_total_layers):
                    token_text = all_token_texts_matrix[row_idx][layer_idx]
                    prob_value = all_probability_matrix[row_idx, layer_idx]

                    # 只标注有效的token
                    if token_text and not np.isnan(prob_value):
                        # 清理token文本，移除换行符和特殊字符，限制长度
                        clean_text = token_text.replace('\n', ' ').replace('\r', ' ').strip()

                        # 检查是否可以正常显示（只包含可打印字符）
                        printable_chars = set(string.printable)
                        if not all(c in printable_chars for c in clean_text) or not clean_text:
                            clean_text = " * "
                        else:
                            # 限制长度，避免文本过长（因为像素点增大了，可以显示更多字符）
                            if len(clean_text) > 15:
                                clean_text = clean_text[:15] + '...'

                        # 根据概率值选择文本颜色（深色或浅色）
                        max_prob = np.nanmax(all_probability_matrix)
                        min_prob = np.nanmin(all_probability_matrix)
                        if max_prob > min_prob:
                            normalized = (prob_value - min_prob) / (max_prob - min_prob)
                            text_color = 'white' if normalized > 0.5 else 'black'
                        else:
                            text_color = 'black'

                        # 在单元格中心标注文本（词汇），旋转45度
                        # 使用半透明背景以提高可读性
                        ax.text(layer_idx + 0.5, row_idx + 0.5, clean_text,
                               ha='center', va='center',
                               fontsize=10, color=text_color, fontweight='bold',  # 字体稍微增大
                               rotation=45,  # 旋转45度
                               bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='white' if text_color == 'black' else 'black',
                                        alpha=0.6, edgecolor='none'))

            # 添加colorbar（减小到原来的十分之一）
            cbar = plt.colorbar(im, ax=ax, fraction=0.042, pad=0.04)
            cbar.set_label('Probability', fontsize=10, fontweight='bold')

            # 设置坐标轴范围
            ax.set_xlim(0, num_total_layers)
            ax.set_ylim(0, num_all_tokens)

            plt.tight_layout()

            # 保存整合的heatmap到输出目录的根目录
            combined_heatmap_file = os.path.join(output_dir, "rank1_probability_heatmap_all_words.png")
            plt.savefig(combined_heatmap_file, dpi=200, bbox_inches='tight')
            plt.close()

            # 统计信息
            valid_count = np.sum(~np.isnan(all_probability_matrix))
            total_count = num_all_tokens * num_total_layers
            print(f"  ✓ 整合所有词汇的Rank1 Probability Heatmap已保存: {os.path.basename(combined_heatmap_file)}")
            print(f"    有效值数量: {valid_count}/{total_count}")
            print(f"    概率值范围: [{all_valid_probabilities.min():.4f}, {all_valid_probabilities.max():.4f}]")


def extract_object_attention_maps(model, tokenizer, image_processor, image_file, prompt, conv_mode, device,
                                  output_ids, input_token_len, all_attentions, object_tokens_info,
                                  image_tensor, output_dir, target_layers=None, all_hidden_states=None, outputs_text=None):
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
        all_hidden_states: 所有生成步骤的hidden states（tuple，每个元素对应一个步骤）
        outputs_text: 生成的完整文本（用于提取每个步骤生成的词汇）
    """
    if not object_tokens_info or all_attentions is None:
        return

    # 解析生成的文本，获取每个步骤生成的token对应的词汇
    step_generated_words = _parse_step_generated_words(output_ids, tokenizer, outputs_text)

    # 加载图像
    image = _load_image(image_file)

    # 拷贝原图到输出目录
    import shutil
    image_filename = os.path.basename(image_file)
    copied_image_path = os.path.join(output_dir, image_filename)
    try:
        shutil.copy2(image_file, copied_image_path)
        print(f"  ✓ 原图已拷贝到: {os.path.basename(copied_image_path)}")
    except Exception as e:
        print(f"  ⚠️  拷贝原图失败: {e}")

    # 额外保存一张resize成正方形的原图
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    else:
        image_pil = image
    # Resize成正方形
    patch_size = 24
    display_size = patch_size * 20  # 480x480
    image_resized = image_pil.resize((display_size, display_size), Image.Resampling.LANCZOS)
    # 保存resize后的原图
    resized_image_path = os.path.join(output_dir, "original_image_resized_square.png")
    try:
        image_resized.save(resized_image_path)
        print(f"  ✓ Resize成正方形的原图已保存: {os.path.basename(resized_image_path)}")
    except Exception as e:
        print(f"  ⚠️  保存resize后的原图失败: {e}")

    # 确定目标层
    selected_layers, num_total_layers = _determine_target_layers(model, target_layers)

    # 获取图像token位置信息
    image_token_start, num_image_tokens = _get_image_token_info(
        model, tokenizer, prompt, image_tensor, device
    )

    # 收集所有目标token位置
    target_token_positions = _collect_target_token_positions(object_tokens_info)

    # 提取各层的logits信息
    step_lm_head_outputs = _extract_layer_logits_for_tokens(
        model, tokenizer, output_ids, all_hidden_states, target_token_positions, device
    )

    # 按步骤组织目标词汇
    step_target_words = _organize_target_words_by_steps(object_tokens_info)

    # 为每个目标词汇提取attention map
    for word, word_info in step_target_words.items():
        attention_data = _extract_attention_maps_for_word(
            model, tokenizer, word, word_info, all_attentions, all_hidden_states,
            selected_layers, image_token_start, num_image_tokens, device
        )

        if attention_data.get('group_attention_maps'):
            _combine_and_visualize_attention_maps(
                word, word_info, attention_data, image, selected_layers, output_dir
            )

    # 生成5×32 heatmap
    _generate_heatmaps_for_words(step_target_words, step_lm_head_outputs, output_dir, num_total_layers)

    # 生成rank1概率heatmap（每个词汇的所有token的rank1行）
    _generate_rank1_heatmaps_for_words(step_target_words, step_lm_head_outputs, output_dir, num_total_layers)


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
                        image_id = int(parts[-1])
                        image_id_list.append(image_id)
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
                    image_id = int(line)
                    image_id_list.append(image_id)
            print(f"✓ 从文本文件读取了 {len(image_id_list)} 个图像 ID")

    # 如果指定了单个图像ID, 使用该ID
    if args.single_image_id is not None:
        image_id_list = [args.single_image_id]
        print(f"📌 使用指定的图像ID: {args.single_image_id}")

    images = get_coco_val2014_images(
        coco_root=args.coco_root,
        image_id_list=image_id_list,
        max_images=args.num_samples if args.num_samples > 0 else 0
    )
    print(f"✓ 找到 {len(images)} 个图像")

    # 如果只处理一个图像, 给出提示
    if len(images) == 1:
        print(f"📝 将处理单个图像: Image ID {images[0]['image_id']}")
        print(f"   - 图像路径: {images[0]['image_path']}")
        if args.extract_object_attention:
            print(f"   - 将提取物体 attention map (目标层: {args.target_layers})")
        else:
            print(f"   - 物体 attention map 提取已禁用")

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

    # 如果启用了提取物体attention map，在循环外部初始化CHAIR评估器（避免重复加载）
    extract_object_attention = getattr(args, 'extract_object_attention', False)
    chair_evaluator = None
    if extract_object_attention:
        coco_annotations_path = os.path.join(args.coco_root, "annotations_trainval2014", "annotations")
        if not os.path.exists(coco_annotations_path):
            if hasattr(project, 'coco_annotations_path'):
                coco_annotations_path = project.coco_annotations_path

        if os.path.exists(coco_annotations_path):
            # 优先使用 eval_tool 目录下已有的缓存文件
            eval_tool_dir = os.path.join(project_root, "eval_tool")
            default_cache_file = os.path.join(eval_tool_dir, "chair_evaluator.pkl")

            # 如果默认缓存文件存在，使用它；否则使用输出目录下的缓存文件
            if os.path.exists(default_cache_file):
                cache_file = default_cache_file
                print(f"\n[初始化 CHAIR 评估器] 使用已有的缓存文件: {cache_file}")
            else:
                cache_file = os.path.join(os.path.dirname(output_file), "chair_evaluator_cache.pkl")
                print(f"\n[初始化 CHAIR 评估器] 正在加载标注数据（这可能需要一些时间，但只会加载一次）...")

            chair_evaluator = get_chair_evaluator(
                coco_path=coco_annotations_path,
                cache_file=cache_file,
                use_cache=True
            )
            print(f"✓ CHAIR 评估器初始化完成")
        else:
            print(f"  ⚠️  无法找到COCO annotations路径，将使用简单的NLTK方法识别名词")

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
        if extract_object_attention and all_attentions is not None and outputs:
            # 使用已在循环外部初始化的CHAIR评估器（避免重复加载）
            # 识别描述中的物体
            object_tokens_info, tokens_detail_info = identify_object_tokens_in_caption(
                outputs, tokenizer, output_ids, input_token_len, chair_evaluator
            )

            # 为这个图像创建输出目录
            image_output_dir = os.path.join(os.path.dirname(output_file), "object_attention_maps", f"image_{image_id}")
            os.makedirs(image_output_dir, exist_ok=True)

            # 获取CHAIR评估信息（如果chair_evaluator可用）
            chair_info = {}
            if chair_evaluator is not None:
                try:
                    # 获取该图像的GT对象
                    gt_objects = list(chair_evaluator.imid_to_objects.get(image_id, []))

                    # 获取生成描述中的对象词汇
                    words, node_words, idxs, raw_words = chair_evaluator.caption_to_words(outputs)

                    # 识别幻视词汇
                    hallucinated_words = []
                    for word, node_word, idx in zip(words, node_words, idxs):
                        if node_word not in gt_objects:
                            hallucinated_words.append((word, node_word))

                    chair_info = {
                        'mscoco_gt_words': sorted(gt_objects),
                        'mscoco_generated_words': list(node_words),
                        'mscoco_hallucinated_words': hallucinated_words
                    }
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  获取CHAIR评估信息失败: {e}")
                    chair_info = {
                        'mscoco_gt_words': [],
                        'mscoco_generated_words': [],
                        'mscoco_hallucinated_words': []
                    }
            else:
                chair_info = {
                    'mscoco_gt_words': [],
                    'mscoco_generated_words': [],
                    'mscoco_hallucinated_words': []
                }

            # 保存详细的token信息到JSON文件
            token_detail_file = os.path.join(image_output_dir, "token_details.json")
            token_detail_data = {
                'image_id': image_id,
                'caption': outputs,
                'full_generated_text': tokens_detail_info['full_generated_text'],
                'total_tokens': tokens_detail_info['total_tokens'],
                'input_token_len': input_token_len,
                'all_tokens': tokens_detail_info['all_tokens_detail'],
                'object_tokens_info': object_tokens_info,
                **chair_info  # 添加CHAIR评估信息
            }
            with open(token_detail_file, 'w', encoding='utf-8') as f:
                json.dump(token_detail_data, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"  ✓ Token详细信息已保存到: {os.path.basename(token_detail_file)}")
                print(f"     - 总Token数: {tokens_detail_info['total_tokens']}")
                print(f"     - 找到的物体数: {len(object_tokens_info)}")

            if object_tokens_info:
                print(f"\n  [物体识别] 找到 {len(object_tokens_info)} 个不同物体名字的词汇(可能是同一个物体不同的描述方式):")
                # object_tokens_info 现在是字典，以 object_word 为 key
                for object_word, obj_info in object_tokens_info.items():
                    print(f"    - {obj_info['object_word']} (规范化: {obj_info['node_word']})")
                    print(f"      Token位置: {obj_info['token_positions']} (共 {len(obj_info['token_positions'])} 个位置， 匹配 {len(obj_info.get('matched_tokens_detail', []))} 个Token)")

                    # 验证：显示这些token位置对应的实际文本
                    matched_tokens = obj_info.get('matched_tokens_detail', [])
                    if matched_tokens:
                        print(f"      [验证] Token文本内容:")
                        for token_detail in matched_tokens:
                            token_text = token_detail.get('token_text', '').strip()
                            token_pos = token_detail.get('absolute_position', 'N/A')
                            char_start = token_detail.get('char_start', 'N/A')
                            char_end = token_detail.get('char_end', 'N/A')
                            # 显示token文本，如果是空白字符则显示转义形式
                            token_display = repr(token_detail.get('token_text', '')) if not token_text else f"'{token_text}'"
                            print(f"        - 位置 {token_pos}: {token_display} (字符位置: {char_start}-{char_end})")

                    else:
                        print(f"      ⚠️  警告: 未找到匹配的token详细信息")

                # 提取物体attention map
                target_layers = getattr(args, 'target_layers', 'even')
                extract_object_attention_maps(
                    model, tokenizer, image_processor, image_file, prompt, conv_mode, device,
                    output_ids, input_token_len, all_attentions, object_tokens_info,
                    image_tensor, image_output_dir, target_layers=target_layers,
                    all_hidden_states=all_hidden_states, outputs_text=outputs
                )

                if verbose:
                    print(f"  📊 已为 {len(object_tokens_info)} 个物体提取 attention map")
                    print(f"     - 输出目录: {image_output_dir}")
                    print(f"     - 目标层: {target_layers}")

                if verbose:
                    print(f"  ✓ 物体attention map已保存到: {image_output_dir}")

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

        # 优先使用 eval_tool 目录下已有的缓存文件
        eval_tool_dir = os.path.join(project_root, "eval_tool")
        default_cache_file = os.path.join(eval_tool_dir, "chair_evaluator.pkl")

        # 如果默认缓存文件存在，使用它；否则使用输出目录下的缓存文件
        if os.path.exists(default_cache_file):
            cache_file = default_cache_file
            print(f"\n[自动计算 CHAIR] 使用已有的缓存文件: {cache_file}")
        else:
            cache_file = os.path.join(results_dir, "chair_evaluator.pkl")
            print(f"\n[自动计算 CHAIR] 使用输出目录下的缓存文件: {cache_file}")

        # 调用 evaluate_chair 函数
        results = evaluate_chair(
            cap_file=output_file,
            coco_path=coco_annotations_path,
            image_id_key="image_id",
            caption_key="caption",
            cache_file=cache_file,
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
                # 按照 mscoco_hallucinated_words 的数量从高到低排序
                error_samples.sort(key=lambda x: len(x.get('mscoco_hallucinated_words', [])), reverse=True)

                # 只保留前10%最严重的样本
                top_10_percent_count = max(1, int(len(error_samples) * 0.1))
                top_error_samples = error_samples[:top_10_percent_count]

                # 自定义JSON格式化函数：所有数组字段都输出到一行
                def format_json_compact_arrays(obj, indent_level=0):
                    """格式化JSON，所有数组字段都输出到一行"""
                    indent = '  ' * indent_level
                    next_indent = '  ' * (indent_level + 1)

                    if isinstance(obj, dict):
                        if not obj:
                            return '{}'
                        lines = []
                        for key, value in obj.items():
                            if isinstance(value, (list, tuple)):
                                # 所有数组都使用紧凑格式（单行）
                                json_value = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
                                lines.append(f'{next_indent}"{key}": {json_value}')
                            elif isinstance(value, dict):
                                # 嵌套字典：递归处理
                                formatted_value = format_json_compact_arrays(value, indent_level + 1)
                                lines.append(f'{next_indent}"{key}": {formatted_value}')
                            else:
                                # 其他类型：正常格式
                                json_value = json.dumps(value, ensure_ascii=False)
                                lines.append(f'{next_indent}"{key}": {json_value}')
                        return '{\n' + ',\n'.join(lines) + '\n' + indent + '}'
                    elif isinstance(obj, (list, tuple)):
                        # 列表：每个元素一行
                        if not obj:
                            return '[]'
                        lines = []
                        for item in obj:
                            if isinstance(item, (dict, list)):
                                formatted_item = format_json_compact_arrays(item, indent_level + 1)
                                lines.append(f'{next_indent}{formatted_item},')
                            else:
                                json_item = json.dumps(item, ensure_ascii=False)
                                lines.append(f'{next_indent}{json_item},')
                        # 移除最后一个逗号
                        if lines and lines[-1].endswith(','):
                            lines[-1] = lines[-1][:-1]
                        return '[\n' + '\n'.join(lines) + '\n' + indent + ']'
                    else:
                        return json.dumps(obj, ensure_ascii=False)

                # 构建输出数据
                output_data = {
                    'error_count': len(error_samples),
                    'total_samples': len(results['sentences']),
                    'top_10_percent_count': top_10_percent_count,
                    'error_samples': top_error_samples
                }

                # 格式化并保存
                formatted_json = format_json_compact_arrays(output_data)
                with open(chair_errors_file, 'w', encoding='utf-8') as f:
                    f.write(formatted_json)

                print(f"错误样本文件: {chair_errors_file} ({top_10_percent_count} 个最严重的幻视样本，共 {len(error_samples)} 个包含幻觉的样本)")

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
        "use_deco": False,
        "alpha": 0.6,
        "threshold_top_p": 0.9,
        "threshold_top_k": 20,
        "start_layer": 20,
        "end_layer": 29,
        "temperature": 0,  # 使用greedy decoding，确保原始输出的token_id是logits最高的
        "top_p": None,
        "max_new_tokens": 512,  # CHAIR 需要详细描述
        "num_beams": 1,
        "num_samples": 500,  # 默认只处理1个图像（用于测试 attention map）
        "seed": 42,
        "extract_object_attention": True,  # 默认启用物体 attention map 提取
        "target_layers": [15, 17, 23, 29, 31]  # 默认只处理偶数层（减少输出）
    }

    # 解析参数(所有参数都有默认值)
    parser = argparse.ArgumentParser(description="CHAIR 评估 - 生成图像描述(所有参数可选)")

    # 数据集参数
    parser.add_argument("--coco-root", type=str, default=default_config["coco_root"],
                       help="COCO 数据集根目录路径(包含 val2014 子目录)")
    parser.add_argument("--image_id_list_file", type=str, default=default_config["image_id_list_file"],
                       help="图像 ID 列表文件, 支持两种格式: 1) JSON 数组格式(如 [\"COCO_val2014_000000001171.jpg\", ...]);2) 文本文件(每行一个 image_id 或图像文件名)。如果提供则只处理这些图像")
    parser.add_argument("--num-samples", type=int, default=default_config["num_samples"],
                       help="处理图像数量(0表示处理所有图像, 非零表示只处理前N个, 默认: 1)")
    parser.add_argument("--single-image-id", type=int, default= 33270,  # 13348,  # 6153,  # 6153
                       help="指定单个图像ID进行处理(如果指定, 将只处理该图像, 忽略其他参数)")

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
    parser.add_argument("--extract-object-attention", action="store_true",
                       default=default_config["extract_object_attention"],
                       help="启用物体 attention map 提取(默认: True)")
    parser.add_argument("--no-extract-object-attention", action="store_false",
                       dest="extract_object_attention",
                       help="禁用物体 attention map 提取")
    parser.add_argument("--target-layers", type=str, default=default_config["target_layers"],
                       help="目标层选择: 'even'(偶数层), 'odd'(奇数层), 'all'(所有层), 或逗号分隔的层索引(如 '20,22,24', 默认: 'even')")

    args = parser.parse_args()
    set_seed(args.seed)

    # 设置 auto_evaluate 参数(默认启用, 除非指定 --no-auto-evaluate)
    args.auto_evaluate = not args.no_auto_evaluate

    # 如果指定了单个图像ID, 只处理该图像
    if args.single_image_id is not None:
        args.num_samples = 1
        # 创建临时的图像ID列表文件内容
        args.image_id_list_file = None  # 清空原有列表
        print(f"📌 指定了单个图像ID: {args.single_image_id}, 将只处理该图像")

    # 如果只处理少量图像(<=3个), 自动启用debug模式
    if args.num_samples > 0 and args.num_samples <= 3:
        if not args.debug:
            print(f"💡 检测到只处理 {args.num_samples} 个图像, 自动启用详细输出模式")
            args.debug = True

    # 自动生成输出文件路径(如果未指定)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "results", "chair")
    os.makedirs(output_dir, exist_ok=True)

    # 如果只处理一个图像, 提示用户这是测试模式
    if args.num_samples == 1 or args.single_image_id is not None:
        print("\n" + "=" * 80)
        print("🧪 测试模式: 只处理单个图像")
        print("=" * 80)
        print("💡 提示: 此模式用于测试和调试, 会生成详细的 attention map")
        if args.use_deco:
            print("   ⚠️  注意: 使用 Deco 时会同时运行 Vanilla 和 Deco 两个版本进行对比")
            print("       (每个版本都会处理同一个图像, 总共会处理 2 次)")
        print("   如需批量处理, 请使用 --num-samples 参数指定更多图像")
        print("=" * 80 + "\n")

    # 如果使用 Deco, 需要同时运行 vanilla 版本进行对比
    vanilla_output_file = None
    vanilla_results = None

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
