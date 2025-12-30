"""
LLaVA v1.5 7B 模型测试脚本

功能说明：
- 加载预训练的 LLaVA v1.5 7B 多模态大语言模型
- 对输入图像进行详细描述生成
- 支持 Deco 早退机制（Deco Early Exit）来加速推理
- 同时输出原生 LLaVA 和 Deco 方法的结果
- 支持单张图片和批量图片处理
- 将结果保存到 JSON 文件

Deco 参数说明：
- use_deco: 是否启用 Deco 早退机制
- alpha: Deco 置信度阈值参数
- threshold_top_p: 早退判断的 top-p 阈值
- threshold_top_k: 早退判断的 top-k 阈值
- early_exit_layers: 允许早退的层索引列表

重要说明 - 如何判断结果是否使用了 Deco：
1. Deco 策略仅在贪婪生成模式下生效（num_beams=1 且 do_sample=False）
2. 如果 use_deco=True 且满足贪婪模式条件，则使用 deco_greedy_search 方法
3. Deco 策略会结合早退层的 logits 和最后一层的 logits 来生成结果
4. 如果 use_deco=False 或不满足贪婪模式，则使用原生的 greedy_search 方法
5. 脚本会自动检测并显示实际使用的生成策略
"""

import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import time
from datetime import datetime
from typing import List, Dict, Optional, Union

# 添加项目根目录到 Python 路径，以便导入 llava 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # tests 的父目录就是项目根目录
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import requests
from io import BytesIO


def load_image(image_file):
    """加载图像文件，支持本地文件和 URL"""
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def parse_args():
    """解析命令行参数"""
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    parser = argparse.ArgumentParser(description="LLaVA v1.5 7B 模型测试脚本")

    # 模型相关参数
    parser.add_argument("--model-path", type=str,
                       default="/home/liying/Documents/llava-v1.5-7b",
                       help="模型路径")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备 (cuda/cpu)")
    parser.add_argument("--conv-mode", type=str, default="llava_v1",
                       help="对话模式")

    # 输入相关参数
    # 默认图片路径（原先的路径）
    # img_default_path = os.path.join(project_root, "Qwen2.5-VL-7B-Instruct_Based/tests/input_image.png")
    # prompt_default_path = "Please describe this image in detail."
    img_default_path = "/home/liying/Documents/dataset/coco/val2014/COCO_val2014_000000065883.jpg"
    prompt_default_path =  "There is a bowl"
    parser.add_argument("--image-file", type=str, default=img_default_path,
                       help=f"单张图片路径（默认: {img_default_path})")
    parser.add_argument("--prompt", type=str, default=None,
                       help="单张图片的提示词（如果不指定，使用 --default-prompt)")
    parser.add_argument("--batch-file", type=str, default=None,
                       help="批量处理 JSON 文件路径，格式: [{\"image_path\": \"...\", \"prompt\": \"...\"}, ...]")
    parser.add_argument("--default-prompt", type=str,
                       default=prompt_default_path,
                       help="默认提示词（当批量文件中 prompt 为空时，或单张图片未指定 --prompt 时使用）")

    # 生成参数
    parser.add_argument("--temperature", type=float, default=-1,
                       help="生成温度 (-1 表示不使用采样)")
    parser.add_argument("--top-p", type=float, default=None,
                       help="Top-p 采样参数")
    parser.add_argument("--num-beams", type=int, default=1,
                       help="Beam search 数量")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                       help="最大生成 token 数")

    # Deco 参数
    parser.add_argument("--use-deco", action="store_true", default=True,
                       help="是否启用 Deco 早退机制")
    parser.add_argument("--no-deco", dest="use_deco", action="store_false",
                       help="禁用 Deco 早退机制")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Deco 置信度阈值参数")
    parser.add_argument("--threshold-top-p", type=float, default=0.9,
                       help="早退判断的 top-p 阈值")
    parser.add_argument("--threshold-top-k", type=int, default=20,
                       help="早退判断的 top-k 阈值")
    parser.add_argument("--early-exit-layers", type=str, default="20,21,22,23,24,25,26,27,28",
                       help="允许早退的层索引列表，用逗号分隔 (例如: 20,21,22)")

    # 输出相关参数
    parser.add_argument("--output-dir", type=str, default=None,
                       help="输出目录（默认为 tests/output）")

    return parser.parse_args()


def parse_early_exit_layers(layers_str: str) -> List[int]:
    """解析早退层字符串为整数列表"""
    return [int(x.strip()) for x in layers_str.split(",") if x.strip()]


def load_batch_data(batch_file: str) -> List[Dict]:
    """加载批量处理数据"""
    with open(batch_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("批量文件必须是 JSON 数组格式")

    return data


def prepare_inputs(model, tokenizer, image_processor, image_file: str, prompt: str,
                   conv_mode: str, device: str, verbose: bool = True):
    """准备模型输入"""
    # 检查图像文件是否存在
    if not os.path.exists(image_file):
        raise FileNotFoundError(f"图像文件不存在: {image_file}")

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

    return input_ids, image_tensor, stopping_criteria, image.size


def generate_response(model, tokenizer, input_ids, image_tensor, stopping_criteria,
                     temperature, top_p, num_beams, max_new_tokens, device,
                     use_deco=False, alpha=None, threshold_top_p=None,
                     threshold_top_k=None, early_exit_layers=None, verbose: bool = True):
    """生成回答"""
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

    # LLaVA 的 generate 方法使用 inputs 而不是 input_ids
    generate_kwargs = {
        "inputs": input_ids,  # 注意：LLaVA 使用 inputs 参数名
        "images": image_tensor.unsqueeze(0).half().to(device),
        "do_sample": do_sample,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p,
        "num_beams": num_beams,
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
            output_dict = model.generate(**generate_kwargs)

    # 解码输出
    if verbose:
        print(f"\n  [生成过程] 检查 output_dict 结构:")
        print(f"    - output_dict 类型: {type(output_dict)}")
        print(f"    - output_dict 属性: {[attr for attr in dir(output_dict) if not attr.startswith('_')]}")
        if hasattr(output_dict, 'sequences'):
            print(f"    - output_dict.sequences 形状: {output_dict.sequences.shape}")
        if hasattr(output_dict, 'scores'):
            print(f"    - output_dict.scores 长度: {len(output_dict.scores) if output_dict.scores else 0}")

    output_ids = output_dict.sequences
    input_token_len = input_ids.shape[1]

    # 检查 output_ids 是否包含 input_ids
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

    # 如果 output_ids 不包含 input_ids，手动拼接
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

        # 安全解码函数（处理无效 token ID）
        def safe_decode(tokenizer, token_ids, skip_special_tokens=False):
            """安全解码，处理无效 token ID"""
            try:
                return tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)[0]
            except (IndexError, ValueError) as e:
                # 如果解码失败，尝试过滤无效的 token ID
                vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)
                valid_ids = token_ids[0].clone()
                # 过滤超出范围的 token ID
                if len(valid_ids.shape) == 1:
                    valid_mask = (valid_ids >= 0) & (valid_ids < vocab_size)
                    if valid_mask.all():
                        raise e  # 如果都在范围内但还是出错，重新抛出异常
                    valid_ids = valid_ids[valid_mask]
                    try:
                        return tokenizer.batch_decode(valid_ids.unsqueeze(0), skip_special_tokens=skip_special_tokens)[0] + f" [包含 {torch.sum(~valid_mask).item()} 个无效token]"
                    except:
                        return f"[解码失败: {str(e)}, token_ids={token_ids[0].tolist()}]"
                return f"[解码失败: {str(e)}]"

        # 输出完整的 output_ids 解码（所有 token）
        print(f"\n  [生成结果] 完整的 output_ids 解码:")
        print(f"    - output_ids 所有 token IDs: {output_ids[0].tolist()}")

        # 解码所有 token（不跳过特殊 token）
        try:
            all_tokens_decoded = safe_decode(tokenizer, output_ids, skip_special_tokens=False)
            print(f"    - output_ids 所有 token 解码（带特殊token）: {repr(all_tokens_decoded)}")
        except Exception as e:
            print(f"    - output_ids 解码失败: {e}")

        # 解码所有 token（跳过特殊 token）
        try:
            all_tokens_decoded_clean = safe_decode(tokenizer, output_ids, skip_special_tokens=True)
            print(f"    - output_ids 所有 token 解码（去特殊token）: {repr(all_tokens_decoded_clean)}")
        except Exception as e:
            print(f"    - output_ids 解码失败: {e}")

        # 对比 input_ids 和 output_ids 的前 input_token_len 部分
        if output_ids.shape[1] >= input_token_len:
            print(f"\n  [生成结果] input_ids vs output_ids 对比:")
            print(f"    - input_ids 前 {input_token_len} 个 token IDs: {input_ids[0].tolist()}")
            print(f"    - output_ids 前 {input_token_len} 个 token IDs: {output_ids[0, :input_token_len].tolist()}")

            # 解码对比（安全解码）
            try:
                input_decoded = safe_decode(tokenizer, input_ids, skip_special_tokens=False)
                print(f"    - input_ids 解码: {repr(input_decoded[:200])}...")
            except Exception as e:
                print(f"    - input_ids 解码失败: {e}")

            try:
                output_prefix_decoded = safe_decode(tokenizer, output_ids[:, :input_token_len], skip_special_tokens=False)
                print(f"    - output_ids 前 {input_token_len} 个 token 解码: {repr(output_prefix_decoded[:200])}...")
            except Exception as e:
                print(f"    - output_ids 前缀解码失败: {e}")

            n_diff = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff > 0:
                print(f"    - ⚠️  警告: {n_diff} 个 output_ids 与 input_ids 不一致")
                # 找出不一致的位置
                diff_mask = (input_ids[0] != output_ids[0, :input_token_len])
                diff_indices = torch.where(diff_mask)[0].tolist()
                print(f"    - 不一致的 token 位置: {diff_indices[:10]}")  # 只显示前10个
                for idx in diff_indices[:5]:  # 只显示前5个不一致的位置
                    print(f"      - 位置 {idx}: input={input_ids[0, idx].item()}, output={output_ids[0, idx].item()}")
            else:
                print(f"    - ✓ input_ids 和 output_ids 前 {input_token_len} 个 token 完全一致")

        # 显示新生成的 token IDs
        if output_token_len > 0:
            generated_ids = output_ids[:, input_token_len:]
            print(f"\n  [生成结果] 新生成的 token:")
            print(f"    - 生成的 token IDs 形状: {generated_ids.shape}")
            print(f"    - 生成的 token IDs: {generated_ids[0].tolist()}")
            try:
                generated_decoded = safe_decode(tokenizer, generated_ids, skip_special_tokens=False)
                print(f"    - 生成的 token 解码（带特殊token）: {repr(generated_decoded)}")
            except Exception as e:
                print(f"    - 生成的 token 解码失败: {e}")
        else:
            print(f"\n  [生成结果] ⚠️  警告: 没有生成新的 token！")
            print(f"    - output_ids 长度 ({output_ids.shape[1]}) <= input_ids 长度 ({input_token_len})")

    # 获取新生成的 token（跳过可能的 BOS token）
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


def process_single_image(args, model, tokenizer, image_processor,
                         image_file: str, prompt: str):
    """处理单张图片"""
    print(f"\n处理图片: {image_file}")
    print(f"提示词: {prompt}")

    # 准备输入
    print("\n" + "=" * 80)
    print("[准备输入]")
    print("=" * 80)
    input_ids, image_tensor, stopping_criteria, image_size = prepare_inputs(
        model, tokenizer, image_processor, image_file, prompt,
        args.conv_mode, args.device, verbose=True
    )

    results = {
        "image_path": image_file,
        "prompt": prompt,
        "image_size": image_size,
        "timestamp": datetime.now().isoformat(),
    }

    # 生成原生 LLaVA 结果
    print("\n" + "=" * 80)
    print("[原生 LLaVA] 正在生成...")
    print("=" * 80)
    native_start = time.time()
    native_output, native_tokens, input_tokens = generate_response(
        model, tokenizer, input_ids, image_tensor, stopping_criteria,
        args.temperature, args.top_p, args.num_beams, args.max_new_tokens,
        args.device, use_deco=False, verbose=True
    )
    native_time = time.time() - native_start

    results["native_llava"] = {
        "output": native_output,
        "input_tokens": int(input_tokens),
        "output_tokens": int(native_tokens),
        "total_tokens": int(input_tokens + native_tokens),
        "generation_time": round(native_time, 2),
        "tokens_per_second": round(native_tokens / native_time, 2) if native_time > 0 else 0,
    }
    print(f"✓ 原生 LLaVA 生成完成 (耗时: {native_time:.2f}秒, 输入: {input_tokens} tokens, 输出: {native_tokens} tokens, 总计: {input_tokens + native_tokens} tokens)")

    # 生成 Deco 结果
    if args.use_deco:
        print("\n" + "=" * 80)
        print("[Deco 策略] 正在生成...")
        print("=" * 80)
        deco_start = time.time()
        early_exit_layers = parse_early_exit_layers(args.early_exit_layers)
        deco_output, deco_tokens, _ = generate_response(
            model, tokenizer, input_ids, image_tensor, stopping_criteria,
            args.temperature, args.top_p, args.num_beams, args.max_new_tokens,
            args.device, use_deco=True, alpha=args.alpha,
            threshold_top_p=args.threshold_top_p, threshold_top_k=args.threshold_top_k,
            early_exit_layers=early_exit_layers, verbose=True
        )
        deco_time = time.time() - deco_start

        results["deco"] = {
            "output": deco_output,
            "input_tokens": int(input_tokens),
            "output_tokens": int(deco_tokens),
            "total_tokens": int(input_tokens + deco_tokens),
            "generation_time": round(deco_time, 2),
            "tokens_per_second": round(deco_tokens / deco_time, 2) if deco_time > 0 else 0,
            "alpha": args.alpha,
            "threshold_top_p": args.threshold_top_p,
            "threshold_top_k": args.threshold_top_k,
            "early_exit_layers": early_exit_layers,
        }
        print(f"✓ Deco 策略生成完成 (耗时: {deco_time:.2f}秒, 输入: {input_tokens} tokens, 输出: {deco_tokens} tokens, 总计: {input_tokens + deco_tokens} tokens)")

        # 计算加速比
        speedup = native_time / deco_time if deco_time > 0 else 0
        results["speedup"] = round(speedup, 2)
        print(f"  加速比: {speedup:.2f}x")
    else:
        results["deco"] = None
        print("\n[Deco 策略] 已禁用，跳过")

    return results


def main():
    args = parse_args()

    # 设置输出目录
    if args.output_dir is None:
        output_dir = os.path.join(current_dir, "output")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 打印配置信息
    print("=" * 80)
    print("LLaVA v1.5 7B 模型测试 - 配置信息")
    print("=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"设备: {args.device}")
    print(f"对话模式: {args.conv_mode}")
    print(f"生成参数: temperature={args.temperature}, top_p={args.top_p}, "
          f"num_beams={args.num_beams}, max_new_tokens={args.max_new_tokens}")
    print(f"\nDeco 早退配置:")
    print(f"  - 启用 Deco: {args.use_deco}")
    if args.use_deco:
        print(f"  - Alpha: {args.alpha}")
        print(f"  - Threshold Top-P: {args.threshold_top_p}")
        print(f"  - Threshold Top-K: {args.threshold_top_k}")
        print(f"  - 早退层: {args.early_exit_layers}")
    print(f"\n输出目录: {output_dir}")
    print("=" * 80)

    # 加载模型
    print("\n[1/4] 正在加载模型...")
    model_load_start = time.time()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=model_name,
        device=args.device,
        device_map=args.device
    )
    model_load_time = time.time() - model_load_start
    print(f"✓ 模型加载完成 (耗时: {model_load_time:.2f}秒)")
    print(f"  - 模型名称: {model_name}")
    print(f"  - 上下文长度: {context_len}")

    # 准备处理数据
    print("\n[2/4] 准备处理数据...")

    # 优先检查批量文件
    batch_file_to_use = None
    if args.batch_file:
        # 如果明确指定了批量文件，检查是否存在
        if os.path.exists(args.batch_file):
            batch_file_to_use = args.batch_file
            print(f"✓ 找到指定的批量文件: {args.batch_file}")
        else:
            print(f"⚠ 指定的批量文件不存在: {args.batch_file}，将使用单张图片模式")
    else:
        # 如果没有指定批量文件，检查默认位置是否有批量文件
        # 使用文件顶部定义的 current_dir 和 project_root
        default_batch_files = [
            os.path.join(current_dir, "batch.json"),
            os.path.join(current_dir, "batch_input.json"),
            os.path.join(project_root, "batch.json"),
        ]
        for batch_file in default_batch_files:
            if os.path.exists(batch_file):
                batch_file_to_use = batch_file
                print(f"✓ 自动发现批量文件: {batch_file}")
                break

    if batch_file_to_use:
        # 批量处理模式
        print(f"批量处理模式: {batch_file_to_use}")
        batch_data = load_batch_data(batch_file_to_use)
        print(f"找到 {len(batch_data)} 个任务")

        all_results = []
        for i, item in enumerate(tqdm(batch_data, desc="处理进度")):
            image_path = item.get("image_path", "")
            prompt = item.get("prompt", "") or args.default_prompt

            if not image_path:
                print(f"⚠ 跳过第 {i+1} 个任务: 缺少 image_path")
                continue

            try:
                result = process_single_image(args, model, tokenizer, image_processor,
                                            image_path, prompt)
                all_results.append(result)
            except Exception as e:
                print(f"⚠ 处理第 {i+1} 个任务时出错: {str(e)}")
                all_results.append({
                    "image_path": image_path,
                    "prompt": prompt,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })

        # 保存批量结果
        output_file = os.path.join(output_dir, f"batch_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 批量处理完成，结果已保存到: {output_file}")

    else:
        # 单张图片模式
        # 确定使用的图片路径和提示词
        image_file_to_use = args.image_file
        prompt_to_use = args.prompt if args.prompt is not None else args.default_prompt

        print(f"单张图片模式")
        print(f"  - 图片路径: {image_file_to_use}")
        print(f"  - 提示词: {prompt_to_use}")

        if not os.path.exists(image_file_to_use):
            print(f"错误: 图片文件不存在: {image_file_to_use}")
            return

        result = process_single_image(args, model, tokenizer, image_processor,
                                     image_file_to_use, prompt_to_use)

        # 保存单张结果
        output_file = os.path.join(output_dir, f"single_result.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 处理完成，结果已保存到: {output_file}")

        # 打印结果摘要
        print("\n" + "=" * 80)
        print("结果摘要")
        print("=" * 80)
        print(f"原生 LLaVA:")
        print(f"  - 输出: {result['native_llava']['output'][:100]}...")
        print(f"  - 生成时间: {result['native_llava']['generation_time']}秒")
        print(f"  - 速度: {result['native_llava']['tokens_per_second']} tokens/秒")
        if result.get('deco'):
            print(f"\nDeco 策略:")
            print(f"  - 输出: {result['deco']['output'][:100]}...")
            print(f"  - 生成时间: {result['deco']['generation_time']}秒")
            print(f"  - 速度: {result['deco']['tokens_per_second']} tokens/秒")
            if 'speedup' in result:
                print(f"  - 加速比: {result['speedup']}x")
        print("=" * 80)

    print("\n测试完成！")


if __name__ == "__main__":
    main()
