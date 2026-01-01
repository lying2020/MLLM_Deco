#!/usr/bin/env python3
"""
CHAIR 评估脚本 - 生成图像描述并保存为 JSONL 格式
参考 run_pope_eval.py 的实现，针对 CHAIR benchmark 优化
自动检测数据集和模型，使用默认参数，无需输入参数即可运行

CHAIR 评估需要：
1. COCO 2014 val2014 图像目录
2. 生成的描述文件(JSONL 格式)：{"image_id": int, "caption": str}
3. COCO annotations 目录(用于后续的 chair.py 评估)

使用步骤：
1. 运行此脚本生成描述文件：
   python run_chair_eval.py --coco-root /path/to/coco --output-file results/chair/captions.jsonl

2. 使用 chair.py 计算 CHAIR 指标：
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
from chair import evaluate_chair


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


def get_coco_val2014_images(coco_root: str, image_id_list: Optional[List[int]] = None, max_images: int = 0):
    """
    获取 COCO val2014 图像列表

    Args:
        coco_root: COCO 数据集根目录(包含 val2014 子目录)
        image_id_list: 可选的图像 ID 列表，如果提供则只返回这些图像
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
        # 如果提供了图像 ID 列表，只处理这些图像
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
            try:
                image_id = int(filename.split("_")[-1])
                images.append({
                    "image_id": image_id,
                    "image_path": str(image_file)
                })
            except ValueError:
                print(f"⚠️  警告: 无法从文件名提取 image_id: {image_file}")
                continue

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

    # 如果 output_ids 不包含 input_ids，手动拼接
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
        # 如果新生成的 token 以 BOS token 开头，跳过它
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

    return outputs, output_token_len, input_token_len


def eval_model(args):
    """评估模型，生成图像描述"""
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

    # 如果提供了图像 ID 列表文件，读取它
    image_id_list = None
    if args.image_id_list_file:
        with open(args.image_id_list_file, 'r') as f:
            image_id_list = [int(line.strip()) for line in f if line.strip()]
        print(f"✓ 从文件读取了 {len(image_id_list)} 个图像 ID")

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
    prompt = "Please describe this image in detail."

    # 计算需要输出详细信息的样本索引(最多10个，均匀分布)
    total_samples = len(images)
    max_debug_samples = min(10, total_samples)
    if total_samples > 0:
        debug_indices = set()
        if total_samples <= max_debug_samples:
            debug_indices = set(range(total_samples))
        else:
            step = total_samples / max_debug_samples
            for i in range(max_debug_samples):
                idx = int(i * step)
                debug_indices.add(idx)

    for sample_idx, image_info in enumerate(tqdm(images, desc="处理进度")):
        image_id = image_info["image_id"]
        image_file = image_info["image_path"]

        # 判断是否需要输出详细信息
        verbose = sample_idx in debug_indices

        if verbose:
            print("\n" + "=" * 80)
            print(f"[样本 {sample_idx + 1}/{total_samples}] Image ID: {image_id}")
            print("=" * 80)
            print(f"图像: {image_file}")

        try:
            # 准备输入
            input_ids, image_tensor, stopping_criteria, stop_str = prepare_inputs(
                model, tokenizer, image_processor, image_file, prompt, conv_mode, device, verbose=verbose
            )

            # 生成回答
            outputs, output_token_len, input_token_len = generate_response(
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

            # 如果输出为空，记录警告
            if not outputs:
                if verbose:
                    print(f"\n  [Warning] 图像 {image_id} 生成结果为空，output_token_len={output_token_len}")
                else:
                    print(f"  [Warning] 图像 {image_id} 生成结果为空，output_token_len={output_token_len}")

            if verbose:
                print(f"\n  [生成结果] 描述:")
                print(f"    - 输出长度: {len(outputs)} 字符")
                print(f"    - 描述预览: {outputs[:200]}...")
                print("=" * 80)

            # 保存结果(CHAIR 格式：image_id 和 caption)
            result = {
                "image_id": image_id,
                "caption": outputs
            }
            output_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_f.flush()

        except Exception as e:
            print(f"\n[Error] 处理图像 {image_id} 时出错: {e}")
            import traceback
            traceback.print_exc()
            # 保存错误信息(caption 为空)
            result = {
                "image_id": image_id,
                "caption": ""
            }
            output_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_f.flush()
            continue

    output_f.close()
    print(f"\n✓ 描述生成完成！结果已保存到: {output_file}")

    # 自动计算 CHAIR 指标（默认启用）
    auto_evaluate = getattr(args, 'auto_evaluate', True)  # 默认为 True
    if auto_evaluate:
        try:
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

            # 生成结果文件路径
            results_dir = os.path.dirname(output_file)
            chair_results_file = output_file.replace('.jsonl', '_chair_results.json')

            # 调用 evaluate_chair 函数
            results = evaluate_chair(
                cap_file=output_file,
                coco_path=coco_annotations_path,
                image_id_key="image_id",
                caption_key="caption",
                cache_file=os.path.join(results_dir, "chair_evaluator.pkl"),
                use_cache=True,
                save_path=chair_results_file,
                verbose=True
            )

            print("\n" + "=" * 80)
            print("✓ CHAIR 指标计算完成！")
            print("=" * 80)
            print(f"详细结果文件: {chair_results_file}")
            print(f"输出文件: {output_file}")

        except Exception as e:
            print(f"\n⚠️  自动计算 CHAIR 指标时出错: {e}")
            import traceback
            traceback.print_exc()
            print("\n可以手动运行以下命令计算 CHAIR 指标：")
            print(f"  python chair.py --cap_file {output_file} --image_id_key image_id --caption_key caption \\")
            print(f"                  --coco_path {args.coco_root}/annotations_trainval2014/annotations/")
    else:
        print(f"\n下一步：使用 chair.py 计算 CHAIR 指标")
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
        print("⚠ 未检测到 CUDA，将使用 CPU(速度较慢)")

    # 默认配置
    default_config = {
        "model_path": project.llava_v15_7b_path,
        "device": device,
        "coco_root": project.coco_data_path,  # 需要根据实际情况修改
        "use_deco": False,
        "alpha": 0.6,
        "threshold_top_p": 0.9,
        "threshold_top_k": 20,
        "start_layer": 20,
        "end_layer": 29,
        "temperature": -1,
        "top_p": None,
        "max_new_tokens": 512,  # CHAIR 需要详细描述
        "num_beams": 1,
        "num_samples": 40,  # 0 表示处理所有图像
        "seed": 42
    }

    # 解析参数(所有参数都有默认值)
    parser = argparse.ArgumentParser(description="CHAIR 评估 - 生成图像描述(所有参数可选)")

    # 数据集参数
    parser.add_argument("--coco-root", type=str, default=default_config["coco_root"],
                       help="COCO 数据集根目录路径(包含 val2014 子目录)")
    parser.add_argument("--image-id-list-file", type=str, default=None,
                       help="图像 ID 列表文件(每行一个 image_id)，如果提供则只处理这些图像")
    parser.add_argument("--num-samples", type=int, default=default_config["num_samples"],
                       help="处理图像数量(0表示处理所有图像，非零表示只处理前N个)")

    # 模型参数
    parser.add_argument("--model-path", type=str, default=default_config["model_path"],
                       help="模型路径")
    parser.add_argument("--model-base", type=str, default=None, help="基础模型路径")
    parser.add_argument("--device", type=str, default=default_config["device"],
                       help="设备 (cuda:0/cpu)")

    # 输出参数
    parser.add_argument("--output-file", type=str, default=None,
                       help="输出描述文件路径(JSONL 格式，如果不指定，将自动生成)")

    # 生成参数
    parser.add_argument("--temperature", type=float, default=default_config["temperature"],
                       help="生成温度(-1表示贪婪生成)")
    parser.add_argument("--top-p", type=float, default=default_config["top_p"], help="Top-p采样")
    parser.add_argument("--max-new-tokens", type=int, default=default_config["max_new_tokens"],
                       help="最大生成 token 数")
    parser.add_argument("--num-beams", type=int, default=default_config["num_beams"],
                       help="Beam search 的 beam 数量")

    # Deco 参数(默认不使用 Deco，只使用原生 LLaVA 模型)
    parser.add_argument("--use-deco", action="store_true", default=default_config["use_deco"],
                       help="启用 Deco 早退机制(默认：False，使用原生 LLaVA 模型)")
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
                       help="禁用自动计算 CHAIR 指标（默认会自动计算）")

    args = parser.parse_args()
    set_seed(args.seed)

    # 自动生成输出文件路径(如果未指定)
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(project_root, "results", "chair")
        os.makedirs(output_dir, exist_ok=True)
        args.output_file = os.path.join(output_dir, f"chair_captions_{timestamp}.jsonl")

    # 设置 auto_evaluate 参数（默认启用，除非指定 --no-auto-evaluate）
    args.auto_evaluate = not args.no_auto_evaluate

    # 运行评估
    eval_model(args)

    print("\n" + "=" * 80)
    print("✓ 所有评估完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
