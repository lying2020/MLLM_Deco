#!/usr/bin/env python3
"""
构建 COCO 训练数据集脚本
从 COCO val2014 中选择 2000 张图片（排除 pope_coco/coco_baseline_500.json 中的图片）
为每张图片生成：
- 6 个 POPE 格式的测试 case（使用 adversarial, popular, random 方式）
- 1 个 CHAIR 格式的测试 case
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Set, Optional
from collections import defaultdict
import argparse
import torch
import warnings
from PIL import Image

warnings.filterwarnings('ignore')

# 导入项目配置
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) != 'MLLM_Deco' else current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import project as project

# 导入LLaVA相关模块（用于生成caption）
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

current_dir = os.path.dirname(os.path.abspath(__file__))
coco_train_json_dir = os.path.join(current_dir, "coco_train_json")
os.makedirs(coco_train_json_dir, exist_ok=True)

# COCO 80 个类别名称（用于生成 POPE 问题）
COCO_CATEGORIES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def load_excluded_images(exclude_file: str) -> Set[str]:
    """
    加载需要排除的图片列表

    Args:
        exclude_file: JSON 文件路径，包含图片文件名列表

    Returns:
        Set[str]: 图片文件名集合（不包含路径，只包含文件名）
    """
    exclude_file = os.path.expanduser(exclude_file)
    if not os.path.exists(exclude_file):
        raise FileNotFoundError(f"排除文件不存在: {exclude_file}")

    with open(exclude_file, 'r', encoding='utf-8') as f:
        image_list = json.load(f)

    # 提取文件名（去掉路径）
    excluded = set()
    for img in image_list:
        if isinstance(img, str):
            # 如果包含路径，只取文件名
            if '/' in img:
                img = img.split('/')[-1]
            excluded.add(img)

    return excluded


def get_coco_val2014_images(coco_root: str, exclude_images: Set[str],
                            imid_to_objects: Dict[int, List[str]],
                            num_images: int = 2000,
                            prioritize_by_hallucination: bool = False,
                            hallucination_scores: Optional[Dict[int, int]] = None,
                            model=None, tokenizer=None, image_processor=None,
                            conv_mode: str = None, device: str = None,
                            chair_evaluator=None) -> List[Dict]:
    """
    从 COCO val2014 目录中获取指定数量的图片（排除已使用的图片）
    优先选择策略：
    1. 按实例数量：3个实例 -> 2个实例 -> 4个实例 -> 1个实例
    2. 如果启用幻视优化：在每个实例数量组内，优先选择有2个幻视、3个幻视或1个幻视的图片

    Args:
        coco_root: COCO 数据集根目录
        exclude_images: 需要排除的图片文件名集合
        imid_to_objects: image_id 到对象列表的映射
        num_images: 需要获取的图片数量
        prioritize_by_hallucination: 是否根据幻视数量进行优化
        hallucination_scores: image_id 到幻视数量的映射（如果启用幻视优化，必须提供）
        model: LLaVA模型（用于生成caption，如果启用幻视优化）
        tokenizer: tokenizer（用于生成caption）
        image_processor: 图像处理器（用于生成caption）
        conv_mode: 对话模式（用于生成caption）
        device: 设备（用于生成caption）
        chair_evaluator: CHAIR评估器（用于计算幻视数量）

    Returns:
        List[Dict]: 包含 image_id 和 image_filename 的字典列表
    """
    coco_root = Path(coco_root)
    val2014_dir = coco_root / "val2014"

    if not val2014_dir.exists():
        raise FileNotFoundError(f"COCO val2014 目录不存在: {val2014_dir}")

    # 根据实例数量分类图片
    # 如果启用幻视优化，还需要在每个组内按幻视数量分类
    images_with_3_instances = []  # 3个实例（最高优先级）
    images_with_2_instances = []  # 2个实例（第二优先级）
    images_with_4_instances = []  # 4个实例（第三优先级）
    images_with_1_instance = []  # 1个实例（第四优先级）
    images_with_0_instances = []  # 0个实例（排除）
    images_with_more_instances = []  # 5个及以上实例（最后选择）

    image_files = sorted(val2014_dir.glob("COCO_val2014_*.jpg"))

    # 如果启用幻视优化，按照新逻辑：先按实例数量排序，然后逐张生成caption并筛选
    selected_images = []
    if prioritize_by_hallucination and model is not None and chair_evaluator is not None:
        print(f"  正在按实例数量排序候选图片，然后逐张生成caption并筛选...")

        # 先收集所有候选图片并按实例数量排序
        candidate_images = []
        for image_file in image_files:
            filename = image_file.name
            if filename not in exclude_images:
                image_id = int(filename.split("_")[-1].replace(".jpg", ""))
                objects = imid_to_objects.get(image_id, [])
                num_instances = len(objects)
                if num_instances > 0:  # 只处理有对象的图片
                    candidate_images.append({
                        "image_id": image_id,
                        "image_filename": filename,
                        "objects": objects,
                        "num_instances": num_instances
                    })

        # 按实例数量排序：3个实例 -> 2个实例 -> 4个实例 -> 1个实例
        def sort_key(img):
            num_inst = img['num_instances']
            if num_inst == 3:
                return 0
            elif num_inst == 2:
                return 1
            elif num_inst == 4:
                return 2
            elif num_inst == 1:
                return 3
            else:
                return 4

        candidate_images.sort(key=sort_key)

        # 定义候选图片数量（通常是目标数量的10倍，但不超过总候选图片数）
        max_candidate_count = min(num_images * 10, len(candidate_images))
        print(f"  已收集 {len(candidate_images)} 张候选图片，将处理前 {max_candidate_count} 张")
        print(f"  开始逐张生成caption并实时校验...")
        from tqdm import tqdm

        # 定义筛选优先级函数
        def check_priority_1(num_grounded, num_hallucinated):
            """优先级1：3个实例词汇 + 2个幻视词汇"""
            return num_grounded == 3 and num_hallucinated == 2

        def check_priority_2(num_grounded, num_hallucinated):
            """优先级2：4个实例词汇 + 2个或3个幻视词汇"""
            return num_grounded == 4 and num_hallucinated in [2, 3]

        def check_priority_3(num_grounded, num_hallucinated):
            """优先级3：2个实例词汇 + 1个幻视词汇"""
            return num_grounded == 2 and num_hallucinated == 1

        def check_priority_4(num_grounded, num_hallucinated):
            """优先级4：1-4个实例词汇 + 1-4个幻视词汇（且实例词汇数量 >= 幻视词汇数量）"""
            return (1 <= num_grounded <= 4 and
                    1 <= num_hallucinated <= 4 and
                    num_grounded >= num_hallucinated)

        # 存储所有处理过的图片信息（用于按优先级筛选）
        processed_images = []  # 存储所有处理过的图片信息
        selected_ids = set()  # 已选中的图片ID集合
        priority_stats = {
            1: 0, 2: 0, 3: 0, 4: 0
        }

        # 第一步：逐张生成caption并实时校验优先级1
        print(f"\n  第一步：逐张生成caption并实时校验优先级1...")
        for img_info in tqdm(candidate_images[:max_candidate_count], desc="生成caption并校验"):
            image_id = img_info['image_id']
            image_filename = img_info['image_filename']
            gt_objects = img_info['objects']
            image_path = os.path.join(str(coco_root), "val2014", image_filename)

            try:
                # 生成caption
                caption = generate_caption(
                    model, tokenizer, image_processor, image_path,
                    conv_mode, device
                )

                # 提取实例词汇和幻视词汇
                grounded_words, hallucinated_words, num_grounded, num_hallucinated = extract_physical_words(
                    caption, gt_objects, chair_evaluator
                )

                # 保存处理结果（无论是否满足条件，都保存下来用于后续优先级筛选）
                proc_img_data = {
                    "img_info": img_info,
                    "num_grounded": num_grounded,
                    "num_hallucinated": num_hallucinated,
                    "grounded_words": grounded_words,
                    "hallucinated_words": hallucinated_words
                }
                processed_images.append(proc_img_data)

                # 实时校验优先级1：如果满足条件且还没选够，立即选中
                if len(selected_images) < num_images:
                    if image_id not in selected_ids:
                        if check_priority_1(num_grounded, num_hallucinated):
                            selected_images.append({
                                "image_id": image_id,
                                "image_filename": image_filename,
                                "num_grounded": num_grounded,
                                "num_hallucinated": num_hallucinated,
                                "grounded_words": list(grounded_words),
                                "hallucinated_words": list(hallucinated_words)
                            })
                            selected_ids.add(image_id)
                            priority_stats[1] += 1

            except Exception as e:
                print(f"  ⚠️  处理图片 {image_filename} 时出错: {e}")
                continue

        print(f"  ✓ 已处理 {len(processed_images)} 张图片的caption")
        print(f"    ✓ 优先级1实时筛选出 {priority_stats[1]} 张图片（当前总计: {len(selected_images)} 张）")

        # 第二步：如果优先级1不够，从已处理的图片中筛选优先级2
        if len(selected_images) < num_images:
            print(f"\n  第二步：从已处理的图片中筛选优先级2...")
            print(f"  优先级2：4个实例词汇 + 2个或3个幻视词汇")
            for proc_img in processed_images:
                if len(selected_images) >= num_images:
                    break
                image_id = proc_img["img_info"]["image_id"]
                if image_id in selected_ids:
                    continue
                if check_priority_2(proc_img["num_grounded"], proc_img["num_hallucinated"]):
                    selected_images.append({
                        "image_id": image_id,
                        "image_filename": proc_img["img_info"]["image_filename"],
                        "num_grounded": proc_img["num_grounded"],
                        "num_hallucinated": proc_img["num_hallucinated"],
                        "grounded_words": list(proc_img["grounded_words"]),
                        "hallucinated_words": list(proc_img["hallucinated_words"])
                    })
                    selected_ids.add(image_id)
                    priority_stats[2] += 1
            print(f"    ✓ 优先级2筛选出 {priority_stats[2]} 张图片（当前总计: {len(selected_images)} 张）")

        # 第三步：如果还不够，筛选优先级3
        if len(selected_images) < num_images:
            print(f"\n  第三步：从已处理的图片中筛选优先级3...")
            print(f"  优先级3：2个实例词汇 + 1个幻视词汇")
            for proc_img in processed_images:
                if len(selected_images) >= num_images:
                    break
                image_id = proc_img["img_info"]["image_id"]
                if image_id in selected_ids:
                    continue
                if check_priority_3(proc_img["num_grounded"], proc_img["num_hallucinated"]):
                    selected_images.append({
                        "image_id": image_id,
                        "image_filename": proc_img["img_info"]["image_filename"],
                        "num_grounded": proc_img["num_grounded"],
                        "num_hallucinated": proc_img["num_hallucinated"],
                        "grounded_words": list(proc_img["grounded_words"]),
                        "hallucinated_words": list(proc_img["hallucinated_words"])
                    })
                    selected_ids.add(image_id)
                    priority_stats[3] += 1
            print(f"    ✓ 优先级3筛选出 {priority_stats[3]} 张图片（当前总计: {len(selected_images)} 张）")

        # 第四步：如果还不够，筛选优先级4
        if len(selected_images) < num_images:
            print(f"\n  第四步：从已处理的图片中筛选优先级4...")
            print(f"  优先级4：1-4个实例词汇 + 1-4个幻视词汇（且实例词汇数量 >= 幻视词汇数量）")
            for proc_img in processed_images:
                if len(selected_images) >= num_images:
                    break
                image_id = proc_img["img_info"]["image_id"]
                if image_id in selected_ids:
                    continue
                if check_priority_4(proc_img["num_grounded"], proc_img["num_hallucinated"]):
                    selected_images.append({
                        "image_id": image_id,
                        "image_filename": proc_img["img_info"]["image_filename"],
                        "num_grounded": proc_img["num_grounded"],
                        "num_hallucinated": proc_img["num_hallucinated"],
                        "grounded_words": list(proc_img["grounded_words"]),
                        "hallucinated_words": list(proc_img["hallucinated_words"])
                    })
                    selected_ids.add(image_id)
                    priority_stats[4] += 1
            print(f"    ✓ 优先级4筛选出 {priority_stats[4]} 张图片（当前总计: {len(selected_images)} 张）")

        # 打印最终筛选统计信息
        print(f"\n  ✓ 筛选完成，共筛选出 {len(selected_images)} 张图片")
        print(f"  筛选统计（按优先级）:")
        for priority in sorted(priority_stats.keys()):
            print(f"    - 优先级{priority}: {priority_stats[priority]} 张图片")

        # 如果选出的图片不够，输出警告并结束
        if len(selected_images) < num_images:
            remaining_needed = num_images - len(selected_images)
            print(f"\n  ⚠️  警告: 筛选出的图片数量不足！")
            print(f"     需求数量: {num_images} 张")
            print(f"     实际筛选: {len(selected_images)} 张")
            print(f"     缺少数量: {remaining_needed} 张")
            print(f"     已结束筛选，不再补充图片")

        # 按 image_id 排序
        selected_images.sort(key=lambda x: x['image_id'])
        return selected_images

    # 否则使用原来的逻辑（不启用幻视优化）
    for image_file in image_files:
        filename = image_file.name
        # 如果不在排除列表中，添加到候选列表
        if filename not in exclude_images:
            # 从文件名提取 image_id
            # 格式: COCO_val2014_000000123456.jpg
            image_id = int(filename.split("_")[-1].replace(".jpg", ""))

            # 获取该图片的对象列表
            objects = imid_to_objects.get(image_id, [])
            num_instances = len(objects)

            img_info = {
                "image_id": image_id,
                "image_filename": filename,
                "num_instances": num_instances
            }

            if num_instances == 0:
                images_with_0_instances.append(img_info)
            elif num_instances == 1:
                images_with_1_instance.append(img_info)
            elif num_instances == 2:
                images_with_2_instances.append(img_info)
            elif num_instances == 3:
                images_with_3_instances.append(img_info)
            elif num_instances == 4:
                images_with_4_instances.append(img_info)
            else:
                images_with_more_instances.append(img_info)

    print(f"  图片分类统计:")
    print(f"    - 3个实例（最高优先级）: {len(images_with_3_instances)} 张")
    print(f"    - 2个实例（第二优先级）: {len(images_with_2_instances)} 张")
    print(f"    - 4个实例（第三优先级）: {len(images_with_4_instances)} 张")
    print(f"    - 1个实例（第四优先级）: {len(images_with_1_instance)} 张")
    print(f"    - 5个及以上实例（最后选择）: {len(images_with_more_instances)} 张")
    print(f"    - 0个实例（将被排除）: {len(images_with_0_instances)} 张")

    # 按优先级选择图片：3个实例 -> 2个实例 -> 4个实例 -> 1个实例 -> 5个及以上实例
    # 如果启用幻视优化，在每个实例数量组内，优先选择：2个幻视 -> 3个幻视 -> 1个幻视 -> 其他
    selected_images = []
    remaining = num_images

    def select_from_group(groups, group_names, remaining_count):
        """从多个组中选择图片，按优先级顺序"""
        selected = []
        for group, name in zip(groups, group_names):
            if remaining_count <= 0:
                break
            if len(group) > 0:
                num_to_select = min(remaining_count, len(group))
                selected_from_group = random.sample(group, num_to_select)
                selected.extend(selected_from_group)
                print(f"  ✓ 从{name}中选择了 {num_to_select} 张")
                remaining_count -= num_to_select
        return selected, remaining_count

    # 1. 优先选择3个实例的图片
    if remaining > 0 and len(images_with_3_instances) > 0:
        num_from_3 = min(remaining, len(images_with_3_instances))
        selected_from_3 = random.sample(images_with_3_instances, num_from_3)
        selected_images.extend(selected_from_3)
        print(f"  ✓ 从3个实例的图片中选择了 {num_from_3} 张")
        remaining -= num_from_3

    # 2. 如果不够，从2个实例的图片中选择
    if remaining > 0 and len(images_with_2_instances) > 0:
        num_from_2 = min(remaining, len(images_with_2_instances))
        selected_from_2 = random.sample(images_with_2_instances, num_from_2)
        selected_images.extend(selected_from_2)
        print(f"  ✓ 从2个实例的图片中选择了 {num_from_2} 张")
        remaining -= num_from_2

    # 3. 如果还不够，从4个实例的图片中选择
    if remaining > 0 and len(images_with_4_instances) > 0:
        num_from_4 = min(remaining, len(images_with_4_instances))
        selected_from_4 = random.sample(images_with_4_instances, num_from_4)
        selected_images.extend(selected_from_4)
        print(f"  ✓ 从4个实例的图片中选择了 {num_from_4} 张")
        remaining -= num_from_4

    # 4. 如果还不够，从1个实例的图片中选择
    if remaining > 0 and len(images_with_1_instance) > 0:
        num_from_1 = min(remaining, len(images_with_1_instance))
        selected_from_1 = random.sample(images_with_1_instance, num_from_1)
        selected_images.extend(selected_from_1)
        print(f"  ✓ 从1个实例的图片中选择了 {num_from_1} 张")
        remaining -= num_from_1

    # 5. 如果还不够，从5个及以上实例的图片中选择
    if remaining > 0 and len(images_with_more_instances) > 0:
        num_from_more = min(remaining, len(images_with_more_instances))
        selected_from_more = random.sample(images_with_more_instances, num_from_more)
        selected_images.extend(selected_from_more)
        print(f"  ✓ 从5个及以上实例的图片中选择了 {num_from_more} 张")
        remaining -= num_from_more

    if remaining > 0:
        print(f"  ⚠️  警告: 无法选择足够的图片，还缺少 {remaining} 张")

    # 按 image_id 排序
    selected_images.sort(key=lambda x: x['image_id'])

    # 移除不需要的字段（保留调试信息字段，但不在最终 JSON 中保存）
    for img in selected_images:
        img.pop('num_instances', None)
        # 注意：保留 num_grounded, num_hallucinated, grounded_words, hallucinated_words 用于调试
        # 但这些字段不会保存到最终的 JSON 文件中（在生成 case 时不会使用）

    return selected_images


def load_coco_instances(coco_annotations_path: str) -> Dict[int, List[str]]:
    """
    加载 COCO instances annotations，构建 image_id 到对象列表的映射

    Args:
        coco_annotations_path: COCO annotations 目录路径

    Returns:
        Dict[int, List[str]]: image_id -> 对象名称列表的映射
    """
    coco_annotations_path = Path(coco_annotations_path)

    # 加载 val2014 instances
    val_instances_file = coco_annotations_path / "instances_val2014.json"
    if not val_instances_file.exists():
        raise FileNotFoundError(f"COCO val2014 instances 文件不存在: {val_instances_file}")

    with open(val_instances_file, 'r', encoding='utf-8') as f:
        val_instances = json.load(f)

    # 构建 category_id -> category_name 映射
    id_to_name = {}
    for cat in val_instances['categories']:
        id_to_name[cat['id']] = cat['name']

    # 构建 image_id -> 对象列表映射
    imid_to_objects = defaultdict(list)
    for annotation in val_instances['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        category_name = id_to_name[category_id]
        imid_to_objects[image_id].append(category_name)

    # 去重并转换为列表
    for image_id in imid_to_objects:
        imid_to_objects[image_id] = sorted(list(set(imid_to_objects[image_id])))

    return dict(imid_to_objects)


def generate_pope_question(object_name: str) -> str:
    """
    生成 POPE 格式的问题

    Args:
        object_name: 对象名称

    Returns:
        str: POPE 格式的问题文本
    """
    # 处理不定冠词
    article = "an" if object_name[0].lower() in ['a', 'e', 'i', 'o', 'u'] else "a"
    return f"Is there {article} {object_name} in the image?  Please answer Yes or No."


def generate_pope_cases(image_id: int, image_filename: str, objects_in_image: List[str],
                       all_objects: List[str], num_cases: int = 6) -> List[Dict]:
    """
    为一张图片生成 POPE 格式的测试 case

    Args:
        image_id: 图片 ID
        image_filename: 图片文件名
        objects_in_image: 图片中存在的对象列表
        all_objects: 所有可能的对象列表（COCO 80 类）
        num_cases: 需要生成的 case 数量（默认 6）

    Returns:
        List[Dict]: POPE 格式的 case 列表
    """
    cases = []
    objects_set = set(objects_in_image)
    all_objects_set = set(all_objects)
    objects_not_in_image = sorted(list(all_objects_set - objects_set))

    # 确保有足够的对象用于生成问题
    if len(objects_in_image) == 0:
        print(f"⚠️  警告: 图片 {image_filename} 没有检测到对象，跳过 POPE case 生成")
        return cases

    # 生成 Yes 和 No 的 case
    # 策略：从图片中的对象中选择一些生成 Yes case，从不在图片中的对象中选择一些生成 No case
    num_yes = num_cases // 2
    num_no = num_cases - num_yes

    # 生成 Yes cases（从图片中的对象选择）
    yes_objects = random.sample(objects_in_image, min(num_yes, len(objects_in_image)))
    for obj in yes_objects:
        cases.append({
            "question_id": len(cases) + 1,  # 将在后续统一编号
            "image": image_filename,
            "text": generate_pope_question(obj),
            "label": ["yes"]
        })

    # 如果 Yes case 不够，从图片中的对象重复选择
    while len(cases) < num_yes:
        obj = random.choice(objects_in_image)
        cases.append({
            "question_id": len(cases) + 1,
            "image": image_filename,
            "text": generate_pope_question(obj),
            "label": ["yes"]
        })

    # 生成 No cases（从不在图片中的对象选择）
    if len(objects_not_in_image) > 0:
        no_objects = random.sample(objects_not_in_image, min(num_no, len(objects_not_in_image)))
        for obj in no_objects:
            cases.append({
                "question_id": len(cases) + 1,
                "image": image_filename,
                "text": generate_pope_question(obj),
                "label": ["no"]
            })

        # 如果 No case 不够，从不在图片中的对象重复选择
        while len(cases) < num_cases:
            obj = random.choice(objects_not_in_image)
            cases.append({
                "question_id": len(cases) + 1,
                "image": image_filename,
                "text": generate_pope_question(obj),
                "label": ["no"]
            })
    else:
        # 如果没有不在图片中的对象（理论上不应该发生），使用图片中的对象但标记为 No
        print(f"⚠️  警告: 图片 {image_filename} 包含所有对象，无法生成 No case")
        while len(cases) < num_cases:
            obj = random.choice(objects_in_image)
            cases.append({
                "question_id": len(cases) + 1,
                "image": image_filename,
                "text": generate_pope_question(obj),
                "label": ["no"]  # 注意：这里标记为 No 但实际上对象存在，需要后续处理
            })

    return cases[:num_cases]  # 确保只返回指定数量的 case


def load_image(image_file):
    """加载图像文件"""
    if not os.path.exists(image_file):
        raise FileNotFoundError(f"图像文件不存在: {image_file}")
    image = Image.open(image_file).convert("RGB")
    return image


def generate_caption(model, tokenizer, image_processor, image_path: str,
                     conv_mode: str, device: str) -> str:
    """
    为图片生成caption

    Args:
        model: LLaVA模型
        tokenizer: tokenizer
        image_processor: 图像处理器
        image_path: 图片路径
        conv_mode: 对话模式
        device: 设备

    Returns:
        str: 生成的caption
    """
    # 加载图像
    image = load_image(image_path)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    # 准备文本输入
    prompt = "Please help me describe the image in detail."
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

    # 准备多模态输入
    images = image_tensor.unsqueeze(0).half().to(device)

    # 生成caption
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        with torch.no_grad():
            output_ids = model.generate(
                inputs=input_ids,
                images=images,
                do_sample=False,
                temperature=1.0,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

    # 解码输出
    generated_ids = output_ids[0]

    # 处理 BOS token
    bos_token_id = tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None else None
    if bos_token_id is not None and len(generated_ids) > 0 and generated_ids[0].item() == bos_token_id:
        generated_ids = generated_ids[1:]

    if len(generated_ids) > 0:
        caption = tokenizer.batch_decode([generated_ids], skip_special_tokens=True)[0].strip()
    else:
        caption = ""

    # 移除停止字符串
    if caption and caption.endswith(stop_str):
        caption = caption[:-len(stop_str)].strip()

    return caption


def extract_physical_words(caption: str, gt_objects: List[str], chair_evaluator) -> tuple:
    """
    使用CHAIR评估器从caption中提取实例词汇和幻视词汇

    Args:
        caption: 生成的caption
        gt_objects: 真实对象列表
        chair_evaluator: CHAIR评估器

    Returns:
        tuple: (实例词汇集合, 幻视词汇集合, 实例词汇数量, 幻视词汇数量)
    """
    if not caption:
        return set(), set(), 0, 0

    # 使用CHAIR接口识别物理词汇
    words, node_words, word_indices, raw_words = chair_evaluator.caption_to_words(caption)

    # 分离实例词汇和幻视词汇
    grounded_words = set()  # 实例词汇（在GT对象中）
    hallucinated_words = set()  # 幻视词汇（不在GT对象中）
    gt_objects_lower = [obj.lower() for obj in gt_objects]

    for word, node_word in zip(words, node_words):
        if node_word.lower() in gt_objects_lower:
            # 实例词汇
            grounded_words.add(node_word.lower())
        else:
            # 幻视词汇
            hallucinated_words.add(node_word.lower())

    return grounded_words, hallucinated_words, len(grounded_words), len(hallucinated_words)


def generate_chair_case(image_id: int, image_filename: str, objects_in_image: List[str]) -> Dict:
    """
    为一张图片生成 CHAIR 格式的测试 case

    Args:
        image_id: 图片 ID
        image_filename: 图片文件名
        objects_in_image: 图片中存在的对象列表

    Returns:
        Dict: CHAIR 格式的 case
    """
    return {
        "question_id": 0,  # 将在后续统一编号
        "image": image_filename,
        "text": "Please help me describe the image in detail.",
        "label": objects_in_image  # 对象列表作为 label
    }


def main():
    parser = argparse.ArgumentParser(description="构建 COCO 训练数据集")
    parser.add_argument("--coco-root", type=str, default=project.coco_data_path,
                       help="COCO 数据集根目录")
    parser.add_argument("--exclude-file", type=str, default="pope_coco/coco_baseline_500.json",
                       help="需要排除的图片列表文件")
    parser.add_argument("--num-images", type=int, default=50,
                       help="需要选择的图片数量")
    parser.add_argument("--output-file", type=str, default=None,
                       help="输出 JSON 文件路径（默认: coco_train_json/coco_train_2000.json）")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--model-path", type=str, default=project.llava_v15_7b_path,
                       help="模型路径（用于生成caption并计算幻视数量）")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="设备（用于生成caption）")
    parser.add_argument("--prioritize-by-hallucination", type=bool, default=True,
                       help="是否根据幻视数量优化图片选择（需要加载模型生成caption）")
    parser.add_argument("--chair-cache", type=str, default="eval_tool/chair_evaluator.pkl",
                       help="CHAIR评估器缓存文件路径")

    args = parser.parse_args()

    # 设置默认输出文件路径
    if args.output_file is None:
        args.output_file = os.path.join(coco_train_json_dir, f"coco_train_{args.num_images}.json")

    # 设置随机种子
    random.seed(args.seed)

    print("=" * 80)
    print("构建 COCO 训练数据集")
    print("=" * 80)
    print(f"COCO 根目录: {args.coco_root}")
    print(f"排除文件: {args.exclude_file}")
    print(f"选择图片数量: {args.num_images}")
    print(f"输出文件: {args.output_file}")
    print(f"随机种子: {args.seed}")
    print("=" * 80)

    # 1. 加载需要排除的图片列表
    print("\n[1/5] 加载需要排除的图片列表...")
    excluded_images = load_excluded_images(args.exclude_file)
    print(f"✓ 加载了 {len(excluded_images)} 张需要排除的图片")

    # 2. 加载 COCO instances annotations（需要在选择图片之前加载）
    print("\n[2/5] 加载 COCO instances annotations...")
    coco_annotations_path = os.path.join(args.coco_root, "annotations")
    imid_to_objects = load_coco_instances(coco_annotations_path)
    print(f"✓ 加载了 {len(imid_to_objects)} 张图片的 annotations")

    # 3. 从 COCO val2014 中选择图片（优先选择有3个实例的图片，然后2个、4个、1个）
    # 如果启用幻视优化，需要加载模型和CHAIR评估器
    model = None
    tokenizer = None
    image_processor = None
    conv_mode = None
    chair_evaluator = None
    prioritize_by_hallucination = args.prioritize_by_hallucination

    if prioritize_by_hallucination:
        print(f"\n[2.5/5] 加载模型和CHAIR评估器（用于生成caption并计算幻视数量）...")

        # 加载模型
        disable_torch_init()
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        device = args.device

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, None, model_name, device=device
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

        # 加载CHAIR评估器
        from eval_tool.chair import get_chair_evaluator
        import pickle

        chair_cache_file = os.path.expanduser(args.chair_cache)
        coco_annotations_path = os.path.join(args.coco_root, "annotations")

        if os.path.exists(chair_cache_file):
            try:
                chair_evaluator = pickle.load(open(chair_cache_file, 'rb'))
                print(f"✓ 从缓存加载 CHAIR 评估器: {chair_cache_file}")
            except Exception as e:
                print(f"⚠️  警告: 加载缓存失败: {e}，将重新创建评估器")
                chair_evaluator = get_chair_evaluator(
                    coco_path=coco_annotations_path,
                    cache_file=chair_cache_file,
                    use_cache=False
                )
        else:
            chair_evaluator = get_chair_evaluator(
                coco_path=coco_annotations_path,
                cache_file=chair_cache_file,
                use_cache=False
            )

    print(f"\n[3/5] 从 COCO val2014 中选择 {args.num_images} 张图片...")
    if prioritize_by_hallucination:
        print(f"  优化策略: 按实例数量 -> 按幻视数量（2个幻视 -> 3个幻视 -> 1个幻视 -> 其他）")
    else:
        print(f"  优化策略: 按实例数量（3个实例 -> 2个实例 -> 4个实例 -> 1个实例）")

    selected_images = get_coco_val2014_images(
        coco_root=args.coco_root,
        exclude_images=excluded_images,
        imid_to_objects=imid_to_objects,
        num_images=args.num_images,
        prioritize_by_hallucination=prioritize_by_hallucination,
        hallucination_scores=None,  # 将在函数内部生成
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        conv_mode=conv_mode,
        device=args.device if prioritize_by_hallucination else None,
        chair_evaluator=chair_evaluator
    )
    print(f"✓ 选择了 {len(selected_images)} 张图片")

    # 4. 为每张图片生成测试 case
    print(f"\n[4/5] 为每张图片生成测试 case...")
    all_cases = []
    question_id_counter = 1
    skipped_count = 0

    for img_info in selected_images:
        image_id = img_info['image_id']
        image_filename = img_info['image_filename']

        # 获取图片中的对象列表
        objects_in_image = imid_to_objects.get(image_id, [])

        # 由于我们已经优先选择了有实例的图片，这里应该不会出现0个实例的情况
        # 但为了安全起见，仍然检查
        if len(objects_in_image) == 0:
            skipped_count += 1
            continue

        # 生成 6 个 POPE 格式的 case
        pope_cases = generate_pope_cases(
            image_id=image_id,
            image_filename=image_filename,
            objects_in_image=objects_in_image,
            all_objects=COCO_CATEGORIES,
            num_cases=6
        )

        # 为 POPE cases 分配 question_id
        for case in pope_cases:
            case['question_id'] = question_id_counter
            question_id_counter += 1
            all_cases.append(case)

        # 生成 1 个 CHAIR 格式的 case
        chair_case = generate_chair_case(
            image_id=image_id,
            image_filename=image_filename,
            objects_in_image=objects_in_image
        )
        chair_case['question_id'] = question_id_counter
        question_id_counter += 1
        all_cases.append(chair_case)

    valid_images = len(selected_images) - skipped_count
    print(f"✓ 生成了 {len(all_cases)} 个测试 case")
    if skipped_count > 0:
        print(f"  ⚠️  跳过了 {skipped_count} 张没有对象的图片")
    print(f"  - POPE cases: {len(all_cases) - valid_images} (每张图片6个)")
    print(f"  - CHAIR cases: {valid_images} (每张图片1个)")

    # 5. 保存到 JSON 文件
    print(f"\n[5/5] 保存结果到 {args.output_file}...")
    output_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)

    print(f"✓ 结果已保存到: {output_file}")

    # 打印统计信息
    print("\n" + "=" * 80)
    print("统计信息")
    print("=" * 80)
    print(f"总图片数: {len(selected_images)}")
    if skipped_count > 0:
        print(f"有效图片数（有对象）: {valid_images}")
        print(f"跳过图片数（无对象）: {skipped_count}")
    print(f"总 case 数: {len(all_cases)}")
    if valid_images > 0:
        print(f"平均每张图片的 case 数: {len(all_cases) / valid_images:.2f}")

    # 统计 label 分布
    yes_count = sum(1 for case in all_cases if case.get('label') == ['yes'])
    no_count = sum(1 for case in all_cases if case.get('label') == ['no'])
    chair_count = sum(1 for case in all_cases if isinstance(case.get('label'), list) and len(case.get('label', [])) > 1)
    print(f"POPE Yes cases: {yes_count}")
    print(f"POPE No cases: {no_count}")
    print(f"CHAIR cases: {chair_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()
