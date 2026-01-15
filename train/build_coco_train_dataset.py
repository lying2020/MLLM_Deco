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
from typing import List, Dict, Set
from collections import defaultdict
import argparse

# 导入项目配置
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) != 'MLLM_Deco' else current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import project as project


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
                            num_images: int = 2000) -> List[Dict]:
    """
    从 COCO val2014 目录中获取指定数量的图片（排除已使用的图片）
    优先选择策略：3个实例 -> 2个实例 -> 4个实例 -> 1个实例

    Args:
        coco_root: COCO 数据集根目录
        exclude_images: 需要排除的图片文件名集合
        imid_to_objects: image_id 到对象列表的映射
        num_images: 需要获取的图片数量

    Returns:
        List[Dict]: 包含 image_id 和 image_filename 的字典列表
    """
    coco_root = Path(coco_root)
    val2014_dir = coco_root / "val2014"

    if not val2014_dir.exists():
        raise FileNotFoundError(f"COCO val2014 目录不存在: {val2014_dir}")

    # 根据实例数量分类图片
    images_with_3_instances = []  # 3个实例（最高优先级）
    images_with_2_instances = []  # 2个实例（第二优先级）
    images_with_4_instances = []  # 4个实例（第三优先级）
    images_with_1_instance = []  # 1个实例（第四优先级）
    images_with_0_instances = []  # 0个实例（排除）
    images_with_more_instances = []  # 5个及以上实例（最后选择）

    image_files = sorted(val2014_dir.glob("COCO_val2014_*.jpg"))

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
    selected_images = []
    remaining = num_images

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

    # 移除 num_instances 字段（不需要在返回结果中）
    for img in selected_images:
        img.pop('num_instances', None)

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
    parser.add_argument("--num-images", type=int, default=10,
                       help="需要选择的图片数量")
    parser.add_argument("--output-file", type=str, default=None,
                       help="输出 JSON 文件路径（默认: train/coco_train_2000.json）")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")

    args = parser.parse_args()

    # 设置默认输出文件路径
    if args.output_file is None:
        train_dir = Path(__file__).parent
        args.output_file = os.path.join(train_dir, f"coco_train_{args.num_images}.json")

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
    print(f"\n[3/5] 从 COCO val2014 中选择 {args.num_images} 张图片（优先顺序：3个实例 -> 2个实例 -> 4个实例 -> 1个实例）...")
    selected_images = get_coco_val2014_images(
        coco_root=args.coco_root,
        exclude_images=excluded_images,
        imid_to_objects=imid_to_objects,
        num_images=args.num_images
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
