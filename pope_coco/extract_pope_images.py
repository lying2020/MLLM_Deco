#!/usr/bin/env python3
"""
从 pope_coco 目录下的三个 JSON 文件中提取图像名称，去重后保存到 coco_baseline_500.json
"""

import json
import os
from pathlib import Path

def extract_unique_images(pope_coco_dir, output_file):
    """
    从 pope_coco 目录下的三个 JSON 文件中提取唯一的图像名称

    Args:
        pope_coco_dir: pope_coco 目录路径
        output_file: 输出文件路径
    """
    pope_coco_path = Path(pope_coco_dir)

    # 三个 JSON 文件名
    json_files = [
        "coco_pope_adversarial.json",
        "coco_pope_popular.json",
        "coco_pope_random.json"
    ]

    # 使用集合来存储唯一的图像名称
    unique_images = set()

    # 遍历每个 JSON 文件
    for json_file in json_files:
        file_path = pope_coco_path / json_file
        if not file_path.exists():
            print(f"⚠️  警告: 文件不存在: {file_path}")
            continue

        print(f"正在处理: {json_file}")
        count = 0

        # 读取 JSONL 格式文件（每行一个 JSON 对象）
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    image_name = data.get("image")
                    if image_name:
                        unique_images.add(image_name)
                        count += 1
                except json.JSONDecodeError as e:
                    print(f"⚠️  警告: 第 {line_num} 行 JSON 解析错误: {e}")
                    continue

        print(f"  - 从 {json_file} 中提取了 {count} 条记录")

    # 转换为排序后的列表
    unique_images_list = sorted(list(unique_images))

    print(f"\n总共提取了 {len(unique_images_list)} 个唯一的图像名称")

    # 保存到输出文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_images_list, f, indent=2, ensure_ascii=False)

    print(f"✓ 结果已保存到: {output_path}")
    print(f"  - 唯一图像数量: {len(unique_images_list)}")

    return unique_images_list


def main():
    """主函数"""
    # 项目根目录
    project_root = Path(__file__).parent

    # pope_coco 目录路径
    pope_coco_dir = project_root / "pope_coco"

    # 输出文件路径
    output_file = project_root / "coco_baseline_500.json"

    # 提取唯一的图像名称
    unique_images = extract_unique_images(pope_coco_dir, output_file)

    # 显示前10个图像名称作为示例
    if unique_images:
        print(f"\n前10个图像名称示例:")
        for i, img in enumerate(unique_images[:10], 1):
            print(f"  {i}. {img}")


if __name__ == "__main__":
    main()
