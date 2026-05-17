"""
将已有的 JSONL 文档中的繁体中文转换为简体中文。

用法：
    python scripts/convert_to_simplified.py

输入：data/processed/wikipedia_articles.jsonl
输出：data/processed/wikipedia_articles.jsonl (原地替换)
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

from opencc import OpenCC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="繁体转简体 JSONL 文档")
    parser.add_argument(
        "--input", type=str,
        default="data/processed/wikipedia_articles.jsonl",
        help="输入 JSONL 路径"
    )
    parser.add_argument(
        "--output", type=str,
        default=None,
        help="输出 JSONL 路径 (默认覆盖输入)"
    )
    parser.add_argument(
        "--keep_backup", action="store_true",
        help="保留原文件的 .bak 备份"
    )
    args = parser.parse_args()

    project_root = os.path.join(os.path.dirname(__file__), "..")
    input_path = os.path.join(project_root, args.input)
    output_path = os.path.join(project_root, args.output) if args.output else input_path

    if not os.path.exists(input_path):
        logger.error(f"找不到文件: {input_path}")
        sys.exit(1)

    logger.info(f"初始化 OpenCC t2s 转换器...")
    converter = OpenCC("t2s")

    # 读取全部文章
    logger.info(f"读取: {input_path}")
    articles = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line.strip()))

    logger.info(f"共 {len(articles):,} 篇文章，开始转换...")

    # 繁简转换
    converted_title = 0
    converted_text = 0
    for article in articles:
        new_title = converter.convert(article["title"])
        if new_title != article["title"]:
            article["title"] = new_title
            converted_title += 1

        new_text = converter.convert(article["text"])
        if new_text != article["text"]:
            article["text"] = new_text
            article["length"] = len(new_text)
            converted_text += 1

    logger.info(f"标题转换: {converted_title}/{len(articles)}")
    logger.info(f"正文转换: {converted_text}/{len(articles)}")

    # 备份原文件
    if args.keep_backup and input_path == output_path:
        backup_path = input_path + ".bak"
        os.rename(input_path, backup_path)
        logger.info(f"已备份原文件: {backup_path}")

    # 写入
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    logger.info(f"已保存: {output_path} ({len(articles):,} 篇文章)")


if __name__ == "__main__":
    main()
