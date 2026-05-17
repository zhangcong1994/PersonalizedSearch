"""
解析中文维基百科 XML dump，流式提取文章内容，清洗 wiki 标记，抽样保存为 JSONL。

用法：
    python scripts/parse_wikipedia.py --num_articles 30000

输入：data/raw/wikipedia/zhwiki-latest-pages-articles.xml (14.7GB)
输出：data/processed/wikipedia_articles.jsonl
"""

import os
import sys
import re
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

_t2s_converter = None


def get_t2s_converter():
    global _t2s_converter
    if _t2s_converter is None:
        _t2s_converter = OpenCC("t2s")
    return _t2s_converter


def to_simplified(text: str) -> str:
    """繁体转简体"""
    try:
        return get_t2s_converter().convert(text)
    except Exception:
        return text


def clean_wiki_text(text: str) -> str:
    """清洗维基百科标记文本为纯文本"""
    if not text:
        return ""

    # 移除注释 <!-- ... -->
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # 移除嵌套模板 {{...}}（处理多层嵌套）
    # 循环移除最内层模板直到没有 {{ }}
    for _ in range(5):
        new_text = re.sub(r"\{\{[^{}]*?\}\}", "", text)
        if new_text == text:
            break
        text = new_text

    # 移除文件/图片引用 [[File:...]], [[Image:...]]
    text = re.sub(r"\[\[(?:File|Image|文件|图像):[^\]]*?\]\]", "", text, flags=re.IGNORECASE)

    # 处理内部链接 [[target|text]] -> text, [[target]] -> target
    text = re.sub(r"\[\[([^|\]]+?)\|([^\]]*?)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+?)\]\]", r"\1", text)

    # 处理外部链接 [url text] -> text, 直接移除裸URL
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]*?)\]", r"\1", text)
    text = re.sub(r"https?://[^\s]+", "", text)

    # 移除引用标签 <ref>...</ref>, <ref name="..." />
    text = re.sub(r"<ref[^>]*?/.*?>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<ref[^>]*?>.*?</ref>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # 移除其他HTML标签
    text = re.sub(r"<[^>]+>", "", text)

    # 移除HTML实体
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)

    # 粗体/斜体标记
    text = re.sub(r"'''?", "", text)

    # 表格标记
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^\s*[|!].*$", "", text, flags=re.MULTILINE)

    # 列表标记
    text = re.sub(r"^\s*[*#:;]+\s*", "", text, flags=re.MULTILINE)

    # 章节标题 == ... == -> 保留文字
    text = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", text, flags=re.MULTILINE)

    # 多个连续空行压缩为两个换行
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # 去掉首尾空行和每行的首尾空格
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    text = "\n".join(lines)

    return text.strip()


def parse_wikipedia_xml(xml_path: str, num_articles: int = 30000, min_text_length: int = 200):
    """
    流式解析维基百科 XML dump。
    逐页顺序处理，跳过无效页面（重定向/消歧义/过短），直到凑够目标数量。

    返回：List[dict]，每篇包含 id, title, text 字段
    """
    try:
        from lxml import etree
        iterparse = etree.iterparse
        logger.info("使用 lxml 加速 XML 解析")
    except ImportError:
        from xml.etree.ElementTree import iterparse
        logger.info("使用标准库 xml.etree.ElementTree 解析")

    articles = []
    ns = "{http://www.mediawiki.org/xml/export-0.11/}"

    page_count = 0
    extracted_count = 0
    skipped_redirect = 0
    skipped_short = 0
    skipped_disambig = 0
    skipped_ns = 0

    logger.info(f"开始流式解析，目标抽取 {num_articles:,} 篇有效文章...")

    for event, elem in iterparse(xml_path, events=("end",), tag=f"{ns}page"):
        page_count += 1

        title_elem = elem.find(f"{ns}title")
        text_elem = elem.find(f"{ns}revision/{ns}text")
        ns_elem_val = elem.find(f"{ns}ns")

        if title_elem is None or text_elem is None:
            elem.clear()
            continue

        title = title_elem.text or ""
        raw_text = text_elem.text or ""

        if ns_elem_val is not None and ns_elem_val.text and ns_elem_val.text != "0":
            skipped_ns += 1
            elem.clear()
            continue

        if raw_text.lower().startswith("#redirect") or raw_text.lower().startswith("#重定向"):
            skipped_redirect += 1
            elem.clear()
            continue

        if "{{消歧义" in raw_text or "{{disambiguation" in raw_text.lower() or "{{disambig" in raw_text.lower():
            skipped_disambig += 1
            elem.clear()
            continue

        clean_text = clean_wiki_text(raw_text)

        if len(clean_text) < min_text_length:
            skipped_short += 1
            elem.clear()
            continue

        clean_text = to_simplified(clean_text)

        articles.append({
            "id": extracted_count,
            "title": title.strip(),
            "text": clean_text,
            "length": len(clean_text),
        })
        extracted_count += 1

        if extracted_count % 1000 == 0:
            elapsed_pct = page_count / max(extracted_count, 1) * num_articles
            logger.info(
                f"  已抽取 {extracted_count:,}/{num_articles:,} 篇 "
                f"(扫描 {page_count:,} 页, 跳过: 重定向{skipped_redirect}, 消歧义{skipped_disambig}, "
                f"过短{skipped_short}, 非正文{skipped_ns})"
            )

        elem.clear()

        if extracted_count >= num_articles:
            break

    logger.info(f"抽取完成: {extracted_count:,} 篇 (共扫描 {page_count:,} 页)")
    logger.info(
        f"过滤统计: 重定向={skipped_redirect}, 消歧义={skipped_disambig}, "
        f"过短={skipped_short}, 非正文命名空间={skipped_ns}"
    )
    return articles


def save_to_jsonl(articles: list, output_path: str):
    """保存文章列表为 JSONL 格式"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")
    logger.info(f"已保存 {len(articles):,} 篇文章到: {output_path}")


def print_stats(articles: list):
    """打印数据集统计信息"""
    if not articles:
        return
    lengths = [a["length"] for a in articles]
    lengths.sort()
    n = len(lengths)

    logger.info("=" * 50)
    logger.info("数据集统计")
    logger.info(f"  总文章数: {n:,}")
    logger.info(f"  最小长度: {lengths[0]:,} 字符")
    logger.info(f"  最大长度: {lengths[-1]:,} 字符")
    logger.info(f"  平均长度: {sum(lengths)/n:,.0f} 字符")
    logger.info(f"  中位数长度: {lengths[n//2]:,} 字符")
    logger.info(f"  P25: {lengths[n//4]:,} 字符")
    logger.info(f"  P75: {lengths[3*n//4]:,} 字符")

    logger.info("  前5篇标题:")
    for a in articles[:5]:
        logger.info(f"    - {a['title']} ({a['length']:,} 字符)")

    logger.info("  后5篇标题:")
    for a in articles[-5:]:
        logger.info(f"    - {a['title']} ({a['length']:,} 字符)")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="解析中文维基百科 XML dump")
    parser.add_argument(
        "--num_articles", type=int, default=30000,
        help="目标抽取文章数量 (默认: 30000)"
    )
    parser.add_argument(
        "--min_length", type=int, default=200,
        help="过滤掉少于此字符数的文章 (默认: 200)"
    )
    parser.add_argument(
        "--xml_path", type=str,
        default="data/raw/wikipedia/zhwiki-latest-pages-articles.xml",
        help="Wikipedia XML dump 路径"
    )
    parser.add_argument(
        "--output_path", type=str,
        default="data/processed/wikipedia_articles.jsonl",
        help="输出 JSONL 文件路径"
    )
    args = parser.parse_args()

    project_root = os.path.join(os.path.dirname(__file__), "..")
    xml_path = os.path.join(project_root, args.xml_path)
    output_path = os.path.join(project_root, args.output_path)

    if not os.path.exists(xml_path):
        logger.error(f"找不到 XML 文件: {xml_path}")
        sys.exit(1)

    logger.info(f"XML 文件: {xml_path}")
    file_size_gb = os.path.getsize(xml_path) / (1024**3)
    logger.info(f"文件大小: {file_size_gb:.1f} GB")

    articles = parse_wikipedia_xml(
        xml_path=xml_path,
        num_articles=args.num_articles,
        min_text_length=args.min_length,
    )

    if articles:
        save_to_jsonl(articles, output_path)
        print_stats(articles)
    else:
        logger.error("未提取到任何文章")
        sys.exit(1)


if __name__ == "__main__":
    main()
