import os
import sys
import time
import argparse
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = PROJECT_ROOT / "data" / "raw" / "t2ranking"

FILES = {
    "collection.tsv":           ("2,303,643 条 passage", "~3.5 GB"),
    "queries.dev.tsv":          ("24,832 条查询（V0 评估主力）", "~1 MB"),
    "queries.test.tsv":         ("24,832 条查询（V1 最终评估）", "~1 MB"),
    "qrels.retrieval.dev.tsv":  ("118,933 条检索 qrels（二元相关）", "~1.5 MB"),
    "qrels.dev.tsv":            ("400,536 条精排 qrels（TREC 格式，4 级标注 0-3）", "~6 MB"),
}

CHUNK_SIZE = 128 * 1024


def _format_size(bytes_val: int) -> str:
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.1f} KB"
    else:
        return f"{bytes_val / (1024 * 1024):.1f} MB"


def download_file(
    url: str,
    dest: Path,
    max_retries: int = 3,
    timeout: int = (15, 300),
) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        size = dest.stat().st_size
        if size > 0:
            print(f"   已存在 ({_format_size(size)})，跳过下载")
            return True
        else:
            dest.unlink()

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })

    for attempt in range(1, max_retries + 1):
        try:
            print(f"   正在连接... (尝试 {attempt}/{max_retries})")
            response = session.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            total_size = int(response.headers.get("Content-Length", 0))
            if total_size > 0:
                print(f"   文件大小: {_format_size(total_size)}")

            downloaded = 0
            start_time = time.time()
            last_report_time = start_time

            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()
                    if total_size > 0 and (now - last_report_time) >= 3:
                        pct = downloaded / total_size * 100
                        elapsed = now - start_time
                        speed = downloaded / elapsed if elapsed > 0 else 0
                        eta = (total_size - downloaded) / speed if speed > 0 else 0
                        print(
                            f"   进度: {pct:.0f}% ({_format_size(downloaded)}/{_format_size(total_size)})"
                            f" | 速度: {_format_size(int(speed))}/s"
                            f" | 预计剩余: {int(eta)}s"
                        )
                        last_report_time = now

            elapsed = time.time() - start_time
            final_size = dest.stat().st_size
            avg_speed = final_size / elapsed if elapsed > 0 else 0
            print(f"   下载完成: {_format_size(final_size)}，耗时 {elapsed:.0f}s ({_format_size(int(avg_speed))}/s)")
            return True

        except Exception as e:
            print(f"   下载失败: {e}")
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"   {wait}s 后重试...")
                time.sleep(wait)
            if dest.exists():
                dest.unlink()

    return False


def main():
    parser = argparse.ArgumentParser(description="下载 T2Ranking 评估数据")
    parser.add_argument(
        "--files",
        nargs="*",
        choices=list(FILES.keys()) + ["all"],
        default=["all"],
        help="要下载的文件 (默认: all)",
    )
    parser.add_argument(
        "--mirror",
        default="https://hf-mirror.com",
        help="HuggingFace 镜像地址 (用于下载大文件)",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="使用 HuggingFace 直连 (小文件更快，大文件可能超时)",
    )
    args = parser.parse_args()

    if "all" in args.files:
        selected = list(FILES.keys())
    else:
        selected = args.files

    mirror = args.mirror.rstrip("/")
    base_url = f"{mirror}/datasets/THUIR/T2Ranking/resolve/main/data"

    print(f"T2Ranking 评估数据下载")
    print(f"镜像: {mirror}")
    print(f"保存目录: {SAVE_DIR}")
    print()

    success_count = 0
    for idx, filename in enumerate(selected):
        desc, size_est = FILES[filename]
        print(f"[{idx + 1}/{len(selected)}] {filename}")
        print(f"   {desc}, 预估 {size_est}")

        url = f"{base_url}/{filename}"

        if download_file(url, SAVE_DIR / filename):
            success_count += 1
        else:
            print(f"   [FAIL] 下载最终失败")
        print()

    print(f"{'=' * 60}")
    print(f"下载完成: {success_count}/{len(selected)} 个文件成功")
    print(f"保存位置: {SAVE_DIR}")

    if success_count == len(selected):
        print("[OK] 所有评估数据下载完毕，可以开始构建 T2Ranking 评估索引")
        return 0
    else:
        print("[WARN] 部分文件下载失败，请检查网络后重试")
        return 1


if __name__ == "__main__":
    sys.exit(main())
