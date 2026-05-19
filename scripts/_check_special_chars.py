import sys, io
sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.evaluation.data_loader import load_passages
from src.utils.config import RAW_DATA_DIR

path = RAW_DATA_DIR / "t2ranking" / "collection.tsv"
pids, texts = load_passages(path, max_passages=10000, show_progress=True)

pua_range = set(range(0xE000, 0xF8FF + 1))
format_chars = {0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0xFEFF}

pua_passages = sum(1 for t in texts if any(ord(c) in pua_range for c in t))
fmt_passages = sum(1 for t in texts if any(ord(c) in format_chars for c in t))
pua_total = sum(1 for t in texts for c in t if ord(c) in pua_range)
fmt_total = sum(1 for t in texts for c in t if ord(c) in format_chars)

print(f"After PUA_RE filter:")
print(f"  PUA chars remaining:   {pua_passages} passages, {pua_total} total (was 229/2080)")
print(f"  Format chars remaining: {fmt_passages} passages, {fmt_total} total (was 5/202)")

if pua_total == 0 and fmt_total == 0:
    print("  => All PUA and format control characters successfully removed.")
