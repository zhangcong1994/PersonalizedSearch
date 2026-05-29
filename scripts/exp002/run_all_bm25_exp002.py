"""
Run all exp-002 experiments with BM25 backend.

Usage:
    python scripts/run_all_bm25_exp002.py                # 2000 queries, top-50
    python scripts/run_all_bm25_exp002.py --sample 500   # quick test
    python scripts/run_all_bm25_exp002.py --top-k 10     # top-10 only
    python scripts/run_all_bm25_exp002.py --bm25-index /path/to/index
"""
import sys
import subprocess
import time
from pathlib import Path


def main():
    sample = 2000
    top_k = 50
    bm25_index = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--sample" and i + 1 < len(args):
            sample = int(args[i + 1])
            i += 2
        elif args[i] == "--top-k" and i + 1 < len(args):
            top_k = int(args[i + 1])
            i += 2
        elif args[i] == "--bm25-index" and i + 1 < len(args):
            bm25_index = args[i + 1]
            i += 2
        elif args[i] in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            print(f"Unknown arg: {args[i]}")
            return 1

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    sys.path.insert(0, str(project_root))

    from src.intent.query_rewrite_prompts import REGISTRY

    all_experiments = sorted(REGISTRY.keys())

    print("=" * 70)
    print("  BM25 Experiments Batch Runner")
    print("=" * 70)
    print(f"  Sample size:  {sample}")
    print(f"  Top-K:        {top_k}")
    print(f"  BM25 index:   {bm25_index or 'auto-detect'}")
    print(f"  Experiments:  {len(all_experiments)}")
    for eid in all_experiments:
        cfg = REGISTRY[eid]
        print(f"    {eid:12s}  {cfg['strategy']:12s}  {cfg['name']}")
    print("=" * 70)

    eval_script = script_dir / "evaluate_exp002.py"
    experiment_str = ",".join(all_experiments)

    cmd = [
        sys.executable, str(eval_script),
        "--experiment", experiment_str,
        "--bm25",
        "--sample", str(sample),
        "--top-k", str(top_k),
    ]
    if bm25_index:
        cmd.extend(["--bm25-index", bm25_index])

    print()
    print(f"Command: {' '.join(cmd)}")
    print()

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(project_root))
    elapsed = time.time() - t0

    print()
    print("=" * 70)
    if result.returncode == 0:
        print(f"  All experiments completed in {elapsed/60:.1f} minutes")
    else:
        print(f"  Failed with exit code {result.returncode} ({elapsed:.0f}s)")
    print("=" * 70)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
