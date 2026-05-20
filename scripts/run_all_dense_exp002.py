"""
Run all exp-002 experiments with Dense (vector) retrieval backend.

Usage:
    python scripts/run_all_dense_exp002.py --device cuda --embedding-model moka-ai/m3e-base --vector-db /path/to/vector_db
    python scripts/run_all_dense_exp002.py --device cpu --embedding-model moka-ai/m3e-base --vector-db /path/to/vector_db
    python scripts/run_all_dense_exp002.py --sample 500 --device cuda --embedding-model moka-ai/m3e-base --vector-db /path/to/vector_db
"""
import sys
import subprocess
import time
from pathlib import Path


def main():
    sample = 2000
    top_k = 50
    device = "cuda"
    embedding_model = None
    vector_db = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--sample" and i + 1 < len(args):
            sample = int(args[i + 1])
            i += 2
        elif args[i] == "--top-k" and i + 1 < len(args):
            top_k = int(args[i + 1])
            i += 2
        elif args[i] == "--device" and i + 1 < len(args):
            device = args[i + 1]
            i += 2
        elif args[i] == "--embedding-model" and i + 1 < len(args):
            embedding_model = args[i + 1]
            i += 2
        elif args[i] == "--vector-db" and i + 1 < len(args):
            vector_db = args[i + 1]
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
    print("  Dense (Vector) Experiments Batch Runner")
    print("=" * 70)
    print(f"  Sample size:  {sample}")
    print(f"  Top-K:        {top_k}")
    print(f"  Device:       {device}")
    print(f"  Embedding:    {embedding_model}")
    print(f"  Vector DB:    {vector_db}")
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
        "--sample", str(sample),
        "--top-k", str(top_k),
        "--device", device,
    ]
    if embedding_model:
        cmd.extend(["--embedding-model", embedding_model])
    if vector_db:
        cmd.extend(["--vector-db", vector_db])

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
