DEFAULT_K_VALUES = [1, 3, 5, 10]
DEFAULT_METRIC_NAMES = ["Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR"]


def compute_metrics(results: list[dict], method_key: str, k_values: list[int] = None):
    if k_values is None:
        k_values = DEFAULT_K_VALUES

    metrics = {}
    for k in k_values:
        recalls = []
        precisions = []
        reciprocal_ranks = []
        hits = 0

        for r in results:
            retrieved_pids = r["retrievals"].get(method_key, [])[:k]
            relevant = r["relevant_pids"]

            hits_in_k = sum(1 for pid in retrieved_pids if pid in relevant)
            recalls.append(hits_in_k / len(relevant) if relevant else 0.0)
            precisions.append(hits_in_k / k if k > 0 else 0.0)

            rr = 0.0
            for rank, pid in enumerate(retrieved_pids, 1):
                if pid in relevant:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

            if hits_in_k > 0:
                hits += 1

        n = len(results) if results else 1
        metrics[f"Recall@{k}"] = sum(recalls) / n
        metrics[f"Precision@{k}"] = sum(precisions) / n
        metrics[f"Hit@{k}"] = hits / n
        if k == max(k_values):
            metrics["MRR"] = sum(reciprocal_ranks) / n

    return metrics


def print_comparison(metrics_map: dict[str, dict], metric_names: list[str] = None):
    if not metric_names:
        metric_names = DEFAULT_METRIC_NAMES

    methods = list(metrics_map.keys())
    col_width = 10

    print()
    print("=" * (16 + len(methods) * (col_width + 2)))
    print("  RESULTS")
    print("=" * (16 + len(methods) * (col_width + 2)))

    header = f"  {'Metric':<16}"
    for m in methods:
        header += f" {m:>{col_width}}"
    print(header)
    print("  " + "-" * (16 + len(methods) * (col_width + 2)))

    for metric_name in metric_names:
        row = f"  {metric_name:<16}"
        for m in methods:
            val = metrics_map[m].get(metric_name, float("nan"))
            row += f" {val:>{col_width}.4f}"
        print(row)

    print("=" * (16 + len(methods) * (col_width + 2)))
