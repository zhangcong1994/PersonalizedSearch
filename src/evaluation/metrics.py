DEFAULT_K_VALUES = [10, 20, 50]
DEFAULT_METRIC_NAMES = ["Recall@10", "Recall@20", "Recall@50", "MRR"]


def _extract_pids(retrieved_items: list) -> list[str]:
    if not retrieved_items:
        return []
    if isinstance(retrieved_items[0], dict):
        return [item["pid"] for item in retrieved_items]
    return retrieved_items


def get_metric_params(top_k: int):
    k_values = sorted(set(DEFAULT_K_VALUES + [top_k]))
    metric_names = [f"Recall@{k}" for k in k_values] + ["MRR"]
    return k_values, metric_names


def compute_metrics(results: list[dict], method_key: str, k_values: list[int] = None):
    if k_values is None:
        k_values = DEFAULT_K_VALUES

    max_k = max(k_values)

    metrics = {}
    for k in k_values:
        recalls = []
        precisions = []
        reciprocal_ranks = []
        hits = 0

        for r in results:
            retrieved_items = r["retrievals"].get(method_key, [])[:max_k]
            retrieved_pids = _extract_pids(retrieved_items)[:k]
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


def _compute_dcg(gains: list[int], k: int) -> float:
    import math
    dcg = 0.0
    for i, g in enumerate(gains[:k]):
        dcg += g / math.log2(i + 2)
    return dcg


def compute_reranker_metrics(
    results: list[dict],
    method_key: str,
    k_values: list[int] = None,
    qrels_graded: dict[str, dict[str, int]] = None,
):
    if k_values is None:
        k_values = DEFAULT_K_VALUES

    max_k = max(k_values)

    metrics = {}
    ndcg_sums_binary = {k: 0.0 for k in k_values}
    ndcg_sums_graded = {k: 0.0 for k in k_values}

    for r in results:
        qid = r["qid"]
        retrieved_items = r["retrievals"].get(method_key, [])[:max_k]
        retrieved_pids = _extract_pids(retrieved_items)
        relevant = r["relevant_pids"]

        binary_gains = [1 if pid in relevant else 0 for pid in retrieved_pids]

        for k in k_values:
            dcg = _compute_dcg(binary_gains, k)
            ideal_gains = sorted([1] * min(len(relevant), k), reverse=True)
            ideal_gains += [0] * max(0, k - len(ideal_gains))
            idcg = _compute_dcg(ideal_gains, k)
            ndcg_sums_binary[k] += dcg / idcg if idcg > 0 else 0.0

            if qrels_graded and qid in qrels_graded:
                graded_rel = qrels_graded[qid]
                graded_gains = [graded_rel.get(pid, 0) for pid in retrieved_pids]
                dcg_graded = _compute_dcg(graded_gains, k)
                ideal_graded = sorted(graded_rel.values(), reverse=True)[:k]
                idcg_graded = _compute_dcg(ideal_graded, k)
                ndcg_sums_graded[k] += dcg_graded / idcg_graded if idcg_graded > 0 else 0.0

    n = len(results) if results else 1
    for k in k_values:
        metrics[f"NDCG@{k}"] = ndcg_sums_binary[k] / n
        if qrels_graded:
            metrics[f"NDCG@{k}_graded"] = ndcg_sums_graded[k] / n

    base_metrics = compute_metrics(results, method_key, k_values=k_values)
    metrics.update(base_metrics)

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
