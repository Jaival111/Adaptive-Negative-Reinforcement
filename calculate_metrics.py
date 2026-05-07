import argparse
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


def read_jsonl(file_path: str) -> List[dict]:
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def group_by_question(data: List[dict]) -> Dict[int, List[dict]]:
    grouped = defaultdict(list)
    for ex in data:
        grouped[ex["question_id"]].append(ex)
    return grouped


def infer_n(grouped: Dict[int, List[dict]], explicit_n: int = None) -> int:
    if explicit_n is not None and explicit_n > 0:
        return explicit_n
    if not grouped:
        return 0
    # Use the smallest per-question count so metrics are comparable and robust to partial runs.
    return min(len(v) for v in grouped.values())


def unbiased_pass_at_k(grouped: Dict[int, List[dict]], k: int, n: int) -> float:
    if n <= 0 or k <= 0:
        return 0.0
    k = min(k, n)

    total_pass_k_prob = 0.0
    total_questions = 0
    for _, examples in grouped.items():
        if len(examples) < n:
            continue
        exs = examples[:n]
        c = sum(1 for ex in exs if bool(ex.get("label", False)))
        if n - c < k:
            prob = 1.0
        else:
            prob = 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
        total_pass_k_prob += float(prob)
        total_questions += 1

    return total_pass_k_prob / total_questions if total_questions > 0 else 0.0


def pass_at_1(grouped: Dict[int, List[dict]], n: int) -> float:
    if n <= 0:
        return 0.0
    total = 0
    correct = 0
    for _, examples in grouped.items():
        if len(examples) < 1:
            continue
        total += 1
        correct += 1 if bool(examples[0].get("label", False)) else 0
    return correct / total if total > 0 else 0.0


def evaluate_file(file_path: str, explicit_n: int = None, ks: List[int] = None) -> Tuple[dict, Dict[int, float]]:
    if ks is None:
        ks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    data = read_jsonl(file_path)
    grouped = group_by_question(data)
    n = infer_n(grouped, explicit_n)

    metrics = {}
    for k in ks:
        if k <= n:
            metrics[k] = unbiased_pass_at_k(grouped, k=k, n=n)
    summary = {
        "file": file_path,
        "num_rows": len(data),
        "num_questions": len(grouped),
        "n": n,
        "pass@1_first_sample": pass_at_1(grouped, n),
    }
    return summary, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        default=None,
        required=False,
        help="Single JSONL path (backward-compatible).",
    )
    parser.add_argument(
        "--file_paths",
        nargs="+",
        default=None,
        required=False,
        help="One or more JSONL paths to evaluate.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=0,
        help="Optional generation count per question. If omitted, inferred as min count per question.",
    )
    args = parser.parse_args()

    file_paths = []
    if args.file_paths:
        file_paths.extend(args.file_paths)
    if args.file_path:
        file_paths.append(args.file_path)
    if not file_paths:
        raise ValueError("Provide --file_path or --file_paths.")

    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    summaries = []
    all_metrics = []

    for fp in file_paths:
        print("=" * 100)
        summary, metrics = evaluate_file(fp, explicit_n=(args.n if args.n > 0 else None), ks=ks)
        summaries.append(summary)
        all_metrics.append(metrics)

        print(f"File: {summary['file']}")
        print(f"Rows: {summary['num_rows']}, Questions: {summary['num_questions']}, n_used: {summary['n']}")
        print(f"pass@1 (first sample): {summary['pass@1_first_sample']:.6f}")
        for k in sorted(metrics.keys()):
            print(f"Unbiased pass@{k}/{summary['n']}: {metrics[k]:.6f}")

    # Macro average over files for shared K values.
    shared_ks = sorted(set.intersection(*[set(m.keys()) for m in all_metrics])) if all_metrics else []
    if len(file_paths) > 1 and shared_ks:
        print("=" * 100)
        print("Macro average across files:")
        for k in shared_ks:
            vals = [m[k] for m in all_metrics]
            print(f"Unbiased pass@{k}: {float(np.mean(vals)):.6f}")
