from __future__ import annotations

import math

import numpy as np


def evaluate_user_record(
    record: tuple[int, int, frozenset[int]],
    p_matrix: np.ndarray,
    q_matrix: np.ndarray,
    topk: int,
) -> tuple[float, float]:
    """Return HR@K and NDCG@K for one user."""
    user_idx, test_item_idx, train_item_set = record
    p_u = p_matrix[user_idx]
    scores = q_matrix.dot(p_u).astype(np.float64, copy=False)

    if train_item_set:
        train_idx = np.fromiter(train_item_set, dtype=np.int64)
        scores[train_idx] = -np.inf

    k = min(topk, scores.shape[0])
    if k <= 0:
        return 0.0, 0.0

    top_k_idx = np.argpartition(scores, -k)[-k:]
    top_k_scores = scores[top_k_idx]
    top_k_sorted = top_k_idx[np.argsort(-top_k_scores)]
    positions = np.where(top_k_sorted == test_item_idx)[0]
    if len(positions) == 0:
        return 0.0, 0.0

    rank = int(positions[0]) + 1  # 1-indexed rank
    return 1.0, 1.0 / math.log2(rank + 1)


def evaluate_model(
    sc,
    test_rdd,
    p_matrix: np.ndarray,
    q_matrix: np.ndarray,
    topk: int = 100,
    eval_user_sample: int | None = None,
    random_seed: int = 42,
) -> dict[str, float | int]:
    """Evaluate leave-one-out with full-item ranking and train masking."""
    eval_rdd = test_rdd
    if eval_user_sample is not None and eval_user_sample > 0:
        sample_records = eval_rdd.takeSample(False, eval_user_sample, random_seed)
        eval_rdd = sc.parallelize(sample_records, numSlices=test_rdd.getNumPartitions())

    p_bc = sc.broadcast(p_matrix)
    q_bc = sc.broadcast(q_matrix)
    metric_rdd = eval_rdd.map(lambda row: evaluate_user_record(row, p_bc.value, q_bc.value, topk))
    hr_sum, ndcg_sum, count = metric_rdd.aggregate(
        (0.0, 0.0, 0),
        lambda acc, value: (acc[0] + value[0], acc[1] + value[1], acc[2] + 1),
        lambda left, right: (left[0] + right[0], left[1] + right[1], left[2] + right[2]),
    )

    try:
        p_bc.unpersist(blocking=False)
        q_bc.unpersist(blocking=False)
    except Exception:
        pass
    try:
        p_bc.destroy()
        q_bc.destroy()
    except Exception:
        pass

    if count == 0:
        return {"hr": 0.0, "ndcg": 0.0, "evaluated_users": 0}
    return {
        "hr": hr_sum / count,
        "ndcg": ndcg_sum / count,
        "evaluated_users": int(count),
    }
