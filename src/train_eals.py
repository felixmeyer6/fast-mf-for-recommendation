from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np

from .schemas import ModelState, PreparedData


def _as_config_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    if hasattr(config, "to_dict") and callable(config.to_dict):
        return config.to_dict()
    if hasattr(config, "__dict__"):
        return vars(config)
    raise TypeError("Config must be a dict-like object or expose to_dict()/__dict__.")


def compute_sq_matrix(q_matrix: np.ndarray, c_array: np.ndarray) -> np.ndarray:
    """Compute S^q = sum_i c_i q_i q_i^T."""
    weighted_q = q_matrix.astype(np.float64) * c_array[:, None]
    return q_matrix.astype(np.float64).T @ weighted_q


def compute_sp_matrix(p_matrix: np.ndarray) -> np.ndarray:
    """Compute S^p = P^T P."""
    return p_matrix.astype(np.float64).T @ p_matrix.astype(np.float64)


def update_single_user(
    p_u: np.ndarray,
    history: list[tuple[int, float]],
    q_matrix: np.ndarray,
    sq_matrix: np.ndarray,
    reg_lambda: float,
    observed_weight: float,
    observed_value: float,
) -> np.ndarray:
    """Coordinate descent update for one user's vector (Eq. 12)."""
    updated = np.copy(p_u).astype(np.float64)
    preds = {item_idx: float(np.dot(updated, q_matrix[item_idx])) for item_idx, _ in history}
    factors = updated.shape[0]

    for f in range(factors):
        num = 0.0
        den = 0.0
        for item_idx, c_i in history:
            q_if = float(q_matrix[item_idx, f])
            pred_f = preds[item_idx] - updated[f] * q_if
            num += (observed_weight * observed_value - (observed_weight - c_i) * pred_f) * q_if
            den += (observed_weight - c_i) * (q_if**2)

        cache_num = float(np.dot(updated, sq_matrix[:, f]) - updated[f] * sq_matrix[f, f])
        old_value = updated[f]
        updated[f] = (num - cache_num) / (den + sq_matrix[f, f] + reg_lambda)

        delta = updated[f] - old_value
        if delta != 0.0:
            for item_idx, _ in history:
                preds[item_idx] += delta * float(q_matrix[item_idx, f])

    return updated


def update_single_item(
    q_i: np.ndarray,
    c_i: float,
    users: list[int],
    p_matrix: np.ndarray,
    sp_matrix: np.ndarray,
    reg_lambda: float,
    observed_weight: float,
    observed_value: float,
) -> np.ndarray:
    """Coordinate descent update for one item's vector (Eq. 13)."""
    updated = np.copy(q_i).astype(np.float64)
    preds = {user_idx: float(np.dot(p_matrix[user_idx], updated)) for user_idx in users}
    factors = updated.shape[0]

    for f in range(factors):
        num = 0.0
        den = 0.0
        for user_idx in users:
            p_uf = float(p_matrix[user_idx, f])
            pred_f = preds[user_idx] - p_uf * updated[f]
            num += (observed_weight * observed_value - (observed_weight - c_i) * pred_f) * p_uf
            den += (observed_weight - c_i) * (p_uf**2)

        cache_num = c_i * float(np.dot(updated, sp_matrix[:, f]) - updated[f] * sp_matrix[f, f])
        old_value = updated[f]
        updated[f] = (num - cache_num) / (den + c_i * sp_matrix[f, f] + reg_lambda)

        delta = updated[f] - old_value
        if delta != 0.0:
            for user_idx in users:
                preds[user_idx] += delta * float(p_matrix[user_idx, f])
    return updated


def _update_user_partition(
    records: Iterable[tuple[int, list[tuple[int, float]]]],
    p_matrix_bc,
    q_matrix_bc,
    sq_matrix_bc,
    config: dict[str, Any],
):
    p_matrix = p_matrix_bc.value
    q_matrix = q_matrix_bc.value
    sq_matrix = sq_matrix_bc.value
    for user_idx, history in records:
        updated = update_single_user(
            p_u=p_matrix[user_idx],
            history=history,
            q_matrix=q_matrix,
            sq_matrix=sq_matrix,
            reg_lambda=config["reg_lambda"],
            observed_weight=config["observed_weight"],
            observed_value=config["observed_value"],
        )
        yield user_idx, updated.astype(p_matrix.dtype, copy=False)


def _update_item_partition(
    records: Iterable[tuple[int, float, list[int]]],
    p_matrix_bc,
    q_matrix_bc,
    sp_matrix_bc,
    config: dict[str, Any],
):
    p_matrix = p_matrix_bc.value
    q_matrix = q_matrix_bc.value
    sp_matrix = sp_matrix_bc.value
    for item_idx, c_i, users in records:
        updated = update_single_item(
            q_i=q_matrix[item_idx],
            c_i=c_i,
            users=users,
            p_matrix=p_matrix,
            sp_matrix=sp_matrix,
            reg_lambda=config["reg_lambda"],
            observed_weight=config["observed_weight"],
            observed_value=config["observed_value"],
        )
        yield item_idx, updated.astype(q_matrix.dtype, copy=False)


def _cleanup_broadcasts(*broadcast_vars) -> None:
    for bc in broadcast_vars:
        if bc is None:
            continue
        try:
            bc.unpersist(blocking=False)
        except Exception:
            pass
        try:
            bc.destroy()
        except Exception:
            pass


def train_eals(sc, prepared_data: PreparedData, config: Any) -> ModelState:
    """Train eALS using map-and-broadcast alternating updates."""
    cfg = _as_config_dict(config)
    dtype = np.dtype(cfg["dtype"])
    rng = np.random.default_rng(cfg["random_seed"])
    p_matrix = rng.normal(
        loc=cfg["init_mean"],
        scale=cfg["init_std"],
        size=(prepared_data.num_users, cfg["factors"]),
    ).astype(dtype)
    q_matrix = rng.normal(
        loc=cfg["init_mean"],
        scale=cfg["init_std"],
        size=(prepared_data.num_items, cfg["factors"]),
    ).astype(dtype)

    train_log: list[dict[str, float | int]] = []
    for iteration in range(1, int(cfg["iterations"]) + 1):
        iter_start = time.time()

        sq_matrix = compute_sq_matrix(q_matrix, prepared_data.c_array)
        p_bc = sc.broadcast(p_matrix)
        q_bc = sc.broadcast(q_matrix)
        sq_bc = sc.broadcast(sq_matrix)
        updated_users = prepared_data.user_history_rdd.mapPartitions(
            lambda records: _update_user_partition(records, p_bc, q_bc, sq_bc, cfg)
        ).collect()
        _cleanup_broadcasts(p_bc, q_bc, sq_bc)
        for user_idx, updated in updated_users:
            p_matrix[user_idx] = updated

        sp_matrix = compute_sp_matrix(p_matrix)
        p_bc = sc.broadcast(p_matrix)
        q_bc = sc.broadcast(q_matrix)
        sp_bc = sc.broadcast(sp_matrix)
        updated_items = prepared_data.item_history_rdd.mapPartitions(
            lambda records: _update_item_partition(records, p_bc, q_bc, sp_bc, cfg)
        ).collect()
        _cleanup_broadcasts(p_bc, q_bc, sp_bc)
        for item_idx, updated in updated_items:
            q_matrix[item_idx] = updated

        train_log.append(
            {
                "iteration": iteration,
                "seconds": time.time() - iter_start,
            }
        )

    return ModelState(p_matrix=p_matrix, q_matrix=q_matrix, train_log=train_log)
