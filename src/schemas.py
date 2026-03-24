from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from pyspark import RDD
except Exception:  # pragma: no cover - only relevant when pyspark is unavailable
    RDD = Any  # type: ignore[assignment]


@dataclass
class PreparedData:
    num_users: int
    num_items: int
    train_ui_rdd: RDD
    test_rdd: RDD
    user_history_rdd: RDD
    item_history_rdd: RDD
    c_array: np.ndarray
    user_ids: list[str]
    item_ids: list[str]
    num_train_interactions: int
    num_test_users: int
    stats: dict[str, float | int]


@dataclass
class ModelState:
    p_matrix: np.ndarray
    q_matrix: np.ndarray
    train_log: list[dict[str, Any]]
