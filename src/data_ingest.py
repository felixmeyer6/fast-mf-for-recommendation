from __future__ import annotations

from collections import Counter
from operator import add
import time
from typing import Any, Iterable

import numpy as np
from pyspark import StorageLevel

from .schemas import PreparedData

PRODUCT_PREFIX = "product/productId: "
USER_PREFIX = "review/userId: "
TIME_PREFIX = "review/time: "


def _as_config_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    if hasattr(config, "to_dict") and callable(config.to_dict):
        return config.to_dict()
    if hasattr(config, "__dict__"):
        return vars(config)
    raise TypeError("Config must be a dict-like object or expose to_dict()/__dict__.")


def storage_level_from_name(name: str) -> StorageLevel:
    upper_name = name.upper()
    if not hasattr(StorageLevel, upper_name):
        raise ValueError(f"Unknown storage level: {name}")
    return getattr(StorageLevel, upper_name)


def compute_c_from_item_counts(
    counts_by_item: dict[int, int],
    num_items: int,
    c0: float,
    alpha: float,
) -> np.ndarray:
    """
    Compute item missing-data weights c_i from item interaction counts.

    c_i = c0 * f_i^alpha / sum_j f_j^alpha, where f_i = |R_i| / sum_j |R_j|.
    """
    c_array = np.zeros(num_items, dtype=np.float64)
    if not counts_by_item:
        return c_array

    total_interactions = float(sum(counts_by_item.values()))
    f_pow_sum = 0.0
    item_f_pow: dict[int, float] = {}
    for item_idx, count in counts_by_item.items():
        f_i = float(count) / total_interactions
        f_pow = f_i**alpha
        item_f_pow[item_idx] = f_pow
        f_pow_sum += f_pow

    if f_pow_sum <= 0.0:
        return c_array

    scale = c0 / f_pow_sum
    for item_idx, f_pow in item_f_pow.items():
        c_array[item_idx] = scale * f_pow
    return c_array


def compute_item_weights_rdd(
    train_ui_rdd,
    num_items: int,
    c0: float,
    alpha: float,
    num_partitions: int | None = None,
):
    """Compute both dense c_array and item->c_i RDD from train interactions."""
    item_counts_rdd = train_ui_rdd.map(lambda ui: (ui[1], 1))
    if num_partitions:
        item_counts_rdd = item_counts_rdd.reduceByKey(add, num_partitions)
    else:
        item_counts_rdd = item_counts_rdd.reduceByKey(add)
    counts_by_item = dict(item_counts_rdd.collect())
    c_array = compute_c_from_item_counts(counts_by_item, num_items=num_items, c0=c0, alpha=alpha)
    c_item_rdd = item_counts_rdd.context.parallelize(
        [(item_idx, float(c_array[item_idx])) for item_idx in counts_by_item.keys()],
        numSlices=item_counts_rdd.getNumPartitions(),
    )
    return c_array, c_item_rdd


def build_histories(
    train_ui_rdd,
    c_item_rdd,
    storage_level_name: str = "MEMORY_AND_DISK",
    num_partitions: int | None = None,
):
    """Build user-history and item-history RDDs used by mapPartitions updates."""
    storage_level = storage_level_from_name(storage_level_name)
    join_partitions = int(num_partitions) if num_partitions else None

    # (item, (user, c_i)) -> (user, (item, c_i))
    user_item_rdd = train_ui_rdd.map(lambda ui: (ui[1], ui[0]))
    if join_partitions:
        user_item_rdd = user_item_rdd.partitionBy(join_partitions)
    user_history_rdd = (
        user_item_rdd.join(c_item_rdd, join_partitions)
        .map(lambda kv: (kv[1][0], (kv[0], kv[1][1])))
        .groupByKey(join_partitions)
        .mapValues(lambda values: list(values))
        .persist(storage_level)
    )

    # (item, [users]) joined with c_i -> (item, c_i, [users])
    item_user_rdd = train_ui_rdd.map(lambda ui: (ui[1], ui[0]))
    if join_partitions:
        item_user_rdd = item_user_rdd.partitionBy(join_partitions)
    item_history_rdd = (
        item_user_rdd.groupByKey(join_partitions)
        .mapValues(lambda users: list(users))
        .join(c_item_rdd, join_partitions)
        .map(lambda kv: (kv[0], kv[1][1], kv[1][0]))
        .persist(storage_level)
    )
    return user_history_rdd, item_history_rdd


def parse_record_block(block: str) -> tuple[str, str, int] | None:
    """Parse one Amazon review block into (user_raw, item_raw, timestamp)."""
    user_id: str | None = None
    product_id: str | None = None
    timestamp: int | None = None
    for line in block.splitlines():
        if line.startswith(PRODUCT_PREFIX):
            product_id = line[len(PRODUCT_PREFIX) :].strip()
        elif line.startswith(USER_PREFIX):
            user_id = line[len(USER_PREFIX) :].strip()
        elif line.startswith(TIME_PREFIX):
            try:
                timestamp = int(line[len(TIME_PREFIX) :].strip())
            except ValueError:
                return None
    if not user_id or not product_id or timestamp is None:
        return None
    return user_id, product_id, timestamp


def _set_record_delimiter(sc) -> None:
    # Amazon raw file has one review record separated by a blank line.
    sc._jsc.hadoopConfiguration().set("textinputformat.record.delimiter", "\n\n")


def load_interactions_rdd(sc, dataset_path: str, num_partitions: int | None = None):
    """Load raw interactions from Amazon review dumps via record-level input."""
    _set_record_delimiter(sc)
    raw_blocks = sc.newAPIHadoopFile(
        dataset_path,
        "org.apache.hadoop.mapreduce.lib.input.TextInputFormat",
        "org.apache.hadoop.io.LongWritable",
        "org.apache.hadoop.io.Text",
    ).map(lambda kv: str(kv[1]))
    if num_partitions:
        raw_blocks = raw_blocks.repartition(num_partitions)
    return raw_blocks.map(parse_record_block).filter(lambda row: row is not None)


def deduplicate_interactions(interactions_rdd, num_partitions: int | None = None):
    """Binary implicit feedback: keep latest timestamp for each (user, item)."""
    by_pair = interactions_rdd.map(lambda x: ((x[0], x[1]), x[2]))
    if num_partitions:
        by_pair = by_pair.reduceByKey(max, num_partitions)
    else:
        by_pair = by_pair.reduceByKey(max)
    return by_pair.map(lambda kv: (kv[0][0], kv[0][1], kv[1]))


def _filter_by_user_min_count(interactions_rdd, min_count: int, num_partitions: int | None = None):
    user_counts = interactions_rdd.map(lambda x: (x[0], 1))
    if num_partitions:
        user_counts = user_counts.reduceByKey(add, num_partitions)
    else:
        user_counts = user_counts.reduceByKey(add)

    valid_users = user_counts.filter(lambda kv: kv[1] >= min_count).keys().map(lambda user: (user, 1))
    user_events = interactions_rdd.map(lambda x: (x[0], (x[1], x[2])))
    if num_partitions:
        user_events = user_events.partitionBy(num_partitions)
        return user_events.join(valid_users, num_partitions).map(
            lambda kv: (kv[0], kv[1][0][0], kv[1][0][1])
        )
    return user_events.join(valid_users).map(lambda kv: (kv[0], kv[1][0][0], kv[1][0][1]))


def _filter_by_item_min_count(interactions_rdd, min_count: int, num_partitions: int | None = None):
    item_counts = interactions_rdd.map(lambda x: (x[1], 1))
    if num_partitions:
        item_counts = item_counts.reduceByKey(add, num_partitions)
    else:
        item_counts = item_counts.reduceByKey(add)

    valid_items = item_counts.filter(lambda kv: kv[1] >= min_count).keys().map(lambda item: (item, 1))
    item_events = interactions_rdd.map(lambda x: (x[1], (x[0], x[2])))
    if num_partitions:
        item_events = item_events.partitionBy(num_partitions)
        return item_events.join(valid_items, num_partitions).map(
            lambda kv: (kv[1][0][0], kv[0], kv[1][0][1])
        )
    return item_events.join(valid_items).map(lambda kv: (kv[1][0][0], kv[0], kv[1][0][1]))


def iterative_k_core_filter(
    interactions_rdd,
    min_user_count: int,
    min_item_count: int,
    storage_level_name: str = "MEMORY_AND_DISK",
    num_partitions: int | None = None,
    max_rounds: int = 30,
):
    """Iteratively enforce user/item interaction minimum counts until stable."""
    storage_level = storage_level_from_name(storage_level_name)
    current = interactions_rdd.persist(storage_level)
    prev_count = current.count()

    for _ in range(max_rounds):
        after_user = _filter_by_user_min_count(
            current, min_user_count, num_partitions=num_partitions
        ).persist(storage_level)
        current.unpersist()
        after_item = _filter_by_item_min_count(
            after_user, min_item_count, num_partitions=num_partitions
        ).persist(storage_level)
        after_user.unpersist()

        new_count = after_item.count()
        if new_count == prev_count:
            return after_item
        current = after_item
        prev_count = new_count

    return current


def iterative_k_core_filter_local(
    interactions: Iterable[tuple[str, str, int]],
    min_user_count: int,
    min_item_count: int,
) -> list[tuple[str, str, int]]:
    """Local reference implementation used in unit tests."""
    current = list(interactions)
    while True:
        user_counts = Counter(u for u, _, _ in current)
        keep_users = {u for u, c in user_counts.items() if c >= min_user_count}
        step = [row for row in current if row[0] in keep_users]

        item_counts = Counter(i for _, i, _ in step)
        keep_items = {i for i, c in item_counts.items() if c >= min_item_count}
        step = [row for row in step if row[1] in keep_items]

        if len(step) == len(current):
            return step
        current = step


def split_user_leave_one_out(events: list[tuple[int, int]]) -> tuple[list[int], int]:
    """Sort by (timestamp, item_idx), hold out latest item."""
    ordered = sorted(events, key=lambda x: (x[1], x[0]))
    if not ordered:
        raise ValueError("User has no events.")
    if len(ordered) == 1:
        return [], ordered[-1][0]
    train_items = [item_idx for item_idx, _ in ordered[:-1]]
    test_item = ordered[-1][0]
    return train_items, test_item


def _collect_ids_by_index(index_rdd) -> list[str]:
    pairs = index_rdd.map(lambda kv: (int(kv[1]), kv[0])).collect()
    ordered_ids = [""] * len(pairs)
    for idx, raw_id in pairs:
        ordered_ids[idx] = raw_id
    return ordered_ids


def _build_indexed_interactions(
    interactions_rdd,
    storage_level: StorageLevel,
    num_partitions: int | None = None,
):
    sort_partitions = int(num_partitions) if num_partitions else None

    users_rdd = interactions_rdd.map(lambda x: x[0])
    items_rdd = interactions_rdd.map(lambda x: x[1])
    if sort_partitions:
        users_rdd = users_rdd.distinct(sort_partitions)
        items_rdd = items_rdd.distinct(sort_partitions)
    else:
        users_rdd = users_rdd.distinct()
        items_rdd = items_rdd.distinct()

    user_index_rdd = (
        users_rdd.sortBy(lambda user: user, numPartitions=sort_partitions)
        .zipWithIndex()
        .map(lambda kv: (kv[0], int(kv[1])))
        .persist(storage_level)
    )
    item_index_rdd = (
        items_rdd.sortBy(lambda item: item, numPartitions=sort_partitions)
        .zipWithIndex()
        .map(lambda kv: (kv[0], int(kv[1])))
        .persist(storage_level)
    )

    user_join_rdd = interactions_rdd.map(lambda x: (x[0], (x[1], x[2])))
    if sort_partitions:
        user_join_rdd = user_join_rdd.partitionBy(sort_partitions)
        with_user_idx = user_join_rdd.join(user_index_rdd, sort_partitions)
    else:
        with_user_idx = user_join_rdd.join(user_index_rdd)

    item_join_rdd = with_user_idx.map(lambda kv: (kv[1][0][0], (kv[1][1], kv[1][0][1])))
    if sort_partitions:
        item_join_rdd = item_join_rdd.partitionBy(sort_partitions)
        indexed_rdd = (
            item_join_rdd.join(item_index_rdd, sort_partitions)
            .map(lambda kv: (kv[1][0][0], kv[1][1], kv[1][0][1]))
            .persist(storage_level)
        )
    else:
        indexed_rdd = (
            item_join_rdd.join(item_index_rdd)
            .map(lambda kv: (kv[1][0][0], kv[1][1], kv[1][0][1]))
            .persist(storage_level)
        )
    return indexed_rdd, user_index_rdd, item_index_rdd


def _build_train_and_test(
    indexed_rdd,
    storage_level: StorageLevel,
    num_partitions: int | None = None,
):
    split_rdd = (
        indexed_rdd.map(lambda x: (x[0], (x[1], x[2])))
        .groupByKey(num_partitions)
        .mapValues(lambda rows: split_user_leave_one_out(list(rows)))
        .persist(storage_level)
    )
    train_ui_rdd = (
        split_rdd.flatMap(lambda kv: ((kv[0], item_idx) for item_idx in kv[1][0])).persist(storage_level)
    )
    test_rdd = (
        split_rdd.map(lambda kv: (kv[0], kv[1][1], frozenset(kv[1][0]))).persist(storage_level)
    )
    return train_ui_rdd, test_rdd, split_rdd


def _resolve_num_partitions(sc, requested_partitions: Any) -> int:
    if requested_partitions is not None:
        partitions = int(requested_partitions)
        if partitions <= 0:
            raise ValueError("runtime_config['num_partitions'] must be a positive integer.")
        return partitions

    # Avoid massive default parallelism on shared clusters generating too many shuffle files.
    default_parallelism = max(int(sc.defaultParallelism), 1)
    return min(max(default_parallelism * 2, 128), 512)


def prepare_data(
    sc,
    dataset_path: str,
    eals_config: Any,
    runtime_config: Any,
    verbose: bool = True,
) -> PreparedData:
    def _log(message: str) -> None:
        if verbose:
            print(message)

    eals = _as_config_dict(eals_config)
    runtime = _as_config_dict(runtime_config)

    stats: dict[str, float | int] = {}
    prep_start = time.time()
    storage_level = storage_level_from_name(runtime["storage_level"])
    preprocess_storage_level_name = str(runtime.get("preprocess_storage_level", "DISK_ONLY"))
    preprocess_storage_level = storage_level_from_name(preprocess_storage_level_name)
    num_partitions = _resolve_num_partitions(sc, runtime.get("num_partitions"))
    stats["num_partitions"] = int(num_partitions)
    _log(
        "[prepare] using preprocess storage level "
        f"{preprocess_storage_level_name} with {num_partitions} partitions"
    )
    if runtime["checkpoint_dir"]:
        sc.setCheckpointDir(runtime["checkpoint_dir"])

    stage_start = time.time()
    interactions_rdd = load_interactions_rdd(
        sc,
        dataset_path,
        num_partitions=num_partitions,
    ).persist(preprocess_storage_level)
    raw_interactions = interactions_rdd.count()
    stats["raw_interactions"] = raw_interactions
    stats["load_seconds"] = time.time() - stage_start
    _log(f"[prepare] loaded interactions: {raw_interactions}")

    stage_start = time.time()
    deduped_rdd = deduplicate_interactions(
        interactions_rdd,
        num_partitions=num_partitions,
    ).persist(preprocess_storage_level)
    deduped_count = deduped_rdd.count()
    stats["deduped_interactions"] = deduped_count
    stats["dedupe_seconds"] = time.time() - stage_start
    _log(f"[prepare] deduped interactions: {deduped_count}")
    interactions_rdd.unpersist()

    stage_start = time.time()
    filtered_rdd = iterative_k_core_filter(
        deduped_rdd,
        min_user_count=eals["min_user_interactions"],
        min_item_count=eals["min_item_interactions"],
        storage_level_name=preprocess_storage_level_name,
        num_partitions=num_partitions,
    ).persist(preprocess_storage_level)
    filtered_count = filtered_rdd.count()
    stats["filtered_interactions"] = filtered_count
    stats["kcore_seconds"] = time.time() - stage_start
    _log(f"[prepare] k-core filtered interactions: {filtered_count}")
    deduped_rdd.unpersist()

    stage_start = time.time()
    indexed_rdd, user_index_rdd, item_index_rdd = _build_indexed_interactions(
        filtered_rdd,
        preprocess_storage_level,
        num_partitions=num_partitions,
    )
    num_users = user_index_rdd.count()
    num_items = item_index_rdd.count()
    indexed_count = indexed_rdd.count()
    stats["num_users"] = num_users
    stats["num_items"] = num_items
    stats["indexed_interactions"] = indexed_count
    stats["index_seconds"] = time.time() - stage_start
    _log(f"[prepare] indexed users/items: {num_users}/{num_items}")
    filtered_rdd.unpersist()

    stage_start = time.time()
    train_ui_rdd, test_rdd, split_rdd = _build_train_and_test(
        indexed_rdd,
        preprocess_storage_level,
        num_partitions=num_partitions,
    )
    num_train_interactions = train_ui_rdd.count()
    num_test_users = test_rdd.count()
    stats["train_interactions"] = num_train_interactions
    stats["test_users"] = num_test_users
    stats["split_seconds"] = time.time() - stage_start
    _log(f"[prepare] train interactions: {num_train_interactions}, test users: {num_test_users}")
    split_rdd.unpersist()
    indexed_rdd.unpersist()

    stage_start = time.time()
    c_array, c_item_rdd = compute_item_weights_rdd(
        train_ui_rdd=train_ui_rdd,
        num_items=num_items,
        c0=eals["c0"],
        alpha=eals["alpha"],
        num_partitions=num_partitions,
    )
    c_item_rdd = c_item_rdd.persist(storage_level)
    weighted_item_count = c_item_rdd.count()
    stats["weighted_items"] = weighted_item_count
    stats["weights_seconds"] = time.time() - stage_start
    _log(f"[prepare] weighted items: {weighted_item_count}")

    stage_start = time.time()
    user_history_rdd, item_history_rdd = build_histories(
        train_ui_rdd=train_ui_rdd,
        c_item_rdd=c_item_rdd,
        storage_level_name=runtime["storage_level"],
        num_partitions=num_partitions,
    )
    user_history_count = user_history_rdd.count()
    item_history_count = item_history_rdd.count()
    stats["user_histories"] = user_history_count
    stats["item_histories"] = item_history_count
    stats["history_seconds"] = time.time() - stage_start
    _log(f"[prepare] user/item histories: {user_history_count}/{item_history_count}")
    c_item_rdd.unpersist()

    stage_start = time.time()
    user_ids = _collect_ids_by_index(user_index_rdd)
    item_ids = _collect_ids_by_index(item_index_rdd)
    user_index_rdd.unpersist()
    item_index_rdd.unpersist()
    stats["collect_ids_seconds"] = time.time() - stage_start

    stats["total_prepare_seconds"] = time.time() - prep_start

    return PreparedData(
        num_users=num_users,
        num_items=num_items,
        train_ui_rdd=train_ui_rdd,
        test_rdd=test_rdd,
        user_history_rdd=user_history_rdd,
        item_history_rdd=item_history_rdd,
        c_array=c_array,
        user_ids=user_ids,
        item_ids=item_ids,
        num_train_interactions=num_train_interactions,
        num_test_users=num_test_users,
        stats=stats,
    )
