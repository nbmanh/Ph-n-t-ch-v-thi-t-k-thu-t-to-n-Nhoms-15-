from __future__ import annotations

import time

import numpy as np

from simple_tsp import (
    build_distance_matrix,
    cheapest_insertion,
    farthest_insertion,
    nearest_insertion,
    nearest_neighbor,
    random_route,
    route_length,
    two_opt,
)


ALGORITHM_METADATA = [
    {"key": "random", "label": "Random", "category": "Baseline"},
    {"key": "nearest_neighbor", "label": "Nearest Neighbor", "category": "Construction"},
    {"key": "nearest_insertion", "label": "Nearest Insertion", "category": "Construction"},
    {"key": "cheapest_insertion", "label": "Cheapest Insertion", "category": "Construction"},
    {"key": "farthest_insertion", "label": "Farthest Insertion", "category": "Construction"},
    {"key": "nn_plus_two_opt", "label": "Nearest Neighbor + 2-opt", "category": "Hybrid"},
    {"key": "best_plus_two_opt", "label": "Best Construction + 2-opt", "category": "Hybrid"},
]
ALGORITHM_ORDER = [item["key"] for item in ALGORITHM_METADATA]
CONSTRUCTION_KEYS = [
    "nearest_neighbor",
    "nearest_insertion",
    "cheapest_insertion",
    "farthest_insertion",
]


def _measure_route(builder) -> tuple[list[int], float]:
    start = time.perf_counter()
    route = builder()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return route, elapsed_ms


def _rank_values(values_by_key: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values_by_key.items(), key=lambda item: item[1])
    return {key: index for index, (key, _) in enumerate(ordered, start=1)}


def _metadata_by_key(key: str) -> dict:
    return next(item for item in ALGORITHM_METADATA if item["key"] == key)


def evaluate_algorithms(points: np.ndarray, seed: int) -> dict:
    distances = build_distance_matrix(points)
    city_count = len(points)

    construction_builders = {
        "random": lambda: random_route(city_count, seed),
        "nearest_neighbor": lambda: nearest_neighbor(distances),
        "nearest_insertion": lambda: nearest_insertion(distances),
        "cheapest_insertion": lambda: cheapest_insertion(distances),
        "farthest_insertion": lambda: farthest_insertion(distances),
    }

    results_by_key: dict[str, dict] = {}
    for key, builder in construction_builders.items():
        route, elapsed_ms = _measure_route(builder)
        metadata = _metadata_by_key(key)
        results_by_key[key] = {
            "key": key,
            "label": metadata["label"],
            "category": metadata["category"],
            "route": route,
            "length": route_length(distances, route),
            "time_ms": elapsed_ms,
            "base_algorithm": "",
        }

    nn_two_opt_route, nn_two_opt_ms = _measure_route(
        lambda: two_opt(distances, results_by_key["nearest_neighbor"]["route"])
    )
    results_by_key["nn_plus_two_opt"] = {
        "key": "nn_plus_two_opt",
        "label": _metadata_by_key("nn_plus_two_opt")["label"],
        "category": _metadata_by_key("nn_plus_two_opt")["category"],
        "route": nn_two_opt_route,
        "length": route_length(distances, nn_two_opt_route),
        "time_ms": results_by_key["nearest_neighbor"]["time_ms"] + nn_two_opt_ms,
        "base_algorithm": "Nearest Neighbor",
    }

    best_construction_key = min(
        CONSTRUCTION_KEYS,
        key=lambda candidate: results_by_key[candidate]["length"],
    )
    best_construction = results_by_key[best_construction_key]
    best_two_opt_route, best_two_opt_ms = _measure_route(
        lambda: two_opt(distances, best_construction["route"])
    )
    results_by_key["best_plus_two_opt"] = {
        "key": "best_plus_two_opt",
        "label": _metadata_by_key("best_plus_two_opt")["label"],
        "category": _metadata_by_key("best_plus_two_opt")["category"],
        "route": best_two_opt_route,
        "length": route_length(distances, best_two_opt_route),
        "time_ms": best_construction["time_ms"] + best_two_opt_ms,
        "base_algorithm": best_construction["label"],
    }

    results = [results_by_key[key] for key in ALGORITHM_ORDER]
    best_length = min(result["length"] for result in results)
    random_length = results_by_key["random"]["length"]

    distance_ranks = _rank_values({result["key"]: result["length"] for result in results})
    runtime_ranks = _rank_values({result["key"]: result["time_ms"] for result in results})

    for result in results:
        result["gap_to_best_pct"] = ((result["length"] - best_length) / best_length * 100.0) if best_length > 0 else 0.0
        result["improvement_vs_random_pct"] = ((random_length - result["length"]) / random_length * 100.0) if random_length > 0 else 0.0
        result["distance_rank"] = distance_ranks[result["key"]]
        result["runtime_rank"] = runtime_ranks[result["key"]]
        result["is_best"] = abs(result["length"] - best_length) <= 1e-12

    best_result = min(results, key=lambda item: (item["length"], item["time_ms"]))
    return {
        "point_count": city_count,
        "seed": seed,
        "algorithms": results,
        "best_result": best_result,
        "best_construction_label": best_construction["label"],
    }
