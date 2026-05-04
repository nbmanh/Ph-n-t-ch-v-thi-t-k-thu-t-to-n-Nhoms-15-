import argparse
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from output_utils import save_with_fallback


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "result"


def generate_unique_points(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = []
    seen = set()

    while len(points) < count:
        x_coord = int(rng.integers(0, 1000))
        y_coord = int(rng.integers(0, 1000))
        key = (x_coord, y_coord)
        if key in seen:
            continue
        seen.add(key)
        points.append((x_coord / 1000.0, y_coord / 1000.0))

    return np.array(points, dtype=float)


def build_distance_matrix(points: np.ndarray) -> np.ndarray:
    deltas = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=2))


def route_length(distances: np.ndarray, route: list[int]) -> float:
    total = 0.0
    for index in range(len(route)):
        total += float(distances[route[index], route[(index + 1) % len(route)]])
    return total


def random_route(city_count: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    route = list(range(city_count))
    rng.shuffle(route)
    return route


def nearest_neighbor(distances: np.ndarray, start_city: int = 0) -> list[int]:
    city_count = distances.shape[0]
    if city_count <= 1:
        return list(range(city_count))

    remaining = set(range(city_count))
    remaining.remove(start_city)
    route = [start_city]

    while remaining:
        current = route[-1]
        next_city = min(remaining, key=lambda candidate: float(distances[current, candidate]))
        route.append(next_city)
        remaining.remove(next_city)

    return route


def _extreme_pair(distances: np.ndarray, *, maximize: bool) -> tuple[int, int]:
    city_count = distances.shape[0]
    best_pair = (0, 1)
    best_value = float(distances[0, 1])

    for start in range(city_count - 1):
        for end in range(start + 1, city_count):
            value = float(distances[start, end])
            if (maximize and value > best_value) or (not maximize and value < best_value):
                best_pair = (start, end)
                best_value = value

    return best_pair


def _distance_to_route(distances: np.ndarray, route: list[int], city: int) -> float:
    return min(float(distances[city, route_city]) for route_city in route)


def _best_insertion(distances: np.ndarray, route: list[int], city: int) -> tuple[int, float]:
    best_index = len(route)
    best_delta = float("inf")

    for index in range(len(route)):
        left_city = route[index]
        right_city = route[(index + 1) % len(route)]
        delta = float(
            distances[left_city, city]
            + distances[city, right_city]
            - distances[left_city, right_city]
        )
        if delta < best_delta:
            best_delta = delta
            best_index = index + 1

    return best_index, best_delta


def nearest_insertion(distances: np.ndarray) -> list[int]:
    city_count = distances.shape[0]
    if city_count <= 2:
        return list(range(city_count))

    route = list(_extreme_pair(distances, maximize=False))
    remaining = set(range(city_count)) - set(route)

    while remaining:
        city = min(remaining, key=lambda candidate: _distance_to_route(distances, route, candidate))
        insert_index, _ = _best_insertion(distances, route, city)
        route.insert(insert_index, city)
        remaining.remove(city)

    return route


def farthest_insertion(distances: np.ndarray) -> list[int]:
    city_count = distances.shape[0]
    if city_count <= 2:
        return list(range(city_count))

    route = list(_extreme_pair(distances, maximize=True))
    remaining = set(range(city_count)) - set(route)

    while remaining:
        city = max(remaining, key=lambda candidate: _distance_to_route(distances, route, candidate))
        insert_index, _ = _best_insertion(distances, route, city)
        route.insert(insert_index, city)
        remaining.remove(city)

    return route


def cheapest_insertion(distances: np.ndarray) -> list[int]:
    city_count = distances.shape[0]
    if city_count <= 2:
        return list(range(city_count))

    route = list(_extreme_pair(distances, maximize=False))
    remaining = set(range(city_count)) - set(route)

    while remaining:
        best_city = None
        best_index = len(route)
        best_delta = float("inf")

        for city in remaining:
            insert_index, delta = _best_insertion(distances, route, city)
            if delta < best_delta:
                best_city = city
                best_index = insert_index
                best_delta = delta

        route.insert(best_index, best_city)
        remaining.remove(best_city)

    return route


def two_opt(distances: np.ndarray, route: list[int]) -> list[int]:
    improved = True
    best_route = route[:]
    city_count = len(best_route)

    while improved:
        improved = False
        for start in range(1, city_count - 2):
            for end in range(start + 1, city_count - 1):
                before_start = best_route[start - 1]
                start_city = best_route[start]
                end_city = best_route[end]
                after_end = best_route[(end + 1) % city_count]

                old_cost = float(distances[before_start, start_city] + distances[end_city, after_end])
                new_cost = float(distances[before_start, end_city] + distances[start_city, after_end])

                if new_cost + 1e-12 < old_cost:
                    best_route[start:end + 1] = reversed(best_route[start:end + 1])
                    improved = True
                    break
            if improved:
                break

    return best_route


def save_points(points: np.ndarray, output_path: Path) -> Path:
    def writer(target_path: Path) -> None:
        with target_path.open("w", encoding="utf-8") as handle:
            handle.write("city_id,x,y\n")
            for index, (x_coord, y_coord) in enumerate(points, start=1):
                handle.write(f"{index},{x_coord:.3f},{y_coord:.3f}\n")

    return save_with_fallback(output_path, writer, "points file")


def save_route(
    route: list[int],
    length: float,
    output_path: Path,
    algorithm_label: str | None = None,
) -> Path:
    def writer(target_path: Path) -> None:
        with target_path.open("w", encoding="utf-8") as handle:
            if algorithm_label is not None:
                handle.write(f"thuật_toán={algorithm_label}\n")
            handle.write(f"total_distance={length:.6f}\n")
            handle.write("route_1_based=" + " ".join(str(city + 1) for city in route) + "\n")

    return save_with_fallback(output_path, writer, "route file")


def figure_to_image(fig: plt.Figure) -> Image.Image:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=140)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGBA")
    frame = image.copy()
    image.close()
    buffer.close()
    return frame


def save_plot(
    points: np.ndarray,
    route: list[int],
    total_distance: float,
    output_path: Path,
    algorithm_label: str | None = None,
) -> Path:
    closed_route = route + [route[0]]
    x_coords = [points[index][0] for index in closed_route]
    y_coords = [points[index][1] for index in closed_route]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(points[:, 0], points[:, 1], c="tab:blue", s=24)
    ax.plot(x_coords, y_coords, c="tab:red", linewidth=1.5)

    for index, (x_coord, y_coord) in enumerate(points, start=1):
        ax.text(x_coord + 0.005, y_coord + 0.005, str(index), fontsize=7)

    title = f"Minh họa TSP với {len(points)} điểm\nTổng độ dài = {total_distance:.4f}"
    if algorithm_label:
        title = (
            f"Minh họa TSP với {len(points)} điểm\n"
            f"{algorithm_label} | Tổng độ dài = {total_distance:.4f}"
        )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    saved_path = save_with_fallback(
        output_path,
        lambda target_path: fig.savefig(target_path, dpi=160),
        "plot image",
    )
    plt.close(fig)
    return saved_path


def save_animation(
    points: np.ndarray,
    route: list[int],
    total_distance: float,
    output_path: Path,
    algorithm_label: str | None = None,
) -> Path:
    closed_route = route + [route[0]]
    frame_total = len(closed_route)
    frame_indices = list(range(1, frame_total + 1))
    if frame_total > 60:
        frame_indices = sorted(set(np.linspace(1, frame_total, num=60, dtype=int).tolist() + [frame_total]))

    frames = []
    for step in frame_indices:
        partial_route = closed_route[:step]
        x_coords = [points[index][0] for index in partial_route]
        y_coords = [points[index][1] for index in partial_route]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(points[:, 0], points[:, 1], c="tab:blue", s=24)
        if len(partial_route) > 1:
            ax.plot(x_coords, y_coords, c="tab:red", linewidth=1.5)

        current_index = partial_route[-1]
        ax.scatter(
            [points[current_index][0]],
            [points[current_index][1]],
            c="gold",
            s=90,
            edgecolors="black",
            zorder=3,
        )

        for index, (x_coord, y_coord) in enumerate(points, start=1):
            ax.text(x_coord + 0.005, y_coord + 0.005, str(index), fontsize=7)

        title = f"Ảnh động đường đi TSP với {len(points)} điểm"
        if algorithm_label:
            title = f"Ảnh động {algorithm_label} với {len(points)} điểm"
        ax.set_title(
            f"{title}\n"
            f"Bước {step}/{frame_total} | Tổng độ dài = {total_distance:.4f}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        frames.append(figure_to_image(fig))
        plt.close(fig)

    return save_with_fallback(
        output_path,
        lambda target_path: frames[0].save(
            target_path,
            save_all=True,
            append_images=frames[1:],
            duration=160,
            loop=0,
        ),
        "route animation",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Tạo và giải một ví dụ TSP đơn giản với 40-200 điểm.")
    parser.add_argument("--points", type=int, default=40, choices=[40, 50, 60, 70, 80, 90, 100, 200])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--animate", action="store_true")
    args = parser.parse_args()

    RESULT_DIR.mkdir(exist_ok=True)

    points = generate_unique_points(args.points, args.seed)
    distances = build_distance_matrix(points)
    route = nearest_neighbor(distances)
    route = two_opt(distances, route)
    total_distance = route_length(distances, route)

    points_path = save_points(points, RESULT_DIR / f"cities_{args.points}.csv")
    route_path = save_route(route, total_distance, RESULT_DIR / f"route_{args.points}.txt", "Nearest Neighbor + 2-opt")
    image_path = save_plot(points, route, total_distance, RESULT_DIR / f"tsp_{args.points}.png", "Nearest Neighbor + 2-opt")

    animation_path = None
    if args.animate:
        animation_path = save_animation(
            points,
            route,
            total_distance,
            RESULT_DIR / f"tsp_{args.points}.gif",
            "Nearest Neighbor + 2-opt",
        )

    print(f"cities={args.points}")
    print(f"total_distance={total_distance:.6f}")
    print(f"points_file={points_path}")
    print(f"route_file={route_path}")
    print(f"image_file={image_path}")
    if animation_path is not None:
        print(f"animation_file={animation_path}")
    print("route_1_based=" + " ".join(str(city + 1) for city in route))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
