from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from output_utils import save_with_fallback
from simple_tsp import RESULT_DIR, generate_unique_points, save_animation, save_plot, save_points, save_route
from tsp_analysis import ALGORITHM_METADATA, evaluate_algorithms


SIZES = [40, 50, 60, 70, 80, 90, 100, 200]
ANIMATED_SIZES = {40, 100, 200}
ALGORITHM_LABELS = [item["label"] for item in ALGORITHM_METADATA]
ALGORITHM_COLORS = {
    "Random": "#94a3b8",
    "Nearest Neighbor": "#3b82f6",
    "Nearest Insertion": "#14b8a6",
    "Cheapest Insertion": "#f59e0b",
    "Farthest Insertion": "#ef4444",
    "Nearest Neighbor + 2-opt": "#8b5cf6",
    "Best Construction + 2-opt": "#22c55e",
}


def benchmark_one(size: int, seed: int, create_animation: bool) -> tuple[list[dict], dict]:
    points = generate_unique_points(size, seed)
    evaluation = evaluate_algorithms(points, seed + size)
    best_result = evaluation["best_result"]

    points_path = save_points(points, RESULT_DIR / f"cities_{size}.csv")
    route_path = save_route(
        best_result["route"],
        best_result["length"],
        RESULT_DIR / f"route_{size}.txt",
        best_result["label"],
    )
    image_path = save_plot(
        points,
        best_result["route"],
        best_result["length"],
        RESULT_DIR / f"tsp_{size}.png",
        best_result["label"],
    )
    animation_path = None
    if create_animation:
        animation_path = save_animation(
            points,
            best_result["route"],
            best_result["length"],
            RESULT_DIR / f"tsp_{size}.gif",
            best_result["label"],
        )

    raw_rows = []
    for algorithm_result in evaluation["algorithms"]:
        raw_rows.append(
            {
                "size": size,
                "point_seed": seed,
                "algorithm_seed": seed + size,
                "algorithm": algorithm_result["label"],
                "algorithm_key": algorithm_result["key"],
                "category": algorithm_result["category"],
                "distance": algorithm_result["length"],
                "time_ms": algorithm_result["time_ms"],
                "gap_to_best_pct": algorithm_result["gap_to_best_pct"],
                "improvement_vs_random_pct": algorithm_result["improvement_vs_random_pct"],
                "distance_rank": algorithm_result["distance_rank"],
                "runtime_rank": algorithm_result["runtime_rank"],
                "is_best": algorithm_result["is_best"],
                "base_algorithm": algorithm_result["base_algorithm"],
                "points_file": points_path.name,
                "route_file": route_path.name if algorithm_result["is_best"] else "",
                "image_file": image_path.name if algorithm_result["is_best"] else "",
                "animation_file": animation_path.name if algorithm_result["is_best"] and animation_path is not None else "",
            }
        )

    best_summary = {
        "size": size,
        "point_seed": seed,
        "best_algorithm": best_result["label"],
        "best_distance": best_result["length"],
        "best_time_ms": best_result["time_ms"],
        "best_construction_label": evaluation["best_construction_label"],
        "points_file": points_path.name,
        "route_file": route_path.name,
        "image_file": image_path.name,
        "animation_file": animation_path.name if animation_path is not None else "",
    }
    return raw_rows, best_summary


def _save_dataframe_csv(dataframe: pd.DataFrame, output_path: Path, label: str) -> Path:
    return save_with_fallback(
        output_path,
        lambda target_path: dataframe.to_csv(target_path, index=False, encoding="utf-8"),
        label,
    )


def _line_chart(
    raw_df: pd.DataFrame,
    value_column: str,
    output_path: Path,
    title: str,
    y_label: str,
) -> Path:
    pivot_df = (
        raw_df.pivot(index="size", columns="algorithm", values=value_column)
        .reindex(columns=ALGORITHM_LABELS)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5.6))
    for algorithm in pivot_df.columns:
        ax.plot(
            pivot_df.index,
            pivot_df[algorithm],
            marker="o",
            linewidth=2,
            label=algorithm,
            color=ALGORITHM_COLORS.get(algorithm),
        )

    ax.set_xlabel("Số địa điểm")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    saved_path = save_with_fallback(
        output_path,
        lambda target_path: fig.savefig(target_path, dpi=160),
        title,
    )
    plt.close(fig)
    return saved_path


def _tradeoff_chart(summary_df: pd.DataFrame, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in summary_df.iterrows():
        ax.scatter(
            row["avg_time_ms"],
            row["avg_gap_to_best_pct"],
            s=120,
            color=ALGORITHM_COLORS.get(row["algorithm"]),
        )
        ax.annotate(
            row["algorithm"],
            (row["avg_time_ms"], row["avg_gap_to_best_pct"]),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Trung bình thời gian (ms)")
    ax.set_ylabel("Trung bình gap so với lời giải tốt nhất (%)")
    ax.set_title("Trade-off giữa tốc độ và chất lượng")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    saved_path = save_with_fallback(
        output_path,
        lambda target_path: fig.savefig(target_path, dpi=160),
        "trade-off chart",
    )
    plt.close(fig)
    return saved_path


def _summary_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    return (
        raw_df.groupby(["algorithm", "category"], as_index=False)
        .agg(
            avg_distance=("distance", "mean"),
            avg_time_ms=("time_ms", "mean"),
            avg_gap_to_best_pct=("gap_to_best_pct", "mean"),
            avg_improvement_vs_random_pct=("improvement_vs_random_pct", "mean"),
            avg_distance_rank=("distance_rank", "mean"),
            avg_runtime_rank=("runtime_rank", "mean"),
            wins=("is_best", "sum"),
        )
        .sort_values(["avg_distance_rank", "avg_time_ms", "algorithm"])
        .reset_index(drop=True)
    )


def _format_for_report(dataframe: pd.DataFrame, float_columns: list[str]) -> pd.DataFrame:
    formatted = dataframe.copy()
    for column in float_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: f"{value:.4f}")
    if "wins" in formatted.columns:
        formatted["wins"] = formatted["wins"].astype(int)
    if "is_best" in formatted.columns:
        formatted["is_best"] = formatted["is_best"].map(lambda value: "Yes" if value else "")
    return formatted


def _rename_columns_for_report(dataframe: pd.DataFrame, rename_map: dict[str, str]) -> pd.DataFrame:
    return dataframe.rename(columns=rename_map)


def _write_excel(raw_df: pd.DataFrame, summary_df: pd.DataFrame, best_df: pd.DataFrame, output_path: Path) -> Path:
    def writer(target_path: Path) -> None:
        with pd.ExcelWriter(target_path) as excel_writer:
            summary_df.to_excel(excel_writer, sheet_name="summary", index=False)
            best_df.to_excel(excel_writer, sheet_name="best_by_size", index=False)
            raw_df.to_excel(excel_writer, sheet_name="raw_results", index=False)

    return save_with_fallback(output_path, writer, "algorithm workbook")


def _write_report(
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    best_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    summary_report = _format_for_report(
        summary_df,
        [
            "avg_distance",
            "avg_time_ms",
            "avg_gap_to_best_pct",
            "avg_improvement_vs_random_pct",
            "avg_distance_rank",
            "avg_runtime_rank",
        ],
    )
    best_report = _format_for_report(best_df, ["best_distance", "best_time_ms"])
    raw_report = _format_for_report(raw_df, ["distance", "time_ms", "gap_to_best_pct", "improvement_vs_random_pct"])
    summary_report = _rename_columns_for_report(
        summary_report,
        {
            "algorithm": "Thuật toán",
            "category": "Nhóm",
            "avg_distance": "Độ dài TB",
            "avg_time_ms": "Thời gian TB (ms)",
            "avg_gap_to_best_pct": "Gap TB (%)",
            "avg_improvement_vs_random_pct": "Cải thiện TB so với Random (%)",
            "avg_distance_rank": "Hạng chất lượng TB",
            "avg_runtime_rank": "Hạng tốc độ TB",
            "wins": "Số lần tốt nhất",
        },
    )
    best_report = _rename_columns_for_report(
        best_report,
        {
            "size": "Số điểm",
            "point_seed": "Seed điểm",
            "best_algorithm": "Thuật toán tốt nhất",
            "best_distance": "Độ dài tốt nhất",
            "best_time_ms": "Thời gian tốt nhất (ms)",
            "best_construction_label": "Thuật toán dựng tour gốc",
            "points_file": "File tọa độ",
            "route_file": "File route",
            "image_file": "Ảnh",
            "animation_file": "GIF",
        },
    )
    raw_report = _rename_columns_for_report(
        raw_report,
        {
            "size": "Số điểm",
            "point_seed": "Seed điểm",
            "algorithm_seed": "Seed thuật toán",
            "algorithm": "Thuật toán",
            "algorithm_key": "Khóa",
            "category": "Nhóm",
            "distance": "Độ dài",
            "time_ms": "Thời gian (ms)",
            "gap_to_best_pct": "Gap so với tốt nhất (%)",
            "improvement_vs_random_pct": "Cải thiện so với Random (%)",
            "distance_rank": "Hạng chất lượng",
            "runtime_rank": "Hạng tốc độ",
            "is_best": "Tốt nhất",
            "base_algorithm": "Tour gốc trước 2-opt",
            "points_file": "File tọa độ",
            "route_file": "File route",
            "image_file": "Ảnh",
            "animation_file": "GIF",
        },
    )

    report_html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <title>Phân tích thuật toán TSP</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f5f7fb; color: #1f2937; }}
    h1, h2, h3 {{ margin: 0 0 12px 0; }}
    p, li {{ line-height: 1.55; }}
    .card {{ background: white; border: 1px solid #d6deeb; border-radius: 14px; padding: 18px; margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .grid3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 18px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #d6deeb; padding: 8px 10px; text-align: left; }}
    th {{ background: #e9f1ff; }}
    img {{ max-width: 100%; border: 1px solid #d6deeb; border-radius: 10px; background: white; }}
    code {{ background: #eef2ff; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Phân tích thuật toán TSP</h1>
    <p>Bộ benchmark này mở rộng từ project demo thành một bài toán phân tích thuật toán rõ ràng hơn. Thay vì chỉ có <code>Random</code>, <code>Nearest Neighbor</code> và <code>Nearest Neighbor + 2-opt</code>, chương trình hiện so sánh thêm <code>Nearest Insertion</code>, <code>Cheapest Insertion</code>, <code>Farthest Insertion</code> và pipeline lai <code>Best Construction + 2-opt</code>.</p>
    <p>Mục tiêu là giữ tinh thần của một bài báo thực nghiệm: mô tả thuật toán, chạy benchmark trên nhiều kích thước, đo thời gian và chất lượng, sau đó tổng hợp bảng và biểu đồ để dễ đọc kết quả.</p>
  </div>

  <div class="card">
    <h2>Thiết lập thí nghiệm</h2>
    <ul>
      <li>Kích thước bài toán: 40, 50, 60, 70, 80, 90, 100 và 200 điểm.</li>
      <li>Dữ liệu được sinh lại bằng seed cố định để có thể lặp lại.</li>
      <li>Chỉ số đo: tổng độ dài tour, thời gian chạy, gap so với lời giải tốt nhất và mức cải thiện so với random.</li>
      <li>Ảnh tour và ảnh động được lưu theo thuật toán tốt nhất trên các kích thước chính.</li>
    </ul>
  </div>

  <div class="card">
    <h2>Tổng hợp theo thuật toán</h2>
    {summary_report.to_html(index=False, escape=False)}
  </div>

  <div class="card">
    <h2>Kết quả tốt nhất theo kích thước</h2>
    {best_report.to_html(index=False, escape=False)}
  </div>

  <div class="grid">
    <div class="card">
      <h2>So sánh thời gian</h2>
      <img src="runtime_comparison.png" alt="Runtime comparison">
    </div>
    <div class="card">
      <h2>So sánh độ dài tour</h2>
      <img src="distance_comparison.png" alt="Distance comparison">
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Gap so với lời giải tốt nhất</h2>
      <img src="gap_comparison.png" alt="Gap comparison">
    </div>
    <div class="card">
      <h2>Trade-off tốc độ và chất lượng</h2>
      <img src="time_quality_tradeoff.png" alt="Tradeoff chart">
    </div>
  </div>

  <div class="grid3">
    <div class="card">
      <h3>Tour 40 điểm</h3>
      <img src="tsp_40.png" alt="TSP 40">
    </div>
    <div class="card">
      <h3>Tour 100 điểm</h3>
      <img src="tsp_100.png" alt="TSP 100">
    </div>
    <div class="card">
      <h3>Tour 200 điểm</h3>
      <img src="tsp_200.png" alt="TSP 200">
    </div>
  </div>

  <div class="grid3">
    <div class="card">
      <h3>Đường đi động 40 điểm</h3>
      <img src="tsp_40.gif" alt="TSP 40 animation">
    </div>
    <div class="card">
      <h3>Đường đi động 100 điểm</h3>
      <img src="tsp_100.gif" alt="TSP 100 animation">
    </div>
    <div class="card">
      <h3>Đường đi động 200 điểm</h3>
      <img src="tsp_200.gif" alt="TSP 200 animation">
    </div>
  </div>

  <div class="card">
    <h2>Bảng dữ liệu đầy đủ</h2>
    {raw_report.to_html(index=False, escape=False)}
  </div>

  <div class="card">
    <h2>File đầu ra</h2>
    <p><a href="algorithm_summary.csv">algorithm_summary.csv</a>, <a href="algorithm_benchmark.csv">algorithm_benchmark.csv</a>, <a href="algorithm_benchmark_raw.csv">algorithm_benchmark_raw.csv</a> và <a href="algorithm_tables.xlsx">algorithm_tables.xlsx</a> chứa dữ liệu tổng hợp. File <a href="algorithm_tables.html">algorithm_tables.html</a> chứa các bảng tách riêng để xem nhanh.</p>
  </div>
</body>
</html>
"""

    return save_with_fallback(
        output_path,
        lambda target_path: target_path.write_text(report_html, encoding="utf-8"),
        "algorithm report",
    )


def _write_tables_html(summary_df: pd.DataFrame, best_df: pd.DataFrame, raw_df: pd.DataFrame, output_path: Path) -> Path:
    summary_report = _format_for_report(
        summary_df,
        [
            "avg_distance",
            "avg_time_ms",
            "avg_gap_to_best_pct",
            "avg_improvement_vs_random_pct",
            "avg_distance_rank",
            "avg_runtime_rank",
        ],
    )
    best_report = _format_for_report(best_df, ["best_distance", "best_time_ms"])
    raw_report = _format_for_report(raw_df, ["distance", "time_ms", "gap_to_best_pct", "improvement_vs_random_pct"])
    summary_report = _rename_columns_for_report(
        summary_report,
        {
            "algorithm": "Thuật toán",
            "category": "Nhóm",
            "avg_distance": "Độ dài TB",
            "avg_time_ms": "Thời gian TB (ms)",
            "avg_gap_to_best_pct": "Gap TB (%)",
            "avg_improvement_vs_random_pct": "Cải thiện TB so với Random (%)",
            "avg_distance_rank": "Hạng chất lượng TB",
            "avg_runtime_rank": "Hạng tốc độ TB",
            "wins": "Số lần tốt nhất",
        },
    )
    best_report = _rename_columns_for_report(
        best_report,
        {
            "size": "Số điểm",
            "point_seed": "Seed điểm",
            "best_algorithm": "Thuật toán tốt nhất",
            "best_distance": "Độ dài tốt nhất",
            "best_time_ms": "Thời gian tốt nhất (ms)",
            "best_construction_label": "Thuật toán dựng tour gốc",
            "points_file": "File tọa độ",
            "route_file": "File route",
            "image_file": "Ảnh",
            "animation_file": "GIF",
        },
    )
    raw_report = _rename_columns_for_report(
        raw_report,
        {
            "size": "Số điểm",
            "point_seed": "Seed điểm",
            "algorithm_seed": "Seed thuật toán",
            "algorithm": "Thuật toán",
            "algorithm_key": "Khóa",
            "category": "Nhóm",
            "distance": "Độ dài",
            "time_ms": "Thời gian (ms)",
            "gap_to_best_pct": "Gap so với tốt nhất (%)",
            "improvement_vs_random_pct": "Cải thiện so với Random (%)",
            "distance_rank": "Hạng chất lượng",
            "runtime_rank": "Hạng tốc độ",
            "is_best": "Tốt nhất",
            "base_algorithm": "Tour gốc trước 2-opt",
            "points_file": "File tọa độ",
            "route_file": "File route",
            "image_file": "Ảnh",
            "animation_file": "GIF",
        },
    )

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <title>Bảng tổng hợp TSP</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; }}
    th {{ background: #e5eefc; }}
  </style>
</head>
<body>
  <h1>Bảng tổng hợp phân tích thuật toán TSP</h1>
  <h2>Summary</h2>
  {summary_report.to_html(index=False, escape=False)}
  <h2>Best by size</h2>
  {best_report.to_html(index=False, escape=False)}
  <h2>Raw results</h2>
  {raw_report.to_html(index=False, escape=False)}
</body>
</html>
"""
    return save_with_fallback(
        output_path,
        lambda target_path: target_path.write_text(html_content, encoding="utf-8"),
        "algorithm tables html",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Chạy benchmark TSP và tạo báo cáo so sánh thuật toán.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RESULT_DIR.mkdir(exist_ok=True)

    raw_rows = []
    best_rows = []
    for size in SIZES:
        size_rows, best_row = benchmark_one(size, args.seed, size in ANIMATED_SIZES)
        raw_rows.extend(size_rows)
        best_rows.append(best_row)
        print(
            f"size={best_row['size']} "
            f"best_algorithm={best_row['best_algorithm']} "
            f"best_distance={best_row['best_distance']:.6f} "
            f"best_time_ms={best_row['best_time_ms']:.2f}"
        )

    raw_df = pd.DataFrame(raw_rows)
    summary_df = _summary_table(raw_df)
    best_df = pd.DataFrame(best_rows)

    _save_dataframe_csv(best_df, RESULT_DIR / "algorithm_benchmark.csv", "benchmark summary CSV")
    _save_dataframe_csv(raw_df, RESULT_DIR / "algorithm_benchmark_raw.csv", "benchmark raw CSV")
    _save_dataframe_csv(summary_df, RESULT_DIR / "algorithm_summary.csv", "benchmark summary by algorithm CSV")
    _line_chart(raw_df, "time_ms", RESULT_DIR / "runtime_comparison.png", "So sánh thời gian chạy", "Thời gian (ms)")
    _line_chart(raw_df, "distance", RESULT_DIR / "distance_comparison.png", "So sánh tổng độ dài tour", "Tổng độ dài")
    _line_chart(raw_df, "gap_to_best_pct", RESULT_DIR / "gap_comparison.png", "Gap so với lời giải tốt nhất", "Gap (%)")
    _tradeoff_chart(summary_df, RESULT_DIR / "time_quality_tradeoff.png")
    _write_excel(raw_df, summary_df, best_df, RESULT_DIR / "algorithm_tables.xlsx")
    _write_tables_html(summary_df, best_df, raw_df, RESULT_DIR / "algorithm_tables.html")
    _write_report(raw_df, summary_df, best_df, RESULT_DIR / "algorithm_report.html")

    print(f"report={RESULT_DIR / 'algorithm_report.html'}")
    print(f"runtime_chart={RESULT_DIR / 'runtime_comparison.png'}")
    print(f"distance_chart={RESULT_DIR / 'distance_comparison.png'}")
    print(f"gap_chart={RESULT_DIR / 'gap_comparison.png'}")
    print(f"tradeoff_chart={RESULT_DIR / 'time_quality_tradeoff.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
