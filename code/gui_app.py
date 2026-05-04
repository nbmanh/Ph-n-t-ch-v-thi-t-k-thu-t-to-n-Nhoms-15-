import argparse
import csv
import os
import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageSequence, ImageTk

from output_utils import save_with_fallback
from simple_tsp import RESULT_DIR, save_animation, save_points, save_route
from tsp_analysis import ALGORITHM_METADATA, evaluate_algorithms


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
DEFAULT_WINDOW_WIDTH = 1460
DEFAULT_WINDOW_HEIGHT = 920
MIN_WINDOW_WIDTH = 1000
MIN_WINDOW_HEIGHT = 620
GIF_PREVIEW_WIDTH = 880
GIF_PREVIEW_HEIGHT = 560
RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def load_points_from_file(file_path: Path) -> np.ndarray:
    rows = []

    if file_path.suffix.lower() == ".csv":
        with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                numbers = []
                for item in row:
                    item = item.strip()
                    if not item:
                        continue
                    try:
                        numbers.append(float(item))
                    except ValueError:
                        numbers = []
                        break
                if len(numbers) >= 2:
                    rows.append(numbers[-2:])
    else:
        with file_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                tokens = [token for token in re.split(r"[\s,;]+", line) if token]
                numbers = []
                for token in tokens:
                    try:
                        numbers.append(float(token))
                    except ValueError:
                        numbers = []
                        break
                if len(numbers) >= 2:
                    rows.append(numbers[-2:])

    if len(rows) < 3:
        raise ValueError("File phải có ít nhất 3 điểm tọa độ hợp lệ.")

    points = np.array(rows, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Không đọc được dữ liệu tọa độ dạng (x, y).")
    return points


def save_uploaded_outputs(points: np.ndarray, analysis: dict, input_path: Path) -> dict:
    RESULT_DIR.mkdir(exist_ok=True)
    stem = input_path.stem
    best_result = analysis["best_result"]

    points_path = save_points(points, RESULT_DIR / f"{stem}_points.csv")
    route_path = save_route(
        best_result["route"],
        best_result["length"],
        RESULT_DIR / f"{stem}_route.txt",
        best_result["label"],
    )

    route_figure = build_route_figure(points, analysis)
    route_image_path = save_with_fallback(
        RESULT_DIR / f"{stem}_route.png",
        lambda target_path: route_figure.savefig(target_path, dpi=160),
        "route image",
    )
    plt.close(route_figure)

    distance_figure = build_distance_figure(analysis)
    distance_chart_path = save_with_fallback(
        RESULT_DIR / f"{stem}_distance.png",
        lambda target_path: distance_figure.savefig(target_path, dpi=160),
        "distance chart",
    )
    plt.close(distance_figure)

    runtime_figure = build_runtime_figure(analysis)
    runtime_chart_path = save_with_fallback(
        RESULT_DIR / f"{stem}_runtime.png",
        lambda target_path: runtime_figure.savefig(target_path, dpi=160),
        "runtime chart",
    )
    plt.close(runtime_figure)

    animation_path = save_animation(
        points,
        best_result["route"],
        best_result["length"],
        RESULT_DIR / f"{stem}_route.gif",
        best_result["label"],
    )

    return {
        "points_path": points_path,
        "route_path": route_path,
        "route_image_path": route_image_path,
        "distance_chart_path": distance_chart_path,
        "runtime_chart_path": runtime_chart_path,
        "animation_path": animation_path,
    }


def build_route_figure(points: np.ndarray, analysis: dict) -> plt.Figure:
    best_result = analysis["best_result"]
    route = best_result["route"]
    closed_route = route + [route[0]]
    x_coords = [points[index][0] for index in closed_route]
    y_coords = [points[index][1] for index in closed_route]

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.scatter(points[:, 0], points[:, 1], c="tab:blue", s=40)
    ax.plot(x_coords, y_coords, c="tab:red", linewidth=1.8)

    for index, (x_coord, y_coord) in enumerate(points, start=1):
        ax.text(x_coord + 0.005, y_coord + 0.005, str(index), fontsize=8)

    ax.set_title(
        f"Đường đi tốt nhất ({analysis['point_count']} điểm)\n"
        f"{best_result['label']} = {best_result['length']:.4f}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def build_distance_figure(analysis: dict) -> plt.Figure:
    labels = [item["label"] for item in analysis["algorithms"]]
    values = [item["length"] for item in analysis["algorithms"]]
    colors = [ALGORITHM_COLORS.get(label, "#64748b") for label in labels]

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("So sánh độ dài tour")
    ax.set_ylabel("Tổng độ dài")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    return fig


def build_runtime_figure(analysis: dict) -> plt.Figure:
    labels = [item["label"] for item in analysis["algorithms"]]
    values = [item["time_ms"] for item in analysis["algorithms"]]
    colors = [ALGORITHM_COLORS.get(label, "#64748b") for label in labels]

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("So sánh thời gian chạy")
    ax.set_ylabel("ms")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    return fig


class TSPGuiApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Phân tích thuật toán TSP")
        self.root.withdraw()
        self._configure_window()

        self.file_path_var = tk.StringVar()
        self.seed_var = tk.StringVar(value="42")
        self.status_var = tk.StringVar(value="Chọn file tọa độ rồi bấm Chạy.")

        self.route_label = None
        self.route_gif_frames = []
        self.route_gif_durations = []
        self.route_gif_after_id = None
        self.route_gif_index = 0
        self.distance_canvas = None
        self.runtime_canvas = None

        self.root.protocol("WM_DELETE_WINDOW", self.close_window)
        self._build_layout()
        self.root.after(0, self._show_window)

    def _configure_window(self) -> None:
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        target_width = min(DEFAULT_WINDOW_WIDTH, max(MIN_WINDOW_WIDTH, screen_width - 80))
        target_height = min(DEFAULT_WINDOW_HEIGHT, max(MIN_WINDOW_HEIGHT, screen_height - 80))
        target_width = min(target_width, screen_width)
        target_height = min(target_height, screen_height)

        offset_x = max((screen_width - target_width) // 2, 0)
        offset_y = max((screen_height - target_height) // 2, 0)

        self.root.geometry(f"{target_width}x{target_height}+{offset_x}+{offset_y}")
        self.root.minsize(min(MIN_WINDOW_WIDTH, target_width), min(MIN_WINDOW_HEIGHT, target_height))

    def _show_window(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.focus_force()
        self.root.after(250, lambda: self.root.attributes("-topmost", False))

    def close_window(self) -> None:
        self._stop_route_gif()
        self.root.destroy()

    def _build_layout(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left_panel = ttk.Frame(self.root, padding=16)
        left_panel.grid(row=0, column=0, sticky="nsw")

        right_panel = ttk.Frame(self.root, padding=12)
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)

        ttk.Label(left_panel, text="Dữ liệu tọa độ", font=("Segoe UI", 15, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Entry(left_panel, textvariable=self.file_path_var, width=44).grid(row=1, column=0, pady=(10, 8), sticky="we")
        ttk.Button(left_panel, text="Chọn file", command=self.choose_file).grid(row=2, column=0, sticky="we")

        ttk.Label(left_panel, text="Seed ngẫu nhiên", font=("Segoe UI", 10, "bold")).grid(row=3, column=0, pady=(18, 6), sticky="w")
        ttk.Entry(left_panel, textvariable=self.seed_var, width=12).grid(row=4, column=0, sticky="w")

        ttk.Label(left_panel, text="GIF đường đi sẽ được tạo và phát tự động.").grid(row=5, column=0, pady=(14, 10), sticky="w")
        ttk.Button(left_panel, text="Chạy phân tích", command=self.run_analysis).grid(row=6, column=0, pady=(6, 8), sticky="we")
        ttk.Button(left_panel, text="Mở thư mục result", command=self.open_result_folder).grid(row=7, column=0, sticky="we")

        self.metrics_text = tk.Text(left_panel, width=46, height=28, wrap="word", font=("Consolas", 9))
        self.metrics_text.grid(row=8, column=0, pady=(16, 0), sticky="nsew")
        left_panel.rowconfigure(8, weight=1)

        ttk.Label(right_panel, textvariable=self.status_var, font=("Segoe UI", 11)).grid(row=0, column=0, sticky="w", pady=(0, 10))

        notebook = ttk.Notebook(right_panel)
        notebook.grid(row=1, column=0, sticky="nsew")

        self.route_tab = ttk.Frame(notebook)
        self.distance_tab = ttk.Frame(notebook)
        self.runtime_tab = ttk.Frame(notebook)

        for tab in [self.route_tab, self.distance_tab, self.runtime_tab]:
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)

        self.route_label = ttk.Label(self.route_tab, anchor="center")
        self.route_label.grid(row=0, column=0, sticky="nsew")

        notebook.add(self.route_tab, text="Đường đi tốt nhất")
        notebook.add(self.distance_tab, text="Khoảng cách")
        notebook.add(self.runtime_tab, text="Thời gian")

    def choose_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Chọn file tọa độ",
            filetypes=[
                ("Coordinate files", "*.csv *.txt"),
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            self.file_path_var.set(file_path)
            self.status_var.set("Đã chọn file. Bấm Chạy phân tích để so sánh thuật toán.")

    def run_analysis(self) -> None:
        raw_path = self.file_path_var.get().strip()
        if not raw_path:
            messagebox.showerror("Thiếu file", "Hãy chọn file tọa độ trước.")
            return

        try:
            seed = int(self.seed_var.get().strip())
        except ValueError:
            messagebox.showerror("Seed không hợp lệ", "Seed phải là số nguyên.")
            return

        file_path = Path(raw_path)
        if not file_path.exists():
            messagebox.showerror("Không tìm thấy file", f"Không tồn tại file:\n{file_path}")
            return

        try:
            points = load_points_from_file(file_path)
            analysis = evaluate_algorithms(points, seed)
            saved_outputs = save_uploaded_outputs(points, analysis, file_path)
        except Exception as exc:
            messagebox.showerror("Không thể chạy", str(exc))
            self.status_var.set("Chạy thất bại. Kiểm tra định dạng file.")
            return

        self.render_route_gif(saved_outputs["animation_path"])
        self.render_figure(self.distance_tab, build_distance_figure(analysis), "distance_canvas")
        self.render_figure(self.runtime_tab, build_runtime_figure(analysis), "runtime_canvas")
        self.render_metrics(file_path, analysis, saved_outputs)
        self.status_var.set("Đã phân tích xong. Biểu đồ và file kết quả đã lưu vào result/.")

    def open_result_folder(self) -> None:
        RESULT_DIR.mkdir(exist_ok=True)
        os.startfile(str(RESULT_DIR))

    def _stop_route_gif(self) -> None:
        if self.route_gif_after_id is not None:
            self.root.after_cancel(self.route_gif_after_id)
            self.route_gif_after_id = None

        if self.route_label is not None:
            self.route_label.configure(image="", text="")
            self.route_label.image = None

        self.route_gif_frames = []
        self.route_gif_durations = []
        self.route_gif_index = 0

    def _route_preview_size(self) -> tuple[int, int]:
        available_width = self.route_tab.winfo_width()
        available_height = self.route_tab.winfo_height()

        if available_width <= 1:
            available_width = self.root.winfo_width() - 420
        if available_height <= 1:
            available_height = self.root.winfo_height() - 140

        preview_width = max(360, min(GIF_PREVIEW_WIDTH, available_width - 20))
        preview_height = max(260, min(GIF_PREVIEW_HEIGHT, available_height - 20))
        return preview_width, preview_height

    def render_route_gif(self, gif_path: Path) -> None:
        if self.route_label is None:
            return

        self._stop_route_gif()
        self.root.update_idletasks()

        preview_width, preview_height = self._route_preview_size()
        frames = []
        durations = []

        with Image.open(gif_path) as gif_image:
            default_duration = max(int(gif_image.info.get("duration", 160)), 80)
            for frame in ImageSequence.Iterator(gif_image):
                frame_image = frame.convert("RGBA")
                frame_image.thumbnail((preview_width, preview_height), RESAMPLE_LANCZOS)
                frames.append(ImageTk.PhotoImage(frame_image))
                durations.append(max(int(frame.info.get("duration", default_duration)), 80))

        if not frames:
            raise ValueError(f"Không đọc được GIF đường đi: {gif_path}")

        self.route_gif_frames = frames
        self.route_gif_durations = durations
        self.route_gif_index = 0
        self.route_label.configure(image=self.route_gif_frames[0], text="")
        self.route_label.image = self.route_gif_frames[0]

        if len(self.route_gif_frames) > 1:
            self.route_gif_after_id = self.root.after(
                self.route_gif_durations[0],
                self._play_route_gif_frame,
            )

    def _play_route_gif_frame(self) -> None:
        if self.route_label is None or not self.route_gif_frames:
            return

        self.route_gif_index = (self.route_gif_index + 1) % len(self.route_gif_frames)
        current_frame = self.route_gif_frames[self.route_gif_index]
        self.route_label.configure(image=current_frame)
        self.route_label.image = current_frame
        self.route_gif_after_id = self.root.after(
            self.route_gif_durations[self.route_gif_index],
            self._play_route_gif_frame,
        )

    def render_figure(self, parent: ttk.Frame, figure: plt.Figure, canvas_attr: str) -> None:
        old_canvas = getattr(self, canvas_attr)
        if old_canvas is not None:
            old_canvas.get_tk_widget().destroy()

        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        setattr(self, canvas_attr, canvas)

    def render_metrics(self, file_path: Path, analysis: dict, saved_outputs: dict) -> None:
        self.metrics_text.delete("1.0", tk.END)
        best_result = analysis["best_result"]

        lines = [
            f"File: {file_path}",
            f"Số điểm: {analysis['point_count']}",
            f"Best algorithm: {best_result['label']}",
            f"Best construction seed: {analysis['seed']}",
            f"Best construction before 2-opt: {analysis['best_construction_label']}",
            "",
            "Bảng xếp hạng theo độ dài tour:",
        ]

        ranked_results = sorted(analysis["algorithms"], key=lambda item: (item["distance_rank"], item["time_ms"]))
        for result in ranked_results:
            lines.append(
                f"{result['distance_rank']}. {result['label']} | "
                f"distance={result['length']:.6f} | "
                f"time={result['time_ms']:.3f} ms | "
                f"gap={result['gap_to_best_pct']:.2f}% | "
                f"improve_vs_random={result['improvement_vs_random_pct']:.2f}%"
            )

        lines.extend(
            [
                "",
                "Thứ tự đường đi tốt nhất (1-based):",
                " ".join(str(index + 1) for index in best_result["route"]),
                "",
                f"Lưu tọa độ: {saved_outputs['points_path']}",
                f"Lưu route: {saved_outputs['route_path']}",
                f"Lưu ảnh route: {saved_outputs['route_image_path']}",
                f"Lưu biểu đồ distance: {saved_outputs['distance_chart_path']}",
                f"Lưu biểu đồ runtime: {saved_outputs['runtime_chart_path']}",
            ]
        )
        if saved_outputs["animation_path"] is not None:
            lines.append(f"Lưu GIF: {saved_outputs['animation_path']}")

        self.metrics_text.insert("1.0", "\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="GUI cho phép chọn file tọa độ và xem biểu đồ so sánh TSP.")
    parser.parse_args()

    RESULT_DIR.mkdir(exist_ok=True)
    root = tk.Tk()
    app = TSPGuiApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
