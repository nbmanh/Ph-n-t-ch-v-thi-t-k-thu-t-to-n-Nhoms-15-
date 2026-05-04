import argparse
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from output_utils import save_with_fallback


def figure_to_image(fig: plt.Figure) -> Image.Image:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=140)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGBA")
    frame = image.copy()
    image.close()
    buffer.close()
    return frame


parser = argparse.ArgumentParser(description="Plot training logs for the simplified 5-city project.")
parser.add_argument("--size", default=5, type=int, choices=[5])
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

size = args.size
result_dir = Path.cwd()
steps = []
losses = []
tour_lengths = []

with (result_dir / f"{size}result.txt").open("r", encoding="utf-8") as result_file:
    for line in result_file:
        line = line.strip()
        if not line:
            continue
        step, score, valid_ratio, average_ans, optimal_len, loss = line.split()
        steps.append(int(step))
        losses.append(float(loss))
        tour_lengths.append(float(average_ans))

plt.figure(figsize=(7, 4))
plt.plot(steps, losses)
plt.xlabel("step")
plt.ylabel("loss")
plt.title(f"tsp{size} loss")
plt.grid()
save_with_fallback(
    result_dir / f"{size}loss.png",
    lambda target_path: plt.savefig(target_path, dpi=160),
    "loss chart",
)
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(steps, tour_lengths)
plt.xlabel("step")
plt.ylabel("tour_len")
plt.title(f"tsp{size} tour_len")
plt.grid()
save_with_fallback(
    result_dir / f"{size}ans.png",
    lambda target_path: plt.savefig(target_path, dpi=160),
    "tour-length chart",
)
plt.close()

if args.animate and steps:
    frame_count = min(len(steps), 30)
    frame_indices = sorted(set(np.linspace(1, len(steps), num=frame_count, dtype=int).tolist() + [len(steps)]))
    frames = []

    for end in frame_indices:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].plot(steps[:end], losses[:end], color="tab:blue")
        axes[0].set_title(f"tsp{size} loss")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("loss")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(steps[:end], tour_lengths[:end], color="tab:red")
        axes[1].set_title(f"tsp{size} tour_len")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("tour_len")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"Training progress up to step {steps[end - 1]}")
        fig.tight_layout()
        frames.append(figure_to_image(fig))
        plt.close(fig)

    save_with_fallback(
        result_dir / f"{size}training.gif",
        lambda target_path: frames[0].save(
            target_path,
            save_all=True,
            append_images=frames[1:],
            duration=220,
            loop=0,
        ),
        "training animation",
    )
