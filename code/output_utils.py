from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar


T = TypeVar("T")


def safe_print(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        sys.stdout.buffer.write((message + os.linesep).encode(encoding, errors="replace"))
        sys.stdout.flush()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def fallback_output_path(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")


def save_with_fallback(path: Path, writer: Callable[[Path], T], label: str) -> Path:
    ensure_parent(path)
    try:
        writer(path)
        safe_print(f"Saved {label} to {path}")
        return path
    except PermissionError:
        fallback_path = fallback_output_path(path)
        writer(fallback_path)
        safe_print(
            f"Could not overwrite locked file {path}. "
            f"Saved {label} to {fallback_path} instead."
        )
        return fallback_path
