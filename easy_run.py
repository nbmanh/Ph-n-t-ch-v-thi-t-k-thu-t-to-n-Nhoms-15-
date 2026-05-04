import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
RESULT_DIR = ROOT / "result"
BUILD_DIR = ROOT / "build"
PACKAGE_NAME = "Phân_tích_thuật_toán"
PACKAGE_STAGE_DIR = BUILD_DIR / PACKAGE_NAME
PACKAGE_ZIP = ROOT / f"{PACKAGE_NAME}.zip"
LEGACY_PACKAGE_NAMES = ["Phan_tich_thuat_toan.zip", "Phân_tích_thuật_toán.zip"]
MODEL_FILE = ROOT / "model" / "5mydata.pt"
TEST_DATA_FILE = ROOT / "test_data" / "tsp5_testdata.txt"
TRAIN_DATA_FILE = ROOT / "train" / "tsp_correct_5.txt"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
GUIDE_FILE = ROOT / "HUONG_DAN_CHAY.txt"
REQUIREMENTS_FILE = ROOT / "requirements-analysis.txt"


def safe_print(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        sys.stdout.buffer.write((message + os.linesep).encode(encoding, errors="replace"))
        sys.stdout.flush()


def project_python() -> str:
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def run_python_script(script: Path, cwd: Path, args: list[str]) -> int:
    command = [project_python(), str(script)] + args
    return subprocess.call(command, cwd=str(cwd))


def open_if_exists(path: Path) -> None:
    if path.exists():
        os.startfile(str(path))


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def clean_generated_outputs() -> None:
    if RESULT_DIR.exists():
        for child in RESULT_DIR.iterdir():
            remove_path(child)

    for cache_dir in [ROOT / "__pycache__", CODE_DIR / "__pycache__"]:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    for package_name in LEGACY_PACKAGE_NAMES:
        package_path = ROOT / package_name
        if package_path.exists():
            package_path.unlink()


def package_project() -> Path:
    if PACKAGE_STAGE_DIR.exists():
        shutil.rmtree(PACKAGE_STAGE_DIR)
    PACKAGE_STAGE_DIR.mkdir(parents=True, exist_ok=True)

    copy_files = [
        "easy_run.py",
        "README.md",
        "LICENSE",
        "RUN_GUI.bat",
        "CLICK_RUN_ALL.bat",
        "OPEN_REPORT.bat",
        "BUILD_PHAN_TICH_THUAT_TOAN.bat",
        "HUONG_DAN_CHAY.txt",
        "requirements-analysis.txt",
        "run_test.bat",
        "run_train.bat",
        "plot_result.bat",
    ]
    copy_dirs = ["code", "model", "train", "test_data", "result"]

    for relative_file in copy_files:
        source = ROOT / relative_file
        if source.exists():
            shutil.copy2(source, PACKAGE_STAGE_DIR / relative_file)

    for relative_dir in copy_dirs:
        source = ROOT / relative_dir
        destination = PACKAGE_STAGE_DIR / relative_dir
        if destination.exists():
            shutil.rmtree(destination)
        if source.exists():
            shutil.copytree(
                source,
                destination,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )

    if PACKAGE_ZIP.exists():
        PACKAGE_ZIP.unlink()

    with zipfile.ZipFile(PACKAGE_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in PACKAGE_STAGE_DIR.rglob("*"):
            archive.write(file_path, file_path.relative_to(BUILD_DIR))

    return PACKAGE_ZIP


def cmd_info(_: argparse.Namespace) -> int:
    safe_print("TSP project đã được nâng cấp thành bộ phân tích thuật toán")
    safe_print(f"- Trainer: {CODE_DIR / 'ptrnet3.py'}")
    safe_print(f"- Tester: {CODE_DIR / 'ptrnet3_test.py'}")
    safe_print(f"- Plotter: {CODE_DIR / 'plot.py'}")
    safe_print(f"- Demo 40-200 điểm: {CODE_DIR / 'simple_tsp.py'}")
    safe_print(f"- Benchmark/report: {CODE_DIR / 'benchmark_tsp.py'}")
    safe_print(f"- GUI phân tích: {CODE_DIR / 'gui_app.py'}")
    safe_print(f"- Train data (100 dòng): {TRAIN_DATA_FILE}")
    safe_print(f"- Test data (40 dòng): {TEST_DATA_FILE}")
    safe_print(f"- Model có sẵn: {MODEL_FILE}")
    safe_print(f"- Thư mục kết quả: {RESULT_DIR}")
    safe_print(f"- File đóng gói: {PACKAGE_ZIP}")
    safe_print("")
    safe_print("Lệnh nhanh")
    safe_print("- python easy_run.py compare --open-report")
    safe_print("- python easy_run.py gui")
    safe_print("- python easy_run.py clean")
    safe_print("- python easy_run.py package")
    safe_print("- python easy_run.py rebuild-package")
    safe_print("- python easy_run.py open-report")
    safe_print("- python easy_run.py open-charts")
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    script_args = [
        "--seq_len",
        "5",
        "--batch_size",
        "40",
        "--model_file",
        str(MODEL_FILE),
        "--data_file",
        str(TEST_DATA_FILE),
    ]
    if args.animate:
        script_args.extend(["--animation_path", str(RESULT_DIR / "test_tour_seq5.gif")])

    exit_code = run_python_script(CODE_DIR / "ptrnet3_test.py", CODE_DIR, script_args)
    if exit_code != 0:
        return exit_code

    if args.open_image:
        open_if_exists(RESULT_DIR / "test_tour_seq5.png")
        if args.animate:
            open_if_exists(RESULT_DIR / "test_tour_seq5.gif")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    script_args = [
        "--seq_len",
        "5",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--model_file",
        str(MODEL_FILE),
        "--result_file",
        str(RESULT_DIR / "5result.txt"),
    ]
    if not args.from_scratch:
        script_args.append("--load")
    if args.save:
        script_args.append("--save")
    return run_python_script(CODE_DIR / "ptrnet3.py", CODE_DIR, script_args)


def cmd_plot(args: argparse.Namespace) -> int:
    script_args = ["--size", "5"]
    if args.animate:
        script_args.append("--animate")

    exit_code = run_python_script(CODE_DIR / "plot.py", RESULT_DIR, script_args)
    if exit_code != 0:
        return exit_code

    if args.open_images:
        open_if_exists(RESULT_DIR / "5loss.png")
        open_if_exists(RESULT_DIR / "5ans.png")
        if args.animate:
            open_if_exists(RESULT_DIR / "5training.gif")
    return 0


def cmd_open_results(_: argparse.Namespace) -> int:
    RESULT_DIR.mkdir(exist_ok=True)
    os.startfile(str(RESULT_DIR))
    return 0


def cmd_open_report(_: argparse.Namespace) -> int:
    report_path = RESULT_DIR / "algorithm_report.html"
    if report_path.exists():
        os.startfile(str(report_path))
        return 0

    safe_print("Chưa có report. Hãy chạy: python easy_run.py compare --open-report")
    return 1


def cmd_open_charts(_: argparse.Namespace) -> int:
    chart_paths = [
        RESULT_DIR / "runtime_comparison.png",
        RESULT_DIR / "distance_comparison.png",
        RESULT_DIR / "gap_comparison.png",
        RESULT_DIR / "time_quality_tradeoff.png",
        RESULT_DIR / "tsp_40.gif",
        RESULT_DIR / "tsp_100.gif",
        RESULT_DIR / "tsp_200.gif",
        RESULT_DIR / "5loss.png",
        RESULT_DIR / "5ans.png",
    ]
    opened = False
    for chart_path in chart_paths:
        if chart_path.exists():
            os.startfile(str(chart_path))
            opened = True

    if opened:
        return 0

    safe_print("Chưa có biểu đồ. Hãy chạy: python easy_run.py compare hoặc python easy_run.py plot")
    return 1


def cmd_gui(_: argparse.Namespace) -> int:
    code_dir = str(CODE_DIR)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    from gui_app import main as gui_main
    original_argv = sys.argv[:]
    try:
        sys.argv = [str(CODE_DIR / "gui_app.py")]
        return gui_main()
    finally:
        sys.argv = original_argv


def cmd_demo(args: argparse.Namespace) -> int:
    script_args = ["--points", str(args.points), "--seed", str(args.seed)]
    if args.animate:
        script_args.append("--animate")

    exit_code = run_python_script(CODE_DIR / "simple_tsp.py", ROOT, script_args)
    if exit_code != 0:
        return exit_code

    if args.open_image:
        open_if_exists(RESULT_DIR / f"tsp_{args.points}.png")
        if args.animate:
            open_if_exists(RESULT_DIR / f"tsp_{args.points}.gif")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    exit_code = run_python_script(CODE_DIR / "benchmark_tsp.py", ROOT, ["--seed", str(args.seed)])
    if exit_code != 0:
        return exit_code

    if args.open_images:
        open_if_exists(RESULT_DIR / "runtime_comparison.png")
        open_if_exists(RESULT_DIR / "distance_comparison.png")
        open_if_exists(RESULT_DIR / "gap_comparison.png")
        open_if_exists(RESULT_DIR / "time_quality_tradeoff.png")
    if args.open_report:
        open_if_exists(RESULT_DIR / "algorithm_report.html")
    return 0


def cmd_clean(_: argparse.Namespace) -> int:
    clean_generated_outputs()
    RESULT_DIR.mkdir(exist_ok=True)
    safe_print("Đã dọn dẹp kết quả cũ, build/ và file zip đóng gói.")
    return 0


def cmd_package(_: argparse.Namespace) -> int:
    package_path = package_project()
    safe_print(f"Đã đóng gói xong: {package_path}")
    return 0


def cmd_rebuild_package(args: argparse.Namespace) -> int:
    clean_generated_outputs()
    RESULT_DIR.mkdir(exist_ok=True)

    compare_exit_code = run_python_script(CODE_DIR / "benchmark_tsp.py", ROOT, ["--seed", str(args.seed)])
    if compare_exit_code != 0:
        return compare_exit_code

    package_path = package_project()
    safe_print(f"Đã build và đóng gói xong: {package_path}")
    if args.open_report:
        open_if_exists(RESULT_DIR / "algorithm_report.html")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple entrypoint for the upgraded TSP analysis project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info_parser = subparsers.add_parser("info", help="Show the project map.")
    info_parser.set_defaults(func=cmd_info)

    test_parser = subparsers.add_parser("test", help="Run the pretrained test script on 40 test samples.")
    test_parser.add_argument("--open-image", action="store_true")
    test_parser.add_argument("--animate", action="store_true")
    test_parser.set_defaults(func=cmd_test)

    train_parser = subparsers.add_parser("train", help="Run the trainer on 100 train samples.")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--from-scratch", action="store_true")
    train_parser.add_argument("--save", action="store_true")
    train_parser.set_defaults(func=cmd_train)

    plot_parser = subparsers.add_parser("plot", help="Plot training logs from result/.")
    plot_parser.add_argument("--open-images", action="store_true")
    plot_parser.add_argument("--animate", action="store_true")
    plot_parser.set_defaults(func=cmd_plot)

    demo_parser = subparsers.add_parser("demo", help="Generate and solve a clear TSP example with 40-200 cities.")
    demo_parser.add_argument("--points", type=int, default=40, choices=[40, 50, 60, 70, 80, 90, 100, 200])
    demo_parser.add_argument("--seed", type=int, default=42)
    demo_parser.add_argument("--open-image", action="store_true")
    demo_parser.add_argument("--animate", action="store_true")
    demo_parser.set_defaults(func=cmd_demo)

    compare_parser = subparsers.add_parser("compare", help="Run all sizes and create algorithm comparison outputs.")
    compare_parser.add_argument("--seed", type=int, default=42)
    compare_parser.add_argument("--open-images", action="store_true")
    compare_parser.add_argument("--open-report", action="store_true")
    compare_parser.set_defaults(func=cmd_compare)

    gui_parser = subparsers.add_parser("gui", help="Open the desktop GUI for coordinate files.")
    gui_parser.set_defaults(func=cmd_gui)

    open_report_parser = subparsers.add_parser("open-report", help="Open the generated HTML report.")
    open_report_parser.set_defaults(func=cmd_open_report)

    open_charts_parser = subparsers.add_parser("open-charts", help="Open generated chart images.")
    open_charts_parser.set_defaults(func=cmd_open_charts)

    clean_parser = subparsers.add_parser("clean", help="Delete generated results, build outputs and package zip.")
    clean_parser.set_defaults(func=cmd_clean)

    package_parser = subparsers.add_parser("package", help="Package the project into a zip bundle.")
    package_parser.set_defaults(func=cmd_package)

    rebuild_parser = subparsers.add_parser("rebuild-package", help="Clean, rebuild the benchmark outputs and package everything.")
    rebuild_parser.add_argument("--seed", type=int, default=42)
    rebuild_parser.add_argument("--open-report", action="store_true")
    rebuild_parser.set_defaults(func=cmd_rebuild_package)

    results_parser = subparsers.add_parser("open-results", help="Open the result folder.")
    results_parser.set_defaults(func=cmd_open_results)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
