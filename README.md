# TSP Pointer Network / Phân Tích Thuật Toán

Project này đã được nâng cấp theo hướng "phân tích thuật toán" cho bài toán TSP:

- So sánh nhiều heuristic:
  - `Random`
  - `Nearest Neighbor`
  - `Nearest Insertion`
  - `Cheapest Insertion`
  - `Farthest Insertion`
  - `Nearest Neighbor + 2-opt`
  - `Best Construction + 2-opt`
- Sinh:
  - report HTML
  - biểu đồ runtime / distance / gap / trade-off
  - bảng CSV / XLSX
  - GUI chọn file tọa độ và so sánh thuật toán trên dữ liệu riêng
- Vẫn giữ phần demo Pointer Network 5 điểm để học thêm.

## Cách chạy nhanh

### Build lại và đóng gói

```powershell
python easy_run.py rebuild-package
```

Hoặc bấm đúp:

- `BUILD_PHAN_TICH_THUAT_TOAN.bat`

File đóng gói sinh ra:

- `Phân_tích_thuật_toán.zip`

### Chạy benchmark và mở report

```powershell
python easy_run.py compare --open-report
```

Hoặc bấm đúp:

- `CLICK_RUN_ALL.bat`

### Mở GUI chọn file tọa độ

```powershell
python easy_run.py gui
```

Hoặc bấm đúp:

- `RUN_GUI.bat`

### Mở report đã sinh

- `OPEN_REPORT.bat`

## Đầu vào hợp lệ cho GUI

GUI nhận file `.csv` hoặc `.txt` chứa ít nhất 3 điểm `(x, y)`.

Ví dụ:

```txt
0.12 0.45
0.33 0.71
0.90 0.10
```

Hoặc:

```csv
city_id,x,y
1,0.12,0.45
2,0.33,0.71
3,0.90,0.10
```

## Thư mục quan trọng

- `code/benchmark_tsp.py`
  - benchmark và report
- `code/gui_app.py`
  - GUI phân tích file tọa độ
- `code/simple_tsp.py`
  - các heuristic TSP cơ bản và hàm vẽ
- `code/tsp_analysis.py`
  - bộ đánh giá nhiều thuật toán
- `result/`
  - report, biểu đồ, bảng kết quả
- `HUONG_DAN_CHAY.txt`
  - hướng dẫn chạy nhanh cho gói đóng gói

## Yêu cầu tối thiểu cho nhánh phân tích

```powershell
pip install -r requirements-analysis.txt
```

Nếu muốn chạy nhánh Pointer Network / PyTorch, cần cài thêm `torch` phù hợp máy của bạn.
