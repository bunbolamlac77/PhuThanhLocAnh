# 01 System Architecture (Kiến Trúc Hệ Thống)

Dự án: **AI Photo Culling App for macOS (M1/Apple Silicon)**
Mô hình hoạt động: **Proxy-based Culling (Mượn JPG chọn RAW)**

## 1. Cấu trúc dữ liệu đầu vào (Input Structure)
Hệ thống nhận một `Thư mục cha` (VD: `Folder_Dam_Cuoi/`) chứa các file RAW và một thư mục con `JPG/` chứa các file JPG proxy đã được xuất nhanh hoặc được máy ảnh tự sinh ra.

```text
Folder_Dam_Cuoi/          <-- Thư mục cha (Input)
  ├── JPG/                <-- Thư mục chứa proxy
  │   ├── DSC0001.JPG
  │   ├── DSC0002.JPG
  │   └── ...
  ├── DSC0001.ARW         <-- RAW gốc
  ├── DSC0002.ARW         <-- RAW gốc
  └── ...
```

## 2. Quy trình xử lý cốt lõi (4 Bước)

### Bước 1: Trích xuất mã số và Scan JPG
- Hệ thống quét file bên trong thư mục `/JPG`. Không load RAW ở bước này để tối ưu IO / Cache.
- Trích xuất tên cơ sở của file (Base name) - VD: `DSC0001`.
- Dùng `OpenCV` để load ảnh JPG vào RAM cực nhanh (< 10ms).

### Bước 2: Phân tích AI trên Proxy
- Đưa khung hình JPG qua hai mạng Neural:
  - **YOLO (ví dụ YOLO11/YOLOv8 chạy qua CoreML)** để lấy Bounding Box khuôn mặt.
  - Sau khi cắt crop khuôn mặt, đưa qua **MediaPipe FaceMesh** để tính tỷ lệ EAR (Eye Aspect Ratio) đo độ mở mắt.

### Bước 3: Định danh nhóm và Xếp hạng
- Phân tích EXIF `DateTimeOriginal` của JPG để nhóm ảnh (ngưỡng 0.1s - 2.0s tuỳ chỉnh).
- Chấm điểm từng tấm (Dựa trên độ nét OpenCV Laplacian + Chỉ số EAR từ AI).
- Chọn ra "Winner" cho mỗi nhóm.

### Bước 4: Ánh xạ đuôi file (Mapping) & Di chuyển (Move/Copy)
- Từ mã số của file "Winner" trong nhóm (vd: `DSC0123`), hệ thống quay lại `Thư mục cha`.
- Tìm kiếm tệp RAW tương ứng (VD `DSC0123.ARW`, `.CR3`, v.v.).
- Lệnh `shutil.copy2` chuyển file RAW này vào thư mục `[AI_SELECTED]`.

## 3. Kiến trúc Đa luồng (Multi-threading)
- GUI Thread: Chạy PySide6. Không bao giờ để bị block.
- Scanner Thread: Quét dữ liệu file và Exif.
- AI & Image Processing Thread: Load ảnh bằng OpenCV và chạy mô hình qua M1 Neural Engine. Đạt hiệu năng tối ưu và gửi `pyqtSignal` cập nhật Progress Bar trên GUI.
