# 02 Implementation Roadmap (Bảng Chi Tiết Công Việc)

Dự án: **AI Photo Culling App for macOS (M1/Apple Silicon)**
Mô hình hoạt động: **Proxy-based Culling (Mượn JPG chọn RAW)**

| STT | Tên Module | Mô tả chi tiết công việc | Mục tiêu kỹ thuật |
| :--- | :--- | :--- | :--- |
| **1** | **Directory Scanner** | Quét thư mục cha. Nhận diện thư mục `/JPG` và danh sách file RAW ở ngoài. Đối chiếu số lượng để đảm bảo khớp mã số. | Đảm bảo tính toàn vẹn dữ liệu (Data Integrity). |
| **2** | **Fast Image Loading** | Sử dụng `OpenCV` để đọc trực tiếp file `.JPG`. Không cần dùng `rawpy` ở bước này. | Tốc độ load ảnh < 10ms. |
| **3** | **Smart Grouping** | Nhóm các file JPG dựa trên Metadata `DateTime`. Cho phép chỉnh ngưỡng (0.1s - 2.0s) từ UI. | Phân loại chùm ảnh chụp liên tiếp. |
| **4** | **Dual-AI Vision (X)** | Chạy **YOLO** (VD: qua `ultralytics` hoặc `coremltools`) tìm mặt, sau đó dùng **MediaPipe FaceMesh** soi độ mở mắt (EAR) trên file JPG. | Độ chính xác cao trên nền tảng JPG. CoreML usage on M1. |
| **5** | **Selection Logic** | Áp dụng quy tắc: Chọn tấm tốt nhất hoặc chọn cả nhóm nếu tất cả đều lỗi. | Ra quyết định chính xác dựa trên mã số (Filename). |
| **6** | **RAW Mapper & Copy** | Chuyển đổi mã số từ `.JPG` sang đuôi RAW tương ứng (`.ARW`, `.CR3`,...). Copy file RAW từ thư mục cha vào `[AI_SELECTED]`. Xuất log `.txt`. | Ánh xạ chính xác 100% mã số file. Không copy dư thừa, tránh tràn cache. |
| **7** | **Multi-threaded UI** | Hiển thị quá trình lọc trên giao diện PySide6 (Dark Mode). Cập nhật Progress Bar và preview hình ảnh phân tích. Slider và setting tuỳ chỉnh. | Trải nghiệm mượt, không treo máy, UX cao cấp. |
