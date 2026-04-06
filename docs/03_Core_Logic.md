# 03 Core Logic (Logic Cốt Lõi)

Dự án: **AI Photo Culling App for macOS (M1/Apple Silicon)**
Mô hình hoạt động: **Proxy-based Culling (Mượn JPG chọn RAW)**

## 1. Logic Quyết Định "Ngon Nhất" (Decision Engine)

Hệ thống sẽ chạy một hàm logic nâng cao để xác định và chọn ra những bức ảnh tốt nhất theo quy tắc sau:

*   **Nếu ảnh đơn (1 tấm duy nhất):** Lấy ngay (Không nằm trong chùm nào do thời gian chụp chênh lệch lớn hơn ngưỡng).
*   **Nếu ảnh thuộc nhóm liên tiếp (2-3+ tấm, cách nhau < Slider ngưỡng, VD: 0.5s):**
    *   **Ưu tiên 1:** Tấm có **Độ nét (Sharpness) cao nhất** VÀ **Tất cả mắt đều mở**.
    *   **Ưu tiên 2 (Trường hợp lỗi):** Nếu TẤT CẢ các tấm trong nhóm đều có ít nhất 1 người nhắm mắt (không tấm nào hoàn hảo) -> **Lấy CẢ NHÓM** (để nhiếp ảnh gia có tư liệu thay thế/ghép mắt).
*   **Nếu ảnh tập thể (nhiều khuôn mặt):** Ưu tiên tấm có số lượng người mở mắt nhiều nhất. (Trường hợp nhiều tấm cùng số lượng người mở mắt, so sánh về độ nét).

## 2. Giao Diện (UI/UX) & Tính năng bổ sung

*   **Slider Ngưỡng Nhóm (0.1s - 2.0s):** Giao diện cung cấp cho người dùng tính năng điều chỉnh khoảng chênh lệch thời gian EXIF để phân định nhóm. Rất hữu ích khi nhiếp ảnh gia chụp chậm hoặc burst (chụp liên tiếp) cực nhanh.
*   **Preview Window:** Cửa sổ hiển thị thời gian thực file JPG đang được AI soi chạy ngầm. Các vùng mắt bị nhắm sẽ được khoanh vùng đỏ để hiển thị tính minh bạch của AI.
*   **Tham số Output:**
    *   Tự động tạo thư mục con `[AI_SELECTED]` nằm trong thư mục cha được chọn.
    *   Tạo file danh sách `catalog.txt` thống kê các file được lọc. File này nằm cạnh thư mục `[AI_SELECTED]`.
