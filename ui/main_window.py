"""
main_window.py — Main Window ergonomic layout 2 cột
Theme: White Luxury, Professional
"""

import os
import traceback
import logging
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QSlider,
    QFrame, QSizePolicy, QSpacerItem
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont
import numpy as np

from ui.components import (
    ImagePreviewWidget, DropZoneWidget, ToastNotification,
    StatCard, LogPanel,
    COLOR_BG, COLOR_SURFACE, COLOR_BORDER, COLOR_BORDER_DARK,
    COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY, COLOR_TEXT_MUTED,
    COLOR_ACCENT, COLOR_ACCENT_LIGHT,
    COLOR_SUCCESS, COLOR_SUCCESS_BG,
    COLOR_ERROR, COLOR_WARNING, COLOR_WARNING_BG,
    COLOR_GAZE_OK, COLOR_GAZE_AWAY,
    _shadow
)
from core.scanner import get_proxy_images
from core.ai_vision import AIVisionPipeline
from core.logic import group_images, select_best_images
from utils.file_ops import find_raw_file, copy_to_selected, export_catalog

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Worker Thread
# ─────────────────────────────────────────────────────────────────────────────

class CullingWorker(QThread):
    progress      = Signal(int, int, str)
    image_preview = Signal(np.ndarray)
    gaze_update   = Signal(int, int, list)   # looking, total, ear_vals — MỚI
    ai_stat       = Signal(dict)              # stats dict để cập nhật StatCards
    log_message   = Signal(str, str)
    finished      = Signal(int, int, int)
    error         = Signal(str)

    def __init__(self, folder_path, threshold, ear_threshold,
                 gaze_yaw=25.0, gaze_pitch=20.0, gaze_priority=0.5):
        super().__init__()
        self.folder_path       = folder_path
        self.threshold         = threshold
        self.ear_threshold     = ear_threshold
        self.gaze_yaw          = gaze_yaw
        self.gaze_pitch        = gaze_pitch
        self.gaze_priority     = gaze_priority

        self._is_paused        = False
        self._is_cancelled     = False

        # Thống kê tích lũy
        self._total_faces     = 0
        self._total_looking   = 0
        self._total_eyes_open = 0
        self._analyzed        = 0

    def progress_callback(self, current, total, filepath, annotated_frame):
        self.progress.emit(current, total, f"Phân tích: {os.path.basename(filepath)}")
        if annotated_frame is not None:
            self.image_preview.emit(annotated_frame)

    def run(self):
        try:
            self.log_message.emit("Khởi tạo AI Engine (YOLO + MediaPipe FaceLandmarker)...", "info")
            self.progress.emit(0, 0, "Nạp AI model...")

            ai = AIVisionPipeline(
                yolo_model_path="yolo26.pt",
                ear_threshold=self.ear_threshold,
                gaze_yaw_threshold=self.gaze_yaw,
                gaze_pitch_threshold=self.gaze_pitch
            )
            self.log_message.emit("AI Engine sẵn sàng ✓", "info")

            self.progress.emit(0, 0, "Quét thư mục JPG...")
            images = get_proxy_images(self.folder_path)
            if not images:
                self.error.emit(
                    "Không tìm thấy file JPG proxy.\n"
                    "Hãy đảm bảo thư mục con /JPG tồn tại và có ảnh."
                )
                return

            total = len(images)
            self.log_message.emit(f"Tìm thấy {total} ảnh JPG proxy.", "info")
            self.ai_stat.emit({'total': total, 'groups': 0, 'selected': 0, 'success': 0})

            self.progress.emit(0, 0, "Phân nhóm theo EXIF time...")
            groups = group_images(images, time_threshold=self.threshold)
            self.log_message.emit(f"Phân nhóm: {total} ảnh → {len(groups)} nhóm.", "info")
            self.ai_stat.emit({'total': total, 'groups': len(groups), 'selected': 0, 'success': 0})

            def check_status():
                if self._is_cancelled:
                    raise Exception("CANCELLED")
                while self._is_paused and not self._is_cancelled:
                    import time
                    time.sleep(0.1)

            # Custom callback đề cập nhật UI real-time từ đa luồng
            def wrapped_callback(current, total_items, filepath, frame, res):
                self.progress.emit(current, total_items, f"Phân tích: {os.path.basename(filepath)}")
                if frame is not None:
                    self.image_preview.emit(frame)
                
                # Emit gaze update từ kết quả phân tích mới
                if res.get('has_face'):
                    self.gaze_update.emit(
                        res.get('looking_count', 0),
                        res.get('faces_count', 0),
                        res.get('ear_values', [])
                    )

            self.progress.emit(0, total, "AI Vision Analysis (Multi-threaded)...")
            selected = select_best_images(
                groups,
                ai_pipeline=ai,
                progress_callback=wrapped_callback,
                gaze_priority=self.gaze_priority
            )
            self.log_message.emit(
                f"AI chọn {len(selected)}/{total} ảnh tốt nhất.", "info"
            )
            self.log_message.emit(
                f"Thống kê: {self._total_looking}/{self._total_faces} mặt nhìn thẳng | "
                f"{self._total_eyes_open} mắt mở", "info"
            )
            self.ai_stat.emit({
                'total': total,
                'groups': len(groups),
                'selected': len(selected),
                'success': 0,
            })

            # Copy RAW
            out_dir = os.path.join(self.folder_path, "[AI_SELECTED]")
            success_c = fail_c = no_space_c = 0
            raws = []
            no_space_warned = False

            self.progress.emit(0, len(selected), "Copy RAW files...")
            for idx, proxy in enumerate(selected):
                check_status()
                base = os.path.splitext(os.path.basename(proxy))[0]
                self.progress.emit(idx + 1, len(selected), f"Copy: {base}")
                raw = find_raw_file(self.folder_path, base)
                if raw:
                    ok, code = copy_to_selected(raw, out_dir)
                    if ok:
                        raws.append(raw)
                        success_c += 1
                    else:
                        fail_c += 1
                        if code == 'no_space':
                            no_space_c += 1
                            if not no_space_warned:
                                self.log_message.emit("⚠ Hết dung lượng đĩa! Một số file bị bỏ qua.", "warn")
                                no_space_warned = True
                        elif code == 'permission':
                            self.log_message.emit(f"Không có quyền ghi: {base}", "warn")
                else:
                    fail_c += 1
                    self.log_message.emit(f"Không tìm được RAW: {base}", "warn")

            if raws:
                export_catalog(raws, out_dir)
                self.log_message.emit(f"Xuất catalog.txt ({len(raws)} files) ✓", "info")

            self.ai_stat.emit({
                'total': total,
                'groups': len(groups),
                'selected': len(selected),
                'success': success_c,
            })
            self.finished.emit(success_c, fail_c, no_space_c)

        except Exception as e:
            if str(e) == "CANCELLED":
                self.log_message.emit("⏹ Tiến trình đã bị người dùng hủy.", "warn")
                return
            logger.error(traceback.format_exc())
            self.error.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_STYLE = f"""
    QMainWindow, QWidget {{
        background-color: {COLOR_BG};
        font-family: 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif;
    }}
    QScrollBar:vertical {{
        background:transparent; width:5px; margin:0;
    }}
    QScrollBar::handle:vertical {{
        background:{COLOR_BORDER_DARK}; border-radius:2px; min-height:24px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
"""

SLIDER_CARD_STYLE = f"""
    QWidget#ctrlCard {{
        background:{COLOR_SURFACE};
        border:1px solid {COLOR_BORDER};
        border-radius:12px;
    }}
    QLabel {{ background:transparent; border:none; }}
    QSlider::groove:horizontal {{
        height:3px; background:{COLOR_BORDER_DARK}; border-radius:1px;
    }}
    QSlider::handle:horizontal {{
        background:{COLOR_ACCENT}; width:14px; height:14px;
        border-radius:7px; margin:-5px 0;
    }}
    QSlider::sub-page:horizontal {{
        background:{COLOR_ACCENT}; border-radius:1px;
    }}
    QSlider::handle:horizontal:disabled {{
        background:{COLOR_TEXT_MUTED};
    }}
"""

BTX_PRIMARY = f"""
    QPushButton {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 {COLOR_ACCENT}, stop:1 #312E81);
        color:white; font-size:13px; font-weight:700;
        border:none; border-radius:12px;
        padding:0 32px; letter-spacing:0.4px;
    }}
    QPushButton:hover {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 #6366F1, stop:1 {COLOR_ACCENT});
    }}
    QPushButton:pressed {{
        background: #312E81;
    }}
    QPushButton:disabled {{
        background:{COLOR_BORDER}; color:{COLOR_TEXT_MUTED};
    }}
"""

PROGRESS_STYLE = f"""
    QProgressBar {{
        background:{COLOR_BORDER}; border:none;
        border-radius:3px; height:5px;
    }}
    QProgressBar::chunk {{
        background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
            stop:0 #2C3E8C, stop:1 #6C63FF);
        border-radius:3px;
    }}
"""

GAZE_CARD_STYLE = f"""
    QWidget#gazeCard {{
        background:{COLOR_SURFACE}; border:1px solid {COLOR_BORDER};
        border-radius:10px;
    }}
    QLabel {{ background:transparent; border:none; }}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Gaze Threshold Card (MỚI — đặt trong sidebar)
# ─────────────────────────────────────────────────────────────────────────────
class GazeSettingCard(QWidget):
    """Card chứa slider điều chỉnh ngưỡng gaze (yaw/pitch)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("gazeCard")
        self.setStyleSheet(GAZE_CARD_STYLE)
        _shadow(self, radius=8, color="#0000000C", offset=(0, 2))

        vl = QVBoxLayout(self)
        vl.setContentsMargins(14, 12, 14, 12)
        vl.setSpacing(10)

        # Title
        title = QLabel("👁  Ngưỡng Gaze Detection")
        title.setStyleSheet(
            f"color:{COLOR_TEXT_PRIMARY}; font-size:12px; font-weight:700;"
        )
        vl.addWidget(title)

        desc = QLabel(
            "Góc quay đầu tối đa để coi là 'nhìn thẳng vào camera'.\n"
            "Giảm xuống để khắt khe hơn."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color:{COLOR_TEXT_MUTED}; font-size:10px; line-height:1.5;")
        vl.addWidget(desc)

        # Yaw slider
        vl.addWidget(self._make_slider_row(
            "Góc ngang (Yaw)", 10, 45, 25, 1.0, "°",
            "_yaw_val", "yaw_slider", self._on_yaw
        ))
        # Pitch slider
        vl.addWidget(self._make_slider_row(
            "Góc dọc (Pitch)", 5, 40, 20, 1.0, "°",
            "_pitch_val", "pitch_slider", self._on_pitch
        ))
        # Gaze Priority slider (MỚI)
        vl.addWidget(self._make_slider_row(
            "Ưu tiên nhìn nhau", 0, 100, 50, 100.0, "",
            "_priority_val", "priority_slider", self._on_priority
        ))

    def _on_priority(self, v):
        pct = v
        self._priority_val.setText(f"{pct}%")

    def _make_slider_row(self, label, s_min, s_max, s_val,
                         scale, unit, attr_val, attr_slider, handler):
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(4)

        hl = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color:{COLOR_TEXT_SECONDARY}; font-size:10px; font-weight:600;")
        val = QLabel(f"{s_val}{unit}")
        val.setStyleSheet(f"color:{COLOR_ACCENT}; font-size:10px; font-weight:700;")
        val.setAlignment(Qt.AlignRight)
        setattr(self, attr_val, val)
        hl.addWidget(lbl)
        hl.addStretch()
        hl.addWidget(val)
        vl.addLayout(hl)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(s_min)
        slider.setMaximum(s_max)
        slider.setValue(s_val)
        slider.valueChanged.connect(handler)
        setattr(self, attr_slider, slider)
        vl.addWidget(slider)
        return w

    def _on_yaw(self, v):
        self._yaw_val.setText(f"{v}°")

    def _on_pitch(self, v):
        self._pitch_val.setText(f"{v}°")


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LocAnh AI — Photo Culling")
        self.resize(1200, 820)
        self.setMinimumSize(960, 680)
        self.setStyleSheet(GLOBAL_STYLE)

        root = QWidget()
        self.setCentralWidget(root)
        root_vl = QVBoxLayout(root)
        root_vl.setContentsMargins(0, 0, 0, 0)
        root_vl.setSpacing(0)

        # ── Header ──────────────────────────────────────────────────────────
        root_vl.addWidget(self._make_header())

        # ── Body: Left col (main) + Right sidebar ───────────────────────────
        body = QWidget()
        body_hl = QHBoxLayout(body)
        body_hl.setContentsMargins(20, 16, 20, 16)
        body_hl.setSpacing(16)

        # LEFT COLUMN
        left = QWidget()
        left_vl = QVBoxLayout(left)
        left_vl.setContentsMargins(0, 0, 0, 0)
        left_vl.setSpacing(14)

        self.drop_zone = DropZoneWidget(self)
        left_vl.addWidget(self.drop_zone)

        left_vl.addWidget(self._make_controls_card())

        self.preview = ImagePreviewWidget()
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_vl.addWidget(self.preview, 1)

        self.log_panel = LogPanel()
        left_vl.addWidget(self.log_panel)

        left_vl.addWidget(self._make_progress_widget())

        body_hl.addWidget(left, 3)

        # RIGHT SIDEBAR
        sidebar = self._make_sidebar()
        body_hl.addWidget(sidebar, 1)

        root_vl.addWidget(body, 1)

        # Toast
        self.toast = ToastNotification(root)
        self.worker = None

    # ─────────────────────────────────────────────────────────────────────────
    # UI Builders
    # ─────────────────────────────────────────────────────────────────────────

    def _make_header(self) -> QWidget:
        hdr = QWidget()
        hdr.setFixedHeight(68)
        hdr.setStyleSheet(f"""
            QWidget {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0F172A, stop:0.5 #1E293B, stop:1 #334155);
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }}
        """)
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(24, 0, 24, 0)

        logo = QLabel("◈  LocAnh AI")
        logo.setStyleSheet(
            "color:white; font-size:21px; font-weight:800; letter-spacing:0.5px; background:transparent;"
        )
        sep = QLabel("|")
        sep.setStyleSheet("color:rgba(255,255,255,0.25); font-size:18px; background:transparent;")
        tag = QLabel("Intelligent Photo Culling  ·  YOLO + MediaPipe + Gaze Detection")
        tag.setStyleSheet("color:rgba(255,255,255,0.5); font-size:11px; background:transparent;")

        badge = QLabel("v3.0")
        badge.setStyleSheet(
            "color:rgba(255,255,255,0.75); font-size:10px; font-weight:700; "
            "background:rgba(255,255,255,0.12); border-radius:8px; padding:2px 9px;"
        )

        hl.addWidget(logo)
        hl.addSpacing(12)
        hl.addWidget(sep)
        hl.addSpacing(12)
        hl.addWidget(tag)
        hl.addStretch()
        hl.addWidget(badge)
        return hdr

    def _make_controls_card(self) -> QWidget:
        card = QWidget()
        card.setObjectName("ctrlCard")
        card.setStyleSheet(SLIDER_CARD_STYLE)
        card.setFixedHeight(74)
        _shadow(card, radius=8, color="#0000000E", offset=(0, 2))

        hl = QHBoxLayout(card)
        hl.setContentsMargins(18, 0, 18, 0)
        hl.setSpacing(0)

        # Slider: Time
        hl.addWidget(self._slider_group(
            "Ngưỡng nhóm burst", "3.0s",
            1, 50, 30, 130,
            self._on_time, "_time_lbl", "time_slider"
        ))
        hl.addWidget(self._vdivider())

        # Slider: EAR
        hl.addWidget(self._slider_group(
            "Độ mở mắt (EAR)", "0.18 · Châu Á",
            10, 35, 18, 120,
            self._on_ear, "_ear_lbl", "ear_slider"
        ))

        hl.addStretch()

        self.btn_start = QPushButton("  ▶   Bắt đầu lọc ảnh")
        self.btn_start.setMinimumHeight(44)
        self.btn_start.setMinimumWidth(180)
        self.btn_start.setStyleSheet(BTX_PRIMARY)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_culling)
        _shadow(self.btn_start, radius=16, color="#4F46E533", offset=(0, 4))
        hl.addWidget(self.btn_start)

        # Nút Tạm Dừng & Hủy (Ẩn lúc đầu)
        pause_style = BTX_PRIMARY.replace("#312E81", "#B45309").replace(COLOR_ACCENT, "#F59E0B")
        cancel_style = BTX_PRIMARY.replace("#312E81", "#991B1B").replace(COLOR_ACCENT, "#EF4444")

        self.btn_pause = QPushButton("⏸ Tạm Dừng")
        self.btn_pause.setMinimumHeight(44)
        self.btn_pause.setStyleSheet(pause_style)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_pause.hide()
        hl.addWidget(self.btn_pause)

        self.btn_cancel = QPushButton("⏹ Hủy")
        self.btn_cancel.setMinimumHeight(44)
        self.btn_cancel.setStyleSheet(cancel_style)
        self.btn_cancel.clicked.connect(self.cancel_culling)
        self.btn_cancel.hide()
        hl.addWidget(self.btn_cancel)

        # Nút Reset (Nhỏ gọn kế bên)
        hl.addSpacing(10)
        self.btn_reset = QPushButton("↺")
        self.btn_reset.setToolTip("Xoá dữ liệu / Làm mới để lọc bộ ảnh khác")
        self.btn_reset.setFixedSize(42, 42)
        reset_style = f"""
            QPushButton {{
                background: white; border: 1px solid {COLOR_BORDER}; border-radius: 8px;
                color: {COLOR_TEXT_SECONDARY}; font-size: 18px;
            }}
            QPushButton:hover {{ background: {COLOR_BG}; border-color: {COLOR_BORDER_DARK}; }}
            QPushButton:pressed {{ background: {COLOR_BORDER}55; }}
        """
        self.btn_reset.setStyleSheet(reset_style)
        self.btn_reset.clicked.connect(self.reset_app)
        hl.addWidget(self.btn_reset)

        # Nút Mở Thư mục Kết quả (Ẩn đi cho đến khi xong)
        hl.addSpacing(10)
        self.btn_open_out = QPushButton("📁")
        self.btn_open_out.setToolTip("Mở thư mục ảnh kết quả ([AI_SELECTED])")
        self.btn_open_out.setFixedSize(42, 42)
        self.btn_open_out.setStyleSheet(reset_style)
        self.btn_open_out.clicked.connect(self.open_output_folder)
        self.btn_open_out.hide()
        hl.addWidget(self.btn_open_out)

        return card

    def _slider_group(self, label, init_val, s_min, s_max, s_val,
                      width, handler, attr_val, attr_slider):
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        vl = QVBoxLayout(w)
        vl.setContentsMargins(10, 10, 10, 10)
        vl.setSpacing(5)

        hl = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet(
            f"color:{COLOR_TEXT_SECONDARY}; font-size:10px; font-weight:700; letter-spacing:0.3px;"
        )
        val = QLabel(init_val)
        val.setStyleSheet(f"color:{COLOR_ACCENT}; font-size:10px; font-weight:800;")
        val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        setattr(self, attr_val, val)
        hl.addWidget(lbl)
        hl.addStretch()
        hl.addWidget(val)
        vl.addLayout(hl)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(s_min)
        slider.setMaximum(s_max)
        slider.setValue(s_val)
        slider.setFixedWidth(width)
        slider.valueChanged.connect(handler)
        setattr(self, attr_slider, slider)
        vl.addWidget(slider)
        return w

    def _vdivider(self):
        ln = QFrame()
        ln.setFixedWidth(1)
        ln.setFixedHeight(40)
        ln.setStyleSheet(f"background:{COLOR_BORDER};")
        return ln

    def _make_progress_widget(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(5)

        self.status_lbl = QLabel("Sẵn sàng — Kéo hoặc chọn thư mục ảnh để bắt đầu")
        self.status_lbl.setStyleSheet(
            f"color:{COLOR_TEXT_MUTED}; font-size:11px; font-weight:500;"
        )

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(5)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(PROGRESS_STYLE)

        vl.addWidget(self.status_lbl)
        vl.addWidget(self.progress_bar)
        return w

    # ─────────────────────────────────────────────────────────────────────────
    # Right Sidebar
    # ─────────────────────────────────────────────────────────────────────────

    def _make_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setStyleSheet("background:transparent;")
        sidebar.setFixedWidth(240)
        vl = QVBoxLayout(sidebar)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(12)

        # Section: Stats
        sec_lbl = QLabel("THỐNG KÊ")
        sec_lbl.setStyleSheet(
            f"color:{COLOR_TEXT_MUTED}; font-size:9px; font-weight:800; letter-spacing:1.2px;"
        )
        vl.addWidget(sec_lbl)

        # 4 stat cards (2x2 grid)
        grid_top = QHBoxLayout()
        grid_top.setSpacing(10)
        self.stat_total = StatCard("📷", "TỔNG ẢNH")
        self.stat_groups = StatCard("📦", "NHÓM", accent="#7C3AED")
        grid_top.addWidget(self.stat_total)
        grid_top.addWidget(self.stat_groups)
        vl.addLayout(grid_top)

        grid_bot = QHBoxLayout()
        grid_bot.setSpacing(10)
        self.stat_selected = StatCard("✅", "ĐÃ CHỌN", accent=COLOR_SUCCESS)
        self.stat_success = StatCard("💾", "ĐÃ COPY", accent="#0891B2")
        grid_bot.addWidget(self.stat_selected)
        grid_bot.addWidget(self.stat_success)
        vl.addLayout(grid_bot)

        # Divider
        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background:{COLOR_BORDER};")
        vl.addWidget(div)

        # Gaze live indicator
        gaze_sec = QLabel("NHÌN VÀO CAMERA")
        gaze_sec.setStyleSheet(
            f"color:{COLOR_TEXT_MUTED}; font-size:9px; font-weight:800; letter-spacing:1.2px;"
        )
        vl.addWidget(gaze_sec)

        self.gaze_card = QWidget()
        self.gaze_card.setObjectName("gazeCard")
        self.gaze_card.setStyleSheet(
            f"QWidget#gazeCard {{ background:{COLOR_SURFACE}; border:1px solid {COLOR_BORDER}; "
            f"border-radius:10px; }}"
        )
        _shadow(self.gaze_card, radius=8, color="#0000000C", offset=(0, 2))
        gaze_vl = QVBoxLayout(self.gaze_card)
        gaze_vl.setContentsMargins(14, 12, 14, 12)
        gaze_vl.setSpacing(8)

        # Live gaze meter
        self.gaze_icon = QLabel("👁")
        self.gaze_icon.setAlignment(Qt.AlignCenter)
        self.gaze_icon.setStyleSheet("font-size:28px; background:transparent; border:none;")

        self.gaze_pct_lbl = QLabel("—")
        self.gaze_pct_lbl.setAlignment(Qt.AlignCenter)
        self.gaze_pct_lbl.setStyleSheet(
            f"color:{COLOR_TEXT_PRIMARY}; font-size:26px; font-weight:800; background:transparent;"
        )

        self.gaze_sub = QLabel("nhìn thẳng vào camera")
        self.gaze_sub.setAlignment(Qt.AlignCenter)
        self.gaze_sub.setStyleSheet(
            f"color:{COLOR_TEXT_MUTED}; font-size:10px; background:transparent; border:none;"
        )

        # Mini progress bar cho gaze %
        self.gaze_bar_widget = QProgressBar()
        self.gaze_bar_widget.setFixedHeight(5)
        self.gaze_bar_widget.setTextVisible(False)
        self.gaze_bar_widget.setMaximum(100)
        self.gaze_bar_widget.setValue(0)
        self.gaze_bar_widget.setStyleSheet(f"""
            QProgressBar {{
                background:{COLOR_BORDER}; border:none; border-radius:2px;
            }}
            QProgressBar::chunk {{
                background:{COLOR_GAZE_OK}; border-radius:2px;
            }}
        """)

        gaze_vl.addWidget(self.gaze_icon)
        gaze_vl.addWidget(self.gaze_pct_lbl)
        gaze_vl.addWidget(self.gaze_sub)
        gaze_vl.addWidget(self.gaze_bar_widget)
        vl.addWidget(self.gaze_card)

        # Divider
        div2 = QFrame()
        div2.setFixedHeight(1)
        div2.setStyleSheet(f"background:{COLOR_BORDER};")
        vl.addWidget(div2)

        # Gaze settings
        setting_sec = QLabel("CÀI ĐẶT NÂNG CAO")
        setting_sec.setStyleSheet(
            f"color:{COLOR_TEXT_MUTED}; font-size:9px; font-weight:800; letter-spacing:1.2px;"
        )
        vl.addWidget(setting_sec)
        self.gaze_setting = GazeSettingCard()
        vl.addWidget(self.gaze_setting)

        vl.addStretch()

        # Credit
        credit = QLabel("LocAnh AI  ·  v3.0 · 2026\nYOLO + MediaPipe + Gaze Detection")
        credit.setAlignment(Qt.AlignCenter)
        credit.setStyleSheet(
            f"color:{COLOR_TEXT_MUTED}; font-size:9px; line-height:1.5; background:transparent;"
        )
        vl.addWidget(credit)
        return sidebar

    # ─────────────────────────────────────────────────────────────────────────
    # Slider callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def _on_time(self, v):
        self._time_lbl.setText(f"{v/10.0:.1f}s")

    def _on_ear(self, v):
        ear = v / 100.0
        hint = " · Châu Á" if ear <= 0.18 else (" · Tây" if ear >= 0.25 else "")
        self._ear_lbl.setText(f"{ear:.2f}{hint}")

    # ─────────────────────────────────────────────────────────────────────────
    # Events
    # ─────────────────────────────────────────────────────────────────────────

    def on_folder_selected(self, path: str):
        self.btn_start.setEnabled(True)
        self.status_lbl.setText(f"Thư mục: {path}")
        self.log_panel.add_log(f"Thư mục đã chọn: {path}", "info")

    def start_culling(self):
        folder = self.drop_zone.folder_path
        if not folder:
            self.toast.show_message("Chưa chọn thư mục ảnh!", "warning")
            return
            
        if hasattr(self, 'btn_open_out'):
            self.btn_open_out.hide()

        threshold     = self.time_slider.value() / 10.0
        ear           = self.ear_slider.value() / 100.0
        gaze_yaw      = float(self.gaze_setting.yaw_slider.value())
        gaze_pitch    = float(self.gaze_setting.pitch_slider.value())
        gaze_priority = float(self.gaze_setting.priority_slider.value()) / 100.0

        self._set_running(True)
        self.status_lbl.setText("Đang khởi động AI Engine...")

        # Reset stats
        for card in (self.stat_total, self.stat_groups, self.stat_selected, self.stat_success):
            card.update_value("—")
        self.gaze_pct_lbl.setText("—")
        self.gaze_bar_widget.setValue(0)

        self.log_panel.add_log(
            f"Bắt đầu | Ngưỡng: {threshold}s | EAR: {ear} | "
            f"Gaze Yaw: {gaze_yaw}° | Pitch: {gaze_pitch}° | Ưu tiên nhìn nhau: {int(gaze_priority*100)}%", "info"
        )

        self.worker = CullingWorker(folder, threshold, ear, gaze_yaw, gaze_pitch, gaze_priority)
        self.worker.progress.connect(self.on_progress)
        self.worker.image_preview.connect(self.preview.set_image)
        self.worker.gaze_update.connect(self._on_gaze_update)
        self.worker.ai_stat.connect(self._on_stat_update)
        self.worker.log_message.connect(self.log_panel.add_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def reset_app(self):
        """Dọn sạch dữ liệu để lọc bộ khác."""
        # 1. Reset logic/folder
        self.drop_zone.reset_state()
        self.folder_path = None
        if hasattr(self, 'btn_open_out'):
            self.btn_open_out.hide()
        
        # 2. Reset Stats
        for card in (self.stat_total, self.stat_groups, self.stat_selected, self.stat_success):
            card.update_value("—")
        self.gaze_pct_lbl.setText("—")
        self.gaze_bar_widget.setValue(0)
        
        # 3. Reset Preview & Progress
        self.preview._show_placeholder()
        self.status_lbl.setText("Sẵn sàng — Kéo hoặc chọn thư mục ảnh để bắt đầu")
        self.progress_bar.setValue(0)
        self.btn_start.setEnabled(False)
        
        # 4. Clear Logs
        self.log_panel.clear_logs()
        self.log_panel.add_log("Đã dọn sạch dữ liệu. Sẵn sàng cho bộ ảnh mới.", "info")
        
        self.toast.show_message("Đã làm mới ứng dụng!", "info")

    def _set_running(self, r: bool):
        self.btn_start.setVisible(not r)
        if r:
            self.btn_pause.setText("⏸ Tạm Dừng")
            self.btn_pause.setEnabled(True)
            self.btn_cancel.setEnabled(True)
            self.btn_pause.show()
            self.btn_cancel.show()
        else:
            self.btn_pause.hide()
            self.btn_cancel.hide()
            self.btn_start.setEnabled(False)

        self.drop_zone.setEnabled(not r)
        self.time_slider.setEnabled(not r)
        self.ear_slider.setEnabled(not r)
        self.gaze_setting.yaw_slider.setEnabled(not r)
        self.gaze_setting.pitch_slider.setEnabled(not r)

    def toggle_pause(self):
        if not self.worker: return
        self.worker._is_paused = not self.worker._is_paused
        
        if self.worker._is_paused:
            self.btn_pause.setText("▶ Tiếp tục")
            self.status_lbl.setText("Đang tạm dừng...")
            self.log_panel.add_log("⏸ Đã tạm dừng tiến trình.", "warn")
        else:
            self.btn_pause.setText("⏸ Tạm Dừng")
            self.status_lbl.setText("Đang tiếp tục phân tích...")
            self.log_panel.add_log("▶ Tiếp tục tiến trình.", "info")

    def cancel_culling(self):
        if not self.worker: return
        self.worker._is_cancelled = True
        self.btn_pause.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.status_lbl.setText("Đang hủy tiến trình, vui lòng chờ...")
        self.log_panel.add_log("Vui lòng chờ, hệ thống đang dừng an toàn...", "warn")

    def closeEvent(self, event):
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker._is_cancelled = True
            self.worker.wait(2000)
        event.accept()

    def on_progress(self, current: int, total: int, text: str):
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
        else:
            self.progress_bar.setMaximum(0)
        self.status_lbl.setText(text)

    def _on_gaze_update(self, looking: int, total: int, ear_vals: list):
        """Cập nhật gaze indicator trong sidebar + preview bar."""
        self.preview.update_gaze_info(looking, total, ear_vals)
        if total > 0:
            pct = int(looking / total * 100)
            self.gaze_pct_lbl.setText(f"{pct}%")
            self.gaze_bar_widget.setValue(pct)
            if pct >= 70:
                color = COLOR_GAZE_OK
                icon = "😊"
            elif pct >= 40:
                color = COLOR_WARNING
                icon = "😐"
            else:
                color = "#EF4444"
                icon = "😶"
            self.gaze_pct_lbl.setStyleSheet(
                f"color:{color}; font-size:26px; font-weight:800; background:transparent;"
            )
            self.gaze_icon.setText(icon)
            self.gaze_bar_widget.setStyleSheet(f"""
                QProgressBar {{ background:{COLOR_BORDER}; border:none; border-radius:2px; }}
                QProgressBar::chunk {{ background:{color}; border-radius:2px; }}
            """)
        else:
            self.gaze_pct_lbl.setText("—")
            self.gaze_bar_widget.setValue(0)

    def _on_stat_update(self, stats: dict):
        if stats.get('total', 0):
            self.stat_total.update_value(str(stats['total']))
        if stats.get('groups', 0):
            self.stat_groups.update_value(str(stats['groups']))
        if stats.get('selected', 0):
            self.stat_selected.update_value(str(stats['selected']), COLOR_SUCCESS)
        if stats.get('success', 0):
            self.stat_success.update_value(str(stats['success']), "#0891B2")

    def on_finished(self, success: int, fail: int, no_space: int):
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(100)
        self._set_running(False)

        if no_space > 0:
            msg = (f"Hoàn thành: {success} ảnh copy thành công.\n"
                   f"⚠ {no_space} file bị bỏ qua do hết dung lượng đĩa.\n"
                   "Vui lòng giải phóng thêm dung lượng và thử lại.")
            self.status_lbl.setText(f"Hoàn thành — {success} OK | ⚠ {no_space} thiếu disk")
            self.log_panel.add_log(f"Hoàn thành: {success} OK, {no_space} disk full, {fail-no_space} lỗi", "warn")
            self.toast.show_message(msg, "warning", 8000)
        elif fail > 0:
            msg = f"Hoàn thành: {success} file RAW đã copy.\n{fail} file không tìm thấy RAW tương ứng."
            self.status_lbl.setText(f"Hoàn thành — {success} OK | {fail} không tìm thấy RAW")
            self.log_panel.add_log(f"Hoàn thành: {success} OK, {fail} lỗi", "warn")
            self.toast.show_message(msg, "warning", 6000)
        else:
            msg = f"Hoàn thành xuất sắc! Đã copy {success} file RAW vào thư mục [AI_SELECTED]."
            self.status_lbl.setText(f"✓ Hoàn thành — {success} files được chọn")
            self.log_panel.add_log(f"✓ Hoàn thành: {success} RAW files", "info")
            self.toast.show_message(msg, "success", 6000)

        if success > 0 and hasattr(self, 'btn_open_out'):
            self.btn_open_out.show()

    def on_error(self, err: str):
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self._set_running(False)
        friendly = self._friendly(err)
        self.status_lbl.setText("Lỗi — xem nhật ký bên dưới")
        self.log_panel.add_log(f"LỖI: {friendly}", "error")
        self.toast.show_message(friendly, "error", 8000)

    def open_output_folder(self):
        """Mở thư mục [AI_SELECTED] bằng trình quản lý tệp tin của OS."""
        import os, subprocess, platform
        if not hasattr(self.drop_zone, 'folder_path') or not self.drop_zone.folder_path:
            return
        out_dir = os.path.join(self.drop_zone.folder_path, "[AI_SELECTED]")
        if not os.path.exists(out_dir):
            self.toast.show_message(f"Không tìm thấy thư mục: {out_dir}", "error")
            return
            
        system = platform.system()
        try:
            if system == "Darwin":
                subprocess.call(["open", out_dir])
            elif system == "Windows":
                os.startfile(out_dir)
            else:
                subprocess.call(["xdg-open", out_dir])
        except Exception as e:
            self.log_panel.add_log(f"Lỗi mở thư mục: {e}", "error")

    @staticmethod
    def _friendly(raw: str) -> str:
        r = raw.lower()
        if "no space" in r or "errno 28" in r:
            return "Hết dung lượng đĩa. Vui lòng giải phóng thêm dung lượng."
        if "permission" in r or "errno 13" in r:
            return "Không có quyền truy cập thư mục đích."
        if "no such file" in r:
            return "Không tìm thấy thư mục hoặc file."
        if "yolo" in r or "model" in r:
            return "Lỗi nạp AI model. Kiểm tra file yolo26.pt và face_landmarker.task."
        if "proxy" in r or "jpg" in r:
            return "Không tìm thấy ảnh JPG proxy trong thư mục /JPG."
        return raw[:200]

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'toast'):
            cw = self.centralWidget()
            if cw:
                self.toast.move(
                    (cw.width() - self.toast.width()) // 2,
                    cw.height() - self.toast.height() - 16
                )
