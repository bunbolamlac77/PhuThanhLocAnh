"""
components.py — UI Components
Theme: White Luxury, Ergonomic Layout
"""

import os
import numpy as np
from PySide6.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QHBoxLayout,
    QFrame, QGraphicsDropShadowEffect, QSizePolicy, QScrollArea
)
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QPixmap, QImage, QColor, QFont

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────
COLOR_BG             = "#F8FAFC"
COLOR_SURFACE        = "#FFFFFF"
COLOR_BORDER         = "#E2E8F0"
COLOR_BORDER_DARK    = "#CBD5E1"
COLOR_TEXT_PRIMARY   = "#0F172A"
COLOR_TEXT_SECONDARY = "#475569"
COLOR_TEXT_MUTED     = "#94A3B8"
COLOR_ACCENT         = "#4F46E5"
COLOR_ACCENT_LIGHT   = "#EEF2FF"
COLOR_SUCCESS        = "#10B981"
COLOR_SUCCESS_BG     = "#ECFDF5"
COLOR_SUCCESS_BORDER = "#D1FAE5"
COLOR_WARNING        = "#F59E0B"
COLOR_WARNING_BG     = "#FFFBEB"
COLOR_WARNING_BORDER = "#FEF3C7"
COLOR_ERROR          = "#EF4444"
COLOR_ERROR_BG       = "#FEF2F2"
COLOR_ERROR_BORDER   = "#FEE2E2"
COLOR_INFO_BG        = "#EFF6FF"
COLOR_INFO_BORDER    = "#DBEAFE"
COLOR_DROP_ACTIVE    = "#4F46E5"
COLOR_GAZE_OK        = "#10B981"   # Màu nhìn thẳng
COLOR_GAZE_AWAY      = "#F97316"   # Màu nhìn đi chỗ khác


def _shadow(widget, radius=24, color="#0000000F", offset=(0, 4)):
    sh = QGraphicsDropShadowEffect(widget)
    sh.setBlurRadius(radius)
    sh.setColor(QColor(color))
    sh.setOffset(*offset)
    widget.setGraphicsEffect(sh)


# ─────────────────────────────────────────────────────────────────────────────
# Toast Notification
# ─────────────────────────────────────────────────────────────────────────────
class ToastNotification(QFrame):
    ICONS   = {'success': '✓', 'error': '✕', 'warning': '⚠', 'info': 'ℹ'}
    PALETTE = {
        'success': (COLOR_SUCCESS,  COLOR_SUCCESS_BG,  COLOR_SUCCESS_BORDER),
        'error':   (COLOR_ERROR,    COLOR_ERROR_BG,    COLOR_ERROR_BORDER),
        'warning': (COLOR_WARNING,  COLOR_WARNING_BG,  COLOR_WARNING_BORDER),
        'info':    (COLOR_ACCENT,   COLOR_INFO_BG,     COLOR_INFO_BORDER),
    }

    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedWidth(460)
        self.setMinimumHeight(56)

        hl = QHBoxLayout(self)
        hl.setContentsMargins(14, 10, 14, 10)
        hl.setSpacing(10)

        self.icon_lbl = QLabel()
        self.icon_lbl.setFixedSize(26, 26)
        self.icon_lbl.setAlignment(Qt.AlignCenter)

        self.msg_lbl = QLabel()
        self.msg_lbl.setWordWrap(True)

        close = QLabel("×")
        close.setFixedSize(18, 18)
        close.setAlignment(Qt.AlignCenter)
        close.setCursor(Qt.PointingHandCursor)
        close.setStyleSheet(f"color:{COLOR_TEXT_MUTED}; font-size:17px;")
        close.mousePressEvent = lambda e: self.hide()

        hl.addWidget(self.icon_lbl)
        hl.addWidget(self.msg_lbl, 1)
        hl.addWidget(close)

        _shadow(self, radius=20, color="#00000022", offset=(0, 6))
        self.hide()

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_message(self, text: str, kind: str = 'info', duration_ms: int = 5000):
        fg, bg, border = self.PALETTE.get(kind, self.PALETTE['info'])
        self.icon_lbl.setText(self.ICONS.get(kind, 'ℹ'))
        self.icon_lbl.setStyleSheet(
            f"background:{fg}18; color:{fg}; border-radius:13px; font-size:13px; font-weight:700;"
        )
        self.msg_lbl.setText(text)
        self.msg_lbl.setStyleSheet(f"color:{COLOR_TEXT_PRIMARY}; font-size:12px; line-height:1.4;")
        self.setStyleSheet(
            f"QFrame{{background:{bg}; border:1.5px solid {border}; border-radius:12px;}}"
        )
        if self.parent():
            pw, ph = self.parent().width(), self.parent().height()
            self.move((pw - self.width()) // 2, ph - self.height() - 20)
        self.show()
        self.raise_()
        self._timer.start(duration_ms)


# ─────────────────────────────────────────────────────────────────────────────
# Stat Card — hiển thị số liệu nhỏ (Ergonomic Stats Panel)
# ─────────────────────────────────────────────────────────────────────────────
class StatCard(QFrame):
    """Card hiển thị 1 chỉ số: icon + giá trị + label."""

    def __init__(self, icon: str, label: str, value: str = "—",
                 accent: str = COLOR_ACCENT, parent=None):
        super().__init__(parent)
        self._accent = accent
        self.setObjectName("statCard")
        self.setStyleSheet(f"""
            QFrame#statCard {{
                background: {COLOR_SURFACE};
                border: 1px solid {COLOR_BORDER};
                border-radius: 10px;
            }}
        """)
        _shadow(self, radius=10, color="#00000010", offset=(0, 2))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(74)

        vl = QVBoxLayout(self)
        vl.setContentsMargins(14, 10, 14, 10)
        vl.setSpacing(2)

        top = QHBoxLayout()
        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet(
            f"background:{accent}18; color:{accent}; border-radius:8px; "
            f"font-size:14px; padding:3px 5px;"
        )
        icon_lbl.setFixedSize(28, 28)
        icon_lbl.setAlignment(Qt.AlignCenter)
        top.addWidget(icon_lbl)
        top.addStretch()
        vl.addLayout(top)

        self.val_lbl = QLabel(value)
        self.val_lbl.setStyleSheet(
            f"color:{COLOR_TEXT_PRIMARY}; font-size:20px; font-weight:800;"
        )

        self.lbl_lbl = QLabel(label)
        self.lbl_lbl.setStyleSheet(
            f"color:{COLOR_TEXT_MUTED}; font-size:10px; font-weight:600; letter-spacing:0.4px;"
        )

        vl.addWidget(self.val_lbl)
        vl.addWidget(self.lbl_lbl)

    def update_value(self, value: str, color: str = None):
        self.val_lbl.setText(value)
        if color:
            self.val_lbl.setStyleSheet(
                f"color:{color}; font-size:20px; font-weight:800;"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Image Preview Widget
# ─────────────────────────────────────────────────────────────────────────────
class ImagePreviewWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("previewCard")
        self.setStyleSheet(f"""
            QFrame#previewCard {{
                background:{COLOR_SURFACE};
                border:1px solid {COLOR_BORDER};
                border-radius:14px;
            }}
        """)
        _shadow(self, radius=14, color="#00000010", offset=(0, 3))

        vl = QVBoxLayout(self)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        # Mac-style header
        hdr = QWidget()
        hdr.setFixedHeight(36)
        hdr.setStyleSheet(f"""
            background:{COLOR_BG};
            border-radius:14px 14px 0 0;
            border-bottom:1px solid {COLOR_BORDER};
        """)
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(12, 0, 12, 0)
        dots = QHBoxLayout()
        dots.setSpacing(6)
        for c in ["#FF5F57", "#FFBD2E", "#28C840"]:
            d = QLabel()
            d.setFixedSize(10, 10)
            d.setStyleSheet(f"background:{c}; border-radius:5px;")
            dots.addWidget(d)
        dots.addStretch()
        title = QLabel("AI Vision Preview · Real-time Analysis")
        title.setStyleSheet(
            f"color:{COLOR_TEXT_MUTED}; font-size:11px; font-weight:600; letter-spacing:0.3px;"
        )
        hl.addLayout(dots)
        hl.addStretch()
        hl.addWidget(title)
        hl.addStretch()
        vl.addWidget(hdr)

        # Image area container
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(self.container, 1)

        self.image_label = QLabel(self.container)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(360, 220)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._show_placeholder()
        self.container_layout.addWidget(self.image_label)

        # Floating Gaze Overlay (MỚI - Glassmorphism)
        self.info_overlay = QFrame(self.container)
        self.info_overlay.setFixedWidth(320)
        self.info_overlay.setFixedHeight(32)
        self.info_overlay.setStyleSheet(f"""
            QFrame {{
                background: rgba(255, 255, 255, 0.7);
                border: 1px solid rgba(255, 255, 255, 0.4);
                border-radius: 16px;
                backdrop-filter: blur(10px);
            }}
        """)
        _shadow(self.info_overlay, radius=12, color="#00000020", offset=(0, 2))

        ol = QHBoxLayout(self.info_overlay)
        ol.setContentsMargins(12, 0, 12, 0)
        ol.setSpacing(10)

        self.gaze_status = QLabel("👁 —")
        self.gaze_status.setStyleSheet(f"color:{COLOR_TEXT_PRIMARY}; font-size:11px; font-weight:700; background:transparent;")
        self.ear_status = QLabel("EAR: —")
        self.ear_status.setStyleSheet(f"color:{COLOR_TEXT_SECONDARY}; font-size:10px; background:transparent;")
        self.face_count_lbl = QLabel("Faces: —")
        self.face_count_lbl.setStyleSheet(f"color:{COLOR_TEXT_MUTED}; font-size:10px; background:transparent;")

        ol.addStretch()
        ol.addWidget(self.gaze_status)
        ol.addSpacing(4)
        ol.addWidget(self.ear_status)
        ol.addSpacing(4)
        ol.addWidget(self.face_count_lbl)
        ol.addStretch()
        
        self.info_overlay.hide()
        self._current_pixmap = None

    def _show_placeholder(self):
        self.image_label.setText(
            f"<div style='text-align:center; padding:20px;'>"
            f"<p style='font-size:32px; margin:0;'>📷</p>"
            f"<p style='color:{COLOR_TEXT_MUTED}; font-size:12px; margin:8px 0 0 0;'>"
            f"Ảnh phân tích AI sẽ xuất hiện ở đây</p>"
            f"<p style='color:{COLOR_BORDER_DARK}; font-size:10px; margin:4px 0 0 0;'>"
            f"Màu xanh = mở mắt + nhìn thẳng · Đỏ = nhắm mắt · Cam = nhìn lệch</p>"
            f"</div>"
        )
        self.image_label.setTextFormat(Qt.RichText)

    def set_image(self, rgb_image: np.ndarray):
        if rgb_image is None:
            return
        h, w, ch = rgb_image.shape
        q = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        self._current_pixmap = QPixmap.fromImage(q)
        self._scale_to_fit()

    def update_gaze_info(self, looking: int, total: int, ear_vals: list):
        """Cập nhật thanh thông tin gaze + EAR trong Floating Overlay."""
        if total == 0:
            self.info_overlay.hide()
            return

        self.info_overlay.show()
        self.face_count_lbl.setText(f"· {total} mặt")

        if looking == total:
            self.gaze_status.setText(f"👁 Nhìn thẳng ({looking}/{total})")
            self.gaze_status.setStyleSheet(
                f"color:{COLOR_GAZE_OK}; font-size:11px; font-weight:700; background:transparent;"
            )
        else:
            not_look = total - looking
            self.gaze_status.setText(f"👁 Nhìn lệch ({not_look}/{total})")
            self.gaze_status.setStyleSheet(
                f"color:{COLOR_GAZE_AWAY}; font-size:11px; font-weight:700; background:transparent;"
            )

        if ear_vals:
            avg = sum(ear_vals) / len(ear_vals)
            self.ear_status.setText(f"EAR avg: {avg:.3f}")

    def _scale_to_fit(self):
        if self._current_pixmap:
            self.image_label.setPixmap(
                self._current_pixmap.scaled(
                    self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_to_fit()
        # Position the info overlay (floating top-center)
        if hasattr(self, 'info_overlay'):
            x = (self.container.width() - self.info_overlay.width()) // 2
            self.info_overlay.move(x, 12)


# ─────────────────────────────────────────────────────────────────────────────
# Drop Zone Widget
# ─────────────────────────────────────────────────────────────────────────────
class DropZoneWidget(QFrame):
    _IDLE = f"""QFrame{{
        background:{COLOR_SURFACE}; border:2px dashed {COLOR_BORDER_DARK};
        border-radius:12px;}}"""
    _HOVER = f"""QFrame{{
        background:{COLOR_INFO_BG}; border:2px dashed {COLOR_DROP_ACTIVE};
        border-radius:12px;}}"""
    _DONE = f"""QFrame{{
        background:{COLOR_SUCCESS_BG}; border:2px solid {COLOR_SUCCESS};
        border-radius:12px;}}"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.folder_path   = None
        self.parent_window = parent
        self.setStyleSheet(self._IDLE)
        self.setMinimumHeight(90)
        self.setMaximumHeight(108)
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        _shadow(self, radius=8, color="#0000000C", offset=(0, 2))

        hl = QHBoxLayout(self)
        hl.setContentsMargins(20, 0, 20, 0)
        hl.setSpacing(14)

        self.icon_lbl = QLabel("📂")
        self.icon_lbl.setStyleSheet(
            "font-size:30px; border:none; background:transparent;"
        )
        self.icon_lbl.setFixedWidth(44)

        txt_col = QVBoxLayout()
        txt_col.setSpacing(3)
        self.title_lbl = QLabel("Kéo & thả thư mục ảnh vào đây")
        self.title_lbl.setStyleSheet(
            f"color:{COLOR_TEXT_PRIMARY}; font-size:14px; font-weight:700; "
            f"border:none; background:transparent;"
        )
        self.sub_lbl = QLabel("Thư mục phải chứa thư mục con /JPG · Hoặc click để chọn")
        self.sub_lbl.setStyleSheet(
            f"color:{COLOR_TEXT_MUTED}; font-size:11px; border:none; background:transparent;"
        )
        txt_col.addWidget(self.title_lbl)
        txt_col.addWidget(self.sub_lbl)

        hl.addWidget(self.icon_lbl)
        hl.addLayout(txt_col, 1)

    def mousePressEvent(self, event):
        from PySide6.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(self, "Chọn thư mục ảnh")
        if path:
            self._set_folder(path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet(self._HOVER)

    def dragLeaveEvent(self, event):
        self.setStyleSheet(self._DONE if self.folder_path else self._IDLE)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if os.path.isdir(path):
                self._set_folder(path)

    def _set_folder(self, path: str):
        self.folder_path = path
        self.setStyleSheet(self._DONE)
        name = os.path.basename(path)
        self.icon_lbl.setText("✅")
        self.title_lbl.setText(f"📁  {name}")
        self.title_lbl.setStyleSheet(
            f"color:{COLOR_SUCCESS}; font-size:14px; font-weight:700; "
            f"border:none; background:transparent;"
        )
        self.sub_lbl.setText(path)
        self.sub_lbl.setStyleSheet(
            f"color:{COLOR_TEXT_SECONDARY}; font-size:10px; "
            f"border:none; background:transparent;"
        )
        if self.parent_window and hasattr(self.parent_window, 'on_folder_selected'):
            self.parent_window.on_folder_selected(path)

    def reset_state(self):
        """Đưa vùng kéo thả về trạng thái ban đầu."""
        self.folder_path = None
        self.setStyleSheet(self._IDLE)
        self.icon_lbl.setText("📂")
        self.title_lbl.setText("Kéo & thả thư mục ảnh vào đây")
        self.title_lbl.setStyleSheet(
            f"color:{COLOR_TEXT_PRIMARY}; font-size:14px; font-weight:700; "
            f"border:none; background:transparent;"
        )
        self.sub_lbl.setText("Thư mục phải chứa thư mục con /JPG · Hoặc click để chọn")
        self.sub_lbl.setStyleSheet(
            f"color:{COLOR_TEXT_MUTED}; font-size:11px; border:none; background:transparent;"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Log Panel
# ─────────────────────────────────────────────────────────────────────────────
class LogPanel(QFrame):
    MAX_LINES = 80

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("logPanel")
        self.setStyleSheet(f"""
            QFrame#logPanel {{
                background:{COLOR_SURFACE}; border:1px solid {COLOR_BORDER};
                border-radius:10px;
            }}
            QScrollArea {{ background:transparent; border:none; }}
        """)
        self.setFixedHeight(110)
        _shadow(self, radius=6, color="#0000000A", offset=(0, 1))

        vl = QVBoxLayout(self)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        hdr = QWidget()
        hdr.setFixedHeight(28)
        hdr.setStyleSheet(
            f"background:{COLOR_BG}; border-radius:10px 10px 0 0; "
            f"border-bottom:1px solid {COLOR_BORDER};"
        )
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(12, 0, 12, 0)
        lbl = QLabel("📋  Nhật ký hoạt động")
        lbl.setStyleSheet(f"color:{COLOR_TEXT_MUTED}; font-size:10px; font-weight:700; letter-spacing:0.3px;")
        hl.addWidget(lbl)
        hl.addStretch()
        vl.addWidget(hdr)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._container = QWidget()
        self._container.setStyleSheet("background:transparent;")
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(12, 6, 12, 6)
        self._layout.setSpacing(2)
        self._layout.addStretch()
        self.scroll.setWidget(self._container)
        vl.addWidget(self.scroll)
        self._lines = []

    def add_log(self, message: str, level: str = "info"):
        STYLE = {
            "info":  f"color:{COLOR_TEXT_SECONDARY};",
            "warn":  f"color:{COLOR_WARNING}; font-weight:600;",
            "error": f"color:{COLOR_ERROR}; font-weight:600;",
        }
        PREFIX = {"info": "·", "warn": "⚠", "error": "✕"}
        lbl = QLabel(f"{PREFIX.get(level,'·')}  {message}")
        lbl.setStyleSheet(f"font-size:10px; {STYLE.get(level, STYLE['info'])} background:transparent;")
        lbl.setWordWrap(True)
        self._layout.insertWidget(self._layout.count() - 1, lbl)
        self._lines.append(lbl)
        if len(self._lines) > self.MAX_LINES:
            old = self._lines.pop(0)
            old.deleteLater()
        QTimer.singleShot(40, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()
        ))

    def clear_logs(self):
        """Xóa toàn bộ nhật ký hiện có."""
        for lbl in self._lines:
            lbl.deleteLater()
        self._lines = []
