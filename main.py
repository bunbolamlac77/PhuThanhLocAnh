import sys
import logging
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    app = QApplication(sys.argv)
    
    # Thiết lập Dark Mode thuần tuý
    app.setStyle("Fusion")
    dark_stylesheet = """
    QWidget {
        background-color: #121212;
        color: #ffffff;
    }
    QProgressBar {
        border: 2px solid #333333;
        border-radius: 5px;
        text-align: center;
        background-color: #2b2b2b;
    }
    QProgressBar::chunk {
        background-color: #05B8CC;
    }
    QSlider::groove:horizontal {
        border: 1px solid #bbb;
        background: #333333;
        height: 10px;
        border-radius: 4px;
    }
    QSlider::sub-page:horizontal {
        background: #05B8CC;
        border: 1px solid #777;
        height: 10px;
        border-radius: 4px;
    }
    QSlider::handle:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 #eee, stop:1 #ccc);
        border: 1px solid #777;
        width: 13px;
        margin-top: -2px;
        margin-bottom: -2px;
        border-radius: 4px;
    }
    """
    app.setStyleSheet(dark_stylesheet)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
