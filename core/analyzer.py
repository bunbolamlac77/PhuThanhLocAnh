import cv2
import exifread
import logging
import os
import numpy as np
from datetime import datetime
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────────
# 1. EXIF Reader
# ───────────────────────────────────────────────────────────────────────────────

def get_exif_datetime(filepath: str) -> float:
    """
    Đọc EXIF DateTimeOriginal + SubSec và chuyển đổi về timestamp (giây, có thập phân).
    Fallback về mtime nếu không có EXIF.
    """
    try:
        with open(filepath, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            if 'EXIF DateTimeOriginal' in tags:
                date_str = str(tags['EXIF DateTimeOriginal']).split('.')[0]
                dt = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                timestamp = dt.timestamp()

                # Sub-second precision để xử lý burst grouping
                for key in ['EXIF SubSecTimeOriginal', 'EXIF SubSecTimeDigitized', 'EXIF SubSecTime']:
                    if key in tags:
                        try:
                            subsec_val = str(tags[key]).replace('\x00', '').strip()
                            if subsec_val.isdigit():
                                timestamp += float("0." + subsec_val)
                                break
                        except Exception:
                            pass

                return timestamp
    except Exception as e:
        logger.warning(f"Không thể đọc EXIF từ {filepath}: {e}")

    try:
        return os.path.getmtime(filepath)
    except Exception as e:
        logger.error(f"Lỗi đọc mtime {filepath}: {e}")
        return 0.0


# ───────────────────────────────────────────────────────────────────────────────
# 2. Sharpness: Face-Crop Laplacian (CẢI TIẾN #3)
# ───────────────────────────────────────────────────────────────────────────────

def calculate_sharpness(filepath: Optional[str] = None,
                        face_bbox: Optional[Tuple[int, int, int, int]] = None,
                        image: Optional[np.ndarray] = None) -> float:
    """
    Tính độ sắc nét (Laplacian variance).
    - Ưu tiên dùng 'image' (numpy array) nếu có để tránh đọc file.
    - Nếu không, đọc từ 'filepath'.
    - Nếu có face_bbox: tính chỉ trên vùng khuôn mặt.
    """
    try:
        if image is None:
            if filepath is None: return 0.0
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            return 0.0

        # Đảm bảo là grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            h, w = image.shape
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                return float(cv2.Laplacian(crop, cv2.CV_64F).var())

        return float(cv2.Laplacian(image, cv2.CV_64F).var())
    except Exception as e:
        logger.error(f"Lỗi tính sharpness: {e}")
        return 0.0


# ───────────────────────────────────────────────────────────────────────────────
# 3. FFT Motion Blur Detection (CẢI TIẾN #8)
# ───────────────────────────────────────────────────────────────────────────────

def detect_motion_blur_fft(filepath: Optional[str] = None,
                           face_bbox: Optional[Tuple[int, int, int, int]] = None,
                           threshold: float = 10.0,
                           image: Optional[np.ndarray] = None) -> Tuple[bool, float]:
    """
    Phát hiện motion blur bằng FFT. Ưu tiên dùng 'image' numpy array.
    """
    try:
        if image is None:
            if filepath is None: return False, 999.0
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            return False, 999.0

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        region = image
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            h, w = image.shape
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 > x1 and y2 > y1:
                region = image[y1:y2, x1:x2]

        f = np.fft.fft2(region.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1e-6)

        rows, cols = magnitude.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones_like(magnitude, dtype=bool)
        r = min(rows, cols) // 5
        if r > 0:
            mask[crow - r:crow + r, ccol - r:ccol + r] = False
        
        fft_score = float(magnitude[mask].mean()) if magnitude[mask].size > 0 else 999.0
        return fft_score < threshold, fft_score

    except Exception as e:
        logger.error(f"Lỗi FFT: {e}")
        return False, 999.0


# ───────────────────────────────────────────────────────────────────────────────
# 4. Exposure Score (phục vụ Composite Scoring)
# ───────────────────────────────────────────────────────────────────────────────

def calculate_exposure_score(filepath: Optional[str] = None,
                             image: Optional[np.ndarray] = None) -> float:
    """
    Đánh giá mức độ phơi sáng. Ưu tiên dùng 'image' numpy array.
    """
    try:
        if image is None:
            if filepath is None: return 0.5
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            return 0.5

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        mean_brightness = float(image.mean())
        score = np.exp(-((mean_brightness - 128) ** 2) / (2 * 60 ** 2))
        return float(score)
    except Exception as e:
        logger.error(f"Lỗi tính exposure: {e}")
        return 0.5
