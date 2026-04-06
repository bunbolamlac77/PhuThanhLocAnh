"""
ai_vision.py — AI Vision Pipeline (YOLO + MediaPipe FaceLandmarker)

Cải tiến đã triển khai:
#1  Fix `all_eyes_open=True` khi no face → dùng `has_face` flag riêng
#9  Adaptive EAR Threshold — calibrate per-session, configurable từ ngoài
#10 Gaze Detection (Head Pose + Iris) — phát hiện nhìn thẳng/nhìn đi chỗ khác
"""

import cv2
import math
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Hằng số
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_EAR_THRESHOLD = 0.18   # Tối ưu người Việt/Châu Á
WESTERN_EAR_THRESHOLD = 0.22   # Người Tây mắt to hơn

# Gaze Detection — Head Pose thresholds (đơn vị: độ)
# Người "nhìn thẳng" khi yaw (ngang) < 25° và pitch (dọc) < 20°
GAZE_YAW_THRESHOLD   = 25.0   # Quay đầu trái/phải quá 25° → không nhìn thẳng
GAZE_PITCH_THRESHOLD = 20.0   # Ngước/cúi đầu quá 20° → không nhìn thẳng

# CẢI TIẾN 3.1: Tăng từ 0.22 lên 0.28 để thả lỏng với các cú liếc nhẹ.
IRIS_OFFSET_THRESHOLD = 0.28

# MediaPipe FaceLandmarker — Chỉ số mắt
LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Iris landmarks (chỉ có khi dùng refinement mode)
# Trung tâm iris: 468 (trái), 473 (phải)
LEFT_IRIS_CENTER  = 468
RIGHT_IRIS_CENTER = 473

# Góc mắt để tính gaze offset
LEFT_EYE_INNER  = 133   # Khóe trong mắt trái
LEFT_EYE_OUTER  = 33    # Khóe ngoài mắt trái
RIGHT_EYE_INNER = 362   # Khóe trong mắt phải
RIGHT_EYE_OUTER = 263   # Khóe ngoài mắt phải

# Điểm mũi để tính head pose đơn giản (khi không có transformation matrix)
NOSE_TIP      = 1
LEFT_EAR_LM   = 234
RIGHT_EAR_LM  = 454


# Gaze States — MỚI
GAZE_STATE_AWAY        = 0
GAZE_STATE_CAMERA      = 1
GAZE_STATE_INTERACTION = 2  # Nhìn nhau (cho dâu rể)

# ─────────────────────────────────────────────────────────────────────────────
# Gaze Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_head_pose_from_landmarks(landmarks, img_w: int, img_h: int) -> Tuple[float, float]:
    """
    Ước tính head pose (yaw, pitch) từ landmarks MediaPipe khi không có
    transformation matrix.

    Phương pháp: So sánh tỷ lệ khoảng cách mũi-tai trái/phải.
    - Nếu mặt quay trái: (nose→left_ear) ngắn hơn (nose→right_ear)
    - Nếu mặt quay phải: ngược lại

    Returns:
        (yaw_deg, pitch_deg) — âm = quay trái/cúi xuống, dương = quay phải/ngước lên
    """
    def lm_pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * img_w, lm.y * img_h])

    nose   = lm_pt(NOSE_TIP)
    l_ear  = lm_pt(LEFT_EAR_LM)
    r_ear  = lm_pt(RIGHT_EAR_LM)

    dist_l = np.linalg.norm(nose - l_ear)
    dist_r = np.linalg.norm(nose - r_ear)
    total  = dist_l + dist_r + 1e-6

    # Yaw: tỷ lệ bất đối xứng trái-phải → ±45°
    asymmetry = (dist_r - dist_l) / total   # -1..1
    yaw_deg   = asymmetry * 45.0

    # Pitch: dùng vị trí y của mũi so với trung điểm 2 tai
    mid_ear_y = (l_ear[1] + r_ear[1]) / 2.0
    img_height = img_h
    dy = nose[1] - mid_ear_y
    pitch_deg = (dy / (img_height * 0.05 + 1e-6)) * 15.0
    pitch_deg = max(-40.0, min(40.0, pitch_deg))

    return float(yaw_deg), float(pitch_deg)


def _estimate_iris_gaze(landmarks, img_w: int, img_h: int) -> Tuple[float, float]:
    """
    Tính gaze offset bằng tỷ lệ lòng trắng (Sclera Ratio).
    Nếu tròng đen lệch quá xa trung tâm mắt -> Liếc.
    """
    if len(landmarks) < 478:
        return 0.0, 0.0

    def lm2d(idx):
        return landmarks[idx].x, landmarks[idx].y

    def get_eye_ratio(iris_idx, inner_idx, outer_idx):
        try:
            iris_x, _  = lm2d(iris_idx)
            inner_x, _ = lm2d(inner_idx)
            outer_x, _ = lm2d(outer_idx)
            
            # Tính khoảng cách từ iris đến 2 khóe mắt
            # Càng cân bằng (ratio ~ 0) thì càng nhìn thẳng
            d1 = abs(iris_x - inner_x)
            d2 = abs(iris_x - outer_x)
            ratio = abs(d1 - d2) / (d1 + d2 + 1e-6)
            return float(ratio)
        except Exception:
            return 0.0

    # 468 = Left Iris, 133 = Inner, 33 = Outer
    offset_l = get_eye_ratio(LEFT_IRIS_CENTER, LEFT_EYE_INNER, LEFT_EYE_OUTER)
    # 473 = Right Iris, 362 = Inner, 263 = Outer
    offset_r = get_eye_ratio(RIGHT_IRIS_CENTER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER)

    return offset_l, offset_r


# ─────────────────────────────────────────────────────────────────────────────
# AIVisionPipeline
# ─────────────────────────────────────────────────────────────────────────────

class AIVisionPipeline:
    def __init__(self,
                 yolo_model_path: str = "yolo26.pt",
                 ear_threshold: float = DEFAULT_EAR_THRESHOLD,
                 gaze_yaw_threshold: float = GAZE_YAW_THRESHOLD,
                 gaze_pitch_threshold: float = GAZE_PITCH_THRESHOLD):
        """
        Khởi tạo AI Vision Pipeline.

        Args:
            yolo_model_path: Đường dẫn YOLO model.
            ear_threshold: Ngưỡng EAR mắt nhắm/mở (từ UI slider).
            gaze_yaw_threshold: Góc quay đầu ngang tối đa để coi là "nhìn thẳng" (độ).
            gaze_pitch_threshold: Góc cúi/ngước đầu tối đa để coi là "nhìn thẳng" (độ).
        """
        self.ear_threshold         = ear_threshold
        self.gaze_yaw_threshold    = gaze_yaw_threshold
        self.gaze_pitch_threshold  = gaze_pitch_threshold
        self.iris_threshold        = IRIS_OFFSET_THRESHOLD # Mặc định 0.22
        self.model_yolo: Optional[YOLO] = None
        self.face_mesh: Optional[mp_vision.FaceLandmarker] = None

        from core.analyzer import (
            calculate_sharpness,
            calculate_exposure_score,
            detect_motion_blur_fft
        )
        from core.dedup import compute_phash_from_image
        self._calc_sharpness = calculate_sharpness
        self._calc_exposure  = calculate_exposure_score
        self._calc_fft       = detect_motion_blur_fft
        self._calc_phash     = compute_phash_from_image

        # Init MediaPipe — BẬT facial_transformation_matrixes để head pose chính xác hơn
        try:
            base_options = mp_python.BaseOptions(model_asset_path='face_landmarker.task')
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=True,   # ← BẬT để head pose
                num_faces=10
            )
            self.face_mesh = mp_vision.FaceLandmarker.create_from_options(options)
            logger.info("MediaPipe FaceLandmarker (+ transformation matrix) OK.")
        except Exception as e:
            logger.error(f"Lỗi MediaPipe: {e}")

        # Init YOLO
        try:
            self.model_yolo = YOLO(yolo_model_path)
            logger.info(f"YOLO '{yolo_model_path}' OK.")
        except Exception as e:
            logger.error(f"Không nạp được YOLO: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # EAR
    # ─────────────────────────────────────────────────────────────────────────

    def _calculate_ear(self, landmarks, eye_indices: List[int]) -> float:
        """Tính Eye Aspect Ratio (EAR) — mắt nhắm → EAR~0, mắt mở → EAR~0.25+"""
        p = [landmarks[i] for i in eye_indices]
        def dist(a, b): return math.hypot(a.x - b.x, a.y - b.y)
        v1 = dist(p[1], p[5])
        v2 = dist(p[2], p[4])
        h  = dist(p[0], p[3])
        return (v1 + v2) / (2.0 * h + 1e-6)

    # ─────────────────────────────────────────────────────────────────────────
    # Head pose từ transformation matrix
    # ─────────────────────────────────────────────────────────────────────────

    def _head_pose_from_matrix(self, transform_matrix) -> Tuple[float, float, float]:
        """
        Trích xuất góc Euler (yaw, pitch, roll) từ transformation matrix 4x4 của MediaPipe.

        MediaPipe trả về world-space matrix. Ta lấy phần rotation 3x3 và decompose.
        Returns: (yaw_deg, pitch_deg, roll_deg)
        """
        try:
            mat = np.array(transform_matrix.data).reshape(4, 4)
            R = mat[:3, :3]

            # Euler angles từ rotation matrix (ZXY convention)
            # Pitch (X-axis rotation)
            pitch = math.degrees(math.asin(max(-1.0, min(1.0, -R[1, 2]))))
            # Yaw (Y-axis rotation)
            yaw   = math.degrees(math.atan2(R[0, 2], R[2, 2]))
            # Roll (Z-axis rotation)
            roll  = math.degrees(math.atan2(R[1, 0], R[1, 1]))

            return float(yaw), float(pitch), float(roll)
        except Exception as e:
            logger.debug(f"Head pose matrix error: {e}")
            return 0.0, 0.0, 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Analyze Image — Full pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Phân tích toàn diện: YOLO + MediaPipe + Sharpness + Exposure + Gaze.
        Đọc ảnh 1 lần duy nhất để tối ưu tốc độ.
        """
        result: Dict[str, Any] = {
            'has_face': False,
            'faces_count': 0,
            'all_eyes_open': False,
            'open_eyes_count': 0,
            'closed_eyes_count': 0,
            'all_looking_at_camera': False,
            'looking_count': 0,
            'not_looking_count': 0,
            'face_bboxes': [],
            'face_details': [],    # List[Dict] chứa info chi tiết từng mặt
            'primary_bbox': None,
            'ear_values': [],
            'sharpness': 0.0,      # Tổng quan hoặc mặt chính
            'exposure_score': 0.5,
            'motion_blur': False,
            'fft_score': 999.0,
            'annotated_frame': None,
        }

        try:
            # ĐỌC ẢNH 1 LẦN DUY NHẤT
            frame = cv2.imread(image_path)
            if frame is None:
                return result

            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 1. Các chỉ số cơ bản (Dùng ảnh đã load)
            result['exposure_score'] = self._calc_exposure(image=gray)

            # 2. YOLO: detect faces
            yolo_bboxes = []
            if self.model_yolo is not None:
                try:
                    # Chạy trên ảnh scale nhỏ hơn để nhanh hơn nếu cần, nhưng YOLO thường tự làm
                    yl = self.model_yolo.predict(source=rgb, device='mps', verbose=False)
                    for r in yl:
                        for box in r.boxes:
                            yolo_bboxes.append(tuple(map(int, box.xyxy[0])))
                except Exception: pass

            # 3. MediaPipe: Face Mesh & Gaze
            if self.face_mesh is None:
                return result

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            fm_res = self.face_mesh.detect(mp_img)

            if not fm_res.face_landmarks:
                # Nếu không có mặt, vẫn tính sharpness full-frame
                result['sharpness'] = self._calc_sharpness(image=gray)
                is_blur, fft = self._calc_fft(image=gray)
                result['motion_blur'] = is_blur
                result['fft_score'] = fft
                return result

            n_faces = len(fm_res.face_landmarks)
            result['has_face'] = True
            result['faces_count'] = n_faces

            has_matrices = (fm_res.facial_transformation_matrixes is not None
                            and len(fm_res.facial_transformation_matrixes) == n_faces)

            all_open = True
            all_looking = True
            face_details = []

            for i, lms in enumerate(fm_res.face_landmarks):
                # EAR
                l_ear = self._calculate_ear(lms, LEFT_EYE_INDICES)
                r_ear = self._calculate_ear(lms, RIGHT_EYE_INDICES)
                avg_ear = (l_ear + r_ear) / 2.0
                is_closed = avg_ear < self.ear_threshold

                # Bbox
                xs = [lm.x * w for lm in lms]
                ys = [lm.y * h for lm in lms]
                bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

                # Head Pose
                if has_matrices:
                    yaw, pitch, roll = self._head_pose_from_matrix(fm_res.facial_transformation_matrixes[i])
                else:
                    yaw, pitch = _estimate_head_pose_from_landmarks(lms, w, h)
                    roll = 0.0

                # Gaze Camera Check
                is_looking_cam = (abs(yaw) <= self.gaze_yaw_threshold and abs(pitch) <= self.gaze_pitch_threshold)
                if is_looking_cam:
                    ir_l, ir_r = _estimate_iris_gaze(lms, w, h)
                    if (ir_l + ir_r) / 2.0 > self.iris_threshold:
                        is_looking_cam = False

                face_info = {
                    'bbox': bbox,
                    'ear': avg_ear,
                    'is_eyes_closed': is_closed,
                    'yaw': yaw,
                    'pitch': pitch,
                    'is_looking_cam': is_looking_cam,
                    'center_x': (bbox[0] + bbox[2]) / 2.0,
                    'body_hash': None,
                }

                # POSE HASHING: Băm vùng cơ thể bên dưới mặt
                # Lấy vùng từ dưới cằm xuống ngực (khoảng 2 lần chiều cao mặt)
                try:
                    fx1, fy1, fx2, fy2 = bbox
                    fw = fx2 - fx1
                    fh = fy2 - fy1
                    # Vùng Pose: rộng như mặt, cao gấp 1.5 lần mặt
                    px1 = max(0, fx1 - int(fw*0.2))
                    py1 = fy2 
                    px2 = min(w, fx2 + int(fw*0.2))
                    py2 = min(h, fy2 + int(fh*1.5))
                    
                    if py2 > py1 and px2 > px1:
                        body_crop = gray[py1:py2, px1:px2]
                        face_info['body_hash'] = self._calc_phash(body_crop)
                except Exception: pass

                face_details.append(face_info)

                if is_closed:
                    result['closed_eyes_count'] += 1
                    all_open = False
                else:
                    result['open_eyes_count'] += 1

                if is_looking_cam:
                    result['looking_count'] += 1
                else:
                    result['not_looking_count'] += 1
                    all_looking = False

            # Gaze Interaction (cho cặp đôi)
            # Nếu có 2 người, kiểm tra xem họ có nhìn nhau không
            if n_faces == 2:
                f1, f2 = (face_details[0], face_details[1]) if face_details[0]['center_x'] < face_details[1]['center_x'] else (face_details[1], face_details[0])
                # f1 bên trái, f2 bên phải
                # f1 nhìn phải (yaw dương), f2 nhìn trái (yaw âm) -> nhìn nhau
                if f1['yaw'] > 15 and f2['yaw'] < -15:
                    f1['is_interacting'] = True
                    f2['is_interacting'] = True
                else:
                    f1['is_interacting'] = f2['is_interacting'] = False
            else:
                for f in face_details: f['is_interacting'] = False

            result['face_details'] = face_details
            result['all_eyes_open'] = all_open
            result['all_looking_at_camera'] = all_looking
            result['face_bboxes'] = yolo_bboxes if yolo_bboxes else [f['bbox'] for f in face_details]
            
            # Primary face stats
            if result['face_bboxes']:
                p_idx = 0
                max_a = 0
                for idx, f in enumerate(face_details):
                    b = f['bbox']
                    area = (b[2]-b[0]) * (b[3]-b[1])
                    if area > max_a:
                        max_a = area
                        p_idx = idx
                
                primary_f = face_details[p_idx]
                result['primary_bbox'] = primary_f['bbox']
                result['ear_values'] = [f['ear'] for f in face_details]
                
                # Tính sharpness vùng mặt chính
                result['sharpness'] = self._calc_sharpness(image=gray, face_bbox=primary_f['bbox'])
                is_blur, fft = self._calc_fft(image=gray, face_bbox=primary_f['bbox'])
                result['motion_blur'] = is_blur
                result['fft_score'] = fft

            # Annotation (Nhanh cho preview)
            for f in face_details:
                b = f['bbox']
                color = (0, 210, 90) if f['is_looking_cam'] else ((0, 140, 255) if f['is_interacting'] else (0, 0, 255))
                label = "CAM" if f['is_looking_cam'] else ("INT" if f['is_interacting'] else "AWAY")
                if f['is_eyes_closed']: 
                    color = (255, 0, 0)
                    label = "CLOSED"
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 2)
                cv2.putText(frame, label, (b[0], b[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            scale = min(1.0, 640.0 / w)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            result['annotated_frame'] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        except Exception as e:
            logger.error(f"Lỗi AI Vision: {e}")

        return result
