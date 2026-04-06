"""
logic.py — Culling Decision Engine (Engine Quyết Định Lọc Ảnh)

Cải tiến đã triển khai:
#2  Chạy AI cho ảnh đơn (minimum quality filter) — không bỏ qua ảnh lẻ nữa
#3  Face-crop Sharpness — dùng primary_bbox từ AI để crop trước khi tính Laplacian
#4  pHash Deduplication — chia time-cluster thành sub-clusters theo nội dung
#5  Two-Pass Grouping — Pass 1: thời gian, Pass 2: pHash similarity
#6  Composite Scoring — điểm tổng hợp nhiều yếu tố thay vì chọn tuần tự
#7  Adaptive Time Threshold — detect burst gap tự động từ dataset
"""

import os
import cv2
import logging
import math
import statistics
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from core.analyzer import (
    get_exif_datetime,
    calculate_sharpness,
    detect_motion_blur_fft,
    calculate_exposure_score,
)
from core.dedup import deduplicate_within_group, hamming_distance

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────────
# Hằng số & trọng số Composite Scoring (CẢI TIẾN #6)
# ───────────────────────────────────────────────────────────────────────────────

# Trọng số W (tổng ≈ 1.0, có thể điều chỉnh)
W_SHARPNESS  = 0.35   # Độ nét vùng mặt (quan trọng nhất)
W_EYE        = 0.25   # Trạng thái mắt mở
W_GAZE       = 0.20   # Hướng nhìn vào camera (MỚI — #10)
W_EXPOSURE   = 0.10   # Phơi sáng hợp lý
W_FACE_SIZE  = 0.10   # Kích thước mặt (nhưu tiên close-up)

MOTION_BLUR_PENALTY = 0.40   # Trừ điểm nặng nếu phát hiện motion blur (FFT)
GAZE_PENALTY        = 0.35   # Trừ điểm khi người liếc đi nơi khác (tăng từ 0.20)
# CẢI TIẾN 3.1: Hạ từ 15.0 xuống 12.0 để giữ lại các ảnh nghệ thuật hơi "soft".
MIN_SHARPNESS_SINGLE = 12.0
FFT_BLUR_THRESHOLD   = 10.0


# ───────────────────────────────────────────────────────────────────────────────
# 1. Adaptive Time Threshold (CẢI TIẾN #7)
# ───────────────────────────────────────────────────────────────────────────────

def detect_adaptive_threshold(timestamps: List[float],
                               user_threshold: float = 0.5) -> float:
    """
    Phát hiện tốc độ burst tự động từ phân phối time gap giữa các ảnh.
    
    Thuật toán:
    - Tính tất cả gaps liên tiếp.
    - Burst shots: gaps rất nhỏ (< 1.0s) → dùng median của vùng này.
    - Nếu không detect được burst rõ ràng → dùng user_threshold.
    
    Returns: threshold tối ưu (float, đơn vị giây).
    """
    if len(timestamps) < 3:
        return user_threshold

    gaps = sorted([
        timestamps[i + 1] - timestamps[i]
        for i in range(len(timestamps) - 1)
        if timestamps[i + 1] > timestamps[i]  # bỏ gaps âm (EXIF lỗi)
    ])

    if not gaps:
        return user_threshold

    # Gaps nhỏ ≤ 2s → nhóm burst
    burst_gaps = [g for g in gaps if 0.01 < g <= 2.0]

    if len(burst_gaps) >= 3:
        # Median của burst gaps + buffer 50%
        adaptive = statistics.median(burst_gaps) * 1.5
        adaptive = max(0.1, min(adaptive, user_threshold * 2))
        logger.info(f"Adaptive threshold: {adaptive:.3f}s (từ {len(burst_gaps)} burst gaps)")
        return adaptive

    return user_threshold


# ───────────────────────────────────────────────────────────────────────────────
# 2. Two-Pass Grouping (CẢI TIẾN #5)
# ───────────────────────────────────────────────────────────────────────────────

def group_images(image_paths: List[str],
                 time_threshold: float = 0.5,
                 use_adaptive: bool = True,
                 phash_threshold: int = 10) -> List[List[Dict[str, Any]]]:
    """
    Nhóm ảnh theo 2 pass:
    
    Pass 1 — Time Clustering:
        Nhóm ảnh liên tiếp có EXIF time gap ≤ time_threshold.
        Nếu use_adaptive=True: tự động detect ngưỡng burst từ dataset.
    
    Pass 2 — pHash Sub-clustering:
        Trong mỗi time-cluster, chia thêm thành sub-cluster dựa trên nội dung ảnh.
        Mục tiêu: tách "burst cùng khoảnh khắc" khỏi "2 tư thế khác nhau gần thời gian".
    
    Returns:
        List các nhóm. Mỗi nhóm là List[Dict] với metadata ảnh.
        (Sub-clusters từ Pass 2 được trả về dưới dạng nhóm riêng biệt.)
    """
    if not image_paths:
        return []

    # Bước 1: Thu thập metadata cơ bản
    metadata_list: List[Dict[str, Any]] = []
    for path in image_paths:
        metadata_list.append({
            'path': path,
            'basename': os.path.splitext(os.path.basename(path))[0],
            'time': get_exif_datetime(path),
            'sharpness': 0.0,
            'has_face': False,
            'faces': 0,
            'all_eyes_open': False,
            'open_eyes_count': 0,
            'closed_eyes_count': 0,
            'all_looking_at_camera': False,   # ← MỚI
            'looking_count': 0,               # ← MỚI
            'not_looking_count': 0,           # ← MỚI
            'gaze_angles': [],                # ← MỚI
            'primary_bbox': None,
            'ear_values': [],
            'exposure_score': 0.5,
            'motion_blur': False,
            'fft_score': 999.0,
            'composite_score': 0.0,
        })

    metadata_list.sort(key=lambda x: x['time'])

    # Adaptive threshold từ timestamps
    timestamps = [m['time'] for m in metadata_list]
    if use_adaptive:
        effective_threshold = detect_adaptive_threshold(timestamps, time_threshold)
    else:
        effective_threshold = time_threshold

    logger.info(f"Time threshold hiệu dụng: {effective_threshold:.3f}s")

    # Pass 1: Time-based clustering
    time_clusters: List[List[Dict[str, Any]]] = []
    current = [metadata_list[0]]
    for i in range(1, len(metadata_list)):
        gap = metadata_list[i]['time'] - metadata_list[i - 1]['time']
        if 0 <= gap <= effective_threshold:
            current.append(metadata_list[i])
        else:
            time_clusters.append(current)
            current = [metadata_list[i]]
    if current:
        time_clusters.append(current)

    logger.info(f"Pass 1 (time): {len(image_paths)} ảnh → {len(time_clusters)} time-clusters")

    # Pass 2: pHash sub-clustering trong mỗi time-cluster
    final_groups: List[List[Dict[str, Any]]] = []
    for cluster in time_clusters:
        if len(cluster) <= 1:
            final_groups.append(cluster)
        else:
            sub_clusters = deduplicate_within_group(cluster, threshold=phash_threshold)
            final_groups.extend(sub_clusters)

    logger.info(f"Pass 2 (pHash): {len(time_clusters)} clusters → {len(final_groups)} sub-groups cuối")
    return final_groups


# ───────────────────────────────────────────────────────────────────────────────
# 3. Composite Scoring (CẢI TIẾN #6)
# ───────────────────────────────────────────────────────────────────────────────

def compute_composite_score(item: Dict[str, Any],
                             max_sharpness_ref: float = 1000.0,
                             image_area: int = 1,
                             gaze_priority: float = 0.5) -> float:
    """
    Tính điểm tổng hợp cho 1 ảnh.
    gaze_priority: 0.0 -> chỉ ưu tiên nhìn Cam, 1.0 -> nhìn Nhau cũng tốt như nhìn Cam.
    """
    # S1: Sharpness
    s_sharp = min(item.get('sharpness', 0) / (max_sharpness_ref + 1e-6), 1.0)

    # S2: Eye score
    if not item.get('has_face'):
        s_eye = 0.5
    else:
        total_eyes = item.get('open_eyes_count', 0) + item.get('closed_eyes_count', 0)
        s_eye = item.get('open_eyes_count', 0) / (total_eyes + 1e-6) if total_eyes > 0 else 0.5

    # S3: Gaze score (CẢI TIẾN: Hỗ trợ Interaction Gaze)
    if not item.get('has_face'):
        s_gaze = 0.5
    else:
        # Tính điểm dựa trên từng khuôn mặt
        face_details = item.get('face_details', [])
        if not face_details:
            s_gaze = 0.5
        else:
            face_scores = []
            for f in face_details:
                if f.get('is_looking_cam'):
                    face_scores.append(1.0)
                elif f.get('is_interacting'):
                    # Interaction score dựa trên priority slider
                    face_scores.append(0.7 + (0.3 * gaze_priority))
                else:
                    face_scores.append(0.0)
            s_gaze = sum(face_scores) / len(face_scores)

    # S4: Exposure
    s_exposure = item.get('exposure_score', 0.5)

    # S5: Face size
    s_face_size = 0.5
    if item.get('primary_bbox') is not None:
        x1, y1, x2, y2 = item['primary_bbox']
        face_area = max(0, (x2 - x1) * (y2 - y1))
        s_face_size = min(face_area / (image_area + 1), 1.0)

    score = (W_SHARPNESS * s_sharp
             + W_EYE       * s_eye
             + W_GAZE      * s_gaze
             + W_EXPOSURE  * s_exposure
             + W_FACE_SIZE * s_face_size)

    # Penalty
    if item.get('motion_blur', False):
        score -= MOTION_BLUR_PENALTY
    
    # Nếu có người liếc đi nơi khác (không nhìn cam và không nhìn nhau)
    if item.get('has_face'):
        has_away = any(not f.get('is_looking_cam') and not f.get('is_interacting') for f in item.get('face_details', []))
        if has_away:
            score -= GAZE_PENALTY

    return max(0.0, min(score, 1.0))


# ───────────────────────────────────────────────────────────────────────────────
# 4. Minimum Quality Check (CẢI TIẾN #2 — ảnh đơn)
# ───────────────────────────────────────────────────────────────────────────────

def passes_minimum_quality(item: Dict[str, Any]) -> bool:
    """
    Kiểm tra ảnh đơn (len(group)==1) có đủ chất lượng tối thiểu không.
    
    Lý do: Ảnh đơn (không có burst buddy) trước đây được giữ lại vô điều kiện.
    Giờ ta áp ngưỡng mềm để loại bỏ ảnh rõ ràng kém chất lượng.
    
    Tiêu chí:
    - Không bị motion blur nghiêm trọng (FFT).
    - Sharpness tối thiểu (nếu có mặt: > MIN_SHARPNESS_SINGLE).
    - Không phải ảnh mà tất cả mặt đều nhắm mắt.
    """
    # Bị motion blur → loại
    if item.get('motion_blur', False) and item.get('fft_score', 999) < FFT_BLUR_THRESHOLD * 0.7:
        logger.debug(f"Ảnh đơn bị blur FFT: {item['basename']}")
        return False

    # Có mặt nhưng quá blur (Laplacian thấp) → xem xét
    if item['has_face'] and item['sharpness'] < MIN_SHARPNESS_SINGLE:
        logger.debug(f"Ảnh đơn sharpness quá thấp ({item['sharpness']:.1f}): {item['basename']}")
        return False

    # Có mặt nhưng tất cả đều nhắm mắt → vẫn giữ (nhiếp ảnh gia cần tham khảo)
    # Không loại bỏ hoàn toàn vì có thể là ảnh duy nhất của moment đó
    return True


# ───────────────────────────────────────────────────────────────────────────────
# 5. Pose & Framing Logic (CẢI TIẾN 3.2)
# ───────────────────────────────────────────────────────────────────────────────

def calculate_face_ratio(item: Dict[str, Any]) -> float:
    """
    Tính tỷ lệ diện tích mặt/ảnh. Dùng để nhận diện cỡ cảnh (Wide/Medium/Close-up).
    """
    if not item.get('face_details'):
        return 0.0
    
    # Lấy diện tích mặt lớn nhất (primary face) làm mốc
    bbox = item.get('primary_bbox')
    if not bbox: return 0.0
    
    # Tính diện tích bbox
    fw = bbox[2] - bbox[0]
    fh = bbox[3] - bbox[1]
    face_area = fw * fh
    
    # Giả định diện tích ảnh từ annotated_frame nếu có, hoặc dùng 1M pixel mặc định
    img_area = 1_000_000
    if item.get('annotated_frame') is not None:
        f = item['annotated_frame']
        img_area = f.shape[0] * f.shape[1]
        
    return face_area / (img_area + 1e-6)

def calculate_pose_diff(item1: Dict[str, Any], item2: Dict[str, Any]) -> int:
    """
    Đếm số người thay đổi tư thế giữa 2 ảnh bằng Body Hash.
    Match người theo vị trí center_x (đơn giản nhưng hiệu quả cho burst).
    """
    f1s = item1.get('face_details', [])
    f2s = item2.get('face_details', [])
    if not f1s or not f2s: return 0

    # Sắp xếp theo vị trí ngang
    f1s = sorted(f1s, key=lambda x: x['center_x'])
    f2s = sorted(f2s, key=lambda x: x['center_x'])

    changes = 0
    # Match theo cặp gần nhất
    for i in range(min(len(f1s), len(f2s))):
        h1 = f1s[i].get('body_hash')
        h2 = f2s[i].get('body_hash')
        if h1 is not None and h2 is not None:
            dist = hamming_distance(h1, h2)
            # Ngưỡng 8: thay đổi tay chân rõ rệt
            if dist > 8:
                changes += 1
    return changes


# ───────────────────────────────────────────────────────────────────────────────
# 6. Engine chọn ảnh chính (CẢI TIẾN 3.0)
# ───────────────────────────────────────────────────────────────────────────────

def select_best_images(groups: List[List[Dict[str, Any]]],
                       ai_pipeline=None,
                       progress_callback=None,
                       gaze_priority: float = 0.5) -> List[str]:
    """
    Kế hoạch chọn ảnh:
    1. Phân tích đa luồng để tối ưu tốc độ.
    2. Phân loại Solo/Couple/Group logic.
    
    progress_callback signature: (current, total, path, frame, result_dict)
    """
    from concurrent.futures import ThreadPoolExecutor
    selected_paths: List[str] = []
    
    # Thu phẳng groups để xử lý hàng loạt
    flat_items = [item for g in groups for item in g]
    total_items = len(flat_items)
    
    if not flat_items:
        return []

    # BƯỚC A: PHÂN TÍCH ĐA LUỒNG
    processed_count = 0

    def analyze_task(item):
        nonlocal processed_count
        res = {}
        if ai_pipeline:
            # analyze_image nay đã tích hợp mọi thứ
            res = ai_pipeline.analyze_image(item['path'])
            item.update(res)
        
        processed_count += 1
        if progress_callback:
            progress_callback(processed_count, total_items, item['path'], item.get('annotated_frame'), res)

    # Dùng 4 luồng (vừa đủ để không quá tải neural engine/GPU trên Mac)
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(analyze_task, flat_items)

    # BƯỚC B: CHỌN WINNER TRONG TỪNG NHÓM
    for group in groups:
        if not group: continue
        
        # Thống kê nhóm
        max_sharp = max((i.get('sharpness', 0) for i in group), default=1.0)
        
        # Lấy area ảnh một lần
        img_area = 1_000_000
        if group[0].get('annotated_frame') is not None:
            f = group[0]['annotated_frame']
            img_area = f.shape[0] * f.shape[1]

        for item in group:
            item['composite_score'] = compute_composite_score(item, max_sharp, img_area, gaze_priority)

        if len(group) == 1:
            if passes_minimum_quality(group[0]):
                selected_paths.append(group[0]['path'])
        else:
            # Phân loại logic dâu rể/portrait
            # TIER 1: Perfect (Tất cả mở mắt, nhìn Cam hoặc nhìn Nhau)
            perfect = []
            for i in group:
                if not i.get('has_face'): continue
                if not i.get('all_eyes_open'): continue
                
                # Kiểm tra tất cả mặt đều "OK" (cam hoặc interaction)
                details = i.get('face_details', [])
                if all(f.get('is_looking_cam') or f.get('is_interacting') for f in details):
                    perfect.append(i)
            
            if perfect:
                winner = max(perfect, key=lambda x: x['composite_score'])
            else:
                # TIER 2: Mắt mở (chấp nhận liếc nhẹ)
                eyes_open = [i for i in group if i.get('all_eyes_open')]
                if eyes_open:
                    winner = max(eyes_open, key=lambda x: x['composite_score'])
                else:
                    # TIER 3: Có mặt
                    has_face = [i for i in group if i.get('has_face')]
                    if has_face:
                        winner = max(has_face, key=lambda x: x['composite_score'])
                    else:
                        winner = max(group, key=lambda x: x['composite_score'])
            
            selected_paths.append(winner['path'])

    # BƯỚC C: GLOBAL DEDUPLICATION (CẢI TIẾN 3.0)
    # So sánh các tấm winner giữa các nhóm gần nhau (trong vòng 10s)
    # Nếu tư thế giống hệt nhau -> Chỉ lấy 1 tấm tốt nhất
    if not selected_paths or len(selected_paths) < 2:
        return selected_paths

    final_unique_paths = []
    # Lấy metadata của các paths đã chọn
    selected_items = []
    for p in selected_paths:
        for it in flat_items:
            if it['path'] == p:
                selected_items.append(it)
                break
    
    if not selected_items: return selected_paths

    last_kept = selected_items[0]
    final_unique_paths.append(last_kept['path'])

    for i in range(1, len(selected_items)):
        curr = selected_items[i]
        
        # 1. Kiểm tra khoảng cách thời gian (chỉ lọc trùng nếu < 10s)
        time_gap = curr['time'] - last_kept['time']
        if time_gap > 10.0:
            final_unique_paths.append(curr['path'])
            last_kept = curr
            continue

        # 2. Kiểm tra thay đổi cỡ cảnh (Toàn -> Bán thân)
        r1 = calculate_face_ratio(last_kept)
        r2 = calculate_face_ratio(curr)
        # Nếu cỡ cảnh thay đổi > 30% thì giữ lại cả hai góc máy
        if r1 > 0 and r2 > 0:
            framing_change = abs(r1 - r2) / max(r1, r2)
            if framing_change > 0.30:
                final_unique_paths.append(curr['path'])
                last_kept = curr
                continue

        # 3. Đếm số người đổi dáng
        total_people = curr.get('faces_count', 1)
        
        # Thả lỏng theo số người (CẢI TIẾN 3.2)
        if total_people > 5:
            # Ảnh đông người: Chỉ cần 10% người đổi dáng là đủ (ít nhất 1 người)
            required_changes = max(1, math.ceil(total_people * 0.10))
        else:
            # Ảnh ít người (Dâu rể): Cần 20% người đổi dáng
            required_changes = max(1, math.ceil(total_people * 0.20))
        
        actual_changes = calculate_pose_diff(last_kept, curr)
        
        # Nếu ít hơn ngưỡng thay đổi -> Coi là trùng góc/trùng dáng
        if actual_changes >= required_changes:
            final_unique_paths.append(curr['path'])
            last_kept = curr
        else:
            # TRÙNG NHAU: Ưu tiên lấy tấm chụp trước (last_kept)
            pass

    logger.info(f"Global Deduplication: {len(selected_paths)} → {len(final_unique_paths)} ảnh cuối.")
    return final_unique_paths
