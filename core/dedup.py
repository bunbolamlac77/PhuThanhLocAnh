"""
dedup.py — Phát hiện và loại bỏ ảnh trùng lặp nội dung (pHash Deduplication)
CẢI TIẾN #4: pHash Deduplication

Vấn đề cần giải quyết:
- Hai burst shots gần giống hệt nhau (99% pixel trùng) có thể thuộc 2 nhóm thời gian khác nhau
  → cả 2 đều được giữ lại dù chúng đồng nhất.
- pHash (Perceptual Hash) so sánh "nội dung cảm nhận" của ảnh, không phải pixel-by-pixel.
  → Phát hiện ảnh trùng ngay cả khi resize, crop nhẹ, hay thay đổi brightness nhỏ.

Thuật toán pHash:
1. Resize ảnh về 32x32 grayscale.
2. Tính DCT (Discrete Cosine Transform).
3. Lấy 8x8 vùng DCT tần số thấp (top-left).
4. So sánh mỗi pixel với giá trị trung bình → tạo hash 64 bit.
5. Khoảng cách Hamming < threshold → ảnh trùng.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Ngưỡng Hamming distance để coi 2 ảnh là "trùng nhau"
# CẢI TIẾN 3.1: Giảm xuống 12 để thả lỏng việc gom nhóm, tránh mất ảnh.
PHASH_THRESHOLD = 20


def compute_phash(filepath: str, hash_size: int = 8, highfreq_factor: int = 4) -> Optional[np.ndarray]:
    """Tính pHash từ file path (giữ nguyên tương thích cũ)."""
    try:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        return compute_phash_from_image(img, hash_size, highfreq_factor)
    except Exception: return None


def compute_phash_from_image(img_gray: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> Optional[np.ndarray]:
    """
    Tính Perceptual Hash (pHash) từ một numpy array đã load sẵn.
    Dùng để băm vùng cơ thể (pose hashing) cực nhanh.
    """
    try:
        if img_gray is None or img_gray.size == 0:
            return None

        img_size = hash_size * highfreq_factor  # 32
        img_resized = cv2.resize(img_gray, (img_size, img_size), interpolation=cv2.INTER_AREA)

        # DCT trên float32
        img_float = img_resized.astype(np.float32)
        dct = cv2.dct(img_float)

        # Lấy vùng top-left 8x8
        dct_low = dct[:hash_size, :hash_size]

        # Tính mean, bỏ DC component
        dct_flat = dct_low.flatten()
        mean_val = dct_flat[1:].mean()

        # Tạo binary hash
        phash = (dct_flat > mean_val).astype(np.uint8)
        return phash
    except Exception as e:
        logger.error(f"Lỗi compute_phash_from_image: {e}")
        return None


def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    """Tính Hamming distance giữa 2 hash (số bit khác nhau)."""
    if hash1 is None or hash2 is None:
        return 64  # Max distance nếu lỗi
    return int(np.count_nonzero(hash1 != hash2))


def deduplicate_within_group(group: List[Dict[str, Any]],
                              threshold: int = PHASH_THRESHOLD) -> List[List[Dict[str, Any]]]:
    """
    Chia 1 nhóm thời gian thành các sub-cluster dựa trên similarity pHash.
    Mỗi sub-cluster chứa các ảnh cực kỳ giống nhau về nội dung.
    
    Thuật toán: Greedy Union-Find đơn giản
    - Tính pHash cho tất cả ảnh trong nhóm.
    - Xây dựng các cluster: ảnh nào gần ảnh nào thì vào cùng sub-cluster.
    - Mỗi sub-cluster → chọn 1 winner → giữ đa dạng góc nhìn.
    
    Args:
        group: Danh sách ảnh trong 1 time-cluster.
        threshold: Hamming distance tối đa để coi là "giống nhau" (mặc định 10).
    
    Returns:
        Danh sách các sub-cluster, mỗi sub-cluster là 1 list ảnh.
    """
    if len(group) <= 1:
        return [group]

    # Tính pHash cho tất cả
    hashes = []
    for item in group:
        h = compute_phash(item['path'])
        hashes.append(h)

    n = len(group)
    # Union-Find labels: ban đầu mỗi ảnh là cluster riêng
    labels = list(range(n))

    def find(x):
        while labels[x] != x:
            labels[x] = labels[labels[x]]
            x = labels[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            labels[ry] = rx

    # So sánh từng cặp ảnh
    for i in range(n):
        for j in range(i + 1, n):
            dist = hamming_distance(hashes[i], hashes[j])
            if dist <= threshold:
                union(i, j)

    # Gom theo root label
    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    # Trả về sub-clusters dưới dạng list of list of items
    result = []
    for indices in clusters.values():
        sub_cluster = [group[i] for i in sorted(indices)]
        result.append(sub_cluster)

    logger.debug(f"pHash split {len(group)} ảnh → {len(result)} sub-clusters (threshold={threshold})")
    return result
