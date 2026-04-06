import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

def get_proxy_dir(parent_dir: str) -> Optional[str]:
    """
    Tìm thư mục 'JPG' hoặc 'jpg' hoặc 'Jpg' bên trong thư mục cha.
    """
    try:
        for item in os.listdir(parent_dir):
            if item.upper() == "JPG" and os.path.isdir(os.path.join(parent_dir, item)):
                return os.path.join(parent_dir, item)
    except Exception as e:
        logger.error(f"Error reading directory {parent_dir}: {e}")
    return None

def get_proxy_images(parent_dir: str) -> List[str]:
    """
    Quét tìm thư mục proxy và trả về danh sách file đường dẫn các hình ảnh JPG.
    """
    proxy_dir = get_proxy_dir(parent_dir)
    if not proxy_dir:
        logger.warning(f"Không tìm thấy thư mục 'JPG' trong {parent_dir}")
        return []

    jpg_extensions = ['.jpg', '.jpeg']
    proxy_files = []
    try:
        for f in os.listdir(proxy_dir):
            _, ext = os.path.splitext(f)
            if ext.lower() in jpg_extensions:
                proxy_files.append(os.path.join(proxy_dir, f))
        
        # Sắp xếp theo tên file (về cơ bản là theo thời gian / STT chụp)
        return sorted(proxy_files)
    except Exception as e:
        logger.error(f"Error scanning proxy directory {proxy_dir}: {e}")
        return []
