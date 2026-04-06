import os
import shutil
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

RAW_EXTENSIONS = ['.arw', '.cr3', '.cr2', '.nef', '.dng', '.raf', '.orf', '.rw2']

def find_raw_file(src_dir: str, basename: str, extensions: List[str] = RAW_EXTENSIONS) -> Optional[str]:
    """
    Tìm file RAW tương ứng với basename (của file JPG proxy) ở thư mục cha.
    Lấy chính xác tên định dạng chữ hoa/chữ thường (ví dụ: .ARW) để không bị sai lệnh copy trên macOS.
    """
    try:
        files = os.listdir(src_dir)
        target_basename_lower = basename.lower()
        for f in files:
            name, ext = os.path.splitext(f)
            if name.lower() == target_basename_lower and ext.lower() in extensions:
                return os.path.join(src_dir, f)
    except Exception as e:
        logger.error(f"Error scanning directory {src_dir}: {e}")

    return None

def copy_to_selected(src_path: str, dest_dir: str) -> tuple:
    """
    Copy một file RAW vào thư mục đích.
    Returns: (success: bool, error_code: str | None)
    error_code: 'no_space' | 'not_found' | 'permission' | 'unknown' | None
    """
    if not os.path.exists(src_path):
        return False, 'not_found'

    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.basename(src_path)
    dest_path = os.path.join(dest_dir, filename)

    try:
        shutil.copy2(src_path, dest_path)
        return True, None
    except OSError as e:
        import errno
        if e.errno == errno.ENOSPC:
            logger.warning(f"Hết dung lượng đĩa khi copy {filename}")
            return False, 'no_space'
        elif e.errno == errno.EACCES:
            logger.error(f"Không có quyền copy {filename}: {e}")
            return False, 'permission'
        else:
            logger.error(f"Lỗi copy {src_path} → {dest_path}: {e}")
            return False, 'unknown'
    except Exception as e:
        logger.error(f"Lỗi không xác định khi copy {src_path}: {e}")
        return False, 'unknown'

def export_catalog(raw_filepaths: List[str], dest_dir: str, filename: str = "catalog.txt"):
    """
    Ghi danh sách file được chọn ra log txt.
    """
    os.makedirs(dest_dir, exist_ok=True)
    catalog_path = os.path.join(dest_dir, filename)
    try:
        with open(catalog_path, 'w', encoding='utf-8') as f:
            for path in raw_filepaths:
                f.write(f"{os.path.basename(path)}\n")
    except Exception as e:
        logger.error(f"Failed to write catalog {catalog_path}: {e}")
