from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

import cv2

if hasattr(sys, '_MEIPASS'):
    _ort_capi = Path(sys._MEIPASS) / 'onnxruntime' / 'capi'
    if _ort_capi.is_dir() and hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(str(_ort_capi))
        except Exception:
            pass
else:
    try:
        import onnxruntime
        _ort_capi = Path(onnxruntime.__file__).parent / 'capi'
        if _ort_capi.is_dir() and hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(str(_ort_capi))
    except Exception:
        pass

from rapidocr_onnxruntime import RapidOCR

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = RapidOCR()
    return _engine

def detect_id(image_path: str | Path) -> str | None:
    img = cv2.imread(str(image_path))

    if img is None:
        print(f'[OCR] Gagal membaca gambar: {image_path}')
        return None

    height, width = img.shape[:2]

    print(f'\n[OCR] Membaca: {image_path}')
    print(f'[OCR] Ukuran gambar: {width}x{height}')

    # ==========================================
    # CROP 50% BAGIAN ATAS
    # ==========================================

    crop_height = height // 2
    cropped = img[0:crop_height, 0:width]

    print(
        f'[OCR] Crop 50% atas: '
        f'x=0:{width}, '
        f'y=0:{crop_height}'
    )

    # ==========================================
    # OCR
    # ==========================================

    engine = _get_engine()
    results, _ = engine(cropped)

    if not results:
        print('[OCR] Tidak ada teks terdeteksi')
        return None

    print('[OCR] Teks yang terbaca:')

    for r in results:
        text = str(r[1])
        confidence = r[2]

        print(f'    - "{text}" (confidence={confidence})')

        # Ambil angka saja
        digits = re.sub(r'\D', '', text)

        # ID PETA = 16 digit
        if len(digits) == 16:
            print(f'[OCR] ID PETA ditemukan: "{digits}"')
            print(f'[OCR] Confidence: {confidence}')

            return digits

    print('[OCR] Tidak ditemukan ID PETA 16 digit')

    return None


# def detect_id(image_path: str | Path) -> str | None:
#     img = cv2.imread(str(image_path))

#     if img is None:
#         print(f'[OCR] Gagal membaca gambar: {image_path}')
#         return None

#     height, width = img.shape[:2]

#     print(f'\n[OCR] Membaca: {image_path}')
#     print(f'[OCR] Ukuran gambar: {width}x{height}')
#     print('[OCR] Mode: membaca seluruh gambar tanpa crop')

#     engine = _get_engine()
#     results, _ = engine(img)

#     if not results:
#         print('[OCR] Tidak ada teks terdeteksi')
#         return None

#     print('[OCR] Teks yang terbaca:')

#     candidates = []

#     for r in results:
#         text = str(r[1])
#         confidence = r[2]

#         print(f'    - "{text}" (confidence={confidence})')

#         # Ambil angka saja
#         digits = re.sub(r'\D', '', text)

#         # Cari ID yang terdiri dari tepat 16 digit
#         if len(digits) == 16:
#             try:
#                 confidence_value = float(confidence)
#             except (TypeError, ValueError):
#                 confidence_value = 0.0

#             candidates.append((digits, confidence_value))

#     if candidates:
#         # Kandidat dengan confidence tertinggi
#         candidates.sort(key=lambda x: x[1], reverse=True)

#         id_peta = candidates[0][0]
#         confidence = candidates[0][1]

#         print(f'[OCR] ID PETA ditemukan: "{id_peta}"')
#         print(f'[OCR] Confidence: {confidence:.2f}')

#         return id_peta

#     print('[OCR] Tidak ditemukan ID PETA 16 digit')

#     return None
    
# def detect_id(image_path: str | Path) -> str | None:
#     img = cv2.imread(str(image_path))
#     if img is None:
#         return None

#     height, width = img.shape[:2]

#     if height > width:
#         img = cv2.resize(img, (904, 1280))
#         h, w = img.shape[:2]
#         crop_h = round(h * 0.04)
#         crop_w = round(w * 0.70)
#     else:
#         img = cv2.resize(img, (1280, 904))
#         h, w = img.shape[:2]
#         crop_h = round(h * 0.05)
#         crop_w = round(w * 0.80)

#     cropped = img[0:crop_h, crop_w:w - 1]
#     engine = _get_engine()
#     results, _ = engine(cropped)
#     if not results:
#         return None
#     all_text = ' '.join(r[1] for r in results)
#     numbers = ''.join(re.findall(r'\d+', all_text))
#     return numbers if numbers else None



def rename_and_copy(source_path: Path, output_dir: Path, idsubsls: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = source_path.suffix
    dest = output_dir / f'{idsubsls}_WSS{suffix}'
    shutil.copy2(source_path, dest)
    return dest


def process_all(input_dir: str | Path, output_dir: str | Path):
    from wss_tool._io import find_images

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = find_images(input_dir)
    if not image_files:
        return

    for img_path in sorted(image_files):
        source = Path(img_path)
        idsubsls = detect_id(str(source))
        if idsubsls:
            rename_and_copy(source, output_path, idsubsls)
            yield source, {'status': 'ok', 'idsubsls': idsubsls}
        else:
            yield source, {'status': 'fail', 'reason': 'Tidak ada nomor ID ditemukan'}

