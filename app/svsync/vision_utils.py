# svsync/vision_utils.py
from PIL import Image
import numpy as np
import cv2 as cv
import io

def load_rgb(image_bytes_or_pil):
    """Charge une image en RGB (np.uint8 HxWx3) depuis bytes ou PIL.Image."""
    if isinstance(image_bytes_or_pil, Image.Image):
        img = image_bytes_or_pil.convert("RGB")
    else:
        if isinstance(image_bytes_or_pil, (bytes, bytearray)):
            image_bytes_or_pil = io.BytesIO(image_bytes_or_pil)
        img = Image.open(image_bytes_or_pil).convert("RGB")
    return np.array(img)

def to_bgr(img_rgb_np):
    """RGB(np) -> BGR(np) pour affichage OpenCV / supervision."""
    return cv.cvtColor(img_rgb_np, cv.COLOR_RGB2BGR)

def apply_mask_white_bg(image_rgb_np, mask_bool):
    """Applique un fond blanc hors masque (comme dans ton notebook)."""
    masked = image_rgb_np.copy()
    masked[~mask_bool] = 255
    return masked

def boxes_to_crops_pil(image_pil, boxes_xyxy):
    """Découpe des crops PIL à partir de boxes xyxy float/list."""
    crops = []
    W, H = image_pil.size
    for box in boxes_xyxy:
        x0, y0, x1, y1 = map(int, box)
        x0 = max(0, min(x0, W-1)); x1 = max(0, min(x1, W))
        y0 = max(0, min(y0, H-1)); y1 = max(0, min(y1, H))
        if x1 > x0 and y1 > y0:
            crops.append(image_pil.crop((x0, y0, x1, y1)))
        else:
            crops.append(image_pil)
    return crops


