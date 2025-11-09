import numpy as np
import cv2 as cv

def extract_hl_features(image_np, box, image_size):
    """Version légère comme en notebook: histogramme couleurs, LBP, AR, aire relative."""
    x0, y0, x1, y1 = map(int, box)
    h, w = image_np.shape[:2]
    x0 = max(0, min(x0, w-1)); x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h-1)); y1 = max(0, min(y1, h))
    crop = image_np[y0:y1, x0:x1, :]

    if crop.size == 0:
        crop = np.full((10, 10, 3), 255, np.uint8)

    # histogramme couleur 8 bins par canal
    hist = []
    for c in range(3):
        hch = cv.calcHist([crop], [c], None, [8], [0, 256]).flatten()
        hch = (hch / (hch.sum() + 1e-8)).astype(float)
        hist.extend(hch.tolist())

    # texture LBP très simple (3x3)
    gray = cv.cvtColor(crop, cv.COLOR_RGB2GRAY)
    lbp = np.zeros_like(gray, dtype=np.uint8)
    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            c = gray[i, j]
            code = 0
            code |= (gray[i-1, j-1] > c) << 7
            code |= (gray[i-1, j  ] > c) << 6
            code |= (gray[i-1, j+1] > c) << 5
            code |= (gray[i,   j+1] > c) << 4
            code |= (gray[i+1, j+1] > c) << 3
            code |= (gray[i+1, j  ] > c) << 2
            code |= (gray[i+1, j-1] > c) << 1
            code |= (gray[i,   j-1] > c) << 0
            lbp[i, j] = code
    hist_lbp, _ = np.histogram(lbp, bins=16, range=(0, 256))
    hist_lbp = (hist_lbp / (hist_lbp.sum() + 1e-8)).astype(float).tolist()

    # ratios
    bw = max(1, x1 - x0); bh = max(1, y1 - y0)
    aspect_ratio = float(bw) / float(bh)
    relative_area = float(bw * bh) / float(image_size[0] * image_size[1])

    return {
        "color_histogram": hist,
        "texture_lbp": hist_lbp,
        "aspect_ratio": aspect_ratio,
        "relative_area": relative_area
    }

