"""
Skin Disease Feature Extractor
Extracts dermatological color, texture, asymmetry, border irregularity,
and spatial gradient features from skin lesion images.
"""

import numpy as np
from PIL import Image

FEATURE_DIM = 125


def extract_features(pil_img):
    """
    Extract a 125-dimensional feature vector from a PIL image.
    Designed for dermatoscopy analysis:
      - ABCDE criteria (Asymmetry, Border, Color, Diameter, Evolution)
      - Color moments in RGB and HSV spaces (melanin, erythema, pigmentation)
      - Radial center-periphery contrast
      - Multi-scale gradient and texture distribution
    """
    img = pil_img.convert("RGB").resize((128, 128))
    arr = np.array(img, dtype=np.float32)
    hsv = np.array(img.convert("HSV"), dtype=np.float32)

    feats = []

    # 1. Color moments (mean, std, median) for RGB and HSV (18 features)
    for ch in range(3):
        c = arr[:, :, ch]
        feats.extend([float(c.mean()), float(c.std()), float(np.median(c))])
        h = hsv[:, :, ch]
        feats.extend([float(h.mean()), float(h.std()), float(np.median(h))])

    # 2. Color ratios & erythema/pigment indices (3 features)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    feats.append(float((r / (g + 1e-5)).mean()))
    feats.append(float((r / (b + 1e-5)).mean()))
    feats.append(float(((2 * r - g - b) / (r + g + b + 1e-5)).mean()))

    # 3. Asymmetry (horizontal & vertical reflection difference) (2 features)
    gray = (0.299 * r + 0.587 * g + 0.114 * b)
    h_diff = float(np.abs(gray - np.fliplr(gray)).mean())
    v_diff = float(np.abs(gray - np.flipud(gray)).mean())
    feats.extend([h_diff, v_diff])

    # 4. Center vs. periphery contrast (lesion core vs surround) (3 features)
    center = gray[32:96, 32:96]
    periphery_mean = float((gray.sum() - center.sum()) / (128 * 128 - 64 * 64))
    feats.extend([float(center.mean()), float(center.std()), float(center.mean() - periphery_mean)])

    # 5. Gradient & border irregularity (Sobel gradients) (3 features)
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx**2 + gy**2)
    feats.extend([float(mag.mean()), float(mag.std()), float(np.percentile(mag, 90))])

    # 6. Local quadrant texture histograms (32 features)
    for q in [gray[:64, :64], gray[:64, 64:], gray[64:, :64], gray[64:, 64:]]:
        hist, _ = np.histogram(q, bins=8, range=(0, 255))
        total = float(hist.sum()) + 1e-6
        feats.extend([float(x / total) for x in hist])

    # 7. Spatial structural thumbnail (8x8 = 64 features)
    thumb = np.array(img.convert("L").resize((8, 8)), dtype=np.float32) / 255.0
    feats.extend([float(x) for x in thumb.flatten()])

    vec = np.array(feats, dtype=np.float32)
    # Replace any NaN/Inf with 0.0
    vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
    return vec
