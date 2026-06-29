from __future__ import annotations
import cv2
import numpy as np

def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def deskew(image: np.ndarray) -> np.ndarray:
    gray = _to_gray(image)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def remove_noise(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21)
    return cv2.fastNlMeansDenoising(image, None, 7, 21, 7)

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    gray = _to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced

def binarize(image: np.ndarray) -> np.ndarray:
    gray = _to_gray(image)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )

def correct_perspective(image: np.ndarray) -> np.ndarray:
    # lightweight fallback, used after contour detection in future versions
    return image

def preprocess_page(image: np.ndarray) -> np.ndarray:
    img = deskew(image)
    img = remove_noise(img)
    img = enhance_contrast(img)
    return img
