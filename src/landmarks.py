"""Extraction des points caracteristiques sur un visage normalise 128x128."""
import cv2
import numpy as np
from . import config

_eye = cv2.CascadeClassifier(config.HAAR_EYE)


def _priors(w, h):
    return {
        "eye_l":     (int(w * 0.32), int(h * 0.38)),
        "eye_r":     (int(w * 0.68), int(h * 0.38)),
        "eye_l_out": (int(w * 0.20), int(h * 0.38)),
        "eye_r_out": (int(w * 0.80), int(h * 0.38)),
    }


def _detect_eyes(face):
    h, w = face.shape
    upper = face[: int(h * 0.55), :]
    eyes = _eye.detectMultiScale(upper, 1.1, 5, minSize=(12, 12))
    if len(eyes) < 2:
        return None
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    eyes = sorted(eyes, key=lambda e: e[0])
    (xg, yg, wg, hg), (xd, yd, wd, hd) = eyes
    return {
        "eye_l":     (xg + wg // 2, yg + hg // 2),
        "eye_r":     (xd + wd // 2, yd + hd // 2),
        "eye_l_out": (xg,            yg + hg // 2),
        "eye_r_out": (xd + wd,       yd + hd // 2),
    }


def _detect_nose(face, eye_l, eye_r):
    h, w = face.shape
    y0, y1 = int(h * 0.45), int(h * 0.72)
    x0, x1 = int(w * 0.30), int(w * 0.70)
    roi = cv2.GaussianBlur(face[y0:y1, x0:x1], (5, 5), 1.0)
    ny, _ = np.unravel_index(np.argmin(roi), roi.shape)
    nose_x = (eye_l[0] + eye_r[0]) // 2
    return (nose_x, y0 + ny)


def _detect_mouth(face):
    h, w = face.shape
    y0, y1 = int(h * 0.65), int(h * 0.92)
    band = face[y0:y1, :]
    horiz_grad = np.abs(cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3))
    proj = horiz_grad.sum(axis=1)
    my = int(np.argmax(proj))
    mouth_y = y0 + my

    row_band = band[max(my - 2, 0):my + 3, :]
    row = cv2.GaussianBlur(row_band, (5, 5), 1.5).mean(axis=0)
    grad = np.abs(np.diff(row))
    mid = w // 2
    left  = int(np.argmax(grad[:mid]))
    right = mid + int(np.argmax(grad[mid:]))
    return (left, mouth_y), (right, mouth_y)


def _detect_chin(snake, w, h):
    if snake is None:
        return (w // 2, int(h * 0.95))
    i = int(np.argmax(snake[:, 1]))
    return (int(snake[i, 0]), int(snake[i, 1]))


def detect_landmarks(face, snake=None):
    """Retourne un dict des 8 points-cles dans face (gris, 128x128)."""
    h, w = face.shape
    eyes = _detect_eyes(face) or _priors(w, h)
    nose = _detect_nose(face, eyes["eye_l"], eyes["eye_r"])
    mouth_l, mouth_r = _detect_mouth(face)
    chin = _detect_chin(snake, w, h)
    return {**eyes, "nose": nose, "mouth_l": mouth_l, "mouth_r": mouth_r, "chin": chin}
