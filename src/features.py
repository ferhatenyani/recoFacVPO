"""Construction du vecteur caracteristique 30D.

14 distances inter-points (normalisees par la diagonale de l'image)
+ 16 rayons du snake echantillonnes a angles fixes (normalises par le rayon moyen).
"""
import numpy as np
from . import config

# Paires choisies pour leur valeur anthropometrique (largeur, hauteur, asymetries)
_PAIRS = [
    ("eye_l_out", "eye_r_out"),
    ("eye_l",     "eye_r"),
    ("eye_l",     "nose"),
    ("eye_r",     "nose"),
    ("eye_l",     "mouth_l"),
    ("eye_r",     "mouth_r"),
    ("nose",      "mouth_l"),
    ("nose",      "mouth_r"),
    ("mouth_l",   "mouth_r"),
    ("nose",      "chin"),
    ("eye_l",     "chin"),
    ("eye_r",     "chin"),
    ("eye_l_out", "mouth_l"),
    ("eye_r_out", "mouth_r"),
]


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _shape_descriptors(snake, n=config.NB_SHAPE):
    if snake is None or len(snake) < 4:
        return np.zeros(n, dtype=np.float32)
    cx, cy = snake.mean(axis=0)
    pts_a = np.arctan2(snake[:, 1] - cy, snake[:, 0] - cx)
    pts_r = np.hypot(snake[:, 0] - cx, snake[:, 1] - cy)
    order = np.argsort(pts_a)
    pa = pts_a[order]
    pr = pts_r[order]
    pa_ext = np.concatenate([pa - 2 * np.pi, pa, pa + 2 * np.pi])
    pr_ext = np.concatenate([pr,             pr, pr])
    angles = np.linspace(-np.pi, np.pi, n, endpoint=False)
    radii = np.interp(angles, pa_ext, pr_ext)
    m = radii.mean()
    if m <= 1e-6:
        return radii.astype(np.float32)
    return (radii / m).astype(np.float32)


def build_feature_vector(landmarks, snake=None, img_size=config.IMG_SIZE):
    diag = float(np.hypot(img_size, img_size))
    distances = np.array(
        [_dist(landmarks[a], landmarks[b]) for a, b in _PAIRS],
        dtype=np.float32
    ) / diag
    shape = _shape_descriptors(snake)
    return np.concatenate([distances, shape])
