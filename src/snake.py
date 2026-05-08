"""Contour actif (Snake) -- formulation Kass, Witkin & Terzopoulos (1988).

Schema semi-implicite:
    (A + gamma * I) v_{t+1} = gamma * v_t + kappa * f_ext(v_t)

ou A est la matrice pentadiagonale circulante issue des differences finies
de l'energie interne (alpha * |v'|^2 + beta * |v''|^2) et f_ext = grad(|grad(I)|^2)
attire le contour vers les bords forts.
"""
import numpy as np
import cv2
from . import config


def _internal_matrix(n, alpha, beta):
    a = beta
    b = -alpha - 4 * beta
    c = 2 * alpha + 6 * beta
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i - 2) % n] = a
        A[i, (i - 1) % n] = b
        A[i,  i          ] = c
        A[i, (i + 1) % n] = b
        A[i, (i + 2) % n] = a
    return A


def _external_field(img):
    f = img.astype(np.float64) / 255.0
    f = cv2.GaussianBlur(f, (5, 5), 1.5)
    gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3)
    edge = cv2.GaussianBlur(gx * gx + gy * gy, (5, 5), 2.0)
    fx = cv2.Sobel(edge, cv2.CV_64F, 1, 0, ksize=3)
    fy = cv2.Sobel(edge, cv2.CV_64F, 0, 1, ksize=3)
    return fx, fy


def _bilinear(field, x, y):
    h, w = field.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0 = np.floor(x).astype(int); x1 = x0 + 1
    y0 = np.floor(y).astype(int); y1 = y0 + 1
    dx = x - x0; dy = y - y0
    return (field[y0, x0] * (1 - dx) * (1 - dy)
          + field[y0, x1] *      dx  * (1 - dy)
          + field[y1, x0] * (1 - dx) *      dy
          + field[y1, x1] *      dx  *      dy)


def _initial_ellipse(shape, n):
    h, w = shape
    cx, cy = w / 2.0, h / 2.0 - 4
    rx, ry = w * 0.36, h * 0.46
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return cx + rx * np.cos(t), cy + ry * np.sin(t)


def fit_snake(img,
              n=config.SNAKE_NB_POINTS,
              alpha=config.SNAKE_ALPHA,
              beta=config.SNAKE_BETA,
              gamma=config.SNAKE_GAMMA,
              kappa=config.SNAKE_KAPPA,
              n_iter=config.SNAKE_ITERATIONS):
    fx, fy = _external_field(img)
    A = _internal_matrix(n, alpha, beta)
    inv = np.linalg.inv(A + gamma * np.eye(n))

    x, y = _initial_ellipse(img.shape, n)
    h, w = img.shape

    for _ in range(n_iter):
        ex = _bilinear(fx, x, y)
        ey = _bilinear(fy, x, y)
        x = inv @ (gamma * x + kappa * ex)
        y = inv @ (gamma * y + kappa * ey)
        x = np.clip(x, 1, w - 2)
        y = np.clip(y, 1, h - 2)

    return np.column_stack([x, y])
