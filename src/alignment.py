import cv2
import numpy as np
from . import config, detection


def align_face(gray):
    """Detection -> rotation des yeux -> resize 128x128 -> egalisation.
    Retourne l'image normalisee, ou None si aucun visage detecte."""
    bbox = detection.detect_face(gray)
    if bbox is None:
        return None
    x, y, w, h = bbox
    roi = gray[y:y+h, x:x+w]

    eyes = detection.detect_eyes(roi)
    if eyes is not None:
        (xg, yg), (xd, yd) = eyes
        angle = np.degrees(np.arctan2(yd - yg, xd - xg))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        roi = cv2.warpAffine(roi, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    face = cv2.resize(roi, (config.IMG_SIZE, config.IMG_SIZE),
                      interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(face)
