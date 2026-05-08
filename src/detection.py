import cv2
from . import config

_face = cv2.CascadeClassifier(config.HAAR_FACE)
_eye  = cv2.CascadeClassifier(config.HAAR_EYE)


def detect_face(gray):
    """Plus grande boite englobante (x, y, w, h) ou None."""
    faces = _face.detectMultiScale(gray, scaleFactor=1.1,
                                   minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda b: b[2] * b[3])


def detect_eyes(face_roi):
    """Centres (oeil_gauche, oeil_droit) en coords du ROI, ou None."""
    h, w = face_roi.shape
    upper = face_roi[: int(h * 0.55), :]
    eyes = _eye.detectMultiScale(upper, scaleFactor=1.1,
                                 minNeighbors=5, minSize=(15, 15))
    if len(eyes) < 2:
        return None
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    eyes = sorted(eyes, key=lambda e: e[0])
    return tuple((int(x + ew / 2), int(y + eh / 2)) for x, y, ew, eh in eyes)
