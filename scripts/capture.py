"""Capture interactive d'images (visages alignes 128x128) pour le dataset.

Usage:
    python scripts/capture.py <nom> [-n NB]

Touches:
    ESPACE  capturer l'image en cours (visage doit etre detecte)
    Q       quitter
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from src import alignment, config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("name", help="Nom de la personne (sans espace).")
    p.add_argument("-n", "--nb", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = config.CAPTURES_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = len(list(out_dir.glob("*.jpg")))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: webcam introuvable.")
        return 1

    print(f"Capture pour '{args.name}': {idx} deja enregistrees, cible {args.nb}.")
    print("ESPACE = capture, Q = quitter.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = alignment.align_face(gray)

        disp = frame.copy()
        if face is not None:
            cv2.putText(disp, f"{idx}/{args.nb}", (10, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (60, 220, 60), 2)
            mini = cv2.cvtColor(cv2.resize(face, (192, 192)), cv2.COLOR_GRAY2BGR)
            disp[10:202, disp.shape[1] - 202:disp.shape[1] - 10] = mini
        else:
            cv2.putText(disp, "Visage non detecte", (10, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (60, 60, 220), 2)

        cv2.imshow("Capture", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord(' ') and face is not None:
            path = out_dir / f"img_{idx:03d}.jpg"
            cv2.imwrite(str(path), face)
            print(f"  + {path.name}")
            idx += 1
            if idx >= args.nb:
                print("Cible atteinte.")
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
