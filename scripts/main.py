"""Application temps reel: webcam OpenCV + porte Tkinter en parallele.

Touches dans la fenetre webcam:
    I  identifier le visage courant
    E  enregistrer une nouvelle personne (saisie du nom dans le terminal)
    Q  quitter
"""
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from src import alignment, config, dataset, door_sim, features
from src import identify, landmarks, snake as snk


WINDOW = "Identification - VPO"


def _compute_vector(face):
    contour = snk.fit_snake(face)
    lm = landmarks.detect_landmarks(face, snake=contour)
    return features.build_feature_vector(lm, snake=contour), contour, lm


def _draw_overlay(disp, face, dataset_size, threshold, last_result, last_t):
    h, w = disp.shape[:2]
    cv2.rectangle(disp, (0, 0), (w, 32), (30, 30, 30), -1)
    cv2.putText(disp, f"Dataset: {dataset_size} vecteurs   seuil = {threshold:.3f}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    cv2.rectangle(disp, (0, h - 28), (w, h), (30, 30, 30), -1)
    cv2.putText(disp, "[I] identifier   [E] enregistrer   [Q] quitter",
                (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    if face is not None:
        thumb = cv2.cvtColor(cv2.resize(face, (160, 160)), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(disp, (w - 172, 38), (w - 8, 202), (200, 200, 200), 1)
        disp[40:200, w - 170:w - 10] = thumb
    else:
        cv2.putText(disp, "Visage non detecte", (10, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (60, 60, 220), 2)

    if last_result and time.time() - last_t < 3.0:
        ok, name, dist, conf, top = last_result
        color = (60, 220, 60) if ok else (60, 60, 220)
        cv2.putText(disp, f"{name}   d={dist:.3f}   conf={conf * 100:.0f}%",
                    (10, 65), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
        for i, (n, d) in enumerate(top[:3]):
            cv2.putText(disp, f"  {i + 1}. {n}  ({d:.3f})",
                        (10, 95 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        cv2.rectangle(disp, (w - 60, 220), (w - 15, 280), color, -1)
        cv2.putText(disp, "OK" if ok else "X", (w - 50, 260),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)


def video_loop(door_q, stop):
    names, vectors = dataset.load()
    print(f"Dataset charge: {len(vectors)} vecteurs, "
          f"{len(set(names))} personne(s).")
    if len(vectors) == 0:
        print("(dataset vide -- la touche I sera sans effet)")
    threshold = config.DEFAULT_THRESHOLD

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam indisponible.")
        stop.set()
        door_q.put(("quit", "", 0.0))
        return

    last_result = None
    last_t = 0.0

    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = alignment.align_face(gray)

        disp = frame.copy()
        _draw_overlay(disp, face, len(vectors), threshold, last_result, last_t)
        cv2.imshow(WINDOW, disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            stop.set()
            door_q.put(("quit", "", 0.0))
            break

        if k == ord('i') and face is not None:
            v, _, _ = _compute_vector(face)
            res = identify.identify(v, names, vectors, threshold)
            last_result = res
            last_t = time.time()
            ok_, name, dist, conf, _ = res
            door_q.put(("open" if ok_ else "denied",
                        name if ok_ else "Inconnu",
                        conf if ok_ else dist))

        elif k == ord('e') and face is not None:
            print("\nNom de la personne a enregistrer (Entree pour annuler): ",
                  end="", flush=True)
            person = sys.stdin.readline().strip()
            if person:
                v, _, _ = _compute_vector(face)
                dataset.append(person, v)
                names.append(person)
                vectors = np.vstack([vectors, v[None, :]]) if len(vectors) \
                          else v[None, :]
                print(f"-> enregistre: {person}  (total {len(vectors)})")
            else:
                print("annule.")

    cap.release()
    cv2.destroyAllWindows()


def main():
    q = queue.Queue()
    stop = threading.Event()

    worker = threading.Thread(target=video_loop, args=(q, stop), daemon=True)
    worker.start()

    door = door_sim.DoorWindow(q)
    try:
        door.run()
    finally:
        stop.set()
        worker.join(timeout=1.5)


if __name__ == "__main__":
    main()
