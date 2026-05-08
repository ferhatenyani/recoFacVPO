"""Construit dataset.csv a partir de captures/<nom>/*.jpg."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from src import config, snake as snk, landmarks, features, dataset


def main():
    if not config.CAPTURES_DIR.exists():
        print(f"Dossier '{config.CAPTURES_DIR}' introuvable.")
        return 1

    names, vectors = [], []
    for person_dir in sorted(p for p in config.CAPTURES_DIR.iterdir() if p.is_dir()):
        imgs = sorted(person_dir.glob("*.jpg"))
        print(f"{person_dir.name:<20s} {len(imgs):>3d} images")
        for path in imgs:
            face = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if face is None or face.shape != (config.IMG_SIZE, config.IMG_SIZE):
                print(f"  ! ignore (shape invalide): {path.name}")
                continue
            try:
                contour = snk.fit_snake(face)
                lm      = landmarks.detect_landmarks(face, snake=contour)
                v       = features.build_feature_vector(lm, snake=contour)
            except Exception as exc:
                print(f"  ! echec {path.name}: {exc}")
                continue
            names.append(person_dir.name)
            vectors.append(v)

    if not vectors:
        print("Aucun vecteur extrait.")
        return 1

    dataset.write_all(names, np.asarray(vectors), config.DATASET_CSV)
    print(f"\n{len(vectors)} vecteurs ({len(set(names))} personnes) "
          f"ecrits dans {config.DATASET_CSV.name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
