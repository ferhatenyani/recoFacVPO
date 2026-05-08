"""Lecture / ecriture du dataset.csv (nom;v0;v1;...;v29)."""
import csv
from pathlib import Path
import numpy as np
from . import config


def load(path=None):
    """Retourne (names, vectors)."""
    path = Path(path or config.DATASET_CSV)
    if not path.exists():
        return [], np.empty((0, config.FEATURE_DIM), dtype=np.float32)
    names, vecs = [], []
    with open(path, newline="") as f:
        for row in csv.reader(f, delimiter=";"):
            if not row or len(row) < 1 + config.FEATURE_DIM:
                continue
            names.append(row[0])
            vecs.append([float(x) for x in row[1:1 + config.FEATURE_DIM]])
    return names, np.asarray(vecs, dtype=np.float32)


def append(name, vector, path=None):
    path = Path(path or config.DATASET_CSV)
    with open(path, "a", newline="") as f:
        csv.writer(f, delimiter=";").writerow(
            [name] + [f"{v:.6f}" for v in vector]
        )


def write_all(names, vectors, path=None):
    path = Path(path or config.DATASET_CSV)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        for n, v in zip(names, vectors):
            w.writerow([n] + [f"{x:.6f}" for x in v])
