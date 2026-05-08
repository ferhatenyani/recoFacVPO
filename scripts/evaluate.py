"""Evaluation hors-ligne du dataset (leave-one-out).

Usage:
    python scripts/evaluate.py                 # seuil par defaut + matrice de confusion
    python scripts/evaluate.py -t 0.15
    python scripts/evaluate.py --sweep         # balayage de seuils
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src import config, dataset, evaluation


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-t", "--threshold", type=float, default=None)
    p.add_argument("--sweep", action="store_true",
                   help="Balayage de seuils 0.05..0.50.")
    return p.parse_args()


def print_confusion(M, classes):
    width = max(8, max(len(c) for c in classes) + 1)
    header = " " * (width + 2) + " ".join(f"{c[:8]:>8}" for c in classes)
    print(header)
    for i, c in enumerate(classes):
        row = " ".join(f"{M[i, j]:>8d}" for j in range(len(classes)))
        print(f"{c[:width]:>{width}}  {row}")


def main():
    args = parse_args()
    names, vectors = dataset.load()
    if len(vectors) == 0:
        print("Dataset vide. Executer build_dataset.py au prealable.")
        return 1
    print(f"Dataset: {len(vectors)} vecteurs, {len(set(names))} personnes.\n")

    if args.sweep:
        ts = np.linspace(0.05, 0.50, 19)
        rows = evaluation.threshold_sweep(names, vectors, ts)
        print(f"{'seuil':>8} {'accuracy':>10} {'rejet':>10}")
        for r in rows:
            print(f"{r['threshold']:>8.3f} "
                  f"{r['accuracy']:>10.3f} "
                  f"{r['rejection_rate']:>10.3f}")
        best = max(rows, key=lambda r: r["accuracy"])
        print(f"\nMeilleur seuil: {best['threshold']:.3f} "
              f"(accuracy {best['accuracy']:.3f}, "
              f"rejet {best['rejection_rate']:.3f})")
        return 0

    t = args.threshold if args.threshold is not None else config.DEFAULT_THRESHOLD
    classes, y_t, y_p = evaluation.leave_one_out(names, vectors, t)
    classes_aug = classes + ["Inconnu"]
    M = evaluation.confusion_matrix(y_t, y_p, classes_aug)

    print(f"Seuil = {t:.3f}\n")
    print("Matrice de confusion (lignes = verite, colonnes = predit):")
    print_confusion(M, classes_aug)

    print("\nMetriques par classe:")
    print(f"{'classe':>14} {'precision':>11} {'rappel':>9} "
          f"{'specificite':>13} {'support':>9}")
    for c, m in zip(classes_aug, evaluation.metrics_per_class(M)):
        print(f"{c[:13]:>14} {m['precision']:>11.3f} {m['rappel']:>9.3f} "
              f"{m['specificite']:>13.3f} {m['support']:>9d}")

    correct = sum(1 for a, b in zip(y_t, y_p) if a == b)
    print(f"\nAccuracy globale: {correct}/{len(y_t)} = {correct / len(y_t):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
