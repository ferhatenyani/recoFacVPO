"""Evaluation: leave-one-out, matrice de confusion, metriques."""
import numpy as np
from . import identify


def confusion_matrix(y_true, y_pred, classes):
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            M[idx[t], idx[p]] += 1
    return M


def metrics_per_class(M):
    n = M.shape[0]
    total = M.sum()
    out = []
    for i in range(n):
        tp = M[i, i]
        fp = M[:, i].sum() - tp
        fn = M[i, :].sum() - tp
        tn = total - tp - fp - fn
        out.append({
            "precision":    tp / (tp + fp) if (tp + fp) else 0.0,
            "rappel":       tp / (tp + fn) if (tp + fn) else 0.0,
            "specificite":  tn / (tn + fp) if (tn + fp) else 0.0,
            "support":      int(M[i, :].sum()),
        })
    return out


def leave_one_out(names, vectors, threshold):
    y_true, y_pred = [], []
    for i in range(len(vectors)):
        n_train = names[:i] + names[i + 1:]
        v_train = np.delete(vectors, i, axis=0)
        ok, pred, *_ = identify.identify(vectors[i], n_train, v_train, threshold)
        y_true.append(names[i])
        y_pred.append(pred if ok else "Inconnu")
    return sorted(set(names)), y_true, y_pred


def threshold_sweep(names, vectors, thresholds):
    rows = []
    for t in thresholds:
        _, y_t, y_p = leave_one_out(names, vectors, t)
        n = len(y_t)
        correct = sum(1 for a, b in zip(y_t, y_p) if a == b)
        rejected = sum(1 for b in y_p if b == "Inconnu")
        rows.append({
            "threshold":      float(t),
            "accuracy":       correct / n if n else 0.0,
            "rejection_rate": rejected / n if n else 0.0,
        })
    return rows
