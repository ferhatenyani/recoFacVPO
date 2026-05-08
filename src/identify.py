"""Recherche du plus proche voisin et calcul de confiance."""
import numpy as np
from . import config


def search(query, names, vectors, top_k=config.TOP_K):
    if len(vectors) == 0:
        return []
    dists = np.linalg.norm(vectors - query[None, :], axis=1)
    order = np.argsort(dists)[:top_k]
    return [(names[i], float(dists[i])) for i in order]


def confidence(distance, threshold):
    """1.0 pour une correspondance parfaite, ~0.0 a la limite du seuil."""
    if distance >= threshold:
        return 0.0
    return float(1.0 - distance / threshold)


def identify(query, names, vectors, threshold=config.DEFAULT_THRESHOLD):
    candidates = search(query, names, vectors)
    if not candidates:
        return False, "Inconnu", float("inf"), 0.0, []
    name, dist = candidates[0]
    if dist <= threshold:
        return True, name, dist, confidence(dist, threshold), candidates
    return False, "Inconnu", dist, 0.0, candidates
