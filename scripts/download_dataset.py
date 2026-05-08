"""Recupere un dataset public de visages frontaux et l'integre a captures/.

Dataset choisi : **Essex Faces94** (Vision Group, University of Essex, 1996).
  - 153 sujets x 20 images, prises en sequence devant un fond uni
  - resolution 180x200, couleur, frontal head-and-shoulders
  - variation : expressions naturelles (parole, sourire, regard) au cours
    des 20 prises ; les variations de pose / eclairage sont apportees par
    les 2 identites capturees a la webcam par les etudiants
  - Licence : libre d'usage pour la recherche academique non commerciale
    (cf. https://cswww.essex.ac.uk/mv/allfaces/index.html)

Pipeline applique a chaque image brute :
  src.alignment.align_face -> 128x128 gris egalise (Haar + rotation des yeux)
Les images rejetees par Haar sont comptees mais non sauvegardees.

Idempotence :
  - les dossiers captures/<nom>/ contenant deja >= MIN_IMAGES jpg sont
    laisses tels quels (utile pour ne pas ecraser les captures etudiants) ;
  - l'archive est mise en cache dans downloads/ ; relancer le script
    n'entraine pas de re-telechargement ni de duplication d'images.

Usage :
    python scripts/download_dataset.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src import alignment, config


# Source officielle (souvent indisponible) :
#   https://cswww.essex.ac.uk/mv/allfaces/faces94.zip
# Miroir academique stable (depot GitHub d'un cours BYU "Foundations of
# Applied Mathematics") qui redistribue le meme zip :
DATASET_URLS = [
    "https://raw.githubusercontent.com/Foundations-of-Applied-Mathematics/"
    "Data/master/FacialRecognition/faces94.zip",
    "https://cswww.essex.ac.uk/mv/allfaces/faces94.zip",
]
ARCHIVE_NAME = "faces94.zip"
MIN_IMAGES = 20            # seuil pour considerer un sujet "complet"
CAP_PER_PERSON = 25        # plafond impose par le cahier des charges
TARGET_NEW_IDENTITIES = 8  # objectif minimum demande
SUBJECT_HARD_CAP = 12      # on s'arrete au-dela pour rester borne

MANUAL_HINT = (
    "\n  -> Telechargement automatique impossible. Solution :\n"
    f"     1. Ouvrir l'une des URLs ci-dessus dans un navigateur\n"
    "     2. Sauvegarder le fichier sous :\n"
    f"        {{ROOT}}/downloads/{ARCHIVE_NAME}\n"
    "     3. Relancer ce script (il detectera l'archive locale).\n"
)


def fetch_archive(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        size_mb = dest.stat().st_size / 1e6
        print(f"Archive deja en cache : {dest.name} ({size_mb:.1f} MB)")
        return dest

    last_error: Exception | None = None
    for url in DATASET_URLS:
        print(f"Telechargement {url} ...")
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(req, timeout=120) as resp:
                data = resp.read()
        except (URLError, TimeoutError, OSError) as exc:
            print(f"  ECHEC : {exc}")
            last_error = exc
            continue
        dest.write_bytes(data)
        print(f"  -> {dest} ({len(data) / 1e6:.1f} MB)")
        return dest

    print(f"\nAucune source accessible (dernier echec : {last_error}).")
    print(MANUAL_HINT.replace("{ROOT}", str(config.ROOT)))
    sys.exit(2)


def iter_subjects(archive: Path):
    """Yield (subject_name, [(filename, bytes), ...]) groupe par dossier parent."""
    with zipfile.ZipFile(archive) as zf:
        by_subject: dict[str, list[str]] = {}
        for n in zf.namelist():
            if not n.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            parts = [p for p in n.split("/") if p]
            if len(parts) < 2:
                continue
            subject = parts[-2]
            by_subject.setdefault(subject, []).append(n)

        for subject, files in sorted(by_subject.items()):
            blobs: list[tuple[str, bytes]] = []
            for f in sorted(files):
                with zf.open(f) as fh:
                    blobs.append((f, fh.read()))
            yield subject, blobs


def existing_jpg_count(person_dir: Path) -> int:
    if not person_dir.exists():
        return 0
    return len(list(person_dir.glob("*.jpg")))


def next_index(person_dir: Path) -> int:
    indices = []
    for p in person_dir.glob("img_*.jpg"):
        try:
            indices.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return max(indices) + 1 if indices else 0


def process_subject(blobs, target_dir: Path) -> tuple[int, int, int]:
    """Retourne (kept_total, added_now, total_seen)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    kept = existing_jpg_count(target_dir)
    idx = next_index(target_dir)
    added = 0
    total = len(blobs)

    for _, blob in blobs:
        if kept >= CAP_PER_PERSON:
            break
        arr = np.frombuffer(blob, dtype=np.uint8)
        gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        face = alignment.align_face(gray)
        if face is None:
            continue
        out = target_dir / f"img_{idx:03d}.jpg"
        cv2.imwrite(str(out), face)
        kept += 1
        added += 1
        idx += 1

    return kept, added, total


def main() -> int:
    captures = config.CAPTURES_DIR
    captures.mkdir(parents=True, exist_ok=True)

    archive = fetch_archive(config.ROOT / "downloads" / ARCHIVE_NAME)

    print("\nIntegration des sujets dans captures/ :\n")
    valid_new = 0
    total_kept = 0
    low_accept = []
    skipped_existing = 0

    for subject, blobs in iter_subjects(archive):
        target = captures / subject
        before = existing_jpg_count(target)
        if before >= MIN_IMAGES:
            skipped_existing += 1
            print(f"  {subject:<20s} deja {before} images, ignore")
            continue

        kept, added, total = process_subject(blobs, target)
        rate = (kept / total) if total else 0.0
        print(f"  {subject:<20s} kept {kept:>3d} / total {total:>3d}  "
              f"(+{added}, {rate:.0%})")

        if kept >= MIN_IMAGES:
            valid_new += 1
            total_kept += kept
            if rate < 0.6:
                low_accept.append((subject, kept, total, rate))

        if valid_new >= SUBJECT_HARD_CAP:
            print("\nPlafond de sujets atteint, arret de l'integration.")
            break

    print(f"\nResume :")
    print(f"  identites publiques completes (>= {MIN_IMAGES} img) : {valid_new}")
    print(f"  identites deja presentes ignorees                  : {skipped_existing}")
    print(f"  total images integrees ce run                      : {total_kept}")
    if valid_new < TARGET_NEW_IDENTITIES:
        print(f"  ATTENTION : objectif {TARGET_NEW_IDENTITIES} non atteint.")
    if low_accept:
        print("\nSujets a faible taux d'acceptation Haar (<60%) :")
        for s, k, t, r in low_accept:
            print(f"  - {s}: {k}/{t} ({r:.0%})")
        print("  (Haar peut echouer si le visage est trop sombre / decentre.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
