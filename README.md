# Identification Faciale -- Approche Geometrique

Master 1 IA, Universite de Bejaia. Module **Vision Artificielle** (Mme S. Boukerram), 2025-2026.

Systeme d'identification de visages base exclusivement sur des techniques classiques
de vision par ordinateur : detection Haar, contour actif (Snake) implemente
de zero a partir de Kass-Witkin-Terzopoulos (1988), et descripteurs geometriques
(distances inter-points + descripteurs de forme).

## Pipeline

```
image -> Haar (visage)         -> ROI
      -> Haar (yeux)           -> rotation alignement
      -> resize 128x128 + egalisation d'histogramme
      -> Snake (Kass-Witkin)   -> contour ovale facial
      -> points caracteristiques (yeux, nez, bouche, menton)
      -> 30D = 14 distances + 16 rayons normalises
      -> recherche du plus proche voisin (distance euclidienne)
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# ou: source .venv/bin/activate  # Linux / macOS
pip install -r requirements.txt
```

## Utilisation

### 1. Construire le dataset

Capturer 20 images par personne (10 personnes minimum) :

```bash
python scripts/capture.py Yanis
python scripts/capture.py Khadidja
# ... etc.
```

Chaque dossier `captures/<nom>/` doit couvrir :
- 5 face neutre
- 5 expressions (sourire, surprise, yeux plisses)
- 5 rotations laterales (+/- 15°)
- 5 conditions d'eclairage variees

Puis encoder en vecteurs 30D :

```bash
python scripts/build_dataset.py
```

### 2. Evaluation hors ligne

```bash
python scripts/evaluate.py --sweep        # choisir le seuil optimal
python scripts/evaluate.py -t 0.18        # matrice de confusion + metriques
```

Reporter le seuil retenu dans `src/config.py` (`DEFAULT_THRESHOLD`).

### 3. Mode temps reel

```bash
python scripts/main.py
```

Une fenetre OpenCV affiche la webcam, une fenetre Tkinter affiche la porte.
Dans la fenetre webcam :

- `I` -- identifier le visage courant
- `E` -- enregistrer une nouvelle personne (saisie du nom dans le terminal)
- `Q` -- quitter

## Arborescence

```
projet_tp/
|-- src/                  modules (config, detection, snake, features, ...)
|-- scripts/              points d'entree (capture, build, main, evaluate)
|-- captures/<nom>/       images alignees 128x128 (gris, jpeg)
|-- dataset.csv           vecteurs 30D + nom (genere)
|-- docs/students_guide.md
|-- requirements.txt
|-- README.md
```

## Choix d'implementation

| Module             | Decision                                                    |
|--------------------|-------------------------------------------------------------|
| Detection visage   | Haar `haarcascade_frontalface_default.xml` (livre OpenCV)   |
| Detection yeux     | Haar `haarcascade_eye.xml` sur la moitie superieure du ROI  |
| Alignement         | Rotation par l'angle des yeux + resize + `equalizeHist`     |
| Snake              | Kass-Witkin de zero, schema semi-implicite (`numpy`)        |
| Points cles        | Yeux (Haar), nez (minimum d'intensite), bouche (gradient)   |
| Vecteur 30D        | 14 distances normalisees + 16 rayons du snake (forme)       |
| Identification     | Plus proche voisin, distance euclidienne, seuil empirique   |
| Evaluation         | Leave-one-out, confusion + precision/rappel/specificite     |
