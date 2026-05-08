from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
CAPTURES_DIR = ROOT / "captures"
DATASET_CSV  = ROOT / "dataset.csv"

IMG_SIZE = 128

# Cascades Haar livrees avec OpenCV
HAAR_FACE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
HAAR_EYE  = cv2.data.haarcascades + "haarcascade_eye.xml"

# Parametres du snake (Kass-Witkin-Terzopoulos)
SNAKE_NB_POINTS  = 80
SNAKE_ITERATIONS = 250
SNAKE_ALPHA      = 0.05   # tension (continuite)
SNAKE_BETA       = 0.10   # rigidite (courbure)
SNAKE_GAMMA      = 1.00   # pas implicite
SNAKE_KAPPA      = 0.08   # poids energie image

# Vecteur caracteristique
NB_DISTANCES = 14
NB_SHAPE     = 16
FEATURE_DIM  = NB_DISTANCES + NB_SHAPE   # = 30

# Identification
DEFAULT_THRESHOLD = 0.18
TOP_K             = 3
