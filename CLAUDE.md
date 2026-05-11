# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Master 1 IA assignment (Universite de Bejaia, *Vision Artificielle*, Mme S. Boukerram, 2025-2026): a face-identification system built **exclusively with classical computer vision** -- Haar cascades, a from-scratch Kass-Witkin-Terzopoulos active contour, geometric landmarks, and nearest-neighbour search on a 30D feature vector. UI text is French.

**Hard constraint**: no ML / no deep learning. Do not introduce dlib, mediapipe, MTCNN, face_recognition, sklearn classifiers, etc. The snake in [src/snake.py](src/snake.py) is implemented from the 1988 paper -- do not replace it with `cv2.GrabCut`, `cv2.findContours`, or `skimage.segmentation.active_contour`.

## Environment & install

Python is the local Anaconda interpreter (3.13, numpy 2.x, opencv-python 4.x). Tkinter ships with it. On Windows, invoke as:

```powershell
& "C:\Users\MyHomehP\anaconda3\python.exe" scripts/<entrypoint>.py
```

Dependencies: see [requirements.txt](requirements.txt) -- `numpy`, `opencv-python`, `Pillow`. Install with `... -m pip install -r requirements.txt`.

## Common commands

```powershell
# Build the dataset from images previously saved under captures/<name>/*.jpg
python scripts/build_dataset.py

# Capture 20 images for a person (webcam, ESPACE = save, Q = quit)
python scripts/capture.py <name> [-n 20]

# Pull the public Essex Faces94 dataset and run alignment over it (idempotent;
# cached in downloads/, skips persons already populated)
python scripts/download_dataset.py

# Offline evaluation (leave-one-out) -- pick a threshold, then report it
python scripts/evaluate.py --sweep          # sweep 0.05..0.50, prints best
python scripts/evaluate.py -t 0.18          # confusion matrix + per-class metrics

# Production demo (OpenCV webcam window + separate Tk door window)
python scripts/main.py

# Single-window Tk debug bench (bento dashboard, threshold slider, snapshot,
# enrolment wizard, identity manager, manual report export)
python scripts/test_app.py
```

There is no test runner, no linter, no CI. Verification is **visual** (does the live app behave?) plus the leave-one-out report from `evaluate.py`. After any add/delete from `test_app.py`, a fresh markdown report is auto-written under `reports/` (see below) -- that file is the artefact to inspect, not stdout.

## Pipeline (read top-down)

```
gray frame
  -> detection.detect_face        (Haar frontalface, biggest bbox)
  -> detection.detect_eyes        (Haar eye on the upper half of the ROI)
  -> alignment.align_face         (rotation by eye angle, resize to 128, equalizeHist)
  -> snake.fit_snake              (Kass-Witkin, semi-implicit, 250 iter, 80 pts)
  -> landmarks.detect_landmarks   (eyes via Haar, nose = darkest pixel,
                                   mouth = horizontal-gradient peak, chin = lowest snake pt)
  -> features.build_feature_vector  (14 distances / diag(128x128) ++ 16 snake radii / mean)
                                                              => 30D float32 vector
  -> identify.identify             (nearest-neighbour L2, threshold from config)
```

The whole 30D contract lives in [src/config.py](src/config.py): `NB_DISTANCES = 14`, `NB_SHAPE = 16`, `FEATURE_DIM = 30`, `IMG_SIZE = 128`. Changing any of these is a breaking change for `dataset.csv`.

## Architecture: what lives where

[src/](src/) is a pure library (no `argparse`, no `cv2.imshow`). Every public function is small and stateless; cascades are loaded once at import time. Scripts in [scripts/](scripts/) are the only entry points and own all I/O, threading, and Tk/OpenCV windows.

| Module | Role |
|---|---|
| [src/config.py](src/config.py) | All tunable constants (snake hyperparameters, `DEFAULT_THRESHOLD`, paths). Edit `DEFAULT_THRESHOLD` after `evaluate.py --sweep`. |
| [src/detection.py](src/detection.py) | Haar face + eye detection. Eyes are searched only in the upper 55% of the face ROI. |
| [src/alignment.py](src/alignment.py) | bbox -> rotated by eye angle -> 128x128 -> `equalizeHist`. Returns `None` if no face. |
| [src/snake.py](src/snake.py) | Kass-Witkin: pentadiagonal circulant matrix `A`, semi-implicit step `(A + gamma*I)^-1 (gamma*v + kappa*f_ext)`, external field = grad(|grad I|^2). Inverse of `(A + gamma I)` is precomputed once. |
| [src/landmarks.py](src/landmarks.py) | 8 keypoints in 128x128 coords, with anthropometric priors as fallback if Haar fails. |
| [src/features.py](src/features.py) | 14 distance pairs (`_PAIRS`) normalised by image diagonal + 16 radii sampled at fixed angles around the snake centroid, normalised by mean radius. |
| [src/dataset.py](src/dataset.py) | Reads/writes `dataset.csv` (`;`-separated, `name;v0;...;v29`). `write_all` for full rewrites (delete), `append` for single rows (enrolment). |
| [src/identify.py](src/identify.py) | NN search + `confidence(d, t) = 1 - d/t`. Returns the 5-tuple `(ok, name, dist, conf, top_k)`. |
| [src/evaluation.py](src/evaluation.py) | Leave-one-out, confusion matrix, per-class precision/rappel/specificite, threshold sweep. Unknown class label is the literal string `"Inconnu"`. |
| [src/door_sim.py](src/door_sim.py) | Tk window for the production demo. Receives `('open'|'denied'|'wait'|'quit', name, value)` via `queue.Queue`. |
| [scripts/identity_ui.py](scripts/identity_ui.py) | Sibling of `test_app.py`. Owns the shared `Theme` palette, `LoadingPopup`, `EnrolWizard` (guided 20-pose capture), `IdentityManager` + `DeleteConfirmDialog`, and `generate_evaluation_report` (markdown). `test_app.py` imports from it; `Theme` is the single source of truth for both files. |

## Concurrency model (both Tk apps)

[scripts/main.py](scripts/main.py) and [scripts/test_app.py](scripts/test_app.py) share the same pattern: a daemon **worker thread** owns the `cv2.VideoCapture` and runs Haar + alignment on every frame; the Tk main thread polls a `queue.Queue` via `root.after(...)` and is the only thread that touches widgets. The expensive snake (250 iter, ~hundreds of ms) runs **only on the `I` keypress**, never per frame -- otherwise the live preview drops below 20 fps. When changing either app, preserve this split: never call Tk from the worker, never call `cap.read()` from the Tk main thread.

`main.py` differs slightly: it uses **two separate `tk.Tk()` roots** (the `door_sim` window) plus a `cv2.imshow` window driven by the worker. `test_app.py` is a single root with the door drawn as a `tk.Canvas` icon -- per the design briefs in [NEXT_AGENT_PROMPT.md](NEXT_AGENT_PROMPT.md) and [NEXT_AGENT_PROMPT_V2.md](NEXT_AGENT_PROMPT_V2.md), `test_app.py` is the everyday interface but must not replace `main.py` (which stays the soutenance door demo).

`test_app.py` extends the basic worker/Tk split with a **three-queue + one-lock** plumbing -- when adding a new long-running operation, plug into it rather than spawning ad-hoc threads:

- `frame_queue` -- worker -> Tk, latest aligned frame for preview.
- `event_queue` -- worker -> Tk, completion events (identification result, encoded vectors, written report path, errors). The Tk thread is the only consumer; modals + progress bars react to events here.
- `cmd_queue` -- Tk -> worker, named commands (e.g. `CMD_ENROL_BATCH`, `CMD_DELETE_PERSON`, `CMD_GENERATE_REPORT`). Heavy work (snake + landmarks + features for 20 images, full `dataset.csv` rewrite, leave-one-out evaluation) runs here, not on the Tk thread.
- `dataset_lock` -- guards `self.names` / `self.vectors` mutation. The wizard / manager hold it only across the actual swap, never across I/O.

The threshold slider must keep working during enrolment (it is independent of the wizard). Closing the wizard with the window manager (`X`) must be treated as **Annuler la session** and clean up any captured files created during that session.

## Dataset conventions

- `captures/<person>/img_NNN.jpg` -- 128x128 grayscale JPEG, already aligned (output of `alignment.align_face`). The brief asks for 20 images per person covering 5 neutre / 5 expressions / 5 rotations / 5 eclairages.
- `dataset.csv` -- semicolon-separated, one row per image: `name;d0;...;d13;s0;...;s15`. `dataset.append` is the *only* sanctioned way to grow it from a running app -- it stays consistent with `dataset.write_all` from `build_dataset.py`.
- `captures/` is gitignored except for `.gitkeep`; `dataset.csv` is also gitignored.

## Reports

`reports/<YYYYMMDD_HHMMSS>_<event>.md` is auto-written at the end of every dataset mutation triggered from `test_app.py`:

- `add_<name>` after the enrolment wizard finishes encoding,
- `delete_<name>` after the identity manager removes a person,
- `manual` when the user clicks **Exporter rapport**.

Generation is dispatched via `cmd_queue` (`CMD_GENERATE_REPORT`) and runs `evaluation.leave_one_out` + `evaluation.confusion_matrix` against the **current session threshold** (slider value, not `DEFAULT_THRESHOLD` unless they coincide). The markdown contract -- header, dataset stats table, threshold note, fenced confusion matrix, per-class precision/rappel/specificite/support table, global accuracy -- is fixed in `identity_ui.generate_evaluation_report` and is what the soutenance report consumes; do not silently change column names or section order. `reports/` is gitignored.

If the dataset has fewer than ~2 vectors per person on average, the leave-one-out section is skipped with a one-line note rather than producing meaningless metrics.

## Threshold workflow

`DEFAULT_THRESHOLD = 0.18` is a placeholder. The intended workflow whenever the dataset changes materially:

1. `python scripts/evaluate.py --sweep` -> read the best threshold.
2. Update `DEFAULT_THRESHOLD` in [src/config.py](src/config.py).
3. `python scripts/evaluate.py -t <new>` -> commit the confusion matrix output to the report.

`test_app.py` exposes a live slider that overrides the threshold for the running session only -- it does **not** write back to `config.py` on purpose.

## Things to avoid

- Don't modify files in `src/` to "fix" something observed only in the apps -- the pipeline contract is shared by `build_dataset.py`, `evaluate.py`, `main.py`, and `test_app.py`. Changing a normalisation in `features.py` invalidates every existing `dataset.csv`.
- Don't add emojis to UI strings, comments, or filenames -- the project's visual identity is dark academic.
- Don't introduce a web stack (PyWebView, Eel, tkhtmlview). Pure Tkinter + ttk + Pillow.
- Don't run the snake on every frame in any live preview -- including inside the enrolment wizard.
- Don't fork a new `Theme` palette in `test_app.py`. The class lives in [scripts/identity_ui.py](scripts/identity_ui.py) and both files share it; duplicating it drifts the dark-academic look between the dashboard and the wizards.
- Don't shell out to `scripts/build_dataset.py` or `scripts/evaluate.py` from the apps -- call the library functions directly so failures surface in-app and so `dataset_lock` actually protects the swap.
