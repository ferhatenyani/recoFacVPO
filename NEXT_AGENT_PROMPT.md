# Prompt pour l'agent suivant -- App Tkinter de test du modele

You're picking up a Master 1 facial-identification project at `c:\Users\MyHomehP\Desktop\cours\S2\VPO\projet_tp`. Read [README.md](README.md), [docs/students_guide.md](docs/students_guide.md), and the assignment PDF (`Projet_Reconnaissance_Faciale_M1 2025-2025.pdf`) before doing anything. The pipeline (Haar -> align -> snake -> 30D vector -> NN, no ML/DL) is fixed.

## What exists already

- `scripts/main.py` -- production demo: OpenCV webcam window + separate Tkinter door window. **Do not modify or replace it.**
- `scripts/capture.py`, `scripts/build_dataset.py`, `scripts/evaluate.py`, `scripts/download_dataset.py` -- data tooling.
- `src/*.py` -- the pipeline (config, detection, alignment, snake, landmarks, features, dataset, identify, evaluation, door_sim). **Do not modify any module here.** Your job is to *consume* them.
- Dataset: `dataset.csv` already populated with 258 Essex Faces94 vectors across 13 subjects (the two student-captured identities will be added later by the team).

## Goal

Create a new **single-window Tkinter test/debug application** at `scripts/test_app.py`. It is a soutenance-grade tool that lets the user (a) see the live model decision in real time, and (b) inspect the intermediate steps so they can show the jury *why* the system identified a face the way it did. It is a complement to `main.py`, not a replacement.

## Use the design skills

You have access to:
- **frontend-design** skill -- invoke it for the overall look and information architecture.
- **ui-ux-pro-max** skill -- invoke it for layout, typography, color palette, spacing, and interaction states.

Both must be used. Tkinter is the constraint -- the skills should produce a design that you can faithfully execute in Tkinter/ttk + Pillow, not a web mock you can't ship. Translate any web-only concepts (CSS gradients, blur filters) into Tkinter equivalents (solid panels, subtle borders, deliberate typography).

## Hard requirements

### Architecture

- Single Tk root window, ~1200x780, resizable down to 1024x680.
- Threading: webcam capture + pipeline run in a daemon worker thread. The worker pushes frames + results into a `queue.Queue`. The Tk main thread polls via `root.after(33, ...)`. Never call Tk from the worker thread.
- Pillow (`Pillow>=10`) for `ImageTk.PhotoImage`. Add it to `requirements.txt` and `pip install` via `C:\Users\MyHomehP\anaconda3\python.exe -m pip install`.
- Frame rate target: ~25 fps for the live preview. The snake (~250 iterations on 128x128) is too slow per frame -- run it **only on demand** (when the user presses Identify), not on every frame. The live preview only needs Haar + alignment.

### Information architecture (bento dashboard)

Lay out the window in a deliberate grid (your choice -- 3 columns x 2 rows, or 2x3, etc.). Every panel below must be visible without scrolling. Treat each as a self-contained module with a label, value, and consistent inner padding:

1. **Webcam tile (largest panel).** Live BGR frame with overlay: green Haar bbox when face detected, two eye markers, and a thin caption row showing FPS and "face detected: yes/no".
2. **Aligned face panel.** The current 128x128 grayscale aligned face, upscaled to ~256x256 with `Image.NEAREST`. Updated live.
3. **Snake + landmarks overlay.** Same aligned face, with the snake contour drawn (closed polyline, semi-transparent or thin colored line) and the 8 landmark dots (eye_l/r, eye_l_out/r_out, nose, mouth_l/r, chin) annotated. Refreshed only when an Identify is performed (not every frame).
4. **Top-3 candidates panel.** Three rows. Each row: name, numeric distance, and a horizontal confidence bar whose width is `1 - dist/threshold` (clamped). The winning row gets a clear visual emphasis (different background or a left accent bar). If empty dataset, show a polite empty state.
5. **30D feature vector panel.** A bar chart of the current query vector. Use two color groups: the first 14 bars (distances) and the last 16 (snake radii). Add tick labels on the X axis (or grouped section labels: "distances" / "forme") so the jury can read it. No matplotlib -- draw it on a `tk.Canvas`.
6. **Decision banner / status panel.** Large readout: "ACCES AUTORISE -- &lt;name&gt;" in a positive accent color, or "ACCES REFUSE -- Inconnu" in a warning color, or "En attente". Show distance, threshold, and confidence numerically below the verdict. A small static door icon is acceptable here -- it does not need to animate (keep `door_sim.py` as the showcase animation in `main.py`).

### Controls

- Buttons: **Identifier** (also bound to `I`), **Capturer (Enrol)** (`E`), **Quitter** (`Q`).
- **Threshold slider** (`tk.Scale` or `ttk.Scale`), 0.05 -> 0.50, step 0.005, live -- moving it instantly recomputes the decision banner and refreshes the top-3 confidence bars without re-running the pipeline. The slider must reflect `DEFAULT_THRESHOLD` from `src/config.py` on startup, but local moves do not write back to config.
- **Snapshot** button: saves the current webcam frame + aligned face + bento screenshot to `screenshots/<timestamp>/` -- handy for the rapport.
- **E (Enrol)** opens a small modal dialog asking for the person's name (no terminal `input()` -- everything in-window). On confirm, recompute the vector and append to `dataset.csv` via `src.dataset.append`, refresh the in-memory dataset.

### Visual design

- The door simulation in `src/door_sim.py` is dark academic (`#1c1c1c`, `#3d3d3d`, red/green accents). The new app must read as the **same visual family** so a juror sees both windows and feels they belong to one product. You may go further (more refined typography, more deliberate spacing) but no jarring style break. No gradients, no rainbow palettes, no emojis, no playful microcopy. All UI text in French (the rest of the project is French).
- One typeface for headings, one for body. Pick deliberately. If you use a monospace, reserve it for technical readouts (distances, threshold values, FPS) -- not body labels.
- Every numeric readout must have a unit or scale indication where relevant (e.g. `distance: 0.183`, `seuil: 0.225`, `confiance: 67%`).
- Hover/focus states on buttons must be visible (not the default ttk grey-on-grey).
- The window must look intentional at the default theme on Windows 11 -- test with both `clam` and the native theme, pick whichever lets your design land.

## Things you must NOT do

- Don't modify any file in `src/`. If you discover a bug in the pipeline, write a one-line note in your final report -- don't patch it.
- Don't add a deep-learning detector (MTCNN, dlib, mediapipe). The whole project is no-ML.
- Don't replace `scripts/main.py`. Add a new file.
- Don't render with web tech (no PyWebView, no Eel, no tkhtmlview tricks). Pure Tkinter + ttk + Pillow. Tk-native widgets only.
- Don't add emojis to UI strings, file names, or comments.
- Don't auto-run the snake every frame -- it will tank the FPS.

## Environment

- Python: `C:\Users\MyHomehP\anaconda3\python.exe` (3.13.9, numpy 2.3.5, opencv-python 4.13.0). Install Pillow if missing.
- Windows / PowerShell. Use `pathlib`, not raw string paths.
- The project is **not** a git repo -- don't init one.
- Tkinter ships with the Anaconda Python -- already verified.

## Verification before claiming done

1. Launch the app: `& "C:\Users\MyHomehP\anaconda3\python.exe" scripts/test_app.py`.
2. Confirm: live webcam tile updates smoothly, Haar bbox appears on a real face, FPS reads >= 20.
3. Press **I** with a face that exists in `dataset.csv` (Faces94 IDs work -- show one of the Essex Faces94 photos on a phone screen if you have no real testers handy). Verify: aligned face panel shows the cropped face, snake overlay draws a sensible oval, landmark dots land in plausible positions, top-3 lists three names with sane distances, decision banner says "ACCES AUTORISE" with a confidence percentage, threshold slider moves the verdict.
4. Press **I** with the camera covered -- decision should remain "En attente" and no crash.
5. Move the threshold slider down to 0.05 -- nearly everything becomes "Inconnu". Move up to 0.50 -- nearly everything becomes "Autorise". Decision banner and confidence bars update live.
6. Press **E**, type a fake name, confirm -- `dataset.csv` gets one new row, in-memory dataset reflects it, and the next Identify on that face matches it.
7. Click **Snapshot** -- a `screenshots/<timestamp>/` folder appears with three images.
8. Press **Q** or close the window -- process exits cleanly, no zombie threads.

If any of those fail, fix before declaring done.

## What to deliver in your final message

In <400 words:

- One-line elevator pitch of the design choices the skills produced (palette, layout, typography rationale).
- The 6 verification checks: pass/fail, and what you saw.
- FPS measured on the user's machine.
- Any panel that didn't make the cut and why.
- Pillow version installed.
- Open issues left for the students (e.g. "the snake oval looks shifted upward on subject 9338543 -- noted but not fixed since `src/` is frozen").
