# Prompt for the next agent -- Identity management in `test_app.py`

You are picking up the Master 1 facial-identification project at `c:\Users\MyHomehP\Desktop\cours\S2\VPO\recoFacVPO`. **Read [CLAUDE.md](CLAUDE.md), [README.md](README.md), [NEXT_AGENT_PROMPT.md](NEXT_AGENT_PROMPT.md), and the existing [scripts/test_app.py](scripts/test_app.py) before writing a single line of code.** The pipeline (Haar -> align -> snake -> 30D vector -> NN, no ML/DL) is fixed and you must consume `src/*` rather than modify it.

## Context

`scripts/test_app.py` is the **main interface** for the project (it has superseded `main.py` for everyday use, even though `main.py` stays as the door-demo). Today it can identify a face and enrol a single image at a time via `E`. We need first-class identity management: **guided multi-pose enrolment** and **deletion**, with the in-memory dataset rebuilt on the fly so the user goes straight back to identifying without restarting the app.

There is no "model" in the ML sense -- "retraining" here means **rebuilding the 30D vectors of all images of the affected person from `captures/<name>/*.jpg` and rewriting `dataset.csv`**, then reloading `(names, vectors)` in memory. Use `src.dataset.write_all` for full rewrites and `src.dataset.append` for single rows. Wrap any mutation in `self.dataset_lock`.

## Hard requirements

### 1. Add face -- guided multi-pose enrolment

Replace the current single-shot `E` flow with a **wizard dialog** (modal `tk.Toplevel`, single `Tk` root preserved) that walks the user through the 20-image protocol from [README.md](README.md):

- 5 face neutre
- 5 expressions (sourire, surprise, yeux plisses)
- 5 rotations laterales (+/- 15 degres)
- 5 conditions d'eclairage variees

For **each step**, the wizard must:

1. Display the current pose label (large, in French) and a one-line instruction (e.g. *"Tournez la tete legerement a droite, yeux vers la camera"*).
2. Show the live webcam feed embedded **inside the wizard** (reuse the existing `frame_queue` -- do not open a second `VideoCapture`).
3. Show a per-step counter (e.g. *"3 / 5 -- expressions"*) and a global counter (e.g. *"13 / 20"*).
4. Require an aligned face on the current frame before the capture button enables (reuse `FrameResult.aligned`). If Haar fails, display *"Visage non detecte"* in danger color and disable capture.
5. On capture (button or `Espace`): write the 128x128 aligned grayscale to `captures/<name>/img_NNN.jpg` exactly as `scripts/capture.py` does, then advance the counters. Allow **Reprendre la derniere prise** (delete the last file, decrement counters) and **Annuler la session** (delete the whole `captures/<name>/` folder created by this wizard if it did not exist before -- never destroy a pre-existing folder).
6. Validate the name **before** any file is written: non-empty, no `;`, no path separators, no leading/trailing whitespace, must not collide with an existing person unless the user explicitly chose *"Ajouter d'autres prises a une identite existante"*.

When the 20 captures are done:

7. Show a **non-dismissable modal loading popup**: *"Encodage des vecteurs en cours..."* with an indeterminate `ttk.Progressbar`. The popup must block input but never freeze the UI thread.
8. In a **background thread**, for each new image: run `snk.fit_snake -> landmarks.detect_landmarks -> features.build_feature_vector` and accumulate the 20 vectors. **Do not run `build_dataset.py` as a subprocess** -- call the library directly so failures are surfacable in-app.
9. On completion, append the 20 rows to `dataset.csv` via `src.dataset.append` (one row per image), then under `dataset_lock` extend `self.names` / `self.vectors`. Update the header label via `_dataset_text()`.
10. Close the popup, close the wizard, **redirect the user to the main testing interface in identification mode** -- the bento dashboard with the live preview already showing -- and pre-fill the *Identifier* button focus so a single `Enter`/`I` runs identification on the freshly enrolled face. Show a success toast: *"Identite <name> enregistree -- 20 vecteurs encodes"*.

If any image fails the snake / landmark step, count it but keep going; report at the end (*"18 / 20 vecteurs encodes -- 2 images rejetees"*) and still redirect. Never leave the dataset in a partially-written state: append only after all encodings have completed (use a temp list, then a single critical section).

### 2. Delete face

Add a **Gerer les identites** button on the control bar (next to *Capturer*). It opens a modal listing every distinct name found in `dataset.csv` with its row count and the on-disk image count under `captures/<name>/`. For each row, two actions:

- **Supprimer** -- requires a confirmation dialog quoting the name and the row count. On confirm:
  1. Filter out every row of `dataset.csv` whose name matches, via `src.dataset.write_all`.
  2. Remove the `captures/<name>/` directory **only if** the user ticked *"Supprimer aussi les images sur le disque"* (default: ticked). Use `shutil.rmtree`. If the box is unticked, leave the folder so the identity can be re-encoded later via `build_dataset.py`.
  3. Under `dataset_lock`, drop the matching entries from `self.names` / `self.vectors`.
  4. Show a loading popup during the rewrite (`dataset.csv` rewrite is fast, but this keeps the UX consistent with add).
  5. Refresh the list and the header label; if `last_identify.candidates` referenced the deleted name, clear it and reset the decision banner to *"En attente"*.

The modal must remain inside the same `Tk` root, follow the existing `Theme` palette, and respect the no-emoji / French-only rule.

### 3. Exportable evaluation report (auto + manual)

Every time the dataset mutates -- end of an **add face** flow *and* end of a **delete face** flow -- automatically generate a fresh evaluation report and write it to `reports/<YYYYMMDD_HHMMSS>_<event>.md`, where `<event>` is `add_<name>`, `delete_<name>`, or `manual`. Also expose an **Exporter rapport** button on the control bar that writes the same report on demand against the current dataset.

The report is GitHub-flavoured Markdown (UTF-8). Use real Markdown headings (`#`, `##`), Markdown tables for tabular data, and fenced code blocks for the confusion matrix so the monospace alignment survives. It must contain, in this order:

1. **Header (`#` title + intro paragraph)** -- timestamp, trigger event (`auto` / `manual`, plus the name when applicable), Python + opencv versions, `IMG_SIZE`, `FEATURE_DIM`, snake hyperparameters from `config.py`.
2. **Dataset stats (`##`)** -- total vectors, total persons, and a Markdown table `| Personne | Vecteurs |` sorted by name.
3. **Threshold in use (`##`)** -- the current session threshold (from the slider, not `DEFAULT_THRESHOLD` unless they coincide). State explicitly that this value is session-only and not written back to `config.py`.
4. **Leave-one-out evaluation at the current threshold (`##`)** -- call `evaluation.leave_one_out` and `evaluation.confusion_matrix`. Render the matrix inside a triple-backtick fenced block (reuse `print_confusion` from [scripts/evaluate.py](scripts/evaluate.py) -- import it or copy the formatter; do not modify `src/`).
5. **Per-class metrics (`##`)** -- a Markdown table with columns `classe | precision | rappel | specificite | support`, one row per class plus `Inconnu`.
6. **Global accuracy and rejection rate (`##`)** -- two short lines, percentages with three decimals.

Behavioural constraints:

- The full leave-one-out can take seconds on a large dataset. Run it in a background thread, behind the same loading popup style as enrolment (*"Generation du rapport d'evaluation..."*). The popup must dismiss itself when the file is written, and a success toast must show the relative path: *"Rapport ecrit -- reports/<file>.md"*.
- For **auto** generation after add/delete, the popup chains directly after the encoding/rewrite popup; the wizard / management modal closes only once the report has been written, so the user always lands back on the main interface with an up-to-date report on disk.
- The `reports/` directory is created on demand and added to `.gitignore`.
- Never block the Tk thread on the evaluation -- always go through `event_queue`.
- If the dataset has fewer than 2 vectors per person on average, skip the leave-one-out section gracefully with a one-line note (`evaluation.leave_one_out` would be meaningless) and still write the rest.

## Behavioural rules (non-negotiable)

- Do not call Tk from the worker thread, do not call `cv2.VideoCapture.read` from the Tk thread. Reuse the existing `frame_queue` / `event_queue` / `cmd_queue` / `dataset_lock` plumbing -- add new command names (e.g. `CMD_ENROL_BATCH`, `CMD_DELETE_PERSON`) rather than ad-hoc threads.
- Every long operation (encoding 20 images, rewriting `dataset.csv`) runs off the Tk thread and reports back via `event_queue`. The Tk thread only drives the modal and the progress bar.
- Snake is still **on demand** -- never run it on every preview frame, including inside the wizard preview.
- The threshold slider must keep working at all times, including during enrolment (it does not interact with the wizard).
- If the user closes the wizard mid-flow with the window manager (`X`), treat it as **Annuler la session** and clean up.
- Pillow, numpy, opencv -- no new dependencies.
- All UI strings in French. No emojis. Same dark-academic palette already in [scripts/test_app.py](scripts/test_app.py) (`Theme` class).

## Verification before claiming done

1. Launch: `& "C:\Users\MyHomehP\anaconda3\python.exe" scripts/test_app.py`.
2. Add a new identity end-to-end -- step through all 20 poses, watch the loading popup, confirm the redirect lands on the main interface with the dataset header showing `+1 personne`.
3. Press `I` immediately after enrolment -- the new person must appear in the top-3 with a small distance.
4. Open *Gerer les identites*, delete the just-enrolled person with the *delete files on disk* box ticked, confirm `captures/<name>/` is gone and `dataset.csv` no longer contains them. Open it again and verify count goes back to the previous baseline.
5. Confirm a fresh `reports/<ts>_add_<name>.md` was written at the end of step 2 and a `reports/<ts>_delete_<name>.md` at the end of step 4. Open both in a Markdown viewer and verify the headings render, the per-person table is well-formed, the confusion matrix sits inside a fenced code block, and the per-class metrics table shows precision / rappel / specificite / support at the current slider threshold.
6. Press *Exporter rapport* manually -- a third file appears in `reports/` named `<ts>_manual.md` with identical structure.
7. Run `python scripts/evaluate.py --sweep` after a real enrolment to make sure `dataset.csv` is still parseable.
8. Try the failure paths: empty name, name with `;`, name colliding with an existing one, webcam unplugged mid-wizard, abandoning the wizard at step 7/20.

If any of these break, fix before reporting done.

Do **not** add features beyond the three above (live face-tracking smoothing, anti-spoofing CNNs, on-disk model caching, multi-camera support, network sync, "AI suggestions" -- all out of scope and drift from the assignment).

## Scope discipline

- Do not modify any file in `src/`. If you find a real bug there, surface it in your final report and stop.
- Do not touch `scripts/main.py` -- it is the door-demo for the soutenance.
- Do not introduce a state machine framework, an MVC refactor, or a settings file. Extend the existing `TestApp` class.
- The wizard, the management modal, the report writer, and any new helpers all live in `scripts/test_app.py`. If the file is getting long, a single sibling helper module under `scripts/` (e.g. `scripts/identity_ui.py`) is acceptable -- still no edits inside `src/`.
- Keep commits small and reviewable. One commit for the add-face wizard, one for delete, one for the auto + manual report.
