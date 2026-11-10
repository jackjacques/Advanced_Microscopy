# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A course labwork ("How to build a virtual microscope", BME imaging, `Fundamentals of Microscopy Labwork.pdf` is the assignment) made of two deliverables:
- `main.py`: a single-file numerical simulation of a widefield microscope (Fourier optics) that produces all figures.
- `Report_Jacques_Angleys.tex`: the LaTeX lab report answering the assignment questions (Q1–Q11), organized by the same Parts 1–6 as the code, and including the PNGs from `figures/`.

Code comments and plot titles are mostly in French; the report is in English.

## Commands

```bash
source .venv/bin/activate            # deps: numpy, matplotlib, opencv-python, scipy (requirements.txt)
pip install -r requirements.txt
python main.py                       # regenerates every figure in figures/ (~2.5 min)
latexmk -pdf Report_Jacques_Angleys.tex
```

There are no tests or linter. `main.py` calls `plt.show()` between parts, so a run blocks on each window; use `MPLBACKEND=Agg python main.py` for a non-interactive run. The `__main__` block runs all parts sequentially with `np.random.seed(0)`, so to regenerate a single figure, call only the relevant method(s) rather than editing seeds.

## Architecture of `main.py`

Everything lives in the `Microscopy` class, with methods grouped by assignment part (section banners like `# ===== PART 2 ... =====`):

- **Object (Part 1)**: `makeGrid` builds the FFT-consistent `N×N` grid `(k - N/2)·dx` with `dx = L/N`. `makeObject` follows the professor's reference `makeObject.docx` (MATLAB): transparent spherical cells whose phases add, plus a slightly absorbing, randomly offset nucleus (amplitude `nucleus_amplitude=0.95`) whose transmissions multiply. It returns `phase_map` and `intensity_map = amplitude²`, so the object is almost a pure phase object. `get_complex_field()` returns `sqrt(I)·exp(iφ)`, which is the input to every imaging step.
- **Imaging (Part 2)**: 4f-system model — FFT (with `fftshift`) → multiply by a pupil mask in frequency space → inverse FFT → `|E|²`. `computePupil` is a circular cutoff at `fc = NA/λ`. `plotPupils` (Q2) and `compareNAs` (Q3) make the Part 2 figures. `resolution()` (Q3.3) builds a synthetic USAF 1951 target at true physical scale (`makeUSAFTarget`, groups 4–9), and prints the finest element whose contrast (`elementContrast`) is above 0.15. `usaf.png` / the SVG are not used: rescaled into the field of view, their physical scale is wrong.
- **Contrast modes (Part 3)**: alternative pupils (`darkfieldPupil`, `phaseContrastPupil`) applied through the generic `imageWithPupil(pupil, field)`. `contrastModes` shows brightfield and darkfield on the real object (~33 rad) and phase contrast on a weak object (phase rescaled to 0.3 rad); fields are always simulated exactly with `exp(iφ)`.
- **Camera (Part 4)**: `addCameraEffects` does pixel integration (`resampleToCamera`, pitch / magnification) plus Poisson noise. SNR is defined on the background: the image is normalized by its median (not its max, which is a thin edge fringe that large pixels average away), so the background gets `SNR²` photons and the output is relative to background. `nucleusCentre()` gives the centroid of the most absorbing nucleus for zooms and profiles.
- **Defocus (Part 5)**: `fresnelPropagate` (angular-spectrum propagation by `dz`). `simulate_two_layer` models two thin layers as `E = t2 · P_dz{t1}` (transmissions multiply, never add fields) and returns the images focused on layer 1 (back-propagate by `-dz`) and on layer 2; layer 2 is generated once and cached (`getLayer2`). Figures: `defocusComparison`, `twoLayerComparison`. With these phase objects, the in-focus layer is nearly invisible (only nuclei) and defocus is what makes cells visible (dark or bright rims depending on the sign of dz).
- **Full pipeline (Part 6)**: `assembleMicroscope` chains defocus → pupil (`mode`: brightfield / darkfield / phase, via `modePupil`) → camera → crop to the sensor field, and returns `(image, extent_um)`. The Q10 components are the module constants `OBJECTIVE` and `CAMERA` (10X/NA 0.25, Zelux CS165MU 3.45 µm; part numbers to be checked on Thorlabs) used by `finalMicroscope` (one brightfield image). `realCellsComparison` (Q11) places `Red Blood Cells DIC 8um.jpg` at true scale (8 µm cell ≈ 70 px → 114 nm/px) on its own `Microscopy(n_cells=0, ...)` grid, padded with background and flipped upright; the `.tif` in the repo is the same image.

Sampling constraint: the pupil radius `fc = NA/λ` must fit inside the grid Nyquist frequency `1/(2·dx)`, i.e. `NA_max = λ/(2·dx)`. Defaults are `L = 1 mm`, `N = 4096` (`dx ≈ 0.24 µm`, `NA_max ≈ 1.1`); with the old `N = 1024`, any NA above ~0.23 meant no filtering at all. Any new grid must respect this. Full-field arrays are 4096² (≈270 MB per complex array), so `makeObject` only computes each cell inside its bounding box.

Images loaded from files (row 0 = top) appear upside down with `imshow(..., origin="lower")`; synthetic maps built on the `makeGrid` grid are fine with `origin="lower"`.

Figure filenames (`figures/partX_*.png`) are referenced directly by `\includegraphics` in the report; keep names in sync when renaming.

## Repo notes

- Commit messages follow a `type: message` prefix style (`code:`, `doc:`, `fix:`, `feat:`, `redaction:`).
- `figures/` is tracked so the Overleaf report and teammates share the same figures: commit regenerated figures together with the code change that produced them.
- LaTeX build artifacts (`.aux`, `.log`, `.fls`, `.fdb_latexmk`, `.synctex.gz`, …) and the report PDF are gitignored; the report is compiled on Overleaf, so only `Report_Jacques_Angleys.tex` is versioned.

## Deliverable and level

Only the PDF of `Report_Jacques_Angleys.tex` is handed in (the code is not). It must read like a student group's work after a 3-hour lab: short answers to each question, few formulas, simple physical explanations, and figures that are plain images (no extra plots, profiles or metric panels).
