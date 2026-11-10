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
mkdir -p figures && python main.py   # regenerates every figure; figures/ is gitignored
latexmk -pdf Report_Jacques_Angleys.tex
```

There are no tests or linter. `main.py` calls `plt.show()` between parts, so a run blocks on each window; use `MPLBACKEND=Agg python main.py` for a non-interactive run. The `__main__` block runs all parts sequentially with `np.random.seed(0)`, so to regenerate a single figure, call only the relevant method(s) rather than editing seeds.

## Architecture of `main.py`

Everything lives in the `Microscopy` class, with methods grouped by assignment part (section banners like `# ===== PART 2 ... =====`):

- **Object (Part 1)**: `__init__` builds an `N×N` grid over a field of view `L` (pixel `dx = L/N`), then `makeObject` generates random cells with nuclei as an `intensity_map` and a `phase_map` (phase from `delta_n`, `lambda_ref`). `absorptive=True` makes the object attenuate, not just phase-shift. `get_complex_field()` returns `sqrt(I)·exp(iφ)`, which is the input to every imaging step.
- **Imaging (Part 2)**: 4f-system model — FFT (with `fftshift`) → multiply by a pupil mask in frequency space → inverse FFT → `|E|²`. `computePupil` is a circular cutoff at `fc = NA/λ`. `resolution()` measures resolution on the USAF 1951 target (`usaf.png`).
- **Contrast modes (Part 3)**: alternative pupils (`darkfieldPupil`, `phaseContrastPupil`) applied through the generic `imageWithPupil(pupil, field)`.
- **Camera (Part 4)**: `addCameraEffects` does pixel integration (pitch / magnification) plus Poisson noise for a target SNR.
- **Defocus (Part 5)**: `fresnelPropagate` (angular-spectrum propagation by `dz`) and `simulate_two_layer`.
- **Full pipeline (Part 6)**: `assembleMicroscope` chains propagate → pupil → normalize → camera; `imageRealCells` feeds a real image (`Red Blood Cells DIC 8um.jpg`) through it as amplitude or phase.

Sampling constraint: the pupil radius `fc = NA/λ` must fit inside the grid Nyquist frequency `1/(2·dx)`. That is why Part 6 builds a second instance with a smaller `L` and larger `N` (`Microscopy(L=0.4e-3, N=2048, ...)`) rather than reusing the default sample.

Figure filenames (`figures/partX_*.png`) are referenced directly by `\includegraphics` in the report; keep names in sync when renaming.

## Repo notes

- Commit messages follow a `type: message` prefix style (`code:`, `doc:`, `fix:`, `feat:`, `redaction:`).
- LaTeX build artifacts (`.aux`, `.log`, `.fls`, `.fdb_latexmk`, `.synctex.gz`, …) and the report PDF are gitignored; the report is compiled on Overleaf, so only `Report_Jacques_Angleys.tex` is versioned.
