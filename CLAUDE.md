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

- **Object (Part 1)**: `makeGrid` builds the FFT-consistent `N×N` grid `(k - N/2)·dx` with `dx = L/N`. `makeObject` follows the professor's reference `makeObject.docx` (MATLAB): transparent spherical cells whose phases add, plus a slightly absorbing, randomly offset nucleus (amplitude `nucleus_amplitude=0.95`) whose transmissions multiply. It returns `phase_map` and `intensity_map = amplitude²`, so the object is almost a pure phase object. `get_complex_field()` returns `sqrt(I)·exp(iφ)`, which is the input to every imaging step.
- **Imaging (Part 2)**: 4f-system model — FFT (with `fftshift`) → multiply by a pupil mask in frequency space → inverse FFT → `|E|²`. `computePupil` is a circular cutoff at `fc = NA/λ`. `plotPupils` (Q2) and `compareNAs` (Q3) make the Part 2 figures. `resolution()` (Q3.3) builds a synthetic USAF 1951 target at true physical scale (`makeUSAFTarget`, groups 4–9), measures Michelson contrast per element from a 1D profile (`elementContrast`), and reports the finest element above the threshold (0.15), to compare with the coherent cutoff `NA/λ`. `usaf.png` / the SVG are not used: rescaled into the field of view, their physical scale is wrong.
- **Contrast modes (Part 3)**: alternative pupils (`darkfieldPupil`, `phaseContrastPupil`) applied through the generic `imageWithPupil(pupil, field)`. `contrastModes` shows them on the real object (~33 rad) and on a weak object (phase rescaled to 0.3 rad); fields are always simulated exactly with `exp(iφ)`, and `1 + iφ` is only the theory used in the report.
- **Camera (Part 4)**: `addCameraEffects` does pixel integration (pitch / magnification) plus Poisson noise for a target SNR.
- **Defocus (Part 5)**: `fresnelPropagate` (angular-spectrum propagation by `dz`) and `simulate_two_layer`.
- **Full pipeline (Part 6)**: `assembleMicroscope` chains propagate → pupil → normalize → camera; `imageRealCells` feeds a real image (`Red Blood Cells DIC 8um.jpg`) through it as amplitude or phase.

Sampling constraint: the pupil radius `fc = NA/λ` must fit inside the grid Nyquist frequency `1/(2·dx)`, i.e. `NA_max = λ/(2·dx)`. Defaults are `L = 1 mm`, `N = 4096` (`dx ≈ 0.24 µm`, `NA_max ≈ 1.1`); with the old `N = 1024`, any NA above ~0.23 meant no filtering at all. Any new grid (e.g. the Part 6 instance `Microscopy(L=0.4e-3, N=2048, ...)`) must respect this. Full-field arrays are 4096² (≈270 MB per complex array), so `makeObject` only computes each cell inside its bounding box.

Images loaded from files (row 0 = top) appear upside down with `imshow(..., origin="lower")`; synthetic maps built on the `makeGrid` grid are fine with `origin="lower"`.

Figure filenames (`figures/partX_*.png`) are referenced directly by `\includegraphics` in the report; keep names in sync when renaming.

## Repo notes

- Commit messages follow a `type: message` prefix style (`code:`, `doc:`, `fix:`, `feat:`, `redaction:`).
- LaTeX build artifacts (`.aux`, `.log`, `.fls`, `.fdb_latexmk`, `.synctex.gz`, …) and the report PDF are gitignored; the report is compiled on Overleaf, so only `Report_Jacques_Angleys.tex` is versioned.
