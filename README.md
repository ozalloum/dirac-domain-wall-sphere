# Dirac domain-wall spectra on the sphere

Reproducible code and numerical outputs for a pole-regular Jacobi–Galerkin study of low-energy modes of an axially symmetric massive Dirac operator on the round sphere. The package includes an independent two-sided shooting benchmark, convergence checks, computational-cost measurements, CSV output tables, and publication-quality figures.

The manuscript PDF and LaTeX source are intentionally excluded while the manuscript is under review.

## Quick start

Requires Python 3.10 or later.

```bash
python -m pip install -r requirements.txt
python run_study.py --output .
```

The full run regenerates the CSV tables, figures, and `simulation_summary.json`. It uses a 112-function-per-component basis and 1000-point Gauss–Legendre quadrature for the main spectra; the first-excited calculation uses 140 functions per component. It also runs an independent ODE shooting check and measures basis construction, matrix assembly, and dense diagonalization. Timings are host- and software-dependent. A shorter smoke test is available with `python run_study.py --output . --quick`; it omits the independent benchmark and cost tables.

## Google Colab

Open [`notebooks/Dirac_Domain_Wall_Sphere_Colab.ipynb`](https://colab.research.google.com/github/ozalloum/dirac-domain-wall-sphere/blob/main/notebooks/Dirac_Domain_Wall_Sphere_Colab.ipynb) in Colab and run the setup cell. It clones this repository, installs the listed Python dependencies, and can rerun the study.

## Repository contents

- `src/sphere_dirac_galerkin.py`: pole-regular Jacobi–Galerkin solver.
- `benchmark_methods.py`: independent two-sided shooting check and cost benchmark.
- `run_study.py`: simulation, validation, CSV, and plotting driver.
- `notebooks/`: Google Colab notebook.
- `data/*.csv`: exact-spectrum validation, wall spectra, convergence, shooting benchmark, cost measurements, and sampled eigenfunction densities.
- `figures/`: six publication-quality vector plots in SVG format.
- `simulation_summary.json`: simulation configuration, benchmark errors, and cost-test environment.
- `CITATION.cff`: citation metadata.
- `MANIFEST.sha256`: file-integrity checksums.

## Model and numerical scope

The dimensionless calculations use sphere radius `R = 1` and mass scale `mu = 1`, with half-integer angular sectors `nu = 1/2` and `3/2`. The single-wall profile is `m(theta) = mu cos(theta)`; the reflection-symmetric two-wall profile is `m(theta) = mu cos(2 theta)`. Component-specific endpoint powers encode the globally regular spinor domain at both poles.

The massless and constant-mass tests compare against exact spectra. The single-wall localized branch is also checked by independently integrating the first-order Dirac ODE from both regular poles and matching at the equator. The calculations resolve algebraic spectral scales; they do not claim to establish an exponentially accurate tunneling correction.

## Data and software availability

The CSV tables and scripts in this repository are the numerical data and software supporting the computations. No external datasets are required. The timing table records the machine and software environment because measured runtimes will differ on other systems.

## Citation and license

Citation metadata are provided in `CITATION.cff`. No license is included. The repository is public for inspection and reproduction, but no permission to reuse, modify, or redistribute the code or data is granted; contact the author to request permission.
