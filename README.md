# GWRForge

[中文说明](README.zh.md)

**GWRForge** is a reproducible experiment and benchmarking project for advanced geographically weighted regression models implemented in [pyGWRx](https://github.com/hujinghaoabcd/pyGWRx). The current study focuses on two user-developed methods:

- **LG-GWR (Latent-Geometry GWR):** learns a shared latent geometry from geographic coordinates and contextual attributes, then performs local weighted regression in the learned space.
- **GR-GWR (Geographic-Regime GWR):** discovers spatially connected regimes from local GWR coefficient patterns and fits regime-aware local regressions with boundary regularization and ICM refinement.

The repository is designed to support transparent method validation, empirical comparison, paper-scale experiments, and reproducible reporting. Model implementations are **not copied into this repository**; experiment code calls the public `pygwrx` API.

## Research design

The experiment framework follows the principle:

> **Real-data model fitting as the main evidence + controlled simulations for structural recovery + limited prediction experiments as supplementary validation.**

### LG-GWR experiments

| Component | Purpose |
|---|---|
| Numerical degeneration | Verify that the separable formulation reduces to ordinary GWR when the attribute effect is disabled. |
| L1–L6 simulations | Test Euclidean geometry, anisotropy, joint geographic–attribute geometry, attribute-dominant structure, noise attributes, and parameter-specific geometry as a negative control. |
| Real-data fitting | Compare OLS, GWR, MGWR, SGWR, LG-GWR Joint, and LG-GWR Separable using AICc, adjusted R², RSS, RMSE, MAE, ENP, and residual spatial autocorrelation. |
| Geometry diagnostics | Evaluate learned-distance correlation, neighborhood overlap, latent contributions, and neighborhood changes. |
| Ablation and sensitivity | Study latent dimension, attribute removal/shuffling, restart count, bandwidth updating, and standardization. |
| Supplementary prediction | Check whether fitting improvements persist on held-out or spatially held-out observations. |

### GR-GWR experiments

| Component | Purpose |
|---|---|
| Numerical degeneration | Verify that a one-regime GR-GWR reduces to ordinary GWR. |
| R1–R6 simulations | Test no-regime structure, regular and irregular connected regimes, imbalanced regimes, smooth transitions, and disconnected identical mechanisms. |
| Real-data fitting | Compare OLS, GWR, MGWR, regional baselines, and GR-GWR using fit diagnostics and regime-specific diagnostics. |
| Regime diagnostics | Report regime count and size, within-regime coefficient variance, between-regime differences, connectivity, boundaries, and stability. |
| Ablation and sensitivity | Evaluate ICM refinement, boundary penalty, connectivity enforcement, coordinate–coefficient balance, initialization, and minimum regime size. |
| Supplementary prediction | Evaluate whether the learned regimes provide limited out-of-sample value. |

## Repository status

| Item | Status |
|---|---|
| Experiment package and command-line scripts | Implemented |
| Numerical degeneration tests | Implemented and validated |
| L1–L6 and R1–R6 generators | Implemented |
| Pilot end-to-end pipeline | Executed |
| Parameter selection, ablation, statistics, figures, and reports | Implemented |
| Full paper-scale configuration | Available in `configs/full.json` |
| Complete paper-scale results | Not yet fully executed; generated outputs must not be replaced by pilot values |

Pilot outputs are engineering evidence only. They confirm that the software pipeline, metrics, file outputs, and reporting contracts work. Formal scientific claims require the complete repeated experiments specified by the full configuration.

## Installation

Python 3.11 or later is recommended.

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The experiment project installs `pyGWRx==0.1.2` from PyPI and installs GWRForge in editable mode.

## Quick start

Run the lightweight pilot profile first:

```bash
python scripts/run_pipeline.py \
  --config configs/pilot.json \
  --profile pilot
```

Run the paper-scale configuration:

```bash
python scripts/run_pipeline.py \
  --config configs/full.json \
  --profile full
```

The full profile is computationally intensive. It contains a 20 × 20 simulation grid, 100 independent test locations, 100 Monte Carlo repetitions per scenario, repeated spatial validation, and multi-start LG-GWR optimization.

## Run individual stages

```bash
python scripts/00_environment_check.py
python scripts/01_correctness.py
python scripts/02_lg_simulation.py --config configs/pilot.json
python scripts/03_gr_simulation.py --config configs/pilot.json
python scripts/04_real_data.py --config configs/pilot.json
python scripts/06_parameter_selection.py --config configs/pilot.json
python scripts/07_ablation.py --config configs/pilot.json
python scripts/08_benchmark.py --config configs/pilot.json
python scripts/09_statistics.py
python scripts/10_figures.py
python scripts/05_report.py --profile pilot
```

## Output policy

Generated artifacts are written under `results/` and are intentionally ignored by Git. Typical outputs include:

- machine-readable CSV result tables;
- environment and execution logs;
- simulation and empirical figures;
- statistical comparisons;
- fixed sampling indices and checkpoints;
- automatically generated Markdown reports.

Failed runs, warnings, and non-convergence records are retained as experimental evidence and must not be silently removed.

## Project structure

```text
GWRForge/
├── configs/                    # Pilot and paper-scale experiment profiles
├── reports/                    # Method plans, audits, and pilot reports
├── scripts/                    # Executable experiment stages
├── src/pygwrx_experiments/     # Simulation, model adapters, metrics, runners, figures
├── tests/                      # Fast engineering and numerical tests
├── .github/workflows/          # Continuous integration
├── pyproject.toml
└── requirements.txt
```

## Reproducibility rules

1. Set and record random seeds for every repeated experiment.
2. Standardize data using training or fitting data only when a hold-out design is used.
3. Use the same tuning budget for comparable models.
4. Keep pilot and formal results in separate output directories.
5. Report failures and convergence rates rather than deleting unsuccessful runs.
6. Do not claim geometry or regime recovery from predictive improvement alone.
7. Treat prediction as supplementary evidence; real-data fitting and structural recovery remain the primary analyses.

## Testing

```bash
pytest
```

GitHub Actions runs the same test suite for pushes and pull requests.

## Relationship to pyGWRx

GWRForge is an experiment repository. The model implementations, datasets, and public APIs are provided by `pyGWRx`. Issues concerning model internals should be reported in the pyGWRx repository; issues concerning experiment design, scripts, configurations, or reporting belong here.

## Author

**Jinghao Hu**
