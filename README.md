# Compounding Vulnerability — Reproducible Code Package

**Paper:** "Compounding Vulnerability: Hub Removal Triggers Cascade Phase Transitions While Degrading Percolation Robustness in Scale-Free Networks"

**Author:** Federico Cachero (federicohernancachero@gmail.com)  
**Target:** Physical Review E / arXiv (cond-mat.dis-nn)  
**Date:** March 2026

---

## Overview

This repository contains simulation code, derived results (`RESULTS_*.json`), and figure-generation scripts to reproduce the main findings:

- Targeted hub removal (top degree-ranked fraction) **raises** the bond-percolation threshold (reduced robustness to random edge failure).
- The same intervention can **expand the Watts cascade window** and trigger a cascade phase transition (subcritical → supercritical) at intermediate thresholds.
- A controlled “hub vulnerability” experiment isolates the firewall mechanism as primarily **dynamical** (threshold-mediated), not purely topological.

## Requirements

```bash
python >= 3.10
pip install networkx numpy matplotlib scipy
```

## Repository Structure (top-level)

- `simulate_production.py` — core synthetic-network simulations (finite-size scaling)
- `phi_sweep_large.py` — φ-sweep at large N (transition sharpening)
- `hub_vulnerability_test.py` — controlled experiment (hub thresholds vs removal)
- `z1_analytical.py` — analytical checks / $z_1$ calculations
- `verify_quick.py` — quick sanity checks
- `generate_figures.py`, `generate_paper_figures.py`, `generate_phi_figure_v4.py` — generate publication figures from `RESULTS_*.json`
- `RESULTS_*.json` — stored outputs used to build the paper’s plots
- `proofs/P2Complete.lean` — Lean 4 formalization of selected analytical inequalities/claims used in the paper

Note: figure PDFs are **generated** by running the `generate_*` scripts (they are not committed to the repo by default).

## Reproducing Results (minimal)

```bash
python verify_quick.py
python simulate_production.py
python hub_vulnerability_test.py
python phi_sweep_large.py

# then generate figures from RESULTS_*.json
python generate_figures.py
python generate_phi_figure_v4.py
```
