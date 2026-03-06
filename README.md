# Compounding Vulnerability — Reproducible Code Package

**Paper:** _Compounding Vulnerability: Hub Removal Triggers Cascade Phase Transitions While Degrading Percolation Robustness in Scale-Free Networks_  
**Author:** Federico Cachero (federicohernancachero@gmail.com)  
**arXiv:** [2603.04838](https://arxiv.org/abs/2603.04838)  
**Target:** Physical Review E  
**Date:** March 2026

---

## Overview

This repository contains simulation code, derived results (`RESULTS_*.json`), and figure-generation scripts to reproduce the main findings:

- Targeted hub removal (top degree-ranked fraction) **raises** the bond-percolation threshold (reduced robustness to random edge failure).
- The same intervention can **expand the Watts cascade window** and trigger a cascade phase transition (subcritical → supercritical) at intermediate thresholds.
- A controlled "hub vulnerability" experiment isolates the firewall mechanism as primarily **dynamical** (threshold-mediated), not purely topological.
- The cascade branching factor z₁(φ, α, m) predicts threshold crossing: pre-removal z₁ = 0.850 (subcritical) → post-removal z₁ = 1.195 (supercritical).

**429,000+ simulation trials** back the claims in the paper.

## What's New (v2 — March 6, 2026)

- **Newman-Ziff susceptibility-peak percolation** replaces S=0.5 as primary method (+347% ΔF)
- **Configuration-model validation**: Effect is STRONGER on config model (38% vs 23%), validating z₁ framework
- **Finite-size scaling**: Effect INCREASES with N (+352% at N=2K → +412% at N=10K)
- **Heterogeneous thresholds**: Effect persists under all distributions (+349% to +1,955%)
- **4 real-world SNAP networks**: Cascade amplification universal (+2,082% to +60,272%)
- **Lean 4 proofs updated**: 5 claims, no `sorry`

## Requirements

```bash
python >= 3.10
pip install -r requirements.txt
```

## Repository Structure

### Core Simulations
- `simulate_production.py` — core synthetic-network simulations (finite-size scaling)
- `phi_sweep_large.py` — φ-sweep at large N (transition sharpening)
- `hub_vulnerability_test.py` — controlled experiment (hub thresholds vs removal)
- `z1_analytical.py` — analytical checks / z₁ calculations
- `verify_quick.py` — quick sanity checks
- `simulate_empirical.py` — empirical network simulations

### V2 Validation (fixes/)
- `newman_ziff_percolation.py` — susceptibility-peak percolation method
- `newman_ziff_multiscale.py` — finite-size scaling at N=2K/5K/10K
- `config_model_validation.py` — configuration-model comparison
- `heterogeneous_thresholds.py` — uniform/beta/truncated-normal threshold tests
- `increased_trials_cascade.py` — 1000-trial cascade sweep with CIs
- `additional_real_networks.py` — 4 SNAP real-world networks

### Real-World Networks (realworld/)
- `realworld_validation.py` — Bitcoin OTC + AS-733 validation
- `realworld_validation_results.json` — results
- `INTERPRETATION.md` — analysis of real-world findings

### Figure Generation
- `generate_figures.py`, `generate_paper_figures.py`, `generate_phi_figure_v4.py`

### Results
- `RESULTS_*.json` — stored outputs from all simulations
- `fixes/*.json` — v2 validation results

### Paper
- `papers/paper-a/arxiv/` — arXiv v1 source
- `papers/paper-a/journal-v2/` — journal submission (v2, PRE format)

### Formal Verification
- `proofs/P2Complete.lean` — Lean 4 formalization (5 claims, no `sorry`)

## Reproducing Results

```bash
# Core simulations
python verify_quick.py
python simulate_production.py
python hub_vulnerability_test.py
python phi_sweep_large.py

# V2 validation
cd fixes/
python newman_ziff_percolation.py
python config_model_validation.py
python heterogeneous_thresholds.py
python newman_ziff_multiscale.py
python additional_real_networks.py

# Empirical
python simulate_empirical.py

# Figures
python generate_figures.py
python generate_phi_figure_v4.py
```

## Key Results

| Metric | Pre-removal | Post-removal | Change |
|--------|-------------|--------------|--------|
| Bond percolation p_c (Newman-Ziff) | 0.146 ± 0.008 | 0.654 ± 0.012 | +347% |
| Cascade size at φ=0.22 | 0.86% | 23.1% [CI: 21.3-24.9] | Phase transition |
| z₁ branching factor at φ=0.22 | 0.850 | 1.195 | Subcritical → supercritical |

## License

- Code: MIT (see `LICENSE`)
- Paper/manuscript assets: CC BY 4.0 (see `LICENSE-PAPER`)
