# The Dual Resilience Paradox — Reproducible Code Package

**Paper:** "Hub Removal Expands the Cascade Window While Improving Percolation Robustness: The Dual Resilience Paradox in Scale-Free Networks"

**Author:** Federico Cachero  
**Target:** Physical Review E  
**Date:** March 2026

---

## Overview

This repository contains all simulation code, data, and figure generation scripts needed to reproduce every result in the paper. All simulations use standard Python libraries (NetworkX, NumPy, Matplotlib) with no proprietary dependencies.

## Requirements

```
python >= 3.10
networkx >= 3.1
numpy >= 1.24
matplotlib >= 3.7
scipy >= 1.10
```

Install: `pip install networkx numpy matplotlib scipy`

## Repository Structure

```
├── README.md                    # This file
├── simulate_production.py       # Core simulation: percolation + cascade on synthetic networks
│                                # N ∈ {500,1000,2000,5000,10000}, 50 realizations, bootstrap CIs
├── simulate_empirical.py        # Empirical validation on real-world networks
│                                # (Power Grid, Email-Eu-Core, Facebook)
├── hub_vulnerability_test.py    # Hub vulnerability experiment (§3.5)
│                                # Discriminates dynamical vs topological firewall
├── phi_sweep_large.py           # φ-sweep for N=10,000 (Regime II verification)
├── susceptibility_fast.py       # Susceptibility analysis near critical point
├── generate_figures.py          # Generates Figures 1-5 from RESULTS_*.json
├── generate_phi_figure_v4.py    # Generates Figure 6 (φ-sweep)
│
├── RESULTS_COMBINED.json        # Main results (all N, all topologies)
├── RESULTS_EMPIRICAL.json       # Empirical network results
├── RESULTS_HUB_VULNERABILITY.json  # Hub vulnerability experiment
├── RESULTS_PHI_SWEEP.json       # φ-sweep N=2000
├── RESULTS_PHI_SWEEP_N10000.json   # φ-sweep N=10,000
├── RESULTS_SUSCEPTIBILITY.json  # Susceptibility near p_c
│
├── fig1_fss_percolation.pdf     # Figure 1: Finite-size scaling (percolation)
├── fig2_fss_cascade.pdf         # Figure 2: Finite-size scaling (cascade)
├── fig3_conservation_scatter.pdf # Figure 3: ΔF_p vs ΔF_c scatter
├── fig4_kappa_mediator.pdf      # Figure 4: κ as mediator
├── fig5_dual_barplot.pdf        # Figure 5: Dual bar plot
└── fig6_phi_sweep_v4.pdf        # Figure 6: φ-sweep with cascade window
```

## Reproducing Results

### 1. Core Simulations (≈45 min on 8-core machine)

```bash
# Run main simulations (N=500,1000,2000,5000,10000)
python simulate_production.py

# Output: RESULTS_COMBINED.json, RESULTS_LARGE.json
```

### 2. Hub Vulnerability Experiment (≈2 min)

```bash
python hub_vulnerability_test.py

# Tests 3 conditions on BA-2000:
# A) Baseline hubs (normal thresholds)
# B) Hubs present but vulnerable (φ=0.01 for hubs)  
# C) Hubs removed
# Prediction: B >> C >> A at φ=0.22
# Result: B=97% cascade, C=28%, A=0.5% (confirms dynamical firewall)
```

### 3. φ-Sweep at N=10,000 (≈20 min)

```bash
python phi_sweep_large.py

# Verifies the cascade phase transition sharpens at large N
# Output: RESULTS_PHI_SWEEP_N10000.json
```

### 4. Empirical Networks (≈10 min)

```bash
python simulate_empirical.py

# Downloads and tests: Power Grid, Email-Eu-Core, Facebook
# Output: RESULTS_EMPIRICAL.json
```

### 5. Generate All Figures

```bash
python generate_figures.py        # Figures 1-5
python generate_phi_figure_v4.py  # Figure 6
```

## Key Results Summary

| Finding | Result | Script | Data File |
|---------|--------|--------|-----------|
| BA ΔF_p (hub removal) | +167% ± 3% | simulate_production.py | RESULTS_COMBINED.json |
| Cascade phase transition at φ=0.22 | 1%→26% cascade | simulate_production.py | RESULTS_COMBINED.json |
| Hub vulnerability (dynamical firewall) | 97% vs 28% vs 0.5% | hub_vulnerability_test.py | RESULTS_HUB_VULNERABILITY.json |
| φ* phase boundary (N=10K) | φ*≈0.17 | phi_sweep_large.py | RESULTS_PHI_SWEEP_N10000.json |
| κ-dependence | κ>10 required | simulate_production.py | RESULTS_COMBINED.json |

## Simulation Parameters

All simulations use:
- **Percolation**: Bond percolation, 50 p-steps (or 200 for production), binary search for p_c
- **Cascade**: Watts 2002 linear threshold model, uniform φ, single random seed
- **Statistics**: 50 network realizations × 100 cascade trials per (network, φ), bootstrap 95% CIs (10,000 resamples)
- **Hub removal**: Top 10% by degree, simultaneous removal, all incident edges deleted
- **Networks**: BA (m=2, ⟨k⟩≈4), ER (p=4/(N-1)), WS (k=4, β=0.1)

## Mathematical Framework

The paper's core claims are:

1. **Percolation ΔF_p**: Relative change in percolation threshold after hub removal
   - ΔF_p = (p_c_after - p_c_before) / p_c_before
   - Analytically derivable from Molloy-Reed criterion

2. **Cascade ΔF_c**: Relative change in cascade size at given φ
   - Regime-dependent: ΔF_c < 0 for φ < φ*, ΔF_c > 0 for φ > φ*

3. **κ-threshold**: The degree heterogeneity ratio κ = ⟨k²⟩/⟨k⟩ mediates the paradox
   - κ > 10 required for observable cascade window expansion

## Lean4 Verification

Four core mathematical claims are formally verified in Lean4:
- See `~/lean-verify/TierSVerify/P2Complete.lean`

## License

MIT License. All code and data freely available for reproduction and extension.

## Contact

Federico Cachero — [email upon publication]
