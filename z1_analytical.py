"""
Analytical z₁ Cascade Condition After Hub Removal
==================================================

Computes the Gleeson z₁ cascade condition for BA networks before and after
targeted hub removal. Provides both exact (binomial convolution) and 
mean-field approximations.

Key result: At φ=0.22, hub removal shifts z₁ from 0.850 (subcritical) to 
1.195 (supercritical), explaining the cascade explosion from 1.1% to 26.4%.

Reference: Gleeson & Cahalane, PRE 75, 056103 (2007)
Author: F. Cachero
Date: 2026-03-01
"""

import numpy as np
from scipy.special import comb
from math import floor


def ba_P(k, m):
    """Barabási-Albert degree distribution P(k) = 2m(m+1)/[k(k+1)(k+2)]."""
    if k < m:
        return 0.0
    return 2 * m * (m + 1) / (k * (k + 1) * (k + 2))


def find_kcut(alpha, m):
    """Find k_cut such that removing nodes with k > k_cut gives fraction ~alpha.
    
    Uses the CLOSEST match to target alpha (not necessarily ≤ alpha).
    For BA: alpha(k_cut) = m(m+1) / [(k_cut+1)(k_cut+2)]
    """
    product = m * (m + 1) / alpha
    k_cut_approx = np.sqrt(product) - 1.5
    k_lo = max(m, int(k_cut_approx))
    k_hi = k_lo + 1
    alpha_lo = m * (m + 1) / ((k_lo + 1) * (k_lo + 2))
    alpha_hi = m * (m + 1) / ((k_hi + 1) * (k_hi + 2))
    # Pick the one closest to target
    if abs(alpha_lo - alpha) <= abs(alpha_hi - alpha):
        return k_lo
    return k_hi


def compute_rho(m, k_cut):
    """Fraction of edges leading to removed hubs: ρ = (m+1)/(k_cut+2)."""
    return (m + 1) / (k_cut + 2)


def harmonic(n):
    """Harmonic number H_n = sum_{k=1}^{n} 1/k."""
    return sum(1.0 / k for k in range(1, n + 1))


def z1_pre(phi, m, k_max=500):
    """Pre-removal z₁ for BA(m) network. Uses Gleeson's formula directly."""
    K = int(1 / phi)
    k_mean = 2 * m
    z1 = 0.0
    for k in range(m, min(K, k_max) + 1):
        z1 += k * (k - 1) * ba_P(k, m) / k_mean
    return z1


def z1_post_exact(phi, m, k_cut):
    """
    Post-removal z₁ with EXACT binomial convolution.
    
    After hub removal, a surviving node with original degree k has
    post-removal degree j ~ Binomial(k, 1-ρ). The vulnerable condition
    becomes j ≤ 1/φ.
    """
    rho = compute_rho(m, k_cut)
    alpha = m * (m + 1) / ((k_cut + 1) * (k_cut + 2))
    K = int(1 / phi)  # vulnerability threshold in post-removal degree
    
    # Mean degree post-removal
    k_mean_post = 0.0
    for k in range(m, k_cut + 1):
        k_mean_post += k * (1 - rho) * ba_P(k, m)
    k_mean_post /= (1 - alpha)
    
    if k_mean_post == 0:
        return 0.0
    
    # z₁ = sum_{j=1}^{K} j(j-1) P'(j) / <k>'
    z1 = 0.0
    for k in range(m, k_cut + 1):
        for j in range(1, min(K, k) + 1):
            binom_coeff = comb(k, j, exact=True)
            prob = binom_coeff * (1 - rho) ** j * rho ** (k - j)
            z1 += j * (j - 1) * prob * ba_P(k, m)
    z1 /= (1 - alpha) * k_mean_post
    
    return z1


def z1_post_meanfield(phi, m, k_cut):
    """
    Mean-field approximation of post-removal z₁.
    
    Closed form: z₁ = (m+1) * [H_{K*+1} - H_m + 3/(K*+2) - 3/(m+1)]
    where K* = min(⌊1/(φ(1-ρ))⌋, k_cut)
    """
    rho = compute_rho(m, k_cut)
    K_eff = floor(1 / (phi * (1 - rho)))
    K_star = min(K_eff, k_cut)
    
    if K_star < m:
        return 0.0
    
    z1 = (m + 1) * (harmonic(K_star + 1) - harmonic(m) 
                     + 3 / (K_star + 2) - 3 / (m + 1))
    return z1


def cascade_window_expansion(m, k_cut):
    """Expansion factor of cascade window: 1/(1-ρ)."""
    rho = compute_rho(m, k_cut)
    return 1.0 / (1.0 - rho)


def full_analysis(m=2, alpha_target=0.10):
    """Run complete z₁ analysis and print results."""
    k_cut = find_kcut(alpha_target, m)
    alpha_actual = m * (m + 1) / ((k_cut + 1) * (k_cut + 2))
    rho = compute_rho(m, k_cut)
    
    print(f"=" * 60)
    print(f"z₁ CASCADE ANALYSIS: BA(m={m}), α={alpha_actual:.3f} (k_cut={k_cut})")
    print(f"ρ = {rho:.3f} (fraction of edges to removed hubs)")
    print(f"Cascade window expansion: {cascade_window_expansion(m, k_cut):.2f}×")
    print(f"=" * 60)
    print()
    
    phis = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.27, 0.30, 0.33, 0.40, 0.50]
    
    print(f"{'φ':>6} | {'z₁_pre':>8} | {'z₁_post_exact':>13} | {'z₁_post_MF':>10} | {'Status':>15}")
    print("-" * 65)
    
    for phi in phis:
        z_pre = z1_pre(phi, m)
        z_post = z1_post_exact(phi, m, k_cut)
        z_mf = z1_post_meanfield(phi, m, k_cut)
        
        pre_casc = z_pre > 1
        post_casc = z_post > 1
        
        if not pre_casc and post_casc:
            status = "*** NEW CASCADE ***"
        elif pre_casc and not post_casc:
            status = "suppressed"
        elif pre_casc and post_casc:
            status = "both cascade"
        else:
            status = "both safe"
        
        print(f"{phi:6.2f} | {z_pre:8.3f} | {z_post:13.3f} | {z_mf:10.3f} | {status:>15}")
    
    print()
    print("KEY RESULT: At φ=0.22, z₁ crosses from subcritical (0.850)")
    print("to supercritical (1.195) after hub removal, explaining the")
    print("cascade explosion from 1.1% to 26.4% observed in simulations.")
    print()
    
    # Find cascade boundaries
    for label, z1_fn in [("Pre-removal", lambda phi: z1_pre(phi, m)),
                          ("Post-removal (exact)", lambda phi: z1_post_exact(phi, m, k_cut))]:
        # Binary search for boundary
        lo, hi = 0.01, 0.99
        while hi - lo > 0.001:
            mid = (lo + hi) / 2
            if z1_fn(mid) > 1:
                lo = mid
            else:
                hi = mid
        print(f"{label} cascade boundary: φ* ≈ {(lo + hi) / 2:.3f}")


if __name__ == "__main__":
    print("BA(m=2), α ≈ 0.10:")
    full_analysis(m=2, alpha_target=0.10)
    print()
    print("BA(m=4), α ≈ 0.10:")
    full_analysis(m=4, alpha_target=0.10)
