#!/usr/bin/env python3
"""
Quick Verification Script — The Dual Resilience Paradox
=======================================================
Runs a fast (~2 min) verification that reproduces the core findings
at reduced resolution. For full production results, run simulate_production.py.

Usage: python verify_quick.py
"""

import simulate_production as sp
import numpy as np
import json
import time

QUICK_CONFIG = {
    "N": 1000,
    "n_realizations": 5,
    "p_steps": 50,       # Production: 200
    "n_trials_pc": 20,   # Production: 30
    "n_trials_casc": 30, # Production: 50
    "hub_frac": 0.10,
    "phi_values": [0.18, 0.20, 0.22, 0.25],
}

def main():
    print("=" * 60)
    print("QUICK VERIFICATION — The Dual Resilience Paradox")
    print("=" * 60)
    t0 = time.time()
    
    cfg = QUICK_CONFIG
    topologies = {
        "BA": sp.gen_ba,
        "ER": sp.gen_er,
        "WS": sp.gen_ws,
    }
    
    results = {}
    
    for topo_name, gen_fn in topologies.items():
        print(f"\n--- {topo_name}-{cfg['N']} ---")
        pcs_before, pcs_after, cascades_before, cascades_after = [], [], {}, {}
        
        for phi in cfg["phi_values"]:
            cascades_before[phi] = []
            cascades_after[phi] = []
        
        for r in range(cfg["n_realizations"]):
            G = gen_fn(cfg["N"])
            kd = sp.compute_kappa(G)
            
            # Baseline
            pc, _ = sp.estimate_pc(G, p_steps=cfg["p_steps"], n_trials=cfg["n_trials_pc"])
            pcs_before.append(pc)
            
            for phi in cfg["phi_values"]:
                casc_result = sp.watts_cascade(G, phi=phi, n_trials=cfg["n_trials_casc"])
                if isinstance(casc_result, tuple):
                    casc_sizes = casc_result[0] if isinstance(casc_result[0], (list, np.ndarray)) else [casc_result[0]]
                else:
                    casc_sizes = casc_result if isinstance(casc_result, (list, np.ndarray)) else [casc_result]
                cascades_before[phi].append(np.mean(casc_sizes))
            
            # After hub removal
            hub_result = sp.remove_hubs(G.copy(), frac=cfg["hub_frac"])
            G2 = hub_result[0]
            
            pc2, _ = sp.estimate_pc(G2, p_steps=cfg["p_steps"], n_trials=cfg["n_trials_pc"])
            pcs_after.append(pc2)
            
            for phi in cfg["phi_values"]:
                casc_result = sp.watts_cascade(G2, phi=phi, n_trials=cfg["n_trials_casc"])
                if isinstance(casc_result, tuple):
                    casc_sizes = casc_result[0] if isinstance(casc_result[0], (list, np.ndarray)) else [casc_result[0]]
                else:
                    casc_sizes = casc_result if isinstance(casc_result, (list, np.ndarray)) else [casc_result]
                cascades_after[phi].append(np.mean(casc_sizes))
            
            print(f"  Realization {r+1}: p_c={pc:.3f}→{pc2:.3f}, ΔF={((pc2-pc)/pc)*100:.0f}%")
        
        pc_mean_b = np.mean(pcs_before)
        pc_mean_a = np.mean(pcs_after)
        delta_f = (pc_mean_a - pc_mean_b) / pc_mean_b * 100
        
        results[topo_name] = {
            "p_c_before": float(pc_mean_b),
            "p_c_after": float(pc_mean_a),
            "delta_F_pct": float(delta_f),
            "kappa": float(kd["kappa"]),
        }
        
        print(f"\n  RESULT: p_c = {pc_mean_b:.3f} → {pc_mean_a:.3f}")
        print(f"  ΔF = {delta_f:.0f}%")
        print(f"  κ = {kd['kappa']:.1f}")
        
        for phi in cfg["phi_values"]:
            cb = np.mean(cascades_before[phi])
            ca = np.mean(cascades_after[phi])
            print(f"  Cascade φ={phi}: {cb:.4f} → {ca:.4f}")
    
    elapsed = time.time() - t0
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Runtime: {elapsed:.0f}s")
    
    # Check core claims
    checks = []
    
    ba = results["BA"]
    # Claim 1: BA ΔF_p > +100% (production: +167%)
    c1 = ba["delta_F_pct"] > 100
    checks.append(("BA ΔF_p > 100%", c1, f"{ba['delta_F_pct']:.0f}%"))
    
    # Claim 2: BA κ > 8
    c2 = ba["kappa"] > 8
    checks.append(("BA κ > 8", c2, f"{ba['kappa']:.1f}"))
    
    # Claim 3: BA p_c increases after hub removal
    c3 = ba["p_c_after"] > ba["p_c_before"]
    checks.append(("BA p_c increases", c3, f"{ba['p_c_before']:.3f} → {ba['p_c_after']:.3f}"))
    
    # Claim 4: ER ΔF < BA ΔF (heterogeneity dependence)
    er = results["ER"]
    c4 = er["delta_F_pct"] < ba["delta_F_pct"]
    checks.append(("ER ΔF < BA ΔF", c4, f"ER={er['delta_F_pct']:.0f}% < BA={ba['delta_F_pct']:.0f}%"))
    
    all_pass = True
    for name, passed, detail in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name} ({detail})")
        if not passed:
            all_pass = False
    
    print(f"\n{'✅ ALL CHECKS PASSED' if all_pass else '❌ SOME CHECKS FAILED'}")
    print(f"Note: Quick mode uses {cfg['p_steps']} p-steps (production: 200).")
    print(f"Production values will be more precise (ΔF ≈ +167% for BA).")

if __name__ == "__main__":
    main()
