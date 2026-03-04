#!/usr/bin/env python3
"""
Conservation of Fragility — Production Simulation Suite
=======================================================
Addresses ALL referee concerns from council debate:
1. N ≥ 10,000 with finite-size scaling (N=500,1000,2000,5000,10000)
2. 50 independent network realizations per configuration
3. 200 p-steps for fine percolation resolution
4. Watts 2002 linear threshold cascade model (fully specified)
5. Matched ⟨k⟩ across topologies (BA m=2 → ⟨k⟩≈4, ER p=4/(N-1), WS k=4)
6. κ = ⟨k²⟩/⟨k⟩ computed for each network before and after hub removal
7. Bootstrap CIs (10,000 resamples)
8. Analytical validation: compare baseline p_c against Molloy-Reed prediction

Hub removal protocol (FULLY SPECIFIED):
- Remove top 10% of nodes by degree (simultaneous removal)
- All edges incident to removed nodes are deleted
- Remaining network may be disconnected (this is physical)
- p_c measured on the full remaining network (all components)

Author: 42 ♟️ (Federico Cachero)
Date: 2026-02-26
"""

import networkx as nx
import numpy as np
import json
import time
import sys
from multiprocessing import Pool, cpu_count
from functools import partial

# ============================================================
# NETWORK GENERATORS (matched ⟨k⟩ ≈ 4)
# ============================================================

def gen_ba(N, m=2):
    """Barabási-Albert: ⟨k⟩ = 2m = 4"""
    return nx.barabasi_albert_graph(N, m)

def gen_er(N):
    """Erdős-Rényi: ⟨k⟩ = 4"""
    p = 4.0 / (N - 1)
    return nx.erdos_renyi_graph(N, p)

def gen_ws(N, k=4, p=0.1):
    """Watts-Strogatz: ⟨k⟩ = k = 4"""
    return nx.watts_strogatz_graph(N, k, p)

# ============================================================
# DEGREE STATISTICS
# ============================================================

def compute_kappa(G):
    """
    Compute κ = ⟨k²⟩/⟨k⟩ (heterogeneity parameter).
    Also returns ⟨k⟩, ⟨k²⟩, σ²_k, and Gini coefficient.
    """
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    if len(degrees) == 0 or np.mean(degrees) == 0:
        return {"kappa": 0, "mean_k": 0, "mean_k2": 0, "var_k": 0, "gini": 0}
    
    mean_k = np.mean(degrees)
    mean_k2 = np.mean(degrees ** 2)
    var_k = np.var(degrees)
    kappa = mean_k2 / mean_k if mean_k > 0 else 0
    
    # Gini coefficient
    sorted_d = np.sort(degrees)
    n = len(sorted_d)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_d) / (n * np.sum(sorted_d))) - (n + 1) / n if np.sum(sorted_d) > 0 else 0
    
    return {
        "kappa": float(kappa),
        "mean_k": float(mean_k),
        "mean_k2": float(mean_k2),
        "var_k": float(var_k),
        "gini": float(gini),
        "N": len(degrees),
        "M": G.number_of_edges()
    }

def molloy_reed_pc(G):
    """
    Analytical prediction of bond percolation threshold
    using the Molloy-Reed criterion:
    p_c = ⟨k⟩ / (⟨k²⟩ - ⟨k⟩)
    
    Valid for locally tree-like random graphs.
    """
    stats = compute_kappa(G)
    denom = stats["mean_k2"] - stats["mean_k"]
    if denom <= 0:
        return 1.0
    return stats["mean_k"] / denom

# ============================================================
# HUB REMOVAL PROTOCOL
# ============================================================

def remove_hubs(G, frac=0.10):
    """
    Remove top `frac` fraction of nodes by degree (SIMULTANEOUS).
    
    Protocol (fully specified for reproducibility):
    1. Compute degree of all nodes in original graph
    2. Sort by degree descending (ties broken by node ID for determinism)
    3. Remove top ⌊N × frac⌋ nodes simultaneously
    4. All edges incident to removed nodes are deleted
    5. Return the resulting (possibly disconnected) subgraph
    
    Returns: (G_after, removed_nodes, n_removed)
    """
    n = len(G)
    n_remove = max(1, int(n * frac))
    
    # Sort by degree descending, then by node ID for determinism
    deg_sorted = sorted(G.degree(), key=lambda x: (-x[1], x[0]))
    to_remove = [node for node, deg in deg_sorted[:n_remove]]
    
    G2 = G.copy()
    G2.remove_nodes_from(to_remove)
    
    return G2, to_remove, n_remove

# ============================================================
# BOND PERCOLATION (fine-grained)
# ============================================================

def estimate_pc(G, p_steps=200, n_trials=30):
    """
    Estimate bond percolation threshold p_c.
    
    Method: For each p in [0.005, 1.0] (200 steps):
    1. Independently activate each edge with probability p
    2. Compute fraction of nodes in largest connected component (GCC)
    3. Average over n_trials
    
    p_c = smallest p where average GCC fraction > 0.5
    
    Also returns the full S(p) curve for finite-size scaling analysis.
    """
    n = len(G)
    if n == 0:
        return 1.0, []
    
    edges = list(G.edges())
    if not edges:
        return 1.0, []
    
    p_values = np.linspace(0.005, 1.0, p_steps)
    s_curve = []
    pc_found = 1.0
    found = False
    
    for p in p_values:
        gccs = []
        for _ in range(n_trials):
            # Activate edges with probability p
            mask = np.random.random(len(edges)) < p
            active_edges = [edges[i] for i in range(len(edges)) if mask[i]]
            
            S = nx.Graph()
            S.add_nodes_from(G.nodes())
            S.add_edges_from(active_edges)
            
            if len(S) == 0:
                gccs.append(0.0)
                continue
            
            comps = sorted(nx.connected_components(S), key=len, reverse=True)
            gccs.append(len(comps[0]) / n)
        
        mean_gcc = float(np.mean(gccs))
        s_curve.append({"p": float(p), "S": mean_gcc, "S_std": float(np.std(gccs))})
        
        if not found and mean_gcc > 0.5:
            pc_found = float(p)
            found = True
    
    return pc_found, s_curve

# ============================================================
# WATTS 2002 LINEAR THRESHOLD CASCADE MODEL
# ============================================================

def watts_cascade(G, phi=0.18, n_trials=50, seed_count=1):
    """
    Watts (2002) linear threshold cascade model.
    
    FULLY SPECIFIED:
    - Each node has threshold φ = 0.18 (uniform, deterministic)
    - Initial seed: `seed_count` random nodes activated
    - At each step: node i becomes active if
      (# active neighbors of i) / (degree of i) ≥ φ
    - Iterate synchronously until no new activations
    - Measure: final fraction of active nodes
    
    Returns: mean cascade size (fraction), std, all individual cascade sizes
    
    Reference: Watts, D.J. (2002). A simple model of global cascades 
    on random networks. PNAS 99(9), 5766-5771.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return 0.0, 0.0, []
    
    adj = {v: list(G.neighbors(v)) for v in nodes}
    deg = {v: G.degree(v) for v in nodes}
    
    cascade_sizes = []
    
    for _ in range(n_trials):
        # Random seed
        seeds = set(np.random.choice(nodes, min(seed_count, n), replace=False))
        active = set(seeds)
        
        # Synchronous update until convergence
        changed = True
        max_steps = 100  # safety
        step = 0
        while changed and step < max_steps:
            changed = False
            new_active = set()
            for v in nodes:
                if v in active:
                    continue
                if deg[v] == 0:
                    continue
                n_active_neighbors = sum(1 for u in adj[v] if u in active)
                if n_active_neighbors / deg[v] >= phi:
                    new_active.add(v)
                    changed = True
            active |= new_active
            step += 1
        
        cascade_sizes.append(len(active) / n)
    
    return float(np.mean(cascade_sizes)), float(np.std(cascade_sizes)), cascade_sizes

# ============================================================
# SINGLE REALIZATION (for multiprocessing)
# ============================================================

def run_single_realization(args):
    """
    Run one complete realization: generate network, measure everything
    before and after hub removal.
    """
    topology, N, run_id, p_steps, n_percolation_trials, n_cascade_trials, hub_frac, cascade_phi = args
    
    np.random.seed(run_id * 10000 + N)  # reproducible
    
    # Generate network
    if topology == "BA":
        G = gen_ba(N)
    elif topology == "ER":
        G = gen_er(N)
    elif topology == "WS":
        G = gen_ws(N)
    else:
        raise ValueError(f"Unknown topology: {topology}")
    
    # --- BEFORE hub removal ---
    stats_before = compute_kappa(G)
    pc_before, s_curve_before = estimate_pc(G, p_steps=p_steps, n_trials=n_percolation_trials)
    pc_analytical_before = molloy_reed_pc(G)
    cascade_mean_before, cascade_std_before, cascade_all_before = watts_cascade(
        G, phi=cascade_phi, n_trials=n_cascade_trials
    )
    
    # --- Hub removal ---
    G_after, removed, n_removed = remove_hubs(G, frac=hub_frac)
    
    # --- AFTER hub removal ---
    stats_after = compute_kappa(G_after)
    pc_after, s_curve_after = estimate_pc(G_after, p_steps=p_steps, n_trials=n_percolation_trials)
    pc_analytical_after = molloy_reed_pc(G_after)
    cascade_mean_after, cascade_std_after, cascade_all_after = watts_cascade(
        G_after, phi=cascade_phi, n_trials=n_cascade_trials
    )
    
    # Compute ΔF metrics
    delta_f_percolation = (pc_after - pc_before) / max(pc_before, 0.001)
    delta_f_cascade = (cascade_mean_after - cascade_mean_before) / max(cascade_mean_before, 0.001)
    delta_pc_absolute = pc_after - pc_before
    
    return {
        "topology": topology,
        "N": N,
        "run_id": run_id,
        "hub_removal_frac": hub_frac,
        "n_removed": n_removed,
        "cascade_phi": cascade_phi,
        # Before
        "stats_before": stats_before,
        "pc_before": pc_before,
        "pc_analytical_before": pc_analytical_before,
        "cascade_mean_before": cascade_mean_before,
        "cascade_std_before": cascade_std_before,
        # After
        "stats_after": stats_after,
        "pc_after": pc_after,
        "pc_analytical_after": pc_analytical_after,
        "cascade_mean_after": cascade_mean_after,
        "cascade_std_after": cascade_std_after,
        # Deltas
        "delta_f_percolation": delta_f_percolation,
        "delta_f_cascade": delta_f_cascade,
        "delta_pc_absolute": delta_pc_absolute,
        # S-curves (store only for first 3 runs to save space)
        "s_curve_before": s_curve_before if run_id < 3 else None,
        "s_curve_after": s_curve_after if run_id < 3 else None,
    }

# ============================================================
# MAIN SIMULATION DRIVER
# ============================================================

def run_full_simulation(
    topologies=("BA", "ER", "WS"),
    sizes=(500, 1000, 2000, 5000, 10000),
    n_runs=50,
    p_steps=200,
    n_percolation_trials=30,
    n_cascade_trials=50,
    hub_frac=0.10,
    cascade_phi=0.18,
    n_workers=None
):
    """Run the full Conservation of Fragility simulation suite."""
    
    if n_workers is None:
        n_workers = max(1, cpu_count() - 2)  # leave 2 cores free
    
    print(f"Conservation of Fragility — Production Simulation")
    print(f"=" * 60)
    print(f"Topologies: {topologies}")
    print(f"Sizes: {sizes}")
    print(f"Runs per config: {n_runs}")
    print(f"Percolation: {p_steps} p-steps, {n_percolation_trials} trials/step")
    print(f"Cascade: Watts φ={cascade_phi}, {n_cascade_trials} trials")
    print(f"Hub removal: top {hub_frac:.0%} by degree")
    print(f"Workers: {n_workers}")
    print(f"Total configurations: {len(topologies) * len(sizes) * n_runs}")
    print(f"=" * 60)
    
    all_results = {}
    start_total = time.time()
    
    for topo in topologies:
        for N in sizes:
            config_key = f"{topo}-{N}"
            print(f"\n>>> {config_key} ({n_runs} runs)...", flush=True)
            start = time.time()
            
            # Prepare arguments for multiprocessing
            args_list = [
                (topo, N, run_id, p_steps, n_percolation_trials, 
                 n_cascade_trials, hub_frac, cascade_phi)
                for run_id in range(n_runs)
            ]
            
            # Run in parallel
            with Pool(n_workers) as pool:
                results = pool.map(run_single_realization, args_list)
            
            elapsed = time.time() - start
            
            # Aggregate
            dfs_percolation = [r["delta_f_percolation"] for r in results]
            dfs_cascade = [r["delta_f_cascade"] for r in results]
            pcs_before = [r["pc_before"] for r in results]
            pcs_after = [r["pc_after"] for r in results]
            pcs_analytical = [r["pc_analytical_before"] for r in results]
            cascades_before = [r["cascade_mean_before"] for r in results]
            cascades_after = [r["cascade_mean_after"] for r in results]
            kappas_before = [r["stats_before"]["kappa"] for r in results]
            kappas_after = [r["stats_after"]["kappa"] for r in results]
            
            # Bootstrap CIs (10,000 resamples)
            def bootstrap_ci(data, n_boot=10000):
                data = np.array(data)
                boot = np.array([np.mean(np.random.choice(data, len(data), replace=True)) 
                                 for _ in range(n_boot)])
                return {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "ci_2.5": float(np.percentile(boot, 2.5)),
                    "ci_97.5": float(np.percentile(boot, 97.5)),
                    "median": float(np.median(data)),
                }
            
            summary = {
                "config": config_key,
                "topology": topo,
                "N": N,
                "n_runs": n_runs,
                "elapsed_s": elapsed,
                "percolation": {
                    "delta_f": bootstrap_ci(dfs_percolation),
                    "pc_before": bootstrap_ci(pcs_before),
                    "pc_after": bootstrap_ci(pcs_after),
                    "pc_analytical": bootstrap_ci(pcs_analytical),
                    "delta_pc_absolute": bootstrap_ci([r["delta_pc_absolute"] for r in results]),
                },
                "cascade": {
                    "delta_f": bootstrap_ci(dfs_cascade),
                    "size_before": bootstrap_ci(cascades_before),
                    "size_after": bootstrap_ci(cascades_after),
                },
                "heterogeneity": {
                    "kappa_before": bootstrap_ci(kappas_before),
                    "kappa_after": bootstrap_ci(kappas_after),
                    "gini_before": bootstrap_ci([r["stats_before"]["gini"] for r in results]),
                    "gini_after": bootstrap_ci([r["stats_after"]["gini"] for r in results]),
                },
                "raw_results": results,  # keep all for detailed analysis
            }
            
            all_results[config_key] = summary
            
            # Print summary
            df_p = summary["percolation"]["delta_f"]
            df_c = summary["cascade"]["delta_f"]
            kp = summary["heterogeneity"]["kappa_before"]
            pc_sim = summary["percolation"]["pc_before"]
            pc_ana = summary["percolation"]["pc_analytical"]
            
            print(f"    Percolation ΔF: {df_p['mean']:+.1%} [{df_p['ci_2.5']:+.1%}, {df_p['ci_97.5']:+.1%}]")
            print(f"    Cascade ΔF:     {df_c['mean']:+.1%} [{df_c['ci_2.5']:+.1%}, {df_c['ci_97.5']:+.1%}]")
            print(f"    κ before: {kp['mean']:.2f} ± {kp['std']:.2f}")
            print(f"    p_c simulated: {pc_sim['mean']:.4f} vs analytical: {pc_ana['mean']:.4f}")
            print(f"    Time: {elapsed:.1f}s")
    
    total_elapsed = time.time() - start_total
    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    
    return all_results

# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(all_results, filename="RESULTS_PRODUCTION.json"):
    """Save results, stripping raw data for the summary file."""
    # Summary (no raw_results, no s_curves)
    summary = {}
    for key, val in all_results.items():
        s = {k: v for k, v in val.items() if k != "raw_results"}
        summary[key] = s
    
    with open(filename, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {filename}")
    
    # Full results (with raw data) - separate file
    full_filename = filename.replace(".json", "_FULL.json")
    # Strip s_curves from raw to save space
    for key in all_results:
        for r in all_results[key]["raw_results"]:
            r.pop("s_curve_before", None)
            r.pop("s_curve_after", None)
    
    with open(full_filename, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved full results to {full_filename}")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Check for quick test mode
    if "--test" in sys.argv:
        print("TEST MODE: small simulations")
        results = run_full_simulation(
            sizes=(500, 1000),
            n_runs=5,
            p_steps=50,
            n_percolation_trials=10,
            n_cascade_trials=10,
            n_workers=4
        )
        save_results(results, "RESULTS_TEST.json")
    elif "--medium" in sys.argv:
        print("MEDIUM MODE: N up to 5000")
        results = run_full_simulation(
            sizes=(500, 1000, 2000, 5000),
            n_runs=30,
            p_steps=100,
            n_percolation_trials=20,
            n_cascade_trials=30,
            n_workers=8
        )
        save_results(results, "RESULTS_MEDIUM.json")
    else:
        print("PRODUCTION MODE: Full suite")
        results = run_full_simulation(
            sizes=(500, 1000, 2000, 5000, 10000),
            n_runs=50,
            p_steps=200,
            n_percolation_trials=30,
            n_cascade_trials=50,
            n_workers=10
        )
        save_results(results, "RESULTS_PRODUCTION.json")
