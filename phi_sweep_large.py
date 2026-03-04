#!/usr/bin/env python3
"""
φ-sweep: Validate the predictive condition for Conservation of Fragility.

Prediction: The cascade trade-off should APPEAR when φ < 1/k_hub 
(hubs become firewalls) and DISAPPEAR when φ > 1/k_hub.

For BA(m=2) with max hub degree ~√N:
- At N=2000: max degree ~90. k_firewall = 1/φ
- φ = 0.05: k_firewall = 20 → many hubs are firewalls → strong trade-off
- φ = 0.18: k_firewall = 5 → moderate (our baseline)
- φ = 0.30: k_firewall = 3 → few firewalls (most hubs too have k>3)
- φ = 0.50: k_firewall = 2 → essentially all nodes vulnerable → no trade-off

This sweep tests the CORE PREDICTION of the theory.
"""

import networkx as nx
import numpy as np
import json
import time
from multiprocessing import Pool, cpu_count

def gen_ba(N, m=2):
    return nx.barabasi_albert_graph(N, m)

def gen_ws(N, k=4, p=0.1):
    return nx.watts_strogatz_graph(N, k, p)

def estimate_pc(G, p_steps=100, n_trials=20):
    n = len(G)
    if n == 0: return 1.0
    edges = list(G.edges())
    if not edges: return 1.0
    
    for p in np.linspace(0.005, 1.0, p_steps):
        gccs = []
        for _ in range(n_trials):
            mask = np.random.random(len(edges)) < p
            S = nx.Graph()
            S.add_nodes_from(G.nodes())
            S.add_edges_from([edges[i] for i in range(len(edges)) if mask[i]])
            comps = sorted(nx.connected_components(S), key=len, reverse=True)
            gccs.append(len(comps[0]) / n if comps else 0)
        if np.mean(gccs) > 0.5:
            return float(p)
    return 1.0

def watts_cascade(G, phi=0.18, n_trials=30, seed_count=1):
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0: return 0.0, []
    
    adj = {v: list(G.neighbors(v)) for v in nodes}
    deg = {v: G.degree(v) for v in nodes}
    
    cascade_sizes = []
    for _ in range(n_trials):
        seeds = set(np.random.choice(nodes, min(seed_count, n), replace=False))
        active = set(seeds)
        changed = True
        step = 0
        while changed and step < 100:
            changed = False
            new_active = set()
            for v in nodes:
                if v in active or deg[v] == 0: continue
                n_active = sum(1 for u in adj[v] if u in active)
                if n_active / deg[v] >= phi:
                    new_active.add(v)
                    changed = True
            active |= new_active
            step += 1
        cascade_sizes.append(len(active) / n)
    
    return float(np.mean(cascade_sizes)), cascade_sizes

def remove_hubs(G, frac=0.10):
    n = len(G)
    n_remove = max(1, int(n * frac))
    deg_sorted = sorted(G.degree(), key=lambda x: (-x[1], x[0]))
    to_remove = [node for node, deg in deg_sorted[:n_remove]]
    G2 = G.copy()
    G2.remove_nodes_from(to_remove)
    return G2

def run_phi_point(args):
    topo, N, phi, run_id = args
    np.random.seed(run_id * 10000 + int(phi * 1000))
    
    if topo == "BA":
        G = gen_ba(N)
    else:
        G = gen_ws(N)
    
    # Before
    pc_before = estimate_pc(G)
    cascade_before, _ = watts_cascade(G, phi=phi)
    
    # After hub removal
    G2 = remove_hubs(G)
    pc_after = estimate_pc(G2)
    cascade_after, _ = watts_cascade(G2, phi=phi)
    
    df_p = (pc_after - pc_before) / max(pc_before, 0.001)
    df_c = (cascade_after - cascade_before) / max(cascade_before, 0.001)
    
    # Count firewall nodes
    degrees = [d for _, d in G.degree()]
    k_firewall = int(1.0 / phi) if phi > 0 else 999
    n_firewalls = sum(1 for d in degrees if d > k_firewall)
    frac_firewalls = n_firewalls / len(degrees)
    
    return {
        "topo": topo, "N": N, "phi": phi, "run_id": run_id,
        "df_p": df_p, "df_c": df_c,
        "pc_before": pc_before, "pc_after": pc_after,
        "cascade_before": cascade_before, "cascade_after": cascade_after,
        "k_firewall": k_firewall, "frac_firewalls": frac_firewalls,
    }

if __name__ == "__main__":
    N = 10000
    n_runs = 20
    phi_values = [0.05, 0.10, 0.15, 0.18, 0.22, 0.25, 0.30, 0.40, 0.50]
    topologies = ["BA", "WS"]
    n_workers = 10
    
    print(f"φ-sweep: Conservation of Fragility Validation")
    print(f"N={N}, {n_runs} runs, φ = {phi_values}")
    print(f"Workers: {n_workers}")
    print("=" * 60)
    
    all_results = {}
    start = time.time()
    
    for topo in topologies:
        for phi in phi_values:
            key = f"{topo}-phi{phi:.2f}"
            print(f"\n>>> {key}...", end=" ", flush=True)
            
            args = [(topo, N, phi, i) for i in range(n_runs)]
            with Pool(n_workers) as pool:
                results = pool.map(run_phi_point, args)
            
            df_p_vals = [r["df_p"] for r in results]
            df_c_vals = [r["df_c"] for r in results]
            frac_fw = np.mean([r["frac_firewalls"] for r in results])
            
            # Bootstrap
            def bci(data):
                data = np.array(data)
                boot = [np.mean(np.random.choice(data, len(data), True)) for _ in range(5000)]
                return {"mean": float(np.mean(data)), "std": float(np.std(data)),
                        "ci_lo": float(np.percentile(boot, 2.5)),
                        "ci_hi": float(np.percentile(boot, 97.5))}
            
            dp = bci(df_p_vals)
            dc = bci(df_c_vals)
            
            all_results[key] = {
                "topo": topo, "N": N, "phi": phi,
                "k_firewall": int(1/phi) if phi > 0 else 999,
                "frac_firewalls": float(frac_fw),
                "delta_f_percolation": dp,
                "delta_f_cascade": dc,
            }
            
            sign = "TRADE-OFF" if dc["mean"] > 0.05 else ("BOTH ↑" if dc["mean"] < -0.05 else "NEUTRAL")
            print(f"ΔF_p={dp['mean']:+.1%}, ΔF_c={dc['mean']:+.1%}, "
                  f"firewalls={frac_fw:.1%}, k_fw={int(1/phi)} → {sign}")
    
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    
    out_path = "/Users/freddycach/.openclaw/workspace/projects/conservation-of-fragility/RESULTS_PHI_SWEEP_N10000.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {out_path}")
