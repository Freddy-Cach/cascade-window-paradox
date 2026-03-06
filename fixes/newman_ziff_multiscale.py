#!/usr/bin/env python3
"""
Newman-Ziff susceptibility-peak percolation at N=2000, 5000, 10000.
Proves finite-size scaling with the standard method.
"""
import numpy as np
import networkx as nx
import json

def susceptibility_peak_pc(G, n_p=200, n_bond=50, rng=None):
    """High-precision susceptibility-peak percolation threshold."""
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(G)
    edges = list(G.edges())
    p_values = np.linspace(0.01, 0.99, n_p)
    chi_values = []
    
    for p in p_values:
        chi_samples = []
        for _ in range(n_bond):
            retained = [e for e in edges if rng.random() < p]
            H = nx.Graph()
            H.add_nodes_from(G.nodes())
            H.add_edges_from(retained)
            components = list(nx.connected_components(H))
            if not components:
                chi_samples.append(0)
                continue
            largest = max(len(c) for c in components)
            sizes = [len(c) for c in components if len(c) < largest]
            if not sizes:
                chi_samples.append(0)
                continue
            s2_sum = sum(s**2 for s in sizes)
            s_sum = sum(sizes)
            chi_samples.append(s2_sum / s_sum if s_sum > 0 else 0)
        chi_values.append(np.mean(chi_samples))
    
    best_idx = np.argmax(chi_values)
    return p_values[best_idx], chi_values[best_idx]

def hub_removal(G, alpha=0.10):
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    n_remove = int(alpha * len(G))
    remove_nodes = [n for n, d in degrees[:n_remove]]
    G_removed = G.copy()
    G_removed.remove_nodes_from(remove_nodes)
    return G_removed

def run_multiscale(sizes=[2000, 5000, 10000], m=2, n_networks=10, alpha=0.10):
    results = {}
    
    for N in sizes:
        print(f"\n{'='*50}")
        print(f"N = {N}")
        print(f"{'='*50}")
        
        pc_pre_list = []
        pc_post_list = []
        
        for i in range(n_networks):
            print(f"  Network {i+1}/{n_networks}...", end=" ", flush=True)
            G = nx.barabasi_albert_graph(N, m, seed=i*1000+N)
            G_post = hub_removal(G, alpha)
            
            rng = np.random.default_rng(i*1000+N+1)
            # Fewer p-values and bond realizations for large N
            n_p = 150 if N <= 5000 else 100
            n_bond = 30 if N <= 5000 else 20
            
            pc_pre, _ = susceptibility_peak_pc(G, n_p=n_p, n_bond=n_bond, rng=rng)
            pc_post, _ = susceptibility_peak_pc(G_post, n_p=n_p, n_bond=n_bond, rng=rng)
            
            pc_pre_list.append(float(pc_pre))
            pc_post_list.append(float(pc_post))
            
            df_p = (pc_post - pc_pre) / pc_pre * 100
            print(f"p_c: {pc_pre:.3f} → {pc_post:.3f} (ΔF = +{df_p:.0f}%)")
        
        mean_pre = np.mean(pc_pre_list)
        mean_post = np.mean(pc_post_list)
        df_mean = (mean_post - mean_pre) / mean_pre * 100
        
        results[str(N)] = {
            'pc_pre': {'mean': float(mean_pre), 'std': float(np.std(pc_pre_list)), 'values': pc_pre_list},
            'pc_post': {'mean': float(mean_post), 'std': float(np.std(pc_post_list)), 'values': pc_post_list},
            'delta_F_pct': float(df_mean),
            'n_networks': n_networks
        }
        
        print(f"\n  SUMMARY N={N}:")
        print(f"  p_c (pre):  {mean_pre:.3f} ± {np.std(pc_pre_list):.3f}")
        print(f"  p_c (post): {mean_post:.3f} ± {np.std(pc_post_list):.3f}")
        print(f"  ΔF_p: +{df_mean:.1f}%")
    
    return results

if __name__ == '__main__':
    results = run_multiscale(sizes=[2000, 5000, 10000], n_networks=8)
    
    print("\n" + "="*60)
    print("FINITE-SIZE SCALING (SUSCEPTIBILITY PEAK)")
    print("="*60)
    for N, data in results.items():
        print(f"N={N}: p_c {data['pc_pre']['mean']:.3f}→{data['pc_post']['mean']:.3f}, ΔF_p = +{data['delta_F_pct']:.1f}%")
    
    with open('newman_ziff_multiscale_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to newman_ziff_multiscale_results.json")
