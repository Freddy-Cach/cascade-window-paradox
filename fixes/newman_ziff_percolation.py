#!/usr/bin/env python3
"""
Newman-Ziff percolation threshold estimation for BA networks.
Replaces the S=0.5 method with the STANDARD susceptibility-peak method.

Newman & Ziff, Phys. Rev. Lett. 85, 4104 (2000); Phys. Rev. E 64, 016706 (2001).

The susceptibility peak method finds p_c as the point where the mean cluster size
(excluding the giant component) is maximized. This is the standard method used
in computational percolation studies.

Also computes: Molloy-Reed analytical, S=0.5 (for comparison), and susceptibility peak.
"""

import numpy as np
import networkx as nx
from collections import defaultdict
import json
import time
import sys

def generate_ba_network(N, m, seed=None):
    """Generate BA network."""
    return nx.barabasi_albert_graph(N, m, seed=seed)

def remove_top_hubs(G, alpha=0.10):
    """Remove top alpha fraction of nodes by degree."""
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    n_remove = int(alpha * G.number_of_nodes())
    nodes_to_remove = [node for node, deg in degrees[:n_remove]]
    G_post = G.copy()
    G_post.remove_nodes_from(nodes_to_remove)
    return G_post

def bond_percolation_sweep(G, p_values, n_samples=50):
    """
    For each p, retain each edge with probability p.
    Return: S(p) = |GCC|/N, susceptibility chi(p), and component size distribution.
    """
    nodes = list(G.nodes())
    N = len(nodes)
    edges = list(G.edges())
    M = len(edges)
    
    results = []
    
    for p in p_values:
        gcc_sizes = []
        susceptibilities = []
        
        for _ in range(n_samples):
            # Retain each edge with probability p
            mask = np.random.random(M) < p
            retained_edges = [edges[i] for i in range(M) if mask[i]]
            
            # Build subgraph
            H = nx.Graph()
            H.add_nodes_from(nodes)
            H.add_edges_from(retained_edges)
            
            # Find components
            components = sorted(nx.connected_components(H), key=len, reverse=True)
            
            if len(components) == 0:
                gcc_sizes.append(0)
                susceptibilities.append(0)
                continue
            
            gcc_size = len(components[0])
            gcc_sizes.append(gcc_size / N)
            
            # Susceptibility = mean size of finite clusters (excluding GCC)
            # chi = sum(s^2 * n_s) / sum(s * n_s) where n_s = number of clusters of size s
            # Excluding the largest component
            if len(components) > 1:
                finite_sizes = [len(c) for c in components[1:]]
                total_finite = sum(finite_sizes)
                if total_finite > 0:
                    chi = sum(s**2 for s in finite_sizes) / total_finite
                else:
                    chi = 0
            else:
                chi = 0
            susceptibilities.append(chi)
        
        results.append({
            'p': float(p),
            'S_mean': float(np.mean(gcc_sizes)),
            'S_std': float(np.std(gcc_sizes)),
            'S_ci95_low': float(np.percentile(gcc_sizes, 2.5)),
            'S_ci95_high': float(np.percentile(gcc_sizes, 97.5)),
            'chi_mean': float(np.mean(susceptibilities)),
            'chi_std': float(np.std(susceptibilities)),
            'n_samples': n_samples
        })
    
    return results

def find_pc_susceptibility_peak(results):
    """Find p_c as the p where susceptibility chi is maximized."""
    chi_values = [r['chi_mean'] for r in results]
    p_values = [r['p'] for r in results]
    idx = np.argmax(chi_values)
    return p_values[idx], chi_values[idx]

def find_pc_S05(results):
    """Find p_c as the p where S(p) crosses 0.5 (old method, for comparison)."""
    S_values = [r['S_mean'] for r in results]
    p_values = [r['p'] for r in results]
    
    for i in range(len(S_values) - 1):
        if S_values[i] < 0.5 and S_values[i+1] >= 0.5:
            # Linear interpolation
            frac = (0.5 - S_values[i]) / (S_values[i+1] - S_values[i])
            return p_values[i] + frac * (p_values[i+1] - p_values[i])
    return None

def molloy_reed_pc(G):
    """Analytical Molloy-Reed threshold: p_c = <k> / (<k^2> - <k>)."""
    degrees = [d for _, d in G.degree()]
    k_mean = np.mean(degrees)
    k2_mean = np.mean([d**2 for d in degrees])
    if k2_mean - k_mean <= 0:
        return float('inf')
    return k_mean / (k2_mean - k_mean)

def compute_kappa(G):
    """Compute degree heterogeneity kappa = <k^2>/<k>."""
    degrees = [d for _, d in G.degree()]
    k_mean = np.mean(degrees)
    k2_mean = np.mean([d**2 for d in degrees])
    return k2_mean / k_mean

def main():
    N = 2000
    m = 2
    alpha = 0.10
    n_networks = 30  # 30 independent network realizations
    n_p_values = 200  # Fine grid for susceptibility peak
    n_percolation_samples = 50  # Per p-value per network
    
    p_values = np.linspace(0.01, 0.99, n_p_values)
    
    print(f"Newman-Ziff Percolation Study")
    print(f"BA(N={N}, m={m}), alpha={alpha}")
    print(f"{n_networks} networks × {n_p_values} p-values × {n_percolation_samples} samples")
    print(f"Total percolation trials: {n_networks * n_p_values * n_percolation_samples:,}")
    print()
    
    all_pre_results = defaultdict(lambda: {'S': [], 'chi': []})
    all_post_results = defaultdict(lambda: {'S': [], 'chi': []})
    
    pre_pc_suscept = []
    pre_pc_S05 = []
    pre_pc_MR = []
    post_pc_suscept = []
    post_pc_S05 = []
    post_pc_MR = []
    
    pre_kappas = []
    post_kappas = []
    
    t0 = time.time()
    
    for net_idx in range(n_networks):
        t_net = time.time()
        
        # Generate network
        G_pre = generate_ba_network(N, m, seed=net_idx * 1000)
        G_post = remove_top_hubs(G_pre, alpha)
        
        pre_kappas.append(compute_kappa(G_pre))
        post_kappas.append(compute_kappa(G_post))
        
        # Molloy-Reed
        pre_pc_MR.append(molloy_reed_pc(G_pre))
        post_pc_MR.append(molloy_reed_pc(G_post))
        
        # Percolation sweep — PRE-REMOVAL
        pre_results = bond_percolation_sweep(G_pre, p_values, n_percolation_samples)
        for r in pre_results:
            p = r['p']
            all_pre_results[p]['S'].append(r['S_mean'])
            all_pre_results[p]['chi'].append(r['chi_mean'])
        
        pc_s = find_pc_susceptibility_peak(pre_results)
        pc_05 = find_pc_S05(pre_results)
        pre_pc_suscept.append(pc_s[0])
        if pc_05: pre_pc_S05.append(pc_05)
        
        # Percolation sweep — POST-REMOVAL
        post_results = bond_percolation_sweep(G_post, p_values, n_percolation_samples)
        for r in post_results:
            p = r['p']
            all_post_results[p]['S'].append(r['S_mean'])
            all_post_results[p]['chi'].append(r['chi_mean'])
        
        pc_s = find_pc_susceptibility_peak(post_results)
        pc_05 = find_pc_S05(post_results)
        post_pc_suscept.append(pc_s[0])
        if pc_05: post_pc_S05.append(pc_05)
        
        elapsed = time.time() - t_net
        total_elapsed = time.time() - t0
        eta = (total_elapsed / (net_idx + 1)) * (n_networks - net_idx - 1)
        print(f"  Network {net_idx+1}/{n_networks}: {elapsed:.1f}s (ETA: {eta/60:.1f}min)")
        sys.stdout.flush()
    
    total_time = time.time() - t0
    
    # Aggregate results
    pre_pc_suscept_mean = np.mean(pre_pc_suscept)
    pre_pc_suscept_ci = (np.percentile(pre_pc_suscept, 2.5), np.percentile(pre_pc_suscept, 97.5))
    post_pc_suscept_mean = np.mean(post_pc_suscept)
    post_pc_suscept_ci = (np.percentile(post_pc_suscept, 2.5), np.percentile(post_pc_suscept, 97.5))
    
    pre_pc_S05_mean = np.mean(pre_pc_S05) if pre_pc_S05 else None
    post_pc_S05_mean = np.mean(post_pc_S05) if post_pc_S05 else None
    
    pre_pc_MR_mean = np.mean(pre_pc_MR)
    post_pc_MR_mean = np.mean(post_pc_MR)
    
    # ΔF calculations
    delta_F_suscept = (post_pc_suscept_mean - pre_pc_suscept_mean) / pre_pc_suscept_mean * 100
    delta_F_S05 = ((post_pc_S05_mean - pre_pc_S05_mean) / pre_pc_S05_mean * 100) if (pre_pc_S05_mean and post_pc_S05_mean) else None
    delta_F_MR = (post_pc_MR_mean - pre_pc_MR_mean) / pre_pc_MR_mean * 100
    
    output = {
        'parameters': {
            'N': N, 'm': m, 'alpha': alpha,
            'n_networks': n_networks,
            'n_p_values': n_p_values,
            'n_percolation_samples': n_percolation_samples,
            'total_trials': n_networks * n_p_values * n_percolation_samples
        },
        'pre_removal': {
            'kappa_mean': float(np.mean(pre_kappas)),
            'kappa_std': float(np.std(pre_kappas)),
            'pc_susceptibility_peak': {
                'mean': float(pre_pc_suscept_mean),
                'std': float(np.std(pre_pc_suscept)),
                'ci95': [float(pre_pc_suscept_ci[0]), float(pre_pc_suscept_ci[1])],
                'all_values': [float(x) for x in pre_pc_suscept]
            },
            'pc_S05': {
                'mean': float(pre_pc_S05_mean) if pre_pc_S05_mean else None,
                'std': float(np.std(pre_pc_S05)) if pre_pc_S05 else None,
                'all_values': [float(x) for x in pre_pc_S05]
            },
            'pc_molloy_reed': {
                'mean': float(pre_pc_MR_mean),
                'std': float(np.std(pre_pc_MR)),
                'all_values': [float(x) for x in pre_pc_MR]
            }
        },
        'post_removal': {
            'kappa_mean': float(np.mean(post_kappas)),
            'kappa_std': float(np.std(post_kappas)),
            'pc_susceptibility_peak': {
                'mean': float(post_pc_suscept_mean),
                'std': float(np.std(post_pc_suscept)),
                'ci95': [float(post_pc_suscept_ci[0]), float(post_pc_suscept_ci[1])],
                'all_values': [float(x) for x in post_pc_suscept]
            },
            'pc_S05': {
                'mean': float(post_pc_S05_mean) if post_pc_S05_mean else None,
                'std': float(np.std(post_pc_S05)) if post_pc_S05 else None,
                'all_values': [float(x) for x in post_pc_S05]
            },
            'pc_molloy_reed': {
                'mean': float(post_pc_MR_mean),
                'std': float(np.std(post_pc_MR)),
                'all_values': [float(x) for x in post_pc_MR]
            }
        },
        'comparison': {
            'delta_F_susceptibility_pct': float(delta_F_suscept),
            'delta_F_S05_pct': float(delta_F_S05) if delta_F_S05 else None,
            'delta_F_molloy_reed_pct': float(delta_F_MR),
            'methods_agree': abs(delta_F_suscept - (delta_F_S05 or 0)) < 30  # within 30% of each other
        },
        'runtime_seconds': float(total_time)
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"NEWMAN-ZIFF PERCOLATION RESULTS")
    print(f"{'='*60}")
    print(f"Runtime: {total_time/60:.1f} minutes")
    print(f"\nPre-removal (intact BA-2000):")
    print(f"  κ = {np.mean(pre_kappas):.2f} ± {np.std(pre_kappas):.2f}")
    print(f"  p_c (susceptibility peak) = {pre_pc_suscept_mean:.4f} ± {np.std(pre_pc_suscept):.4f}")
    print(f"  p_c (S=0.5 method)        = {pre_pc_S05_mean:.4f}" if pre_pc_S05_mean else "  p_c (S=0.5) = N/A")
    print(f"  p_c (Molloy-Reed)          = {pre_pc_MR_mean:.4f}")
    print(f"\nPost-removal (top 10% hubs removed):")
    print(f"  κ = {np.mean(post_kappas):.2f} ± {np.std(post_kappas):.2f}")
    print(f"  p_c (susceptibility peak) = {post_pc_suscept_mean:.4f} ± {np.std(post_pc_suscept):.4f}")
    print(f"  p_c (S=0.5 method)        = {post_pc_S05_mean:.4f}" if post_pc_S05_mean else "  p_c (S=0.5) = N/A")
    print(f"  p_c (Molloy-Reed)          = {post_pc_MR_mean:.4f}")
    print(f"\nΔF (percolation fragility change):")
    print(f"  Susceptibility peak: +{delta_F_suscept:.1f}%")
    if delta_F_S05: print(f"  S=0.5 method:        +{delta_F_S05:.1f}%")
    print(f"  Molloy-Reed:         +{delta_F_MR:.1f}%")
    print(f"\nMethods agreement: {'YES' if output['comparison']['methods_agree'] else 'NO'}")
    
    # Save
    outpath = '/Users/freddycach/.openclaw/workspace/projects/science-program/papers/paper-a/fixes/newman_ziff_results.json'
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")

if __name__ == '__main__':
    main()
