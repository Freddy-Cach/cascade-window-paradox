#!/usr/bin/env python3
"""
Heterogeneous threshold sensitivity analysis for Paper A.
Tests whether the compounding vulnerability effect persists when nodes have
heterogeneous cascade thresholds drawn from Beta distributions centered on φ.

Tests three distributions:
1. Uniform φ (baseline, as in paper)
2. Beta(α=5, β=5) centered on φ (moderate heterogeneity, std ≈ 0.15φ)
3. Beta(α=2, β=2) centered on φ (high heterogeneity, std ≈ 0.22φ)
"""

import numpy as np
import networkx as nx
import json
import time
import sys

def generate_ba_network(N, m, seed=None):
    return nx.barabasi_albert_graph(N, m, seed=seed)

def remove_top_hubs(G, alpha=0.10):
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    n_remove = int(alpha * G.number_of_nodes())
    nodes_to_remove = [node for node, deg in degrees[:n_remove]]
    G_post = G.copy()
    G_post.remove_nodes_from(nodes_to_remove)
    return G_post

def watts_cascade(G, phi_dict, seed_node, max_steps=150):
    """
    Watts cascade with per-node thresholds.
    phi_dict: {node: threshold} mapping.
    """
    active = {seed_node}
    newly_active = {seed_node}
    nodes = set(G.nodes())
    
    for step in range(max_steps):
        next_active = set()
        for node in nodes - active:
            neighbors = set(G.neighbors(node))
            if len(neighbors) == 0:
                continue
            active_neighbors = len(neighbors & active)
            fraction = active_neighbors / len(neighbors)
            if fraction >= phi_dict.get(node, 1.0):
                next_active.add(node)
        
        if not next_active:
            break
        active |= next_active
        newly_active = next_active
    
    return len(active) / len(nodes)

def assign_thresholds(G, phi_mean, distribution='uniform', rng=None):
    """
    Assign per-node thresholds.
    
    distribution:
    - 'uniform': all nodes get phi_mean (paper baseline)
    - 'beta_low': Beta distribution, moderate heterogeneity (std ≈ 0.15*phi)
    - 'beta_high': Beta distribution, high heterogeneity (std ≈ 0.22*phi)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    nodes = list(G.nodes())
    
    if distribution == 'uniform':
        return {n: phi_mean for n in nodes}
    
    elif distribution == 'beta_low':
        # Beta(5,5) has mean=0.5, std≈0.15
        # Scale to have mean=phi_mean
        a, b = 5, 5
        samples = rng.beta(a, b, size=len(nodes))
        # Rescale: mean of Beta(5,5) is 0.5, we want phi_mean
        # Use affine transform: threshold = phi_mean + (sample - 0.5) * 2 * phi_mean * 0.3
        # This gives std ≈ 0.15 * phi_mean
        thresholds = phi_mean + (samples - 0.5) * 2 * phi_mean * 0.3
        thresholds = np.clip(thresholds, 0.01, 0.99)
        return {n: float(t) for n, t in zip(nodes, thresholds)}
    
    elif distribution == 'beta_high':
        # Higher heterogeneity
        a, b = 2, 2
        samples = rng.beta(a, b, size=len(nodes))
        thresholds = phi_mean + (samples - 0.5) * 2 * phi_mean * 0.5
        thresholds = np.clip(thresholds, 0.01, 0.99)
        return {n: float(t) for n, t in zip(nodes, thresholds)}
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

def run_sweep(N=2000, m=2, alpha=0.10, n_networks=15, n_seeds=40, phi_values=None):
    """Run complete φ-sweep for all distributions, pre and post removal."""
    
    if phi_values is None:
        phi_values = [0.05, 0.10, 0.12, 0.14, 0.15, 0.16, 0.17, 0.18, 
                      0.19, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.35, 0.40, 0.50]
    
    distributions = ['uniform', 'beta_low', 'beta_high']
    conditions = ['pre_removal', 'post_removal']
    
    results = {dist: {cond: {phi: [] for phi in phi_values} 
                      for cond in conditions} 
               for dist in distributions}
    
    t0 = time.time()
    
    for net_idx in range(n_networks):
        t_net = time.time()
        
        G_pre = generate_ba_network(N, m, seed=net_idx * 1000)
        G_post = remove_top_hubs(G_pre, alpha)
        
        rng = np.random.default_rng(seed=net_idx * 1000 + 42)
        
        for phi in phi_values:
            for dist in distributions:
                for cond, G in [('pre_removal', G_pre), ('post_removal', G_post)]:
                    phi_dict = assign_thresholds(G, phi, dist, rng)
                    nodes = list(G.nodes())
                    
                    for seed_idx in range(n_seeds):
                        seed_node = nodes[rng.integers(len(nodes))]
                        cascade_size = watts_cascade(G, phi_dict, seed_node)
                        results[dist][cond][phi].append(cascade_size)
        
        elapsed = time.time() - t_net
        total_elapsed = time.time() - t0
        eta = (total_elapsed / (net_idx + 1)) * (n_networks - net_idx - 1)
        print(f"  Network {net_idx+1}/{n_networks}: {elapsed:.1f}s (ETA: {eta/60:.1f}min)")
        sys.stdout.flush()
    
    total_time = time.time() - t0
    
    # Compile output
    output = {
        'parameters': {
            'N': N, 'm': m, 'alpha': alpha,
            'n_networks': n_networks, 'n_seeds': n_seeds,
            'phi_values': phi_values,
            'distributions': distributions,
            'total_cascades': n_networks * n_seeds * len(phi_values) * len(distributions) * 2
        },
        'results': {},
        'summary': {},
        'runtime_seconds': total_time
    }
    
    for dist in distributions:
        output['results'][dist] = {}
        output['summary'][dist] = {}
        for phi in phi_values:
            phi_key = f"{phi:.2f}"
            pre_vals = results[dist]['pre_removal'][phi]
            post_vals = results[dist]['post_removal'][phi]
            
            pre_mean = np.mean(pre_vals)
            post_mean = np.mean(post_vals)
            
            amplification = ((post_mean - pre_mean) / max(pre_mean, 1e-6)) * 100
            
            output['results'][dist][phi_key] = {
                'pre_mean': float(pre_mean),
                'pre_std': float(np.std(pre_vals)),
                'pre_ci95': [float(np.percentile(pre_vals, 2.5)), float(np.percentile(pre_vals, 97.5))],
                'post_mean': float(post_mean),
                'post_std': float(np.std(post_vals)),
                'post_ci95': [float(np.percentile(post_vals, 2.5)), float(np.percentile(post_vals, 97.5))],
                'amplification_pct': float(amplification),
                'n_trials': len(pre_vals)
            }
        
        # Summary: find transition point and max amplification
        phi_amps = [(phi, output['results'][dist][f"{phi:.2f}"]['amplification_pct']) for phi in phi_values]
        max_amp_phi, max_amp = max(phi_amps, key=lambda x: x[1])
        
        # Find φ* (first φ where amplification > 100%)
        phi_star = None
        for phi, amp in phi_amps:
            if amp > 100:
                phi_star = phi
                break
        
        output['summary'][dist] = {
            'max_amplification_pct': float(max_amp),
            'max_amplification_phi': float(max_amp_phi),
            'phi_star': float(phi_star) if phi_star else None,
            'effect_persists': max_amp > 100  # compounding vulnerability still exists
        }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"HETEROGENEOUS THRESHOLD SENSITIVITY ANALYSIS")
    print(f"{'='*60}")
    print(f"Runtime: {total_time/60:.1f} minutes")
    
    for dist in distributions:
        s = output['summary'][dist]
        print(f"\n{dist.upper()}:")
        print(f"  Max amplification: +{s['max_amplification_pct']:.0f}% at φ={s['max_amplification_phi']:.2f}")
        print(f"  φ* (transition): {s['phi_star']}")
        print(f"  Effect persists: {s['effect_persists']}")
    
    # Key comparison
    uniform_max = output['summary']['uniform']['max_amplification_pct']
    beta_low_max = output['summary']['beta_low']['max_amplification_pct']
    beta_high_max = output['summary']['beta_high']['max_amplification_pct']
    
    print(f"\n{'='*60}")
    print(f"CRITICAL VERDICT:")
    print(f"  Uniform:        +{uniform_max:.0f}%")
    print(f"  Beta (low het):  +{beta_low_max:.0f}%")
    print(f"  Beta (high het): +{beta_high_max:.0f}%")
    
    if beta_low_max > 100 and beta_high_max > 100:
        print(f"  ✅ COMPOUNDING VULNERABILITY PERSISTS under heterogeneous thresholds")
    else:
        print(f"  ⚠️ EFFECT MAY NOT PERSIST — investigate further")
    
    outpath = '/Users/freddycach/.openclaw/workspace/projects/science-program/papers/paper-a/fixes/heterogeneous_threshold_results.json'
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")

if __name__ == '__main__':
    run_sweep()
