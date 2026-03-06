#!/usr/bin/env python3
"""
High-precision cascade sweep with 1000+ trials per (φ, condition).
Addresses P1-STAT-04: "200 trials marginal for phase transition characterization."

Uses unified protocol:
- 50 networks × 20 seeds = 1000 total trials per φ
- Reports mean, SD, CI, and number of global cascades
- Both pre-removal and post-removal
"""

import numpy as np
import networkx as nx
import json
import time
import sys

def generate_ba(N, m, seed):
    return nx.barabasi_albert_graph(N, m, seed=seed)

def remove_hubs(G, alpha=0.10):
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    n_remove = int(alpha * G.number_of_nodes())
    G2 = G.copy()
    G2.remove_nodes_from([n for n, d in degrees[:n_remove]])
    return G2

def watts_cascade(G, phi, seed_node, max_steps=150):
    active = {seed_node}
    nodes = set(G.nodes())
    for _ in range(max_steps):
        new = set()
        for n in nodes - active:
            nbrs = set(G.neighbors(n))
            if nbrs and len(nbrs & active) / len(nbrs) >= phi:
                new.add(n)
        if not new:
            break
        active |= new
    return len(active) / len(nodes)

def main():
    N = 2000
    m = 2
    alpha = 0.10
    n_networks = 50
    n_seeds = 20
    total_per_phi = n_networks * n_seeds  # 1000
    
    phi_values = [0.05, 0.10, 0.12, 0.14, 0.15, 0.16, 0.17, 0.18, 
                  0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.26, 0.28, 0.30, 0.35, 0.40, 0.50]
    
    print(f"HIGH-PRECISION CASCADE SWEEP")
    print(f"BA(N={N}, m={m}), α={alpha}")
    print(f"{n_networks} networks × {n_seeds} seeds = {total_per_phi} trials per φ")
    print(f"{len(phi_values)} φ values, 2 conditions = {total_per_phi * len(phi_values) * 2:,} total cascades")
    print()
    
    results_pre = {f"{phi:.2f}": [] for phi in phi_values}
    results_post = {f"{phi:.2f}": [] for phi in phi_values}
    
    t0 = time.time()
    
    for net_idx in range(n_networks):
        t_net = time.time()
        
        G_pre = generate_ba(N, m, seed=net_idx)
        G_post = remove_hubs(G_pre, alpha)
        
        rng = np.random.default_rng(seed=net_idx * 1000)
        nodes_pre = list(G_pre.nodes())
        nodes_post = list(G_post.nodes())
        
        for phi in phi_values:
            phi_key = f"{phi:.2f}"
            for _ in range(n_seeds):
                # Pre-removal
                seed_pre = nodes_pre[rng.integers(len(nodes_pre))]
                results_pre[phi_key].append(watts_cascade(G_pre, phi, seed_pre))
                
                # Post-removal
                seed_post = nodes_post[rng.integers(len(nodes_post))]
                results_post[phi_key].append(watts_cascade(G_post, phi, seed_post))
        
        elapsed = time.time() - t_net
        total_elapsed = time.time() - t0
        eta = (total_elapsed / (net_idx + 1)) * (n_networks - net_idx - 1)
        print(f"  Net {net_idx+1}/{n_networks}: {elapsed:.1f}s (ETA: {eta/60:.1f}min)")
        sys.stdout.flush()
    
    total_time = time.time() - t0
    
    # Compile
    output = {
        'parameters': {
            'N': N, 'm': m, 'alpha': alpha,
            'n_networks': n_networks, 'n_seeds': n_seeds,
            'total_per_phi': total_per_phi,
            'phi_values': phi_values
        },
        'results': {},
        'runtime_seconds': total_time
    }
    
    print(f"\n{'='*70}")
    print(f"HIGH-PRECISION CASCADE RESULTS ({total_per_phi} trials per φ)")
    print(f"{'='*70}")
    print(f"{'φ':>6} {'Pre-mean':>10} {'Post-mean':>10} {'Amp(%)':>10} {'Pre-CI95':>20} {'Post-CI95':>20} {'#Global':>8}")
    print("-" * 90)
    
    for phi in phi_values:
        phi_key = f"{phi:.2f}"
        pre = np.array(results_pre[phi_key])
        post = np.array(results_post[phi_key])
        
        pre_mean = float(np.mean(pre))
        post_mean = float(np.mean(post))
        amp = ((post_mean - pre_mean) / max(pre_mean, 1e-6)) * 100
        
        pre_se = float(np.std(pre) / np.sqrt(len(pre)))
        post_se = float(np.std(post) / np.sqrt(len(post)))
        
        entry = {
            'pre_mean': pre_mean,
            'pre_std': float(np.std(pre)),
            'pre_se': pre_se,
            'pre_ci95': [float(pre_mean - 1.96*pre_se), float(pre_mean + 1.96*pre_se)],
            'post_mean': post_mean,
            'post_std': float(np.std(post)),
            'post_se': post_se,
            'post_ci95': [float(post_mean - 1.96*post_se), float(post_mean + 1.96*post_se)],
            'amplification_pct': amp,
            'n_global_pre': int(sum(1 for c in pre if c > 0.10)),
            'n_global_post': int(sum(1 for c in post if c > 0.10)),
            'n_trials': total_per_phi
        }
        output['results'][phi_key] = entry
        
        print(f"{phi:>6.2f} {pre_mean*100:>9.2f}% {post_mean*100:>9.2f}% {amp:>+9.0f}% "
              f"[{entry['pre_ci95'][0]*100:.2f},{entry['pre_ci95'][1]*100:.2f}] "
              f"[{entry['post_ci95'][0]*100:.2f},{entry['post_ci95'][1]*100:.2f}] "
              f"{entry['n_global_post']:>6}/{total_per_phi}")
    
    print(f"\nRuntime: {total_time/60:.1f} minutes")
    
    # Key headline numbers for abstract
    phi_022 = output['results']['0.22']
    print(f"\n{'='*60}")
    print(f"HEADLINE NUMBERS (φ=0.22, {total_per_phi} trials):")
    print(f"  Pre:  {phi_022['pre_mean']*100:.2f}% ± {phi_022['pre_se']*100:.3f}% (SE)")
    print(f"  Post: {phi_022['post_mean']*100:.2f}% ± {phi_022['post_se']*100:.3f}% (SE)")
    print(f"  CI95 Pre:  [{phi_022['pre_ci95'][0]*100:.2f}%, {phi_022['pre_ci95'][1]*100:.2f}%]")
    print(f"  CI95 Post: [{phi_022['post_ci95'][0]*100:.2f}%, {phi_022['post_ci95'][1]*100:.2f}%]")
    print(f"  Amplification: +{phi_022['amplification_pct']:.0f}%")
    print(f"  Global cascades: {phi_022['n_global_post']}/{total_per_phi} ({phi_022['n_global_post']/total_per_phi*100:.1f}%)")
    
    outpath = '/Users/freddycach/.openclaw/workspace/projects/science-program/papers/paper-a/fixes/high_precision_cascade_results.json'
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outpath}")

if __name__ == '__main__':
    main()
