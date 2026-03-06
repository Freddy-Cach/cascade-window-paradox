#!/usr/bin/env python3
"""
Additional real-world network validation for Paper A.
Tests compounding vulnerability on multiple real networks beyond Bitcoin OTC.

Networks tested:
1. Email-Enron (email communication, N≈36,692, scale-free)
2. CA-HepPh (co-authorship in High-Energy Physics, N≈12,008)
3. p2p-Gnutella (peer-to-peer file sharing, N≈10,876)
4. Wiki-Vote (Wikipedia adminship votes, N≈7,115)

All from Stanford SNAP (https://snap.stanford.edu/data/).
"""

import numpy as np
import networkx as nx
import json
import time
import sys
import urllib.request
import gzip
import os
import io

SNAP_URLS = {
    'email-enron': 'https://snap.stanford.edu/data/email-Enron.txt.gz',
    'ca-hepph': 'https://snap.stanford.edu/data/ca-HepPh.txt.gz',
    'p2p-gnutella': 'https://snap.stanford.edu/data/p2p-Gnutella08.txt.gz',
    'wiki-vote': 'https://snap.stanford.edu/data/wiki-Vote.txt.gz',
}

DATA_DIR = '/Users/freddycach/.openclaw/workspace/projects/science-program/papers/paper-a/fixes/data'

def download_snap_network(name, url):
    """Download and parse a SNAP edge list."""
    os.makedirs(DATA_DIR, exist_ok=True)
    local_path = os.path.join(DATA_DIR, f'{name}.txt.gz')
    
    if not os.path.exists(local_path):
        print(f"  Downloading {name} from SNAP...")
        urllib.request.urlretrieve(url, local_path)
    
    # Parse edge list
    G = nx.Graph()
    with gzip.open(local_path, 'rt', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    u, v = int(parts[0]), int(parts[1])
                    if u != v:  # no self-loops
                        G.add_edge(u, v)
                except ValueError:
                    continue
    
    # Take largest connected component
    if not nx.is_connected(G):
        gcc_nodes = max(nx.connected_components(G), key=len)
        G = G.subgraph(gcc_nodes).copy()
    
    return G

def compute_network_stats(G, name):
    """Compute key network statistics."""
    degrees = [d for _, d in G.degree()]
    k_mean = np.mean(degrees)
    k2_mean = np.mean([d**2 for d in degrees])
    kappa = k2_mean / k_mean
    k_max = max(degrees)
    
    # Estimate power-law exponent via MLE on tail
    # Using Clauset et al. (2009) method
    k_min_candidates = sorted(set(degrees))
    k_min = max(2, int(np.percentile(degrees, 50)))  # rough estimate
    tail = [d for d in degrees if d >= k_min]
    if len(tail) > 10:
        gamma = 1 + len(tail) / sum(np.log(np.array(tail) / (k_min - 0.5)))
    else:
        gamma = None
    
    return {
        'name': name,
        'N': G.number_of_nodes(),
        'M': G.number_of_edges(),
        'k_mean': float(k_mean),
        'k_max': int(k_max),
        'k2_mean': float(k2_mean),
        'kappa': float(kappa),
        'gamma_mle': float(gamma) if gamma else None,
        'clustering': float(nx.average_clustering(G)),
    }

def remove_top_hubs(G, alpha=0.10):
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    n_remove = int(alpha * G.number_of_nodes())
    nodes_to_remove = [node for node, deg in degrees[:n_remove]]
    G_post = G.copy()
    G_post.remove_nodes_from(nodes_to_remove)
    return G_post

def bond_percolation_pc(G, n_p=100, n_samples=30):
    """Estimate p_c using susceptibility peak method."""
    nodes = list(G.nodes())
    N = len(nodes)
    edges = list(G.edges())
    M = len(edges)
    p_values = np.linspace(0.01, 0.99, n_p)
    
    chi_curve = []
    S_curve = []
    
    for p in p_values:
        chis = []
        Ss = []
        for _ in range(n_samples):
            mask = np.random.random(M) < p
            H = nx.Graph()
            H.add_nodes_from(nodes)
            H.add_edges_from([edges[i] for i in range(M) if mask[i]])
            
            components = sorted(nx.connected_components(H), key=len, reverse=True)
            gcc_size = len(components[0]) if components else 0
            Ss.append(gcc_size / N)
            
            if len(components) > 1:
                finite_sizes = [len(c) for c in components[1:]]
                total = sum(finite_sizes)
                chi = sum(s**2 for s in finite_sizes) / max(total, 1)
            else:
                chi = 0
            chis.append(chi)
        
        chi_curve.append(np.mean(chis))
        S_curve.append(np.mean(Ss))
    
    # p_c = argmax(chi)
    pc_idx = np.argmax(chi_curve)
    pc = p_values[pc_idx]
    
    # Also S=0.5 for comparison
    pc_S05 = None
    for i in range(len(S_curve) - 1):
        if S_curve[i] < 0.5 and S_curve[i+1] >= 0.5:
            frac = (0.5 - S_curve[i]) / (S_curve[i+1] - S_curve[i])
            pc_S05 = p_values[i] + frac * (p_values[i+1] - p_values[i])
            break
    
    return pc, pc_S05, p_values.tolist(), chi_curve, S_curve

def watts_cascade_uniform(G, phi, seed_node, max_steps=150):
    """Standard Watts cascade with uniform threshold."""
    active = {seed_node}
    nodes = set(G.nodes())
    
    for step in range(max_steps):
        next_active = set()
        for node in nodes - active:
            neighbors = set(G.neighbors(node))
            if len(neighbors) == 0:
                continue
            if len(neighbors & active) / len(neighbors) >= phi:
                next_active.add(node)
        if not next_active:
            break
        active |= next_active
    
    return len(active) / len(nodes)

def cascade_sweep(G, phi_values, n_seeds=40, rng=None):
    """Run cascade sweep across φ values."""
    if rng is None:
        rng = np.random.default_rng()
    
    nodes = list(G.nodes())
    results = {}
    
    for phi in phi_values:
        cascades = []
        for _ in range(n_seeds):
            seed = nodes[rng.integers(len(nodes))]
            c = watts_cascade_uniform(G, phi, seed)
            cascades.append(c)
        results[f"{phi:.2f}"] = {
            'mean': float(np.mean(cascades)),
            'std': float(np.std(cascades)),
            'ci95': [float(np.percentile(cascades, 2.5)), float(np.percentile(cascades, 97.5))],
            'n_global': int(sum(1 for c in cascades if c > 0.1)),
            'n_trials': len(cascades)
        }
    
    return results

def analyze_network(name, url):
    """Full analysis pipeline for one network."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {name}")
    print(f"{'='*60}")
    
    t0 = time.time()
    
    # Download/load
    G = download_snap_network(name, url)
    stats = compute_network_stats(G, name)
    print(f"  N={stats['N']}, M={stats['M']}, <k>={stats['k_mean']:.1f}, κ={stats['kappa']:.1f}, C={stats['clustering']:.3f}")
    
    # Skip if too large (> 50k nodes → percolation takes too long)
    if stats['N'] > 50000:
        print(f"  WARNING: Large network (N={stats['N']}). Using subsampled percolation.")
        n_p = 50
        n_perc_samples = 10
    else:
        n_p = 100
        n_perc_samples = 30
    
    # Hub removal
    G_post = remove_top_hubs(G, alpha=0.10)
    stats_post = compute_network_stats(G_post, f"{name}_post")
    print(f"  Post-removal: N={stats_post['N']}, <k>={stats_post['k_mean']:.1f}, κ={stats_post['kappa']:.1f}")
    
    # Percolation
    print(f"  Running percolation (susceptibility peak method)...")
    pc_pre, pc_pre_S05, _, chi_pre, S_pre = bond_percolation_pc(G, n_p=n_p, n_samples=n_perc_samples)
    pc_post, pc_post_S05, _, chi_post, S_post = bond_percolation_pc(G_post, n_p=n_p, n_samples=n_perc_samples)
    
    delta_F_p = (pc_post - pc_pre) / max(pc_pre, 0.001) * 100
    print(f"  p_c (suscept): {pc_pre:.3f} → {pc_post:.3f} (ΔF = +{delta_F_p:.1f}%)")
    if pc_pre_S05 and pc_post_S05:
        delta_F_p_S05 = (pc_post_S05 - pc_pre_S05) / max(pc_pre_S05, 0.001) * 100
        print(f"  p_c (S=0.5):   {pc_pre_S05:.3f} → {pc_post_S05:.3f} (ΔF = +{delta_F_p_S05:.1f}%)")
    
    # Cascade sweep
    phi_values = [0.05, 0.10, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.35, 0.40]
    rng = np.random.default_rng(seed=42)
    
    print(f"  Running cascade sweep ({len(phi_values)} φ values × 40 seeds)...")
    cascade_pre = cascade_sweep(G, phi_values, n_seeds=40, rng=rng)
    cascade_post = cascade_sweep(G_post, phi_values, n_seeds=40, rng=rng)
    
    # Find max amplification
    max_amp = 0
    max_amp_phi = None
    cascade_comparison = {}
    for phi in phi_values:
        phi_key = f"{phi:.2f}"
        pre_c = cascade_pre[phi_key]['mean']
        post_c = cascade_post[phi_key]['mean']
        amp = ((post_c - pre_c) / max(pre_c, 1e-6)) * 100
        cascade_comparison[phi_key] = {
            'pre': pre_c, 'post': post_c, 'amplification_pct': amp
        }
        if amp > max_amp:
            max_amp = amp
            max_amp_phi = phi
    
    elapsed = time.time() - t0
    
    print(f"  Max cascade amplification: +{max_amp:.0f}% at φ={max_amp_phi}")
    print(f"  Runtime: {elapsed:.0f}s")
    
    # Compile φ=0.22 headline numbers for comparison with paper
    phi_022_pre = cascade_pre.get('0.22', {}).get('mean', 0)
    phi_022_post = cascade_post.get('0.22', {}).get('mean', 0)
    phi_022_amp = ((phi_022_post - phi_022_pre) / max(phi_022_pre, 1e-6)) * 100
    
    return {
        'network': name,
        'stats_pre': stats,
        'stats_post': stats_post,
        'percolation': {
            'pc_pre_suscept': float(pc_pre),
            'pc_post_suscept': float(pc_post),
            'pc_pre_S05': float(pc_pre_S05) if pc_pre_S05 else None,
            'pc_post_S05': float(pc_post_S05) if pc_post_S05 else None,
            'delta_F_suscept_pct': float(delta_F_p),
        },
        'cascade_pre': cascade_pre,
        'cascade_post': cascade_post,
        'cascade_comparison': cascade_comparison,
        'headline': {
            'phi_022_pre': float(phi_022_pre),
            'phi_022_post': float(phi_022_post),
            'phi_022_amplification_pct': float(phi_022_amp),
            'max_amplification_pct': float(max_amp),
            'max_amplification_phi': float(max_amp_phi) if max_amp_phi else None,
            'compounding_vulnerability_confirmed': max_amp > 100 and delta_F_p > 10
        },
        'runtime_seconds': float(elapsed)
    }

def main():
    print("ADDITIONAL REAL-WORLD NETWORK VALIDATION")
    print(f"Testing {len(SNAP_URLS)} networks from Stanford SNAP\n")
    
    all_results = {}
    t0 = time.time()
    
    for name, url in SNAP_URLS.items():
        try:
            result = analyze_network(name, url)
            all_results[name] = result
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[name] = {'error': str(e)}
    
    total_time = time.time() - t0
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — ALL NETWORKS")
    print(f"{'='*60}")
    print(f"{'Network':<20} {'N':>8} {'κ':>8} {'Δp_c':>10} {'MaxAmp':>10} {'Confirmed':>10}")
    print("-" * 70)
    
    for name, r in all_results.items():
        if 'error' in r:
            print(f"{name:<20} ERROR: {r['error']}")
            continue
        h = r['headline']
        s = r['stats_pre']
        print(f"{name:<20} {s['N']:>8} {s['kappa']:>8.1f} {r['percolation']['delta_F_suscept_pct']:>+9.1f}% {h['max_amplification_pct']:>+9.0f}% {'✅' if h['compounding_vulnerability_confirmed'] else '❌':>10}")
    
    print(f"\nTotal runtime: {total_time/60:.1f} minutes")
    
    output = {
        'networks': all_results,
        'total_runtime_seconds': total_time,
        'n_networks_tested': len(SNAP_URLS),
        'n_confirmed': sum(1 for r in all_results.values() 
                          if 'headline' in r and r['headline'].get('compounding_vulnerability_confirmed', False))
    }
    
    outpath = '/Users/freddycach/.openclaw/workspace/projects/science-program/papers/paper-a/fixes/additional_networks_results.json'
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")

if __name__ == '__main__':
    main()
