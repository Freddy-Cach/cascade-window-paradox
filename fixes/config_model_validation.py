#!/usr/bin/env python3
"""
Configuration Model Validation: Test the z₁ derivation on config-model networks
(where it should be EXACT) vs BA networks (where it's approximate).
This validates the analytical framework independently of BA-specific correlations.
"""
import numpy as np
import networkx as nx
import json, sys

def create_config_model_from_ba(N, m, seed=None):
    """Create a configuration-model network with same degree sequence as BA."""
    rng = np.random.default_rng(seed)
    ba = nx.barabasi_albert_graph(N, m, seed=int(rng.integers(1e9)))
    degree_seq = sorted([d for _, d in ba.degree()], reverse=True)
    # Make sum even
    if sum(degree_seq) % 2 == 1:
        degree_seq[-1] += 1
    G = nx.configuration_model(degree_seq, seed=int(rng.integers(1e9)))
    G = nx.Graph(G)  # Remove multi-edges and self-loops
    G.remove_edges_from(nx.selfloop_edges(G))
    return G, ba

def hub_removal(G, alpha=0.10):
    """Remove top alpha fraction of nodes by degree."""
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    n_remove = int(alpha * len(G))
    remove_nodes = [n for n, d in degrees[:n_remove]]
    G_removed = G.copy()
    G_removed.remove_nodes_from(remove_nodes)
    return G_removed

def watts_cascade(G, phi, seed_node):
    """Run Watts threshold cascade from a single seed."""
    active = {seed_node}
    newly_active = {seed_node}
    nodes = set(G.nodes())
    
    for _ in range(150):
        next_active = set()
        for node in nodes - active:
            neighbors = set(G.neighbors(node))
            if len(neighbors) == 0:
                continue
            frac_active = len(neighbors & active) / len(neighbors)
            if frac_active >= phi:
                next_active.add(node)
        if not next_active:
            break
        active |= next_active
        newly_active = next_active
    
    return len(active) / len(G)

def susceptibility_peak_pc(G, n_p=100, n_bond=30):
    """Estimate percolation threshold using susceptibility peak."""
    N = len(G)
    edges = list(G.edges())
    rng = np.random.default_rng(42)
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
    return p_values[best_idx]

def run_validation(N=2000, m=2, n_networks=20, n_seeds=20, alpha=0.10):
    """Compare cascade behavior on BA vs config-model networks."""
    phi_values = [0.15, 0.18, 0.20, 0.22, 0.25, 0.27, 0.30]
    
    results = {
        'ba': {str(phi): {'pre': [], 'post': []} for phi in phi_values},
        'config': {str(phi): {'pre': [], 'post': []} for phi in phi_values},
        'percolation': {'ba_pre': [], 'ba_post': [], 'config_pre': [], 'config_post': []},
        'kappa': {'ba': [], 'config': []},
        'params': {'N': N, 'm': m, 'n_networks': n_networks, 'n_seeds': n_seeds, 'alpha': alpha}
    }
    
    for i in range(n_networks):
        print(f"  Network {i+1}/{n_networks}", flush=True)
        config_G, ba_G = create_config_model_from_ba(N, m, seed=i*1000)
        
        # Compute kappa
        for label, G in [('ba', ba_G), ('config', config_G)]:
            degrees = [d for _, d in G.degree()]
            k1 = np.mean(degrees)
            k2 = np.mean([d**2 for d in degrees])
            results['kappa'][label].append(k2/k1)
        
        for label, G in [('ba', ba_G), ('config', config_G)]:
            G_post = hub_removal(G, alpha)
            
            # Percolation (every 5th network to save time)
            if i % 5 == 0:
                pc_pre = susceptibility_peak_pc(G, n_p=80, n_bond=20)
                pc_post = susceptibility_peak_pc(G_post, n_p=80, n_bond=20)
                results['percolation'][f'{label}_pre'].append(float(pc_pre))
                results['percolation'][f'{label}_post'].append(float(pc_post))
            
            # Cascades
            nodes_pre = list(G.nodes())
            nodes_post = list(G_post.nodes())
            rng = np.random.default_rng(i * 1000 + 1)
            
            for phi in phi_values:
                pre_cascades = []
                post_cascades = []
                seeds_pre = rng.choice(nodes_pre, min(n_seeds, len(nodes_pre)), replace=False)
                seeds_post = rng.choice(nodes_post, min(n_seeds, len(nodes_post)), replace=False)
                
                for s in seeds_pre:
                    pre_cascades.append(watts_cascade(G, phi, s))
                for s in seeds_post:
                    post_cascades.append(watts_cascade(G_post, phi, s))
                
                results[label][str(phi)]['pre'].extend(pre_cascades)
                results[label][str(phi)]['post'].extend(post_cascades)
    
    # Compute summary
    summary = {'params': results['params'], 'kappa': {}}
    for label in ['ba', 'config']:
        summary['kappa'][label] = {
            'mean': float(np.mean(results['kappa'][label])),
            'std': float(np.std(results['kappa'][label]))
        }
    
    summary['percolation'] = {}
    for key in ['ba_pre', 'ba_post', 'config_pre', 'config_post']:
        vals = results['percolation'][key]
        if vals:
            summary['percolation'][key] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals))
            }
    
    summary['cascades'] = {}
    for label in ['ba', 'config']:
        summary['cascades'][label] = {}
        for phi in phi_values:
            pre = results[label][str(phi)]['pre']
            post = results[label][str(phi)]['post']
            summary['cascades'][label][str(phi)] = {
                'pre_mean': float(np.mean(pre)) if pre else 0,
                'post_mean': float(np.mean(post)) if post else 0,
                'pre_std': float(np.std(pre)) if pre else 0,
                'post_std': float(np.std(post)) if post else 0,
                'amplification': float((np.mean(post) - np.mean(pre)) / max(np.mean(pre), 1e-6) * 100) if pre and post else 0
            }
    
    return summary

if __name__ == '__main__':
    print("Configuration Model Validation")
    print("=" * 50)
    summary = run_validation(N=2000, m=2, n_networks=20, n_seeds=20)
    
    print("\n=== RESULTS ===")
    print(f"κ (BA): {summary['kappa']['ba']['mean']:.2f} ± {summary['kappa']['ba']['std']:.2f}")
    print(f"κ (Config): {summary['kappa']['config']['mean']:.2f} ± {summary['kappa']['config']['std']:.2f}")
    
    if summary['percolation']:
        for key, vals in summary['percolation'].items():
            print(f"p_c ({key}): {vals['mean']:.3f} ± {vals['std']:.3f}")
    
    print("\nCascade amplification at φ=0.22:")
    for label in ['ba', 'config']:
        d = summary['cascades'][label]['0.22']
        print(f"  {label}: {d['pre_mean']*100:.2f}% → {d['post_mean']*100:.2f}% (amp: {d['amplification']:.0f}%)")
    
    with open('config_model_validation_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nSaved to config_model_validation_results.json")
