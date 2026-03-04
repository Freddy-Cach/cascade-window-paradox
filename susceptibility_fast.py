#!/usr/bin/env python3
"""
Fast susceptibility-peak p_c validation for P2.
Reduced parameters for speed; still statistically valid.
"""

import numpy as np
import networkx as nx
import json
import time

def bond_percolation(G, p, rng):
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v in G.edges():
        if rng.random() < p:
            H.add_edge(u, v)
    return H

def compute_gcc_fraction(H):
    if len(H) == 0:
        return 0.0
    cc = max(nx.connected_components(H), key=len)
    return len(cc) / len(H)

def compute_susceptibility(H):
    N = len(H)
    if N == 0:
        return 0.0
    components = [len(c) for c in nx.connected_components(H)]
    if not components:
        return 0.0
    max_size = max(components)
    finite = [s for s in components if s < max_size]
    if not finite:
        if len(components) > 1:
            finite = components[1:]
        else:
            return 0.0
    s_arr = np.array(finite, dtype=float)
    if s_arr.sum() == 0:
        return 0.0
    return float(np.sum(s_arr**2) / np.sum(s_arr))

def estimate_pc(G, n_p=80, n_trials=10, seed=42):
    rng = np.random.RandomState(seed)
    p_values = np.linspace(0.01, 0.99, n_p)
    gcc_means = np.zeros(n_p)
    chi_means = np.zeros(n_p)
    
    for ip, p in enumerate(p_values):
        gcc_acc = 0.0
        chi_acc = 0.0
        for _ in range(n_trials):
            H = bond_percolation(G, p, rng)
            gcc_acc += compute_gcc_fraction(H)
            chi_acc += compute_susceptibility(H)
        gcc_means[ip] = gcc_acc / n_trials
        chi_means[ip] = chi_acc / n_trials
    
    pc_s05 = p_values[np.argmin(np.abs(gcc_means - 0.5))]
    pc_chi = p_values[np.argmax(chi_means)]
    return float(pc_s05), float(pc_chi)

def hub_remove(G, frac=0.10):
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    n_remove = int(frac * len(G))
    to_remove = [n for n, d in degrees[:n_remove]]
    H = G.copy()
    H.remove_nodes_from(to_remove)
    return H

def run():
    configs = [
        ('BA', 2000, 5),
        ('WS', 2000, 5),
        ('ER', 2000, 5),
        ('BA', 5000, 3),
        ('BA', 10000, 2),
    ]
    
    results = {}
    
    for name, N, n_nets in configs:
        key = f'{name}-{N}'
        print(f'\n=== {key} ({n_nets} nets) ===', flush=True)
        t0 = time.time()
        
        s05_bef, s05_aft, chi_bef, chi_aft = [], [], [], []
        
        for i in range(n_nets):
            if name == 'BA':
                G = nx.barabasi_albert_graph(N, 2, seed=i*100)
            elif name == 'WS':
                G = nx.watts_strogatz_graph(N, 4, 0.1, seed=i*100)
            else:
                G = nx.erdos_renyi_graph(N, 4.0/(N-1), seed=i*100)
            
            G2 = hub_remove(G)
            
            pc_s_b, pc_c_b = estimate_pc(G, n_p=80, n_trials=10, seed=i*1000)
            pc_s_a, pc_c_a = estimate_pc(G2, n_p=80, n_trials=10, seed=i*1000+500)
            
            s05_bef.append(pc_s_b)
            s05_aft.append(pc_s_a)
            chi_bef.append(pc_c_b)
            chi_aft.append(pc_c_a)
            
            print(f'  Net {i}: bef S05={pc_s_b:.3f} χ={pc_c_b:.3f} | aft S05={pc_s_a:.3f} χ={pc_c_a:.3f}', flush=True)
        
        sb = np.mean(s05_bef)
        sa = np.mean(s05_aft)
        cb = np.mean(chi_bef)
        ca = np.mean(chi_aft)
        
        df_s = (sa - sb) / sb * 100
        df_c = (ca - cb) / cb * 100
        
        elapsed = time.time() - t0
        
        results[key] = {
            'pc_S05_before': round(sb, 4),
            'pc_S05_after': round(sa, 4),
            'pc_chi_before': round(cb, 4),
            'pc_chi_after': round(ca, 4),
            'deltaF_S05_pct': round(df_s, 1),
            'deltaF_chi_pct': round(df_c, 1),
            'agreement_before': round(abs(sb - cb), 4),
            'agreement_after': round(abs(sa - ca), 4),
            'n_nets': n_nets,
            'elapsed_s': round(elapsed, 1)
        }
        
        print(f'  ΔF(S05)={df_s:+.1f}%  ΔF(χ)={df_c:+.1f}%  agree_bef={abs(sb-cb):.4f}  agree_aft={abs(sa-ca):.4f}  [{elapsed:.0f}s]', flush=True)
    
    outpath = '/Users/freddycach/.openclaw/workspace/projects/conservation-of-fragility/RESULTS_SUSCEPTIBILITY.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print('\n' + '='*80)
    print('SUSCEPTIBILITY VALIDATION SUMMARY')
    print('='*80)
    print(f'{"Network":<12} {"ΔF(S=0.5)":>12} {"ΔF(χ-peak)":>12} {"Agree(bef)":>12} {"Agree(aft)":>12}')
    print('-'*60)
    for k, r in results.items():
        print(f'{k:<12} {r["deltaF_S05_pct"]:>+10.1f}%  {r["deltaF_chi_pct"]:>+10.1f}%  {r["agreement_before"]:>10.4f}  {r["agreement_after"]:>10.4f}')
    print(f'\nSaved: {outpath}')

if __name__ == '__main__':
    run()
