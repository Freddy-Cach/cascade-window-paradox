"""
Universality Test: Does the z₁ cascade window expansion hold beyond BA networks?

Tests the cascade window expansion under hub removal for:
1. BA (Barabási-Albert) — power-law degree distribution
2. Configuration model with power-law P(k) ~ k^(-gamma) for different gamma
3. Holme-Kim (BA + clustering)
4. Chung-Lu (arbitrary degree sequence)

The key question: Is cascade window expansion universal for networks with κ > κ_c?

Author: F. Cachero
Date: 2026-03-01
"""

import networkx as nx
import numpy as np
import json
import time
from collections import Counter

def generate_powerlaw_config(N, gamma, k_min=2, k_max=None):
    """Generate configuration model with power-law degree sequence."""
    if k_max is None:
        k_max = int(np.sqrt(N))
    
    # Generate degree sequence from power-law
    ks = np.arange(k_min, k_max+1)
    probs = ks.astype(float)**(-gamma)
    probs /= probs.sum()
    
    degrees = np.random.choice(ks, size=N, p=probs)
    # Ensure even sum
    if degrees.sum() % 2 == 1:
        degrees[np.argmin(degrees)] += 1
    
    try:
        G = nx.configuration_model(degrees.tolist())
        G = nx.Graph(G)  # Remove multi-edges
        G.remove_edges_from(nx.selfloop_edges(G))
        return G
    except:
        return None

def generate_holme_kim(N, m=2, p_triangle=0.5):
    """Holme-Kim model: BA + triangle closure (adds clustering)."""
    return nx.powerlaw_cluster_graph(N, m, p_triangle)

def compute_kappa(G):
    """Degree heterogeneity κ = <k²>/<k>."""
    degrees = np.array([d for _, d in G.degree()])
    if len(degrees) == 0 or degrees.mean() == 0:
        return 0
    return (degrees**2).mean() / degrees.mean()

def remove_hubs(G, frac=0.10):
    """Remove top frac% of nodes by degree."""
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    n_remove = max(1, int(len(G) * frac))
    to_remove = [n for n, d in degrees[:n_remove]]
    G2 = G.copy()
    G2.remove_nodes_from(to_remove)
    return G2

def estimate_pc_fast(G, n_trials=15, n_p=50):
    """Quick bond percolation threshold estimate via S=0.5."""
    N = len(G)
    if N < 10:
        return 1.0
    
    def giant_frac(p):
        sizes = []
        for _ in range(n_trials):
            edges = [(u, v) for u, v in G.edges() if np.random.random() < p]
            H = nx.Graph()
            H.add_nodes_from(G.nodes())
            H.add_edges_from(edges)
            if len(H) > 0:
                largest = max(nx.connected_components(H), key=len)
                sizes.append(len(largest) / N)
            else:
                sizes.append(0)
        return np.mean(sizes)
    
    # Binary search for S=0.5
    lo, hi = 0.0, 1.0
    for _ in range(12):  # ~4 digits precision
        mid = (lo + hi) / 2
        s = giant_frac(mid)
        if s > 0.5:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2

def run_cascade(G, phi, seed_node=None, max_steps=100):
    """Run Watts threshold cascade from single seed."""
    N = len(G)
    if N == 0:
        return 0.0
    
    nodes = list(G.nodes())
    if seed_node is None:
        seed_node = np.random.choice(nodes)
    
    active = {seed_node}
    newly_active = {seed_node}
    
    degrees = dict(G.degree())
    
    for step in range(max_steps):
        next_active = set()
        for node in G.nodes():
            if node in active:
                continue
            k = degrees[node]
            if k == 0:
                continue
            active_neighbors = sum(1 for n in G.neighbors(node) if n in active)
            if active_neighbors / k >= phi:
                next_active.add(node)
        
        if not next_active:
            break
        active |= next_active
    
    return len(active) / N

def cascade_sweep(G, phi, n_seeds=20):
    """Average cascade size over random seeds."""
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return 0.0
    cascades = []
    for _ in range(n_seeds):
        seed = np.random.choice(nodes)
        c = run_cascade(G, phi, seed)
        cascades.append(c)
    return np.mean(cascades)


def test_network(name, G, phi_test=0.22, hub_frac=0.10, n_seeds=20):
    """Test cascade window expansion for a given network."""
    N = len(G)
    kappa = compute_kappa(G)
    mean_k = np.mean([d for _, d in G.degree()])
    
    # Before hub removal
    pc_before = estimate_pc_fast(G)
    cascade_before = cascade_sweep(G, phi_test, n_seeds)
    
    # After hub removal
    G2 = remove_hubs(G, hub_frac)
    N2 = len(G2)
    kappa_after = compute_kappa(G2)
    pc_after = estimate_pc_fast(G2)
    cascade_after = cascade_sweep(G2, phi_test, n_seeds)
    
    delta_f_p = (pc_after - pc_before) / pc_before * 100 if pc_before > 0 else 0
    delta_f_c = (cascade_after - cascade_before) / max(cascade_before, 0.001) * 100
    
    result = {
        'name': name,
        'N': N,
        'mean_k': round(mean_k, 2),
        'kappa_before': round(kappa, 2),
        'kappa_after': round(kappa_after, 2),
        'pc_before': round(pc_before, 3),
        'pc_after': round(pc_after, 3),
        'delta_f_p_pct': round(delta_f_p, 1),
        'cascade_before': round(cascade_before * 100, 2),
        'cascade_after': round(cascade_after * 100, 2),
        'delta_f_c_pct': round(delta_f_c, 1),
        'compounding': delta_f_p > 0 and cascade_after > cascade_before * 1.5,
        'phase_transition': cascade_before < 0.05 and cascade_after > 0.10,
    }
    return result


def main():
    N = 1000  # Quick test
    np.random.seed(42)
    
    print("=" * 70)
    print("UNIVERSALITY TEST: Cascade Window Expansion")
    print(f"N={N}, φ=0.22, hub removal=10%")
    print("=" * 70)
    print()
    
    tests = []
    
    # 1. BA (reference)
    print("Testing BA(m=2)...", end=" ", flush=True)
    t0 = time.time()
    G = nx.barabasi_albert_graph(N, 2)
    r = test_network("BA(m=2)", G)
    tests.append(r)
    print(f"done ({time.time()-t0:.1f}s)")
    
    # 2. BA(m=4) — higher connectivity
    print("Testing BA(m=4)...", end=" ", flush=True)
    t0 = time.time()
    G = nx.barabasi_albert_graph(N, 4)
    r = test_network("BA(m=4)", G, phi_test=0.15)  # Lower phi for higher m
    tests.append(r)
    print(f"done ({time.time()-t0:.1f}s)")
    
    # 3. Config model gamma=2.5
    print("Testing CM(γ=2.5)...", end=" ", flush=True)
    t0 = time.time()
    G = generate_powerlaw_config(N, 2.5)
    if G:
        r = test_network("CM(γ=2.5)", G)
        tests.append(r)
    print(f"done ({time.time()-t0:.1f}s)")
    
    # 4. Config model gamma=3.0
    print("Testing CM(γ=3.0)...", end=" ", flush=True)
    t0 = time.time()
    G = generate_powerlaw_config(N, 3.0)
    if G:
        r = test_network("CM(γ=3.0)", G)
        tests.append(r)
    print(f"done ({time.time()-t0:.1f}s)")
    
    # 5. Holme-Kim (BA + clustering)
    print("Testing Holme-Kim(m=2, p=0.5)...", end=" ", flush=True)
    t0 = time.time()
    G = generate_holme_kim(N, 2, 0.5)
    r = test_network("Holme-Kim(m=2)", G)
    tests.append(r)
    print(f"done ({time.time()-t0:.1f}s)")
    
    # 6. ER (control — should NOT show effect)
    print("Testing ER(⟨k⟩=4)...", end=" ", flush=True)
    t0 = time.time()
    G = nx.erdos_renyi_graph(N, 4/N)
    r = test_network("ER(⟨k⟩=4)", G)
    tests.append(r)
    print(f"done ({time.time()-t0:.1f}s)")
    
    # 7. WS (control — should NOT show effect)
    print("Testing WS(k=4, p=0.1)...", end=" ", flush=True)
    t0 = time.time()
    G = nx.watts_strogatz_graph(N, 4, 0.1)
    r = test_network("WS(k=4)", G)
    tests.append(r)
    print(f"done ({time.time()-t0:.1f}s)")
    
    # Results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"{'Network':<20} {'κ':>6} {'ΔF_p%':>7} {'Casc_b':>8} {'Casc_a':>8} {'ΔF_c%':>8} {'Compound':>10} {'PhTrans':>8}")
    print("-" * 75)
    for r in tests:
        print(f"{r['name']:<20} {r['kappa_before']:>6.1f} {r['delta_f_p_pct']:>7.1f} "
              f"{r['cascade_before']:>7.2f}% {r['cascade_after']:>7.2f}% {r['delta_f_c_pct']:>8.1f} "
              f"{'YES' if r['compounding'] else 'no':>10} {'YES' if r['phase_transition'] else 'no':>8}")
    
    # Universality assessment
    print()
    n_compound = sum(1 for r in tests if r['compounding'])
    n_high_kappa = sum(1 for r in tests if r['kappa_before'] > 8)
    n_compound_high_k = sum(1 for r in tests if r['compounding'] and r['kappa_before'] > 8)
    n_no_compound_low_k = sum(1 for r in tests if not r['compounding'] and r['kappa_before'] < 6)
    
    print(f"Networks with κ > 8: {n_high_kappa}")
    print(f"Networks with compounding vulnerability: {n_compound}")
    print(f"High-κ networks WITH compounding: {n_compound_high_k}/{n_high_kappa}")
    print(f"Low-κ networks WITHOUT compounding: {n_no_compound_low_k}/{sum(1 for r in tests if r['kappa_before'] < 6)}")
    
    if n_compound_high_k == n_high_kappa and n_high_kappa > 0:
        print("\n✅ UNIVERSALITY SUPPORTED: All high-κ networks show compounding vulnerability")
    elif n_compound_high_k > 0:
        print(f"\n⚠️ PARTIAL SUPPORT: {n_compound_high_k}/{n_high_kappa} high-κ networks show effect")
    else:
        print("\n❌ UNIVERSALITY NOT SUPPORTED")
    
    # Save results
    with open('RESULTS_UNIVERSALITY.json', 'w') as f:
        json.dump(tests, f, indent=2)
    print(f"\nResults saved to RESULTS_UNIVERSALITY.json")


if __name__ == '__main__':
    main()
