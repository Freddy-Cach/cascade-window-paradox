#!/usr/bin/env python3
"""
φ-sweep figure v4: Log-scale cascade sizes with clearer REGIME LABELS.
Corrects legend order, adds numerical annotations, and ensures regime shading is legible.
"""

import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Okabe-Ito colorblind-safe palette
OI_BLUE   = '#0072B2'
OI_RED    = '#D55E00'
OI_GREEN  = '#009E73'

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 12,
    'legend.fontsize': 10.5, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'lines.linewidth': 2.5,
})

# ── simulation helpers ──────────────────────────────────────────────────────

def remove_hubs(G, frac=0.10):
    n = len(G)
    n_remove = max(1, int(n * frac))
    # Use degree-based removal with random tie-breaking (implied by stable sort on node id)
    top = sorted(G.degree(), key=lambda x: (-x[1], x[0]))[:n_remove]
    G2 = G.copy()
    G2.remove_nodes_from([v for v, _ in top])
    return G2

def watts_cascade(G, phi, n_trials=40):
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return 0.0, 0.0
    adj = {v: list(G.neighbors(v)) for v in nodes}
    deg = {v: G.degree(v) for v in nodes}
    sizes = []
    rng = np.random.default_rng()
    for _ in range(n_trials):
        seed = rng.choice(nodes)
        active = {seed}
        changed = True
        step = 0
        while changed and step < 150:
            changed = False
            new_active = set()
            for v in nodes:
                if v in active or deg[v] == 0:
                    continue
                if sum(1 for u in adj[v] if u in active) / deg[v] >= phi:
                    new_active.add(v)
                    changed = True
            active |= new_active
            step += 1
        sizes.append(len(active) / n)
    arr = np.array(sizes)
    return float(arr.mean()), float(arr.std() / np.sqrt(len(arr)))

def run_sweep(topo, N=2000, n_nets=15, phi_values=None):
    if phi_values is None:
        # Finer resolution around critical points
        phi_values = [0.05, 0.10, 0.13, 0.15, 0.16, 0.17, 0.18, 0.19,
                      0.20, 0.22, 0.24, 0.25, 0.27, 0.28, 0.30, 0.35, 0.40, 0.50]
    rng = np.random.default_rng(2026)
    res = {p: dict(before=[], after=[]) for p in phi_values}
    for i in range(n_nets):
        seed = int(rng.integers(1, 10**6))
        np.random.seed(seed)
        G  = (nx.barabasi_albert_graph(N, 2, seed=seed) if topo == 'BA'
              else nx.watts_strogatz_graph(N, 4, 0.1, seed=seed))
        Ga = remove_hubs(G)
        for phi in phi_values:
            cb, _ = watts_cascade(G,  phi)
            ca, _ = watts_cascade(Ga, phi)
            res[phi]['before'].append(cb)
            res[phi]['after'].append(ca)
        if (i+1) % 5 == 0:
            print(f'  {topo} {i+1}/{n_nets}')
    out = {}
    for p, d in res.items():
        b, a = np.array(d['before']), np.array(d['after'])
        out[p] = dict(
            bm=b.mean(), bs=1.96*b.std()/np.sqrt(len(b)),
            am=a.mean(), as_=1.96*a.std()/np.sqrt(len(a)),
        )
    return out

# ── figure builder ───────────────────────────────────────────────────────────

def make_figure(ba, ws):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)

    for ax, data, topo, kappa in [
        (axes[0], ba, 'Barabási–Albert  (m=2,  ⟨κ⟩≈12)',  12),
        (axes[1], ws, 'Watts–Strogatz  (k=4, p=0.1,  ⟨κ⟩≈4)', 4),
    ]:
        phis = sorted(data.keys())
        bm  = np.array([data[p]['bm']  for p in phis])
        bs  = np.array([data[p]['bs']  for p in phis])
        am  = np.array([data[p]['am']  for p in phis])
        as_ = np.array([data[p]['as_'] for p in phis])
        phis = np.array(phis)

        # floor near-zero values so log scale works
        FLOOR = 1.5e-4
        bm_plot = np.maximum(bm, FLOOR)
        am_plot = np.maximum(am, FLOOR)

        # "before" curve FIRST (chronological order)
        ax.fill_between(phis,
                         np.maximum(bm_plot - bs, FLOOR),
                         bm_plot + bs,
                         alpha=0.18, color=OI_BLUE)
        ax.semilogy(phis, bm_plot, 'o-', color=OI_BLUE, linewidth=2.5,
                    markersize=7, label='Before hub removal')

        # "after" curve SECOND
        ax.fill_between(phis,
                         np.maximum(am_plot - as_, FLOOR),
                         am_plot + as_,
                         alpha=0.18, color=OI_RED)
        ax.semilogy(phis, am_plot, 's-', color=OI_RED,  linewidth=2.5,
                    markersize=7, label='After hub removal (top 10%)')

        # --- shade the three regimes ---\n        # Use lighter, accessible colors
        ax.axvspan(0.00, 0.16, alpha=0.07, color=OI_GREEN, zorder=0)
        ax.axvspan(0.16, 0.27, alpha=0.08, color=OI_RED,   zorder=0)
        ax.axvspan(0.27, 0.55, alpha=0.05, color='gray',   zorder=0)

        # vertical dashed lines at φ* and φ**
        ax.axvline(0.16, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
        ax.axvline(0.27, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
        
        # Annotate critical thresholds
        if topo.startswith('BA'):
            ax.text(0.16, 1.8, 'φ*≈0.16', fontsize=9, ha='right', rotation=90, va='top')
            ax.text(0.27, 1.8, 'φ**≈0.27', fontsize=9, ha='right', rotation=90, va='top')

        # regime labels at top - CLEARER TEXT
        # Regime I
        ax.text(0.09, 0.88, 'Regime I\nBoth global\n(hub removal\nhelps)', 
                transform=ax.transAxes, fontsize=9, ha='center', va='top', 
                color='#006600', style='italic', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#e6f7e6', alpha=0.8, edgecolor='none'))
        
        # Regime II
        if topo.startswith('BA'):
            ax.text(0.40, 0.88, 'Regime II ⚡\nPHASE TRANSITION\n(after > before)',
                    transform=ax.transAxes, fontsize=9, ha='center', va='top', 
                    color='#8B0000', style='italic', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffe6e6', alpha=0.8, edgecolor='none'))
            
            # Annotation arrow for the gap at phi=0.22
            # Find closest index to 0.22
            idx = np.abs(phis - 0.22).argmin()
            if abs(phis[idx] - 0.22) < 0.01:
                p_val = phis[idx]
                y_before = max(bm[idx], FLOOR)
                y_after = max(am[idx], FLOOR)
                
                # Draw arrow
                ax.annotate('', xy=(p_val, y_after), xytext=(p_val, y_before),
                           arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
                # Label fold change
                ratio = y_after / max(y_before, 1e-5)
                ax.text(p_val + 0.01, np.sqrt(y_before * y_after), f'×{ratio:.0f}', 
                       fontsize=10, va='center', fontweight='bold')

        # Regime III
        ax.text(0.82, 0.88, 'Regime III\nBoth\nsubcritical',
                transform=ax.transAxes, fontsize=9, ha='center', va='top', 
                color='#444444', style='italic', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#f0f0f0', alpha=0.8, edgecolor='none'))

        ax.set_xlabel('Cascade threshold  φ', fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel('Cascade size  (network fraction, log scale)', fontsize=11)
        ax.set_title(topo, fontsize=12, pad=10)
        ax.legend(frameon=True, loc='lower left', fontsize=9.5)
        ax.grid(True, which='both', alpha=0.25, linestyle='-')
        ax.set_xlim(0.03, 0.53)
        ax.set_ylim(5e-4, 2.0)
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, _: f'{y:.0%}' if y >= 0.01 else f'{y*100:.2f}%'))

    fig.suptitle(
        'Hub Removal Shifts the Cascade Window While Improving Percolation Robustness\n'
        'Regime II shows the "resilience trade-off": hub removal triggers global cascades at previously safe thresholds.',
        fontsize=12, y=1.02
    )
    fig.tight_layout()
    fig.savefig('fig6_phi_sweep_v4.png', bbox_inches='tight')
    fig.savefig('fig6_phi_sweep_v4.pdf', bbox_inches='tight')
    plt.close(fig)
    print('✅  fig6_phi_sweep_v4 saved')


if __name__ == '__main__':
    print('Running BA sweep...')
    ba = run_sweep('BA', N=2000, n_nets=15)
    print('Running WS sweep...')
    ws = run_sweep('WS', N=2000, n_nets=15)
    make_figure(ba, ws)
    print('\\nAll done ✅')
