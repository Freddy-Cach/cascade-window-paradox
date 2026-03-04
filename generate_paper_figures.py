"""
Generate all figures for Paper A: "Compounding Vulnerability"
Reads from RESULTS_*.json files and produces publication-quality figures.

Author: F. Cachero
Date: 2026-03-01
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

# Style for PRE
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

FIGDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'science-program', 'modules', 'figures')
os.makedirs(FIGDIR, exist_ok=True)


def load_json(name):
    path = os.path.join(os.path.dirname(__file__), name)
    with open(path) as f:
        return json.load(f)


def fig1_hub_vulnerability():
    """Figure 1: Hub vulnerability experiment bar chart."""
    data = load_json('RESULTS_HUB_VULNERABILITY.json')
    
    conditions = ['A\n(baseline)', 'B\n(φ_h=0.01)', 'B2\n(φ_h=0.05)', 'C\n(removed)', 'D\n(φ_h=0.50)']
    keys = ['A_baseline', 'B_hubs_vulnerable_001', 'B2_hubs_moderate_005', 'C_hubs_removed', 'D_hubs_extra_resistant']
    means = [data[k]['mean'] * 100 for k in keys]
    stds = [data[k].get('std', 0) * 100 for k in keys]
    
    colors = ['#2ecc71', '#e74c3c', '#e67e22', '#3498db', '#95a5a6']
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(range(5), means, yerr=stds, capsize=4, 
                  color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.set_xticks(range(5))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel('Mean Cascade Size (%)')
    ax.set_ylim(0, 120)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Annotations (inside bars for large values, above for small)
    for i, (mv, s) in enumerate(zip(means, stds)):
        if mv > 50:
            ax.text(i, mv/2, f'{mv:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        else:
            ax.text(i, mv + s + 2, f'{mv:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'fig1_hub_vulnerability.png'))
    plt.savefig(os.path.join(FIGDIR, 'fig1_hub_vulnerability.pdf'))
    plt.close()
    print("Fig 1: Hub vulnerability → saved")


def fig2_phi_sweep():
    """Figure 2: φ-sweep cascade phase diagram."""
    data = load_json('RESULTS_PHI_SWEEP_N10000.json')
    
    phis = []
    df_casc = []
    df_casc_err = []
    
    # Parse BA phi-sweep data
    for key in sorted(data.keys()):
        if key.startswith('BA-phi'):
            phi = float(key.split('phi')[1])
            phis.append(phi)
            df_casc.append(data[key]['delta_f_cascade']['mean'])
            df_casc_err.append(data[key]['delta_f_cascade'].get('std', 0))
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(phis, df_casc, yerr=df_casc_err, fmt='o-', color='#e74c3c', 
                markersize=5, linewidth=1.5, capsize=3, label='BA(m=2, N=10000)')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvspan(0.18, 0.28, alpha=0.1, color='red', label='Phase transition zone')
    
    # Mark peak
    peak_idx = np.argmax(df_casc)
    ax.annotate(f'Peak: φ={phis[peak_idx]}\nΔF={df_casc[peak_idx]:.0f}',
                xy=(phis[peak_idx], df_casc[peak_idx]),
                xytext=(phis[peak_idx]+0.05, df_casc[peak_idx]-30),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9)
    
    ax.set_xlabel('Cascade threshold φ')
    ax.set_ylabel('ΔF_cascade (%)')
    ax.set_title('Cascade Fragility Change After Hub Removal')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'fig2_phi_sweep.png'))
    plt.savefig(os.path.join(FIGDIR, 'fig2_phi_sweep.pdf'))
    plt.close()
    print("Fig 2: φ-sweep → saved")


def fig3_z1_crossing():
    """Figure 3: z₁ pre vs post showing the crossing at φ=0.22."""
    from scipy.special import comb
    
    def ba_P(k, m=2):
        return 2*m*(m+1)/(k*(k+1)*(k+2)) if k >= m else 0
    
    def z1_pre(phi, m=2):
        K = int(1/phi)
        return sum(k*(k-1)*ba_P(k,m)/(2*m) for k in range(m, K+1))
    
    def z1_post_exact(phi, m=2, k_cut=6):
        rho = (m+1)/(k_cut+2)
        alpha = m*(m+1)/((k_cut+1)*(k_cut+2))
        K = int(1/phi)
        k_mean_post = sum(k*(1-rho)*ba_P(k,m) for k in range(m, k_cut+1))/(1-alpha)
        if k_mean_post == 0: return 0
        z1 = 0
        for k in range(m, k_cut+1):
            for j in range(1, min(K,k)+1):
                z1 += j*(j-1)*comb(k,j,exact=True)*(1-rho)**j*rho**(k-j)*ba_P(k,m)
        return z1/((1-alpha)*k_mean_post)
    
    phis = np.arange(0.08, 0.55, 0.005)
    z1_pres = [z1_pre(p) for p in phis]
    z1_posts = [z1_post_exact(p) for p in phis]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(phis, z1_pres, 'b-', linewidth=2, label='z₁ (pre-removal)')
    ax.plot(phis, z1_posts, 'r-', linewidth=2, label='z₁ (post-removal)')
    # Add markers at key computed points
    key_phis = [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.27, 0.30, 0.40, 0.50]
    for kp in key_phis:
        idx = np.argmin(np.abs(phis - kp))
        ax.plot(phis[idx], z1_pres[idx], 'bo', markersize=4)
        ax.plot(phis[idx], z1_posts[idx], 'ro', markersize=4)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='z₁ = 1 (critical)')
    
    # Shade the new cascade window
    # Find where pre < 1 and post > 1
    for i in range(len(phis)):
        if z1_pres[i] < 1 and z1_posts[i] > 1:
            ax.axvspan(phis[max(0,i-1)], phis[min(len(phis)-1,i+1)], 
                      alpha=0.15, color='orange')
    
    ax.annotate('New cascade\nwindow', xy=(0.23, 1.08), fontsize=9, 
                fontstyle='italic', color='darkorange', ha='center')
    
    ax.set_xlabel('Cascade threshold φ')
    ax.set_ylabel('Branching factor z₁')
    ax.set_title('Cascade Branching Factor: Pre vs Post Hub Removal')
    ax.set_xlim(0.08, 0.55)
    ax.set_ylim(0, 2.5)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'fig3_z1_crossing.png'))
    plt.savefig(os.path.join(FIGDIR, 'fig3_z1_crossing.pdf'))
    plt.close()
    print("Fig 3: z₁ crossing → saved")


def fig4_topology_comparison():
    """Figure 4: ΔF_p across topologies (bar chart)."""
    data = load_json('RESULTS_COMBINED.json')
    
    topos = ['BA', 'ER', 'WS', 'CM']
    labels = ['BA\n(m=2)', 'ER\n(⟨k⟩=4)', 'WS\n(k=4)', 'Config\n(exp)']
    df_percs = []
    kappas = []
    
    for t in topos:
        if t in data:
            d = data[t]
            df_percs.append(d.get('delta_F_pct', 0))
            kappas.append(d.get('kappa', 0))
        else:
            df_percs.append(0)
            kappas.append(0)
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    bars = ax.bar(range(len(topos)), df_percs, color=colors, edgecolor='black', 
                  linewidth=0.5, alpha=0.85)
    
    for i, (df, k) in enumerate(zip(df_percs, kappas)):
        ax.text(i, df + 3, f'{df:.0f}%\n(κ={k:.1f})', ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(range(len(topos)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('ΔF_percolation (%)')
    ax.set_title('Percolation Fragility Change by Topology (N=2000)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'fig4_topology.png'))
    plt.savefig(os.path.join(FIGDIR, 'fig4_topology.pdf'))
    plt.close()
    print("Fig 4: Topology comparison → saved")


if __name__ == '__main__':
    print("Generating Paper A figures...")
    print()
    fig1_hub_vulnerability()
    fig2_phi_sweep()
    fig3_z1_crossing()
    fig4_topology_comparison()
    print()
    print(f"All figures saved to: {os.path.abspath(FIGDIR)}")
