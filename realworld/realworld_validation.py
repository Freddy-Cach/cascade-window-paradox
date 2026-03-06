"""
Real-World Network Validation for Paper A (arXiv:2603.04838)
=============================================================
Validates hub-removal dual effects (percolation + cascade) on real-world networks.

Networks:
  1. Bitcoin OTC Trust Network (Stanford SNAP, ~5k nodes)
  2. AS-733 Autonomous Systems (Stanford SNAP, ~1.5k nodes)

Experiments (matching paper methodology):
  a. Bond percolation threshold p_c (intact vs hub-removed top-10% by degree)
  b. Watts threshold model at φ=0.18, 0.22, 0.25, 0.30 (200 trials each)

Reference results (BA synthetic, N=2000, m=2):
  p_c: 0.34 → 0.90  (+165%)
  Cascade at φ=0.22: 0.29% → 20.6%  (intact→hub-removed)
"""

import os
import sys
import json
import time
import random
import urllib.request
import gzip
import math
import collections
import numpy as np
import networkx as nx

RESULTS_DIR = os.path.expanduser(
    "~/.openclaw/workspace/projects/science-program/papers/paper-a/realworld-results/"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# BA reference values from the paper
BA_REF = {
    "p_c_intact": 0.34,
    "p_c_hub_removed": 0.90,
    "cascade_phi022_intact_pct": 0.29,
    "cascade_phi022_hub_removed_pct": 20.6,
}

# ─────────────────────────────────────────────
# DOWNLOAD HELPERS
# ─────────────────────────────────────────────

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"  [cache] {os.path.basename(dest_path)} already exists, skipping download.")
        return dest_path
    print(f"  [download] {url}")
    print(f"          → {dest_path}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest_path, "wb") as f:
            total = 0
            while True:
                chunk = r.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
                print(f"  ... {total//1024} KB", end="\r")
        print(f"  [done] {total//1024} KB downloaded")
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise
    return dest_path


# ─────────────────────────────────────────────
# NETWORK LOADERS
# ─────────────────────────────────────────────

def load_bitcoin_otc(cache_dir):
    """Load Bitcoin OTC trust network (positive edges only, undirected LCC)."""
    csv_path = os.path.join(cache_dir, "soc-sign-bitcoinotc.csv.gz")
    url = "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz"

    # Check if already exists in topological-uncertainty project
    alt_path = os.path.expanduser(
        "~/.openclaw/workspace/projects/paper-topological-uncertainty/empirical/soc-sign-bitcoinotc.csv"
    )
    alt_gz = os.path.expanduser(
        "~/.openclaw/workspace/projects/paper-topological-uncertainty/empirical/soc-sign-bitcoinotc.csv.gz"
    )

    raw_edges = []

    if os.path.exists(alt_path):
        print(f"  [found] Using existing CSV at {alt_path}")
        with open(alt_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        u, v, rating = int(parts[0]), int(parts[1]), int(parts[2])
                        if rating > 0:
                            raw_edges.append((u, v))
                    except ValueError:
                        continue
    elif os.path.exists(alt_gz):
        print(f"  [found] Using existing GZ at {alt_gz}")
        with gzip.open(alt_gz, "rt") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        u, v, rating = int(parts[0]), int(parts[1]), int(parts[2])
                        if rating > 0:
                            raw_edges.append((u, v))
                    except ValueError:
                        continue
    else:
        download_file(url, csv_path)
        with gzip.open(csv_path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        u, v, rating = int(parts[0]), int(parts[1]), int(parts[2])
                        if rating > 0:
                            raw_edges.append((u, v))
                    except ValueError:
                        continue

    G_dir = nx.DiGraph()
    G_dir.add_edges_from(raw_edges)
    G = nx.Graph(G_dir)  # make undirected (drops self-loops automatically)
    G.remove_edges_from(nx.selfloop_edges(G))

    # Keep largest connected component
    lcc_nodes = max(nx.connected_components(G), key=len)
    G = G.subgraph(lcc_nodes).copy()
    print(f"  Bitcoin OTC: N={G.number_of_nodes()}, E={G.number_of_edges()}")
    return G


def load_as733(cache_dir):
    """
    Load a real-world scale-free network for second validation.
    Primary: AS-733 Autonomous Systems (tries multiple SNAP URLs)
    Fallback 1: email-Enron (SNAP) — known scale-free ~36k nodes
    Fallback 2: BA synthetic N=1500 (note as synthetic in results)
    """
    # Try AS-733 (several possible URLs)
    as_urls = [
        ("as19971108.txt.gz", "https://snap.stanford.edu/data/as19971108.txt.gz"),
        ("as20000102.txt.gz", "https://snap.stanford.edu/data/as20000102.txt.gz"),
    ]
    for fname, url in as_urls:
        gz_path = os.path.join(cache_dir, fname)
        try:
            download_file(url, gz_path)
            raw_edges = []
            with gzip.open(gz_path, "rt") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            raw_edges.append((int(parts[0]), int(parts[1])))
                        except ValueError:
                            continue
            if raw_edges:
                G = nx.Graph()
                G.add_edges_from(raw_edges)
                G.remove_edges_from(nx.selfloop_edges(G))
                lcc_nodes = max(nx.connected_components(G), key=len)
                G = G.subgraph(lcc_nodes).copy()
                print(f"  AS-733: N={G.number_of_nodes()}, E={G.number_of_edges()}")
                return G, "AS-733 Autonomous Systems (SNAP)"
        except Exception as e:
            print(f"  [WARN] {url} failed: {e}")
            continue

    # Fallback 1: email-Enron (smaller LCC version) from SNAP
    print("  [TRY] email-Enron network...")
    enron_urls = [
        ("email-Enron.txt.gz", "https://snap.stanford.edu/data/email-Enron.txt.gz"),
        ("Email-Enron.txt.gz", "https://snap.stanford.edu/data/Email-Enron.txt.gz"),
    ]
    for fname, url in enron_urls:
        gz_path = os.path.join(cache_dir, fname)
        try:
            download_file(url, gz_path)
            raw_edges = []
            with gzip.open(gz_path, "rt") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            raw_edges.append((int(parts[0]), int(parts[1])))
                        except ValueError:
                            continue
            if raw_edges:
                G = nx.Graph()
                G.add_edges_from(raw_edges)
                G.remove_edges_from(nx.selfloop_edges(G))
                lcc_nodes = max(nx.connected_components(G), key=len)
                G = G.subgraph(lcc_nodes).copy()
                print(f"  email-Enron: N={G.number_of_nodes()}, E={G.number_of_edges()}")
                return G, "email-Enron (SNAP)"
        except Exception as e:
            print(f"  [WARN] {url} failed: {e}")
            continue

    # Fallback 2: BA synthetic (explicitly labeled as synthetic)
    print("  [FALLBACK] All real-world downloads failed. Using BA synthetic N=1500 for reference.")
    G = nx.barabasi_albert_graph(1500, 2, seed=2603)
    print(f"  BA-fallback: N={G.number_of_nodes()}, E={G.number_of_edges()}")
    return G, "BA-1500 (SYNTHETIC FALLBACK — real network download failed)"


# ─────────────────────────────────────────────
# NETWORK CHARACTERIZATION
# ─────────────────────────────────────────────

def characterize_network(G, name):
    """Compute degree distribution statistics and κ = <k²>/<k>."""
    degrees = [d for _, d in G.degree()]
    k_mean = np.mean(degrees)
    k2_mean = np.mean([d**2 for d in degrees])
    kappa = k2_mean / k_mean  # degree heterogeneity κ = <k²>/<k>
    k_max = max(degrees)
    k_min = min(degrees)
    k_std = np.std(degrees)

    # Power-law exponent via MLE (Clauset et al.)
    # Use k_min = max(1, int(k_mean/2)) as lower cutoff
    k_min_fit = max(1, int(k_mean / 2))
    tail = [d for d in degrees if d >= k_min_fit]
    if len(tail) > 10:
        gamma_mle = 1 + len(tail) / sum(math.log(d / (k_min_fit - 0.5)) for d in tail)
    else:
        gamma_mle = float("nan")

    # z1 branching factor (analytical): z1 = <k²>/<k> - 1 (Watts-Strogatz formula)
    z1 = kappa - 1

    info = {
        "name": name,
        "N": G.number_of_nodes(),
        "E": G.number_of_edges(),
        "k_mean": round(k_mean, 3),
        "k_max": k_max,
        "k_min": k_min,
        "k_std": round(k_std, 3),
        "kappa": round(kappa, 2),   # degree heterogeneity
        "z1": round(z1, 2),          # branching factor
        "gamma_mle": round(gamma_mle, 3) if not math.isnan(gamma_mle) else None,
        "is_scale_free": kappa > 5,  # scale-free proxy: κ > 5
        "scale_free_threshold_met": kappa > 10,  # paper's κ>10 threshold
    }

    print(f"\n  {name}:")
    print(f"    N={info['N']}, E={info['E']}")
    print(f"    <k>={k_mean:.2f}, k_max={k_max}, κ=<k²>/<k>={kappa:.1f}, z1={z1:.2f}")
    print(f"    γ_MLE={gamma_mle:.3f}" if not math.isnan(gamma_mle) else "    γ_MLE=N/A")
    print(f"    Scale-free (κ>10): {info['scale_free_threshold_met']}")
    return info


# ─────────────────────────────────────────────
# HUB REMOVAL
# ─────────────────────────────────────────────

def remove_top_hubs(G, fraction=0.10):
    """Remove top `fraction` nodes by degree. Returns (G_removed, removed_nodes)."""
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    n_remove = max(1, int(len(degrees) * fraction))
    hub_nodes = [n for n, d in degrees[:n_remove]]
    G2 = G.copy()
    G2.remove_nodes_from(hub_nodes)
    # Keep LCC
    if len(G2) > 0:
        lcc = max(nx.connected_components(G2), key=len)
        G2 = G2.subgraph(lcc).copy()
    return G2, hub_nodes


# ─────────────────────────────────────────────
# PERCOLATION EXPERIMENT
# ─────────────────────────────────────────────

def bond_percolation_curve(G, p_values, n_trials=200, rng=None):
    """
    Bond percolation: for each p, keep each edge independently with prob p.
    Returns mean giant component fraction at each p.
    """
    if rng is None:
        rng = random.Random(42)

    N = G.number_of_nodes()
    edges = list(G.edges())
    gc_fracs = []

    for p in p_values:
        gc_sum = 0.0
        for _ in range(n_trials):
            # Sample edges
            kept = [e for e in edges if rng.random() < p]
            G_sub = nx.Graph()
            G_sub.add_nodes_from(G.nodes())
            G_sub.add_edges_from(kept)
            if G_sub.number_of_edges() == 0:
                gc_sum += 1.0 / N
                continue
            comps = sorted(nx.connected_components(G_sub), key=len, reverse=True)
            gc_sum += len(comps[0]) / N
        gc_fracs.append(gc_sum / n_trials)

    return gc_fracs


def find_percolation_threshold(p_values, gc_fracs):
    """
    Estimate p_c as the p where GC fraction exceeds 0.05 (onset of giant component).
    Uses linear interpolation between the crossing point.
    """
    threshold = 0.05
    for i in range(1, len(p_values)):
        if gc_fracs[i] >= threshold and gc_fracs[i - 1] < threshold:
            # Linear interpolation
            p0, p1 = p_values[i - 1], p_values[i]
            f0, f1 = gc_fracs[i - 1], gc_fracs[i]
            if f1 > f0:
                p_c = p0 + (threshold - f0) / (f1 - f0) * (p1 - p0)
                return round(p_c, 3)
    # If GC is always above threshold (robust network), return ~0
    if gc_fracs[0] >= threshold:
        return p_values[0]
    return p_values[-1]


def run_percolation_experiment(G_intact, G_removed, n_trials=200, n_p_steps=40):
    """Run percolation on both intact and hub-removed networks."""
    print(f"  Running bond percolation ({n_trials} trials × {n_p_steps} p-values)...")
    p_values = [round(i / n_p_steps, 3) for i in range(1, n_p_steps + 1)]

    rng = random.Random(2603)
    t0 = time.time()
    gc_intact = bond_percolation_curve(G_intact, p_values, n_trials=n_trials, rng=rng)
    t1 = time.time()
    print(f"    Intact done in {t1-t0:.1f}s")

    rng2 = random.Random(2604)
    gc_removed = bond_percolation_curve(G_removed, p_values, n_trials=n_trials, rng=rng2)
    t2 = time.time()
    print(f"    Hub-removed done in {t2-t1:.1f}s")

    p_c_intact = find_percolation_threshold(p_values, gc_intact)
    p_c_removed = find_percolation_threshold(p_values, gc_removed)
    delta_pct = round((p_c_removed - p_c_intact) / p_c_intact * 100, 1) if p_c_intact > 0 else None

    print(f"    p_c intact={p_c_intact:.3f}, hub-removed={p_c_removed:.3f}, Δ={delta_pct}%")

    return {
        "p_values": p_values,
        "gc_intact": [round(x, 4) for x in gc_intact],
        "gc_hub_removed": [round(x, 4) for x in gc_removed],
        "p_c_intact": p_c_intact,
        "p_c_hub_removed": p_c_removed,
        "delta_pct": delta_pct,
        "n_trials": n_trials,
    }


# ─────────────────────────────────────────────
# CASCADE EXPERIMENT (WATTS THRESHOLD MODEL)
# ─────────────────────────────────────────────

def watts_cascade(G, phi, seed_node=None, rng=None):
    """
    Watts threshold model:
    - Each node i adopts if fraction of neighbors already adopted >= phi
    - Seed: 1 random node (or specified)
    - Returns fraction of network that adopts (cascade size)
    """
    if rng is None:
        rng = random.Random()
    nodes = list(G.nodes())
    if not nodes:
        return 0.0
    if seed_node is None or seed_node not in G:
        seed_node = rng.choice(nodes)

    adopted = {seed_node}
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if node in adopted:
                continue
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue
            frac = sum(1 for nb in neighbors if nb in adopted) / len(neighbors)
            if frac >= phi:
                adopted.add(node)
                changed = True

    return len(adopted) / len(nodes)


def run_cascade_experiment(G_intact, G_removed, phi_values, n_trials=200):
    """Run Watts threshold model for each phi on intact and hub-removed networks."""
    results = {}

    for phi in phi_values:
        print(f"  φ={phi}: running {n_trials} trials on intact...", end=" ", flush=True)
        rng = random.Random(2603 + int(phi * 1000))
        cascade_intact = []
        for _ in range(n_trials):
            frac = watts_cascade(G_intact, phi, rng=rng)
            cascade_intact.append(frac)
        mean_intact = np.mean(cascade_intact)
        std_intact = np.std(cascade_intact)
        print(f"mean={mean_intact*100:.2f}%", end=" | ")

        rng2 = random.Random(2604 + int(phi * 1000))
        cascade_removed = []
        for _ in range(n_trials):
            frac = watts_cascade(G_removed, phi, rng=rng2)
            cascade_removed.append(frac)
        mean_removed = np.mean(cascade_removed)
        std_removed = np.std(cascade_removed)
        print(f"hub-removed mean={mean_removed*100:.2f}%")

        delta_pct = round((mean_removed - mean_intact) / (mean_intact + 1e-9) * 100, 1)

        results[str(phi)] = {
            "phi": phi,
            "n_trials": n_trials,
            "intact_mean_frac": round(float(mean_intact), 5),
            "intact_std_frac": round(float(std_intact), 5),
            "intact_mean_pct": round(float(mean_intact) * 100, 3),
            "hub_removed_mean_frac": round(float(mean_removed), 5),
            "hub_removed_std_frac": round(float(std_removed), 5),
            "hub_removed_mean_pct": round(float(mean_removed) * 100, 3),
            "delta_absolute_pct": round(float(mean_removed - mean_intact) * 100, 3),
            "delta_relative_pct": delta_pct,
        }

    return results


# ─────────────────────────────────────────────
# MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────

def analyze_network(G, name, cache_dir, n_perc_trials=200, n_cas_trials=200):
    """Full analysis for one real-world network."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {name}")
    print(f"{'='*60}")

    # Characterize
    info = characterize_network(G, name)

    # Hub removal (top 10% by degree)
    print(f"\n  Removing top 10% hubs...")
    G_removed, hub_nodes = remove_top_hubs(G, fraction=0.10)
    n_removed = len(hub_nodes)
    info_removed = {
        "n_hubs_removed": n_removed,
        "pct_removed": round(n_removed / G.number_of_nodes() * 100, 1),
        "G_removed_N": G_removed.number_of_nodes(),
        "G_removed_E": G_removed.number_of_edges(),
    }
    print(f"  Removed {n_removed} hubs → G_removed: N={G_removed.number_of_nodes()}, E={G_removed.number_of_edges()}")

    if G_removed.number_of_nodes() < 10:
        print("  [WARN] Hub-removed graph too small, skipping analysis")
        return None

    # Percolation experiment
    print(f"\n  --- PERCOLATION EXPERIMENT ---")
    perc_results = run_percolation_experiment(G, G_removed, n_trials=n_perc_trials)

    # Cascade experiment
    print(f"\n  --- CASCADE EXPERIMENT (Watts threshold model) ---")
    phi_values = [0.18, 0.22, 0.25, 0.30]
    cascade_results = run_cascade_experiment(G, G_removed, phi_values, n_trials=n_cas_trials)

    # Compile full results
    result = {
        "network": info,
        "hub_removal": info_removed,
        "percolation": perc_results,
        "cascade": cascade_results,
    }
    return result


# ─────────────────────────────────────────────
# SUMMARY COMPARISON
# ─────────────────────────────────────────────

def print_summary(all_results):
    """Print comparison table between real-world and BA synthetic results."""
    print("\n")
    print("=" * 70)
    print("SUMMARY: REAL-WORLD vs BA SYNTHETIC VALIDATION")
    print("=" * 70)
    print(f"\n{'Network':<25} {'κ':>6} {'p_c↑':>8} {'Δp_c%':>8} {'Cascade φ=0.22 intact':>22} {'hub-rmv':>10} {'Δ%':>8}")
    print("-" * 70)

    # BA reference
    print(f"{'BA synthetic (paper)':25} {'>>10':>6} {'0.34→0.90':>8} {'+165%':>8} {'0.29%':>22} {'20.6%':>10} {'+7007%':>8}")
    print("-" * 70)

    for net_name, res in all_results.items():
        if res is None:
            print(f"  {net_name}: FAILED")
            continue
        kappa = res["network"]["kappa"]
        p_c_i = res["percolation"]["p_c_intact"]
        p_c_r = res["percolation"]["p_c_hub_removed"]
        delta_pc = res["percolation"]["delta_pct"]
        phi022 = res["cascade"].get("0.22", {})
        cas_i = phi022.get("intact_mean_pct", 0)
        cas_r = phi022.get("hub_removed_mean_pct", 0)
        delta_cas = phi022.get("delta_relative_pct", 0)
        pc_str = f"{p_c_i:.2f}→{p_c_r:.2f}"
        cas_str = f"{cas_i:.2f}%"
        casr_str = f"{cas_r:.2f}%"
        delta_str = f"+{delta_cas:.0f}%" if delta_cas > 0 else f"{delta_cas:.0f}%"
        delta_pc_str = f"+{delta_pc:.0f}%" if delta_pc and delta_pc > 0 else f"{delta_pc:.0f}%" if delta_pc else "N/A"
        print(f"  {net_name[:23]:<23} {kappa:>6.1f} {pc_str:>8} {delta_pc_str:>8} {cas_str:>22} {casr_str:>10} {delta_str:>8}")

    print("\nKey: p_c = percolation threshold (higher = more fragile to random failures)")
    print("     Cascade = mean fraction of network adopting in Watts threshold model")
    print("     Hub removal = top 10% by degree\n")

    print("INTERPRETATION:")
    for net_name, res in all_results.items():
        if res is None:
            continue
        kappa = res["network"]["kappa"]
        sf_met = res["network"]["scale_free_threshold_met"]
        p_c_i = res["percolation"]["p_c_intact"]
        p_c_r = res["percolation"]["p_c_hub_removed"]
        delta_pc = res["percolation"]["delta_pct"]
        phi022 = res["cascade"].get("0.22", {})
        delta_cas = phi022.get("delta_relative_pct", 0)

        print(f"\n  [{net_name}] κ={kappa:.1f} {'✓ SCALE-FREE' if sf_met else '✗ homogeneous'}")
        if sf_met:
            if delta_pc and delta_pc > 50:
                print(f"    ✓ Percolation degraded: Δp_c=+{delta_pc:.0f}% (paper predicts: +165%)")
            elif delta_pc:
                print(f"    ~ Percolation partially degraded: Δp_c=+{delta_pc:.0f}%")
            if delta_cas > 50:
                print(f"    ✓ Cascade amplified at φ=0.22: Δ=+{delta_cas:.0f}% (paper predicts: +7007%)")
            elif delta_cas > 0:
                print(f"    ~ Cascade mildly amplified at φ=0.22: Δ=+{delta_cas:.0f}%")
            else:
                print(f"    ✗ Cascade NOT amplified at φ=0.22: Δ={delta_cas:.0f}%")
        else:
            print(f"    → Paper predicts minimal effect in homogeneous networks (κ<10)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    cache_dir = RESULTS_DIR
    all_results = {}

    print("=" * 70)
    print("REAL-WORLD NETWORK VALIDATION — arXiv:2603.04838")
    print("=" * 70)
    print(f"\nResults will be saved to: {RESULTS_DIR}")
    print(f"\nBA reference: p_c 0.34→0.90 (+165%), cascade φ=0.22: 0.29%→20.6%")

    # ── NETWORK 1: Bitcoin OTC ──────────────────────────────────────────
    print("\n[1/2] Loading Bitcoin OTC Trust Network...")
    try:
        G_btc = load_bitcoin_otc(cache_dir)
        res_btc = analyze_network(
            G_btc, "Bitcoin OTC", cache_dir,
            n_perc_trials=200, n_cas_trials=200
        )
        all_results["Bitcoin OTC"] = res_btc
    except Exception as e:
        print(f"  [ERROR] Bitcoin OTC analysis failed: {e}")
        import traceback; traceback.print_exc()
        all_results["Bitcoin OTC"] = None

    # ── NETWORK 2: AS-733 ───────────────────────────────────────────────
    print("\n[2/2] Loading AS-733 Autonomous Systems Network...")
    try:
        result = load_as733(cache_dir)
        if isinstance(result, tuple):
            G_as, as_label = result
        else:
            G_as = result
            as_label = "AS-733 Autonomous Systems"
        res_as = analyze_network(
            G_as, as_label, cache_dir,
            n_perc_trials=200, n_cas_trials=200
        )
        all_results[as_label] = res_as
    except Exception as e:
        print(f"  [ERROR] AS-733 analysis failed: {e}")
        import traceback; traceback.print_exc()
        all_results[as_label if 'as_label' in locals() else "AS-733"] = None

    # ── SAVE RESULTS ────────────────────────────────────────────────────
    output_path = os.path.join(RESULTS_DIR, "realworld_validation_results.json")
    output = {
        "metadata": {
            "paper": "arXiv:2603.04838",
            "date": time.strftime("%Y-%m-%d"),
            "methodology": {
                "hub_removal_fraction": 0.10,
                "n_percolation_trials": 200,
                "n_cascade_trials": 200,
                "cascade_model": "Watts threshold model",
                "phi_values_tested": [0.18, 0.22, 0.25, 0.30],
                "percolation_type": "bond percolation",
                "lcc_only": True,
            },
            "ba_reference": BA_REF,
        },
        "results": all_results,
    }

    # Custom JSON encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\n[saved] {output_path}")

    # ── PRINT SUMMARY ───────────────────────────────────────────────────
    print_summary(all_results)

    print("\n[DONE] Real-world validation complete.")
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()
