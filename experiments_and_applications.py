# ============================================================
# ADITYA's Module: experiments_and_applications.py
# Part 3: Applications, Experiments, Error Analysis & Visualization
# Paper: Halko, Martinsson, Tropp (2009) — Section 7
# Uses: randomized_svd_core.py (Aryan), srft_and_decompositions.py (Naman)
# ============================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import svd, qr
import time
import os

# ------------------------------------------------------------
# Import teammates' modules
# ------------------------------------------------------------
from randomized_svd_core import (
    randomized_svd,
    randomized_range_finder_power,
    error_estimator,
)
from srft_and_decompositions import srft_range_finder

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# HELPER: build a matrix with controlled singular value decay
# ============================================================
def _make_matrix(m, n, decay_exp, seed=42):
    """Create an m×n matrix whose singular values decay as 1/(i+1)^decay_exp."""
    rng = np.random.default_rng(seed)
    r = min(m, n)
    sigma = np.array([(1.0 / (i + 1)) ** decay_exp for i in range(r)])
    U_t, _ = np.linalg.qr(rng.standard_normal((m, r)))
    V_t, _ = np.linalg.qr(rng.standard_normal((n, r)))
    return (U_t * sigma) @ V_t.T, sigma


# ============================================================
# Experiment 1: Image Compression via Randomized SVD
# Reproduces Section 7 image-compression demonstration
# ============================================================
def experiment_image_compression(image_path=None, k_values=None):
    """
    Compress a grayscale image using Randomized SVD at multiple ranks k.
    If no image path supplied, a synthetic smooth image-like matrix is used.
    Produces: experiment_image_compression.png
    """
    if k_values is None:
        k_values = [5, 20, 50, 100]

    if image_path and os.path.exists(image_path):
        from PIL import Image
        A = np.array(Image.open(image_path).convert('L'), dtype=float)
        print(f"  Loaded image: {image_path}  shape={A.shape}")
    else:
        # Synthetic: outer-product + noise simulates texture gradients
        rng = np.random.default_rng(7)
        x = np.linspace(0, 4 * np.pi, 256)
        A = (np.outer(np.sin(x), np.cos(x))
             + 0.3 * np.outer(np.cos(2 * x), np.sin(3 * x))
             + 0.15 * rng.standard_normal((256, 256)))
        print("  Using synthetic 256×256 image-like matrix")

    n_plots = len(k_values) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4.5))
    axes[0].imshow(A, cmap='gray', aspect='auto')
    axes[0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    errors = []
    for i, k in enumerate(k_values):
        U, S, Vt = randomized_svd(A, k=k, p=10, q=2)
        A_approx = (U * S) @ Vt
        rel_err = np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro')
        compression_ratio = (k * (A.shape[0] + A.shape[1])) / (A.shape[0] * A.shape[1])
        errors.append(rel_err)
        axes[i + 1].imshow(np.clip(A_approx, A.min(), A.max()), cmap='gray', aspect='auto')
        axes[i + 1].set_title(f'k={k}\nerr={rel_err:.3f}\nCR={compression_ratio:.2%}',
                               fontsize=9)
        axes[i + 1].axis('off')

    fig.suptitle('Randomized SVD Image Compression (Section 7)\n'
                 'CR = stored values / original pixels', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'experiment_image_compression.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out}")
    return errors


# ============================================================
# Experiment 2: Oversampling Effect — Theorem 1.1 Verification
# Eq. 1.8: E‖A − QQ^T A‖₂ → σ_{k+1} as p → ∞
# ============================================================
def experiment_oversampling_effect(m=500, n=400, k=20, q=2, n_trials=25,
                                   p_range=None):
    """
    Verifies Theorem 1.1: expected error ‖A − QQ^T A‖₂ vs oversampling p.
    For each p, runs n_trials trials and plots mean ± 1σ band.
    Produces: experiment_oversampling.png
    """
    if p_range is None:
        p_range = range(0, 51, 5)

    A, sigma = _make_matrix(m, n, decay_exp=1.0)
    sigma_k1 = sigma[k]          # theoretical lower bound = σ_{k+1}
    sigma_k  = sigma[k - 1]      # σ_k (last captured)

    p_list = list(p_range)
    means, stds = [], []

    print(f"  Running oversampling experiment: m={m}, n={n}, k={k}, q={q} …")
    for p in p_list:
        l = k + p
        if l > min(m, n):
            means.append(np.nan); stds.append(np.nan)
            continue
        errs = []
        for trial in range(n_trials):
            Q = randomized_range_finder_power(A, l=l, q=q, random_state=trial)
            err = np.linalg.norm(A - Q @ (Q.T @ A), ord=2)
            errs.append(err)
        means.append(float(np.mean(errs)))
        stds.append(float(np.std(errs)))

    p_arr = np.array(p_list[:len(means)])
    m_arr = np.array(means)
    s_arr = np.array(stds)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(p_arr, m_arr - s_arr, m_arr + s_arr, alpha=0.25,
                    color='steelblue', label='±1 std')
    ax.plot(p_arr, m_arr, 'o-', color='steelblue', linewidth=2, label='Mean error')
    ax.axhline(sigma_k1, color='crimson', linestyle='--', linewidth=1.8,
               label=f'σ_{{k+1}} = {sigma_k1:.4f}  (Thm 1.1 lower bound)')
    ax.axhline(sigma_k,  color='orange',  linestyle=':',  linewidth=1.5,
               label=f'σ_k = {sigma_k:.4f}')
    ax.set_xlabel('Oversampling parameter  p', fontsize=12)
    ax.set_ylabel('‖A − QQᵀA‖₂', fontsize=12)
    ax.set_title(f'Oversampling Effect — Theorem 1.1\n'
                 f'k={k}, q={q},  m×n={m}×{n},  decay 1/i', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'experiment_oversampling.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out}")
    print(f"    σ_{{k+1}} = {sigma_k1:.6f}  |  best mean error = {min(m for m in means if not np.isnan(m)):.6f}")


# ============================================================
# Experiment 3: Power Iteration Effect — Theorem 1.2 Verification
# ============================================================
def experiment_power_iteration(m=400, n=300, k=20, p=10, n_trials=10):
    """
    Verifies Theorem 1.2: power iteration q reduces error on matrices with
    slowly-decaying spectra.  Compares q = 0,1,2,3 for fast vs flat decay.
    Produces: experiment_power_iteration.png
    """
    q_vals = [0, 1, 2, 3]
    decay_configs = [
        (0.5, 'Flat spectrum  (decay ∝ 1/√i)'),
        (2.0, 'Moderate spectrum  (decay ∝ 1/i²)'),
        (5.0, 'Fast spectrum  (decay ∝ 1/i⁵)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (decay_exp, label) in zip(axes, decay_configs):
        q_means = []
        A, sigma = _make_matrix(m, n, decay_exp=decay_exp)
        sigma_k1 = sigma[k]

        for q in q_vals:
            errs = []
            for trial in range(n_trials):
                Q = randomized_range_finder_power(A, l=k + p, q=q, random_state=trial)
                err = np.linalg.norm(A - Q @ (Q.T @ A), ord=2)
                errs.append(err)
            q_means.append(np.mean(errs))

        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
        bars = ax.bar([f'q={q}' for q in q_vals], q_means, color=colors, edgecolor='k',
                      linewidth=0.8)
        ax.axhline(sigma_k1, color='crimson', linestyle='--', linewidth=1.8,
                   label=f'σ_{{k+1}} = {sigma_k1:.4f}')
        for bar, val in zip(bars, q_means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlabel('Power iterations  q', fontsize=10)
        ax.set_ylabel('‖A − QQᵀA‖₂', fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Power Iteration Effect — Theorem 1.2  (k=20, p=10)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'experiment_power_iteration.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out}")


# ============================================================
# Experiment 4: PCA via Randomized SVD
# Reproduces Section 7.1 / 7.3 low-rank-structure experiment
# ============================================================
def experiment_pca(n_samples=2000, n_features=500, true_rank=10):
    """
    PCA on a synthetic low-rank + noise dataset.
    Compares Classical SVD vs Randomized SVD via explained-variance curves
    and reconstructed eigenvalue spectra.
    Produces: experiment_pca.png
    """
    rng = np.random.default_rng(5)
    W = rng.standard_normal((n_features, true_rank))
    H = rng.standard_normal((true_rank, n_samples))
    X = W @ H + 0.1 * rng.standard_normal((n_features, n_samples))
    X -= X.mean(axis=1, keepdims=True)   # center

    # Classical PCA
    t0 = time.time()
    _, S_c, _ = svd(X, full_matrices=False)
    t_c = time.time() - t0

    k_pca = true_rank + 5
    # Randomized PCA
    t0 = time.time()
    _, S_r, _ = randomized_svd(X, k=k_pca, p=10, q=2)
    t_r = time.time() - t0

    total_var = np.sum(S_c ** 2)
    ev_c  = np.cumsum(S_c[:20] ** 2) / total_var
    ev_r  = np.cumsum(S_r ** 2)      / total_var
    comps = np.arange(1, 21)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # -- Explained variance --
    ax1.plot(comps, ev_c, color='b', marker='o', linestyle='-', linewidth=2, markersize=5, label=f'Classical SVD  ({t_c:.2f}s)')
    ax1.plot(range(1, len(ev_r)+1), ev_r, color='r', marker='s', linestyle='--', linewidth=2, markersize=5,
             label=f'Randomized SVD  ({t_r:.2f}s)')
    ax1.axvline(true_rank, color='gray', linestyle=':', label=f'True rank={true_rank}')
    ax1.set_xlabel('# components', fontsize=11)
    ax1.set_ylabel('Cumulative explained variance', fontsize=11)
    ax1.set_title('PCA: Explained Variance Curve', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.4)

    # -- Singular value spectrum --
    ax2.semilogy(comps, S_c[:20], color='b', marker='o', linestyle='-', linewidth=2, markersize=5, label='Classical SVD')
    ax2.semilogy(range(1, len(S_r)+1), S_r, color='r', marker='s', linestyle='--', linewidth=2, markersize=5,
                 label='Randomized SVD')
    ax2.axvline(true_rank, color='gray', linestyle=':', label=f'True rank={true_rank}')
    ax2.set_xlabel('Component index', fontsize=11)
    ax2.set_ylabel('Singular value  (log scale)', fontsize=11)
    ax2.set_title('Singular Value Spectrum', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.4, which='both')

    fig.suptitle(f'PCA via Randomized SVD  —  {n_features}×{n_samples} matrix, true rank={true_rank}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'experiment_pca.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out}")
    print(f"    Classical SVD: {t_c:.3f}s  |  Randomized SVD: {t_r:.3f}s  "
          f"(speedup ×{t_c / max(t_r, 1e-9):.1f})")


# ============================================================
# Experiment 5: Final Dashboard — Classical vs Randomized vs SRFT
# Benchmarks across matrix sizes: time, error, singular-value accuracy
# ============================================================
def experiment_final_dashboard(k=30, p=10, q=2,
                               sizes=None):
    """
    Benchmark Classical SVD vs Randomized SVD vs SRFT range finder
    across various matrix sizes.
    Plots: time vs size, relative error vs size, error vs k.
    Produces: experiment_dashboard.png
    """
    if sizes is None:
        sizes = [200, 400, 600, 800, 1000, 1200]

    times_classical, times_rand, times_srft = [], [], []
    errs_classical, errs_rand, errs_srft = [], [], []

    print(f"  Running dashboard benchmark — k={k}, p={p}, q={q}")
    for sz in sizes:
        m, n = sz, max(sz - 50, k + p + 10)
        rng = np.random.default_rng(0)
        # Reproducible low-rank + noise matrix
        Ut = rng.standard_normal((m, k))
        Vt = rng.standard_normal((n, k))
        A  = Ut @ Vt.T + 0.01 * rng.standard_normal((m, n))
        l  = k + p

        # --- Classical SVD ---
        t0 = time.perf_counter()
        Uc, Sc, Vtc = svd(A, full_matrices=False)
        t_c = time.perf_counter() - t0
        Ac  = (Uc[:, :k] * Sc[:k]) @ Vtc[:k, :]
        ec  = np.linalg.norm(A - Ac) / np.linalg.norm(A)
        times_classical.append(t_c); errs_classical.append(ec)

        # --- Randomized SVD ---
        t0 = time.perf_counter()
        Ur, Sr, Vtr = randomized_svd(A, k=k, p=p, q=q)
        t_r = time.perf_counter() - t0
        Ar  = (Ur * Sr) @ Vtr
        er  = np.linalg.norm(A - Ar) / np.linalg.norm(A)
        times_rand.append(t_r); errs_rand.append(er)

        # --- SRFT (range finder only, then Stage B) ---
        t0 = time.perf_counter()
        Q_s = srft_range_finder(A, l=l)
        from randomized_svd_core import stage_b_svd
        Us, Ss, Vts = stage_b_svd(A, Q_s)
        t_s = time.perf_counter() - t0
        As  = (Us[:, :k] * Ss[:k]) @ Vts[:k, :]
        es  = np.linalg.norm(A - As) / np.linalg.norm(A)
        times_srft.append(t_s); errs_srft.append(es)

        print(f"    {m}×{n}:  Classical {t_c:.3f}s ({ec:.2e})  "
              f"Rand {t_r:.3f}s ({er:.2e})  SRFT {t_s:.3f}s ({es:.2e})")

    # ---- Final error-vs-rank experiment ----
    rank_list = list(range(5, 55, 5))
    A_fixed, sigma_fixed = _make_matrix(600, 500, decay_exp=2.0, seed=99)
    sv_errors_rand, sv_errors_srft, sv_true = [], [], []
    for kr in rank_list:
        Ur, Sr, _ = randomized_svd(A_fixed, k=kr, p=p, q=q)
        _, Sc_f, _ = svd(A_fixed, full_matrices=False)
        sv_errors_rand.append(abs(Sr[0] - Sc_f[0]) / Sc_f[0])
        Q_s = srft_range_finder(A_fixed, l=kr + p)
        _, Ss_f, _ = stage_b_svd(A_fixed, Q_s)
        sv_errors_srft.append(abs(Ss_f[0] - Sc_f[0]) / Sc_f[0])
        sv_true.append(sigma_fixed[kr - 1])

    # ---- Plot dashboard ----
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    label_c    = 'Classical SVD'
    label_r    = 'Randomized SVD'
    label_s    = 'SRFT + Stage-B'
    colors     = {'c': '#1f77b4', 'r': '#2ca02c', 's': '#d62728'}
    markers    = {'c': 'o', 'r': 's', 's': '^'}

    # (0,0) Time vs matrix size
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.plot(sizes, times_classical, color=colors['c'], marker=markers['c'], linestyle='-', label=label_c, linewidth=2)
    ax00.plot(sizes, times_rand,      color=colors['r'], marker=markers['r'], linestyle='-', label=label_r, linewidth=2)
    ax00.plot(sizes, times_srft,      color=colors['s'], marker=markers['s'], linestyle='-', label=label_s, linewidth=2)
    ax00.set_xlabel('Matrix size (m=n±50)', fontsize=10)
    ax00.set_ylabel('Wall time (s)', fontsize=10)
    ax00.set_title('🕒  Time vs Matrix Size', fontsize=11, fontweight='bold')
    ax00.legend(fontsize=8); ax00.grid(True, alpha=0.4)

    # (0,1) Relative error vs matrix size
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.semilogy(sizes, errs_classical, color=colors['c'], marker=markers['c'], linestyle='-', label=label_c, linewidth=2)
    ax01.semilogy(sizes, errs_rand,      color=colors['r'], marker=markers['r'], linestyle='-', label=label_r, linewidth=2)
    ax01.semilogy(sizes, errs_srft,      color=colors['s'], marker=markers['s'], linestyle='-', label=label_s, linewidth=2)
    ax01.set_xlabel('Matrix size (m=n±50)', fontsize=10)
    ax01.set_ylabel('Relative error ‖A−Ã‖/‖A‖  (log)', fontsize=10)
    ax01.set_title('📈  Rel. Error vs Matrix Size', fontsize=11, fontweight='bold')
    ax01.legend(fontsize=8); ax01.grid(True, alpha=0.4, which='both')

    # (0,2) Speedup ratio vs matrix size
    ax02 = fig.add_subplot(gs[0, 2])
    speedup_r = [c / r for c, r in zip(times_classical, times_rand)]
    speedup_s = [c / s for c, s in zip(times_classical, times_srft)]
    ax02.plot(sizes, speedup_r, color=colors['r'], marker=markers['r'], linestyle='-', label=label_r + ' speedup', linewidth=2)
    ax02.plot(sizes, speedup_s, color=colors['s'], marker=markers['s'], linestyle='-', label=label_s + ' speedup', linewidth=2)
    ax02.axhline(1.0, color='k', linestyle='--', linewidth=1)
    ax02.set_xlabel('Matrix size', fontsize=10)
    ax02.set_ylabel('Speedup factor  (×)', fontsize=10)
    ax02.set_title('🚀  Speedup vs Classical SVD', fontsize=11, fontweight='bold')
    ax02.legend(fontsize=8); ax02.grid(True, alpha=0.4)

    # (1,0) Relative top singular value error vs k
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.semilogy(rank_list, sv_errors_rand, color=colors['r'], marker=markers['r'], linestyle='-', label=label_r, linewidth=2)
    ax10.semilogy(rank_list, sv_errors_srft, color=colors['s'], marker=markers['s'], linestyle='-', label=label_s, linewidth=2)
    ax10.set_xlabel('Target rank  k', fontsize=10)
    ax10.set_ylabel('Rel. error in σ₁ (log)', fontsize=10)
    ax10.set_title('🎯  Singular Value Accuracy vs k', fontsize=11, fontweight='bold')
    ax10.legend(fontsize=8); ax10.grid(True, alpha=0.4, which='both')

    # (1,1) Bar chart: method comparison at largest size
    ax11 = fig.add_subplot(gs[1, 1])
    methods = [label_c, label_r, label_s]
    t_vals  = [times_classical[-1], times_rand[-1], times_srft[-1]]
    e_vals  = [errs_classical[-1], errs_rand[-1],  errs_srft[-1]]
    x_pos   = np.arange(len(methods))
    bar_cols = [colors['c'], colors['r'], colors['s']]
    bars = ax11.bar(x_pos, t_vals, color=bar_cols, edgecolor='k', linewidth=0.8, width=0.5)
    ax11_r = ax11.twinx()
    ax11_r.plot(x_pos, e_vals, 'k^--', markersize=8, linewidth=1.5, label='Error (right)')
    ax11.set_xticks(x_pos)
    ax11.set_xticklabels(['Classical', 'Rand SVD', 'SRFT'], fontsize=9)
    ax11.set_ylabel('Time (s)', fontsize=10, color='k')
    ax11_r.set_ylabel('Rel. Error', fontsize=10, color='k')
    ax11.set_title(f'📊  Summary at size {sizes[-1]}×{sizes[-1]-50}', fontsize=11, fontweight='bold')
    for bar, val in zip(bars, t_vals):
        ax11.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                  f'{val:.3f}s', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # (1,2) Singular value spectrum comparison
    ax12 = fig.add_subplot(gs[1, 2])
    _, Sc_f, _ = svd(A_fixed, full_matrices=False)
    _, Sr_f, _ = randomized_svd(A_fixed, k=40, p=p, q=q)
    Q_sf = srft_range_finder(A_fixed, l=40 + p)
    _, Ss_f, _ = stage_b_svd(A_fixed, Q_sf)
    idx_c = np.arange(1, 41)
    ax12.semilogy(idx_c,             Sc_f[:40], color=colors['c'], marker=markers['c'], linestyle='-', label=label_c,    linewidth=2)
    ax12.semilogy(np.arange(1, 41),  Sr_f,      color=colors['r'], marker=markers['r'], linestyle='--', label=label_r,  linewidth=2)
    ax12.semilogy(np.arange(1, 41),  Ss_f[:40], color=colors['s'], marker=markers['s'], linestyle=':', label=label_s,   linewidth=2)
    ax12.set_xlabel('Singular value index', fontsize=10)
    ax12.set_ylabel('σᵢ  (log scale)', fontsize=10)
    ax12.set_title('🔢  Singular Value Spectra', fontsize=11, fontweight='bold')
    ax12.legend(fontsize=8); ax12.grid(True, alpha=0.4, which='both')

    fig.suptitle('Final Benchmark Dashboard — Classical SVD vs Randomized SVD vs SRFT\n'
                 'Halko–Martinsson–Tropp (2009)',
                 fontsize=14, fontweight='bold')
    out = os.path.join(OUTPUT_DIR, 'experiment_dashboard.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out}")


# ============================================================
# Entry Point — run all experiments
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ADITYA — LAAO Mini Project: HMT 2009, Part 3")
    print("=" * 60)

    print("\n[1/5] Image Compression Experiment …")
    experiment_image_compression()

    print("\n[2/5] Oversampling Effect — Theorem 1.1 …")
    experiment_oversampling_effect()

    print("\n[3/5] Power Iteration Effect — Theorem 1.2 …")
    experiment_power_iteration()

    print("\n[4/5] PCA via Randomized SVD …")
    experiment_pca()

    print("\n[5/5] Final Dashboard Benchmark …")
    experiment_final_dashboard()

    print("\n" + "=" * 60)
    print("All experiments complete.  Output PNGs saved to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)
