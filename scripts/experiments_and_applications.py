# ============================================================
# ADITYA's Module: experiments_and_applications.py
# Part 3: Applications, Experiments, Error Analysis & Visualization
# Paper: Halko, Martinsson, Tropp (2009) — Section 7
# Uses: randomized_svd_core.py (Aryan), srft_and_decompositions.py (Naman)
# ============================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI window nahi chahiye — directly file mein save karo
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import svd, qr
import time
import os

# ------------------------------------------------------------
# Import teammates' modules
# Aryan ka randomized_svd and Naman ka srft yahan use hoga
# ------------------------------------------------------------
from randomized_svd_core import (
    randomized_svd,
    randomized_range_finder_power,
    error_estimator,
)
from srft_and_decompositions import srft_range_finder

# output PNGs isi folder mein save honge jahan ye script hai
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# HELPER: controlled singular value decay wali matrix banana
# ============================================================
def _make_matrix(m, n, decay_exp, seed=42):
    """
    Ek custom matrix banao jiske exact singular values hume pata hain.
    Power iteration experiment ke liye zaroori hai — hume "flat" aur "fast" dono
    spectra test karne hain, toh khud banao controlled environment mein.
    """
    rng = np.random.default_rng(seed)
    r = min(m, n)

    # sigma array manually set karo — decay_exp control karta hai kitni jaldi girta hai
    # decay_exp = 0.5 → (1, 0.7, 0.5, 0.4...) — bohot slow, flat spectrum
    # decay_exp = 5.0 → (1, 0.03, 0.003...) — fast, most info top 2-3 components mein
    sigma = np.array([(1.0 / (i + 1)) ** decay_exp for i in range(r)])

    # random orthogonal matrices U aur V — basically random "directions"
    # qr(random) gives us a proper unitary matrix — standard trick
    U_t, _ = np.linalg.qr(rng.standard_normal((m, r)))
    V_t, _ = np.linalg.qr(rng.standard_normal((n, r)))

    # A = U * diag(sigma) * V^T — yahi SVD structure hai, hum ise construct kar rahe hain
    # U_t * sigma means multiply each column of U_t by corresponding sigma value (broadcasting)
    return (U_t * sigma) @ V_t.T, sigma


# ============================================================
# Experiment 1: Image Compression via Randomized SVD
# Reproduces Section 7 image-compression demonstration
# ============================================================
def experiment_image_compression(image_path=None, k_values=None):
    """
    Randomized SVD se image compress karo at different ranks k.
    Intuition: ek 1080p image ka mostly saara structure top 50-100 singular vectors mein hota hai.
    Baaki sab noise/fine details hain jo compress karke remove ho jaate hain.
    """
    if k_values is None:
        k_values = [5, 20, 50, 100]

    if image_path and os.path.exists(image_path):
        from PIL import Image
        # grayscale mein convert kiya — 2D matrix mil jaati hai, easy for SVD
        A = np.array(Image.open(image_path).convert('L'), dtype=float)
        print(f"  Loaded image: {image_path}  shape={A.shape}")
    else:
        # actual image nahi hai toh synthetic banao — sin/cos outer products
        # outer product se low-rank "texture" bands milti hain jaisi real images mein hoti hain
        # phir thoda Gaussian noise add kiya realism ke liye
        rng = np.random.default_rng(7)
        x = np.linspace(0, 4 * np.pi, 256)
        A = (np.outer(np.sin(x), np.cos(x))
             + 0.3 * np.outer(np.cos(2 * x), np.sin(3 * x))
             + 0.15 * rng.standard_normal((256, 256)))
        print("  Using synthetic 256×256 image-like matrix")

    n_plots = len(k_values) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4.5))

    # pehla plot = original image
    axes[0].imshow(A, cmap='gray', aspect='auto')
    axes[0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    errors = []
    for i, k in enumerate(k_values):
        # Aryan ka pipeline yahan call kar rahe hain
        U, S, Vt = randomized_svd(A, k=k, p=10, q=2)

        # A_approx = U @ diag(S) @ Vt — ye approximate image hai
        # (U * S) matlab element-wise multiply each column of U by corresponding S value
        A_approx = (U * S) @ Vt

        # Frobenius norm = root of sum of squares of all elements
        # relative error = kitna fraction quality lost hua original se
        rel_err = np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro')

        # compression ratio = kitne numbers store kiye / original total
        # k components ke liye: U has k cols (m*k), V has k rows (k*n) → total = k*(m+n)
        # vs original m*n — ye ratio batata hai kitna compress hua
        compression_ratio = (k * (A.shape[0] + A.shape[1])) / (A.shape[0] * A.shape[1])
        errors.append(rel_err)

        # np.clip zaroori hai kyunki float approximation mein kuch values range se bahar ja sakti hain
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
    Theorem 1.1 prove karna empirically — oversampling p badhane ka effect dekhna.
    Paper ka claim: bas 5-10 extra random vectors lene se error sigma_{k+1} ke paas aa jaata hai.
    Ye 'sigma_{k+1}' ek lower bound hai — Eckart-Young theorem se iska pata hai.
    """
    if p_range is None:
        p_range = range(0, 51, 5)

    # decay_exp=1.0 → sigma_i = 1/i — moderately decaying spectrum
    A, sigma = _make_matrix(m, n, decay_exp=1.0)

    # sigma[k] = sigma_{k+1} in 0-indexed — ye woh value hai jo best possible error define karta hai
    # Eckart-Young theorem: koi bhi rank-k approximation se error >= sigma_{k+1} hoga
    # toh ye humara "floor" hai — isse neeche nahi ja sakte chahe koi bhi algorithm use karo
    sigma_k1 = sigma[k]    # theoretical lower bound = σ_{k+1}
    sigma_k  = sigma[k - 1]  # σ_k (last captured singular value)

    p_list = list(p_range)
    means, stds = [], []

    print(f"  Running oversampling experiment: m={m}, n={n}, k={k}, q={q} …")
    for p in p_list:
        l = k + p
        if l > min(m, n):  # rank exceed nahi karna chahiye matrix dimensions se
            means.append(np.nan); stds.append(np.nan)
            continue
        errs = []
        for trial in range(n_trials):
            # different random seed per trial taaki average meaningful ho
            Q = randomized_range_finder_power(A, l=l, q=q, random_state=trial)

            # actual error = ||A - QQ^T A||_2 (spectral norm)
            # QQ^T A = projection of A onto column space of Q
            # jo bacha (A - QQ^T A) woh Q mein capture nahi hua = error
            err = np.linalg.norm(A - Q @ (Q.T @ A), ord=2)
            errs.append(err)

        means.append(float(np.mean(errs)))
        stds.append(float(np.std(errs)))  # spread of trials — randomness ka effect

    p_arr = np.array(p_list[:len(means)])
    m_arr = np.array(means)
    s_arr = np.array(stds)

    fig, ax = plt.subplots(figsize=(8, 5))
    # shaded band = ±1 standard deviation across trials
    ax.fill_between(p_arr, m_arr - s_arr, m_arr + s_arr, alpha=0.25,
                    color='steelblue', label='±1 std')
    ax.plot(p_arr, m_arr, 'o-', color='steelblue', linewidth=2, label='Mean error')

    # ye red line = theoretical minimum — hum dekhenge ki humara error iske kitna close aata hai
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
    Theorem 1.2: power iterations (q) ka effect on accuracy.
    Flat spectrum matrices pe q=0 fail hota hai — q=2 easily fix kar deta hai.
    Humne teen types ke spectra test kiye: flat, moderate, fast.
    """
    q_vals = [0, 1, 2, 3]
    decay_configs = [
        (0.5, 'Flat spectrum  (decay ∝ 1/√i)'),   # random SVD ke liye worst case
        (2.0, 'Moderate spectrum  (decay ∝ 1/i²)'),
        (5.0, 'Fast spectrum  (decay ∝ 1/i⁵)'),   # random SVD ke liye best case
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (decay_exp, label) in zip(axes, decay_configs):
        q_means = []
        A, sigma = _make_matrix(m, n, decay_exp=decay_exp)
        sigma_k1 = sigma[k]  # optimal target for this matrix

        for q in q_vals:
            errs = []
            for trial in range(n_trials):
                # exactly same as Aryan's code — power iteration steps vary karein
                Q = randomized_range_finder_power(A, l=k + p, q=q, random_state=trial)
                err = np.linalg.norm(A - Q @ (Q.T @ A), ord=2)
                errs.append(err)
            q_means.append(np.mean(errs))  # mean across trials

        # bar chart — x-axis = q values, y-axis = mean error
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
        bars = ax.bar([f'q={q}' for q in q_vals], q_means, color=colors, edgecolor='k',
                      linewidth=0.8)

        # red line = optimal bound — bars should approach this as q increases
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
    PCA = Principal Component Analysis.
    Internally ye SVD hi hai — aur sklearn ka TruncatedSVD exactly Aryan ka algorithm use karta hai.
    Hum yahan prove kar rahe hain ki randomized SVD = classical SVD ke almost same results deta hai
    lekin kaafi faster.
    """
    rng = np.random.default_rng(5)

    # synthetic dataset: W @ H = low rank signal (actual information)
    # ye imagine karo jaise 500 features hain but only 10 underlying patterns/topics
    W = rng.standard_normal((n_features, true_rank))   # feature patterns (500 x 10)
    H = rng.standard_normal((true_rank, n_samples))    # sample weights (10 x 2000)
    X = W @ H + 0.1 * rng.standard_normal((n_features, n_samples))  # + noise

    # centering — PCA ke liye zaroori hai, mean 0 pe laana
    # axis=1 means rowwise mean nikala (per feature across all samples)
    X -= X.mean(axis=1, keepdims=True)

    # Classical PCA
    t0 = time.time()
    _, S_c, _ = svd(X, full_matrices=False)
    t_c = time.time() - t0

    k_pca = true_rank + 5  # k = true rank se thoda zyada taaki saara signal capture ho

    # Randomized PCA — Aryan ka algorithm
    t0 = time.time()
    _, S_r, _ = randomized_svd(X, k=k_pca, p=10, q=2)
    t_r = time.time() - t0

    # total variance = sum of squared singular values (by definition of Frobenius norm)
    total_var = np.sum(S_c ** 2)

    # cumulative explained variance — kitne components mein kitna % information hai
    # agar true_rank = 10 pe elbow aata hai toh matlab humne sahi rank identify kiya
    ev_c = np.cumsum(S_c[:20] ** 2) / total_var
    ev_r = np.cumsum(S_r ** 2)      / total_var
    comps = np.arange(1, 21)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # -- Explained variance curve (elbow should be at true_rank) --
    ax1.plot(comps, ev_c, color='b', marker='o', linestyle='-', linewidth=2, markersize=5,
             label=f'Classical SVD  ({t_c:.2f}s)')
    ax1.plot(range(1, len(ev_r)+1), ev_r, color='r', marker='s', linestyle='--', linewidth=2,
             markersize=5, label=f'Randomized SVD  ({t_r:.2f}s)')
    ax1.axvline(true_rank, color='gray', linestyle=':', label=f'True rank={true_rank}')
    ax1.set_xlabel('# components', fontsize=11)
    ax1.set_ylabel('Cumulative explained variance', fontsize=11)
    ax1.set_title('PCA: Explained Variance Curve', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.4)

    # -- Singular value spectrum (log scale) --
    # log scale isliye kyunki singular values exponentially drop hoti hain
    # agar dono lines overlap karti hain toh prove hua ki randomized = classical yahan
    ax2.semilogy(comps, S_c[:20], color='b', marker='o', linestyle='-', linewidth=2,
                 markersize=5, label='Classical SVD')
    ax2.semilogy(range(1, len(S_r)+1), S_r, color='r', marker='s', linestyle='--',
                 linewidth=2, markersize=5, label='Randomized SVD')
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
# Experiment 5: Final Dashboard — three methods head-to-head
# Classical SVD vs Randomized SVD vs SRFT across matrix sizes
# ============================================================
def experiment_final_dashboard(k=30, p=10, q=2, sizes=None):
    """
    Sab experiments ka summary dashboard — real scaling behavior dekhna.
    As matrix grows: Classical SVD O(mn^2) → slow, Randomized O(mnk) → scales, SRFT → fastest.
    """
    if sizes is None:
        sizes = [200, 400, 600, 800, 1000, 1200]

    times_classical, times_rand, times_srft = [], [], []
    errs_classical, errs_rand, errs_srft   = [], [], []

    print(f"  Running dashboard benchmark — k={k}, p={p}, q={q}")
    for sz in sizes:
        m, n = sz, max(sz - 50, k + p + 10)
        rng = np.random.default_rng(0)

        # reproducible test matrix (same seed every size so comparison is fair)
        Ut = rng.standard_normal((m, k))
        Vt = rng.standard_normal((n, k))
        A  = Ut @ Vt.T + 0.01 * rng.standard_normal((m, n))
        l  = k + p

        # --- Classical SVD ---
        t0 = time.perf_counter()  # perf_counter better than time() for short intervals
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
        # SRFT gives us Q (Stage A), Aryan's stage_b_svd gives us U,S,Vt from Q
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
    # ek fixed matrix pe rank vary karo — dekhna hai accuracy kaise improve hoti hai k badhane pe
    rank_list = list(range(5, 55, 5))
    A_fixed, sigma_fixed = _make_matrix(600, 500, decay_exp=2.0, seed=99)
    sv_errors_rand, sv_errors_srft, sv_true = [], [], []
    for kr in rank_list:
        Ur, Sr, _ = randomized_svd(A_fixed, k=kr, p=p, q=q)
        _, Sc_f, _ = svd(A_fixed, full_matrices=False)
        # top singular value relative error — sigma_1 accurate hai toh baaki bhi hoga
        sv_errors_rand.append(abs(Sr[0] - Sc_f[0]) / Sc_f[0])
        Q_s = srft_range_finder(A_fixed, l=kr + p)
        _, Ss_f, _ = stage_b_svd(A_fixed, Q_s)
        sv_errors_srft.append(abs(Ss_f[0] - Sc_f[0]) / Sc_f[0])
        sv_true.append(sigma_fixed[kr - 1])

    # ---- Plot dashboard ----
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    label_c = 'Classical SVD'
    label_r = 'Randomized SVD'
    label_s = 'SRFT + Stage-B'
    colors  = {'c': '#1f77b4', 'r': '#2ca02c', 's': '#d62728'}
    markers = {'c': 'o', 'r': 's', 's': '^'}

    # (0,0) Time vs matrix size
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.plot(sizes, times_classical, color=colors['c'], marker=markers['c'], linestyle='-', label=label_c, linewidth=2)
    ax00.plot(sizes, times_rand,      color=colors['r'], marker=markers['r'], linestyle='-', label=label_r, linewidth=2)
    ax00.plot(sizes, times_srft,      color=colors['s'], marker=markers['s'], linestyle='-', label=label_s, linewidth=2)
    ax00.set_xlabel('Matrix size (m=n±50)', fontsize=10)
    ax00.set_ylabel('Wall time (s)', fontsize=10)
    ax00.set_title('Time vs Matrix Size', fontsize=11, fontweight='bold')
    ax00.legend(fontsize=8); ax00.grid(True, alpha=0.4)

    # (0,1) Relative error vs matrix size
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.semilogy(sizes, errs_classical, color=colors['c'], marker=markers['c'], linestyle='-', label=label_c, linewidth=2)
    ax01.semilogy(sizes, errs_rand,      color=colors['r'], marker=markers['r'], linestyle='-', label=label_r, linewidth=2)
    ax01.semilogy(sizes, errs_srft,      color=colors['s'], marker=markers['s'], linestyle='-', label=label_s, linewidth=2)
    ax01.set_xlabel('Matrix size (m=n±50)', fontsize=10)
    ax01.set_ylabel('Relative error ‖A−Ã‖/‖A‖  (log)', fontsize=10)
    ax01.set_title('Rel. Error vs Matrix Size', fontsize=11, fontweight='bold')
    ax01.legend(fontsize=8); ax01.grid(True, alpha=0.4, which='both')

    # (0,2) Speedup ratio vs matrix size — 1.0 line = classical ke barabar, usse upar = faster
    ax02 = fig.add_subplot(gs[0, 2])
    speedup_r = [c / r for c, r in zip(times_classical, times_rand)]
    speedup_s = [c / s for c, s in zip(times_classical, times_srft)]
    ax02.plot(sizes, speedup_r, color=colors['r'], marker=markers['r'], linestyle='-', label=label_r + ' speedup', linewidth=2)
    ax02.plot(sizes, speedup_s, color=colors['s'], marker=markers['s'], linestyle='-', label=label_s + ' speedup', linewidth=2)
    ax02.axhline(1.0, color='k', linestyle='--', linewidth=1)  # baseline = same as classical
    ax02.set_xlabel('Matrix size', fontsize=10)
    ax02.set_ylabel('Speedup factor  (×)', fontsize=10)
    ax02.set_title('Speedup vs Classical SVD', fontsize=11, fontweight='bold')
    ax02.legend(fontsize=8); ax02.grid(True, alpha=0.4)

    # (1,0) Relative top singular value error vs k
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.semilogy(rank_list, sv_errors_rand, color=colors['r'], marker=markers['r'], linestyle='-', label=label_r, linewidth=2)
    ax10.semilogy(rank_list, sv_errors_srft, color=colors['s'], marker=markers['s'], linestyle='-', label=label_s, linewidth=2)
    ax10.set_xlabel('Target rank  k', fontsize=10)
    ax10.set_ylabel('Rel. error in σ₁ (log)', fontsize=10)
    ax10.set_title('Singular Value Accuracy vs k', fontsize=11, fontweight='bold')
    ax10.legend(fontsize=8); ax10.grid(True, alpha=0.4, which='both')

    # (1,1) Bar chart: side-by-side comparison at largest size
    ax11 = fig.add_subplot(gs[1, 1])
    methods = [label_c, label_r, label_s]
    t_vals  = [times_classical[-1], times_rand[-1], times_srft[-1]]
    e_vals  = [errs_classical[-1],  errs_rand[-1],  errs_srft[-1]]
    x_pos   = np.arange(len(methods))
    bar_cols = [colors['c'], colors['r'], colors['s']]
    bars = ax11.bar(x_pos, t_vals, color=bar_cols, edgecolor='k', linewidth=0.8, width=0.5)
    ax11_r = ax11.twinx()  # dual y-axis — left = time, right = error
    ax11_r.plot(x_pos, e_vals, 'k^--', markersize=8, linewidth=1.5, label='Error (right)')
    ax11.set_xticks(x_pos)
    ax11.set_xticklabels(['Classical', 'Rand SVD', 'SRFT'], fontsize=9)
    ax11.set_ylabel('Time (s)', fontsize=10, color='k')
    ax11_r.set_ylabel('Rel. Error', fontsize=10, color='k')
    ax11.set_title(f'Summary at size {sizes[-1]}x{sizes[-1]-50}', fontsize=11, fontweight='bold')
    for bar, val in zip(bars, t_vals):
        ax11.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                  f'{val:.3f}s', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # (1,2) Singular value spectrum — do dono methods wahi sigma_i reproduce karte hain?
    ax12 = fig.add_subplot(gs[1, 2])
    _, Sc_f, _ = svd(A_fixed, full_matrices=False)
    _, Sr_f, _ = randomized_svd(A_fixed, k=40, p=p, q=q)
    Q_sf = srft_range_finder(A_fixed, l=40 + p)
    _, Ss_f, _ = stage_b_svd(A_fixed, Q_sf)
    idx_c = np.arange(1, 41)
    ax12.semilogy(idx_c,            Sc_f[:40], color=colors['c'], marker=markers['c'], linestyle='-',  label=label_c,  linewidth=2)
    ax12.semilogy(np.arange(1, 41), Sr_f,      color=colors['r'], marker=markers['r'], linestyle='--', label=label_r,  linewidth=2)
    ax12.semilogy(np.arange(1, 41), Ss_f[:40], color=colors['s'], marker=markers['s'], linestyle=':',  label=label_s,  linewidth=2)
    ax12.set_xlabel('Singular value index', fontsize=10)
    ax12.set_ylabel('σᵢ  (log scale)', fontsize=10)
    ax12.set_title('Singular Value Spectra', fontsize=11, fontweight='bold')
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
