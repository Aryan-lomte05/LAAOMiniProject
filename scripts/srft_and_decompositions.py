# ============================================================
# NAMAN's Module: srft_and_decompositions.py
# Implements: Algorithm 4.5 (SRFT), ID, CUR
# Paper: Halko, Martinsson, Tropp (2009)
# ============================================================

import numpy as np
from scipy.linalg import qr, svd, lu
from scipy.fft import fft, ifft
import time


# -----------------------------------------------------------
# SRFT Test Matrix (Section 4.6, Algorithm 4.5)
# Omega = sqrt(n/l) * R * F * D  (three structured components)
# -----------------------------------------------------------
def srft_range_finder(A, l):
    """
    Algorithm 4.5 — Aryan ke Gaussian Omega ki jagah SRFT use karta hai.
    Main gain: Gaussian multiply = O(mnl), SRFT = O(mn log l) — FFT ki wajah se.
    """
    m, n = A.shape
    rng = np.random.default_rng(42)

    # ==========================================================
    # SRFT has 3 components: D (diagonal signs), F (FFT), R (row sampler)
    # inhe explicitly nahi banaate — ek ek kaam karte hain seedha A pe
    # ==========================================================

    # Step 1: D — random sign diagonal matrix
    # basically randomly kuch columns ka sign flip karo (+1 ya -1)
    # ye isliye kiya kyunki agar data mein koi column "spiky" hai (ek jagah concentrated),
    # toh FFT pe dalne se pehle usse spread karna zaroori hai
    # bina D ke FFT sirf ek hi frequency bucket mein sab daal deta
    signs = rng.choice([-1.0, 1.0], size=n)
    AD = A * signs[np.newaxis, :]   # broadcast se multiply, D matrix explicitly nahi banaya

    # Step 2: F — Fast Fourier Transform (ye hi asli speedup hai)
    # humne pehle real FFT try kiya tha (np.real(fft())) directly
    # lekin clarity ke liye pehle complex FFT karo, phir subsample karo, phir real lo
    # reason: agar pehle real lo toh subsample ke waqt phase information lose hoti hai
    # Aryan ne ye confusion point out kiya — cleared it!
    AF_complex = fft(AD, axis=1) / np.sqrt(n)

    # Step 3: R — random row selection (subsampling)
    # n rows mein se randomly l rows select karo bina repeat ke
    # ye effectively Omega ke l random columns choose karna hai
    idx = rng.choice(n, size=l, replace=False)

    # scaling factor sqrt(n/l) paper se (preserves expected norms)
    # ab real part lo — complex part discard, FYI ye mathematically equivalent hai
    # kyunki original A real matrix hai, toh imaginary parts cancel out hote hain
    Y = np.sqrt(n / l) * np.real(AF_complex[:, idx])

    # Y ready hai — ab QR se orthonormal basis nikalo exactly jaisa Algorithm 4.1 mein
    Q, _ = qr(Y, mode='economic')
    return Q


# -----------------------------------------------------------
# Interpolative Decomposition (Section 5.2)
# A ≈ A[:, J] @ X  (actual columns of A, no abstract vectors)
# -----------------------------------------------------------
def interpolative_decomposition(A, k, p=10, use_srft=False):
    """
    SVD se ek alag idea — actual columns of A use karo approximation ke liye.
    SVD mein U, V imaginary directions hote hain, ID mein real data columns hote hain.
    This is useful when you want interpretable compression.
    """
    l = k + p

    # Stage A: Q nikalo (same as before — Gaussian ya SRFT)
    if use_srft:
        Q = srft_range_finder(A, l)
    else:
        rng = np.random.default_rng(42)
        Omega = rng.standard_normal((A.shape[1], l))
        Y = A @ Omega
        Q, _ = qr(Y, mode='economic')

    # Section 5.2 ka core idea:
    # Q A ka approximate column space represent karta hai
    # agar hum Q^T A pe pivoted QR run karein toh woh automatically sort karta hai
    # ki kaunse columns of A sabse zyada linearly independent hain
    # (pivoting = True matlab columns ko swap karo importance ke hisaab se)
    B = Q.T @ A        # B = (l x n) — chota matrix
    _, _, piv = qr(B, pivoting=True)   # ye pivot indices dega, most important columns pehle

    # top k pivot indices = most important actual columns of A
    J = piv[:k]

    # C = selected actual columns
    C = A[:, J]

    # X solve karo: A ≈ C @ X
    # lstsq use kiya kyunki direct inverse numerically unstable hota hai
    # ye least squares mein X find karta hai such that C @ X ≈ A with min ||C@X - A||
    X, _, _, _ = np.linalg.lstsq(C, A, rcond=None)

    # J = actual column indices, X = coefficient matrix, C = selected columns
    return J, X, C


# -----------------------------------------------------------
# CUR Decomposition (Section 2.1.4)
# A ≈ C @ U @ R — actual rows AND columns, not abstract vectors
# -----------------------------------------------------------
def cur_decomposition(A, k, p=5):
    """
    CUR = Column-linkage-Row decomposition.
    Advantage over ID: rows AND columns are real data, fully interpretable.
    """
    m, n = A.shape
    rng = np.random.default_rng(0)

    # columns select karne ke liye uniform sampling nahi kiya — leverage-score style kiya
    # idea: jo columns mein zyada variation (high norm) hai wo zyada important hain
    # squared norm = "energy" of that column — higher energy = higher probability of selection
    col_norms = np.linalg.norm(A, axis=0) ** 2
    col_probs = col_norms / col_norms.sum()  # probabilities normalize karo
    col_idx = rng.choice(n, size=k, replace=False, p=col_probs)

    # same treatment for rows
    row_norms = np.linalg.norm(A, axis=1) ** 2
    row_probs = row_norms / row_norms.sum()
    row_idx = rng.choice(m, size=k, replace=False, p=row_probs)

    C = A[:, col_idx]   # selected columns (m x k)
    R = A[row_idx, :]   # selected rows (k x n)

    # W = C aur R ka intersection — A ka ek small k x k subblock
    # ye CUR mein "linkage" ke liye zaroori hai
    W = A[np.ix_(row_idx, col_idx)]

    # U = W ka pseudoinverse — ye "glue" matrix hai jo C aur R ko sahi tarah jodata hai
    # mathematically: agar A = C M R for some M, toh M = W^+ (pseudoinverse)
    U = np.linalg.pinv(W)

    return C, U, R, col_idx, row_idx


# -----------------------------------------------------------
# Benchmarking: Gaussian vs SRFT — speed aur accuracy compare karo
# -----------------------------------------------------------
def benchmark_srft_vs_gaussian(m=2000, n=1500, k=50, p=10):
    rng = np.random.default_rng(1)

    # standard test matrix — rank k signal + small noise
    U_t = rng.standard_normal((m, k))
    V_t = rng.standard_normal((n, k))
    A = U_t @ V_t.T + 0.01 * rng.standard_normal((m, n))
    l = k + p

    # Gaussian benchmark
    t0 = time.time()
    Omega_g = rng.standard_normal((n, l))
    Y_g = A @ Omega_g
    Q_g, _ = qr(Y_g, mode='economic')
    t_gauss = time.time() - t0
    # error = how well Q_g approximates A's column space: ||(I - QQ^T)A||
    err_g = np.linalg.norm(A - Q_g @ Q_g.T @ A) / np.linalg.norm(A)

    # SRFT benchmark
    t0 = time.time()
    Q_s = srft_range_finder(A, l)
    t_srft = time.time() - t0
    err_s = np.linalg.norm(A - Q_s @ Q_s.T @ A) / np.linalg.norm(A)

    print(f"Matrix {m}x{n}, target rank l={l}")
    print(f"Gaussian Range Finder | Time: {t_gauss:.4f}s | Error: {err_g:.4e}")
    print(f"SRFT Range Finder     | Time: {t_srft:.4f}s | Error: {err_s:.4e}")

    # ID test
    J, X, C = interpolative_decomposition(A, k=k, p=p)
    err_id = np.linalg.norm(A - C @ X) / np.linalg.norm(A)
    print(f"Interpolative Decomp  | Error: {err_id:.4e} | Cols used: {len(J)}")

    # CUR test
    C_cur, U_cur, R_cur, _, _ = cur_decomposition(A, k=k)
    A_cur = C_cur @ U_cur @ R_cur
    err_cur = np.linalg.norm(A - A_cur) / np.linalg.norm(A)
    print(f"CUR Decomposition     | Error: {err_cur:.4e}")


if __name__ == "__main__":
    benchmark_srft_vs_gaussian()
