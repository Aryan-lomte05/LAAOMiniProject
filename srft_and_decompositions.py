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
# Omega = (l/n)^{1/2} * D * F * R
#   D: random ±1 diagonal
#   F: DFT matrix (applied via FFT)
#   R: random row selection of l rows from n
# -----------------------------------------------------------
def srft_range_finder(A, l):
    """
    Algorithm 4.5 — Fast Randomized Range Finder using SRFT.
    Computes Y = A * Omega where Omega is an SRFT matrix.
    Cost: O(mn log l) vs O(mnl) for Gaussian.

    Params:
        A: (m, n) ndarray
        l: int — number of samples (k + oversampling)
    Returns:
        Q: (m, l) orthonormal matrix
    """
    m, n = A.shape
    rng = np.random.default_rng(42)

    # Step 1: Build SRFT: Omega = sqrt(n/l) * R * F * D
    # D: random ±1 signs applied to columns of A
    signs = rng.choice([-1.0, 1.0], size=n)
    AD = A * signs[np.newaxis, :]        # A * D  (broadcast)

    # F: Apply FFT along axis=1 (columns)
    AF = np.real(fft(AD, axis=1)) / np.sqrt(n)

    # R: Randomly select l rows (columns of Omega = rows of Omega^T)
    idx = rng.choice(n, size=l, replace=False)
    Y = np.sqrt(n / l) * AF[:, idx]     # m x l sample matrix

    # Step 3: Orthonormalize
    Q, _ = qr(Y, mode='economic')
    return Q


# -----------------------------------------------------------
# Interpolative Decomposition (Section 5.2, eq. 5.3-5.5)
# A ≈ A[:, J] @ X  where J is an index set of k columns
# Uses randomized range finder + pivoted QR on Q
# -----------------------------------------------------------
def interpolative_decomposition(A, k, p=10, use_srft=False):
    """
    Randomized Interpolative Decomposition.
    Returns A ≈ C @ X where C = A[:, J] (actual columns of A).

    Params:
        A: (m, n) ndarray
        k: target rank
        p: oversampling
        use_srft: use SRFT (fast) or Gaussian (accurate)
    Returns:
        J: array of k column indices
        X: (k, n) coefficient matrix (X[:, J] = I_k, entries <= 2)
        C: (m, k) — selected columns A[:, J]
    """
    l = k + p
    # Stage A: get Q
    if use_srft:
        Q = srft_range_finder(A, l)
    else:
        rng = np.random.default_rng(42)
        Omega = rng.standard_normal((A.shape[1], l))
        Y = A @ Omega
        Q, _ = qr(Y, mode='economic')
    
    # Section 5.2: Form B = Q^T A and perform pivoted QR to find important columns
    B = Q.T @ A                          # (l, n)
    _, _, piv = qr(B, pivoting=True)     # B P = Q R
    J = piv[:k]                          # k most important column indices
    
    # Solve for X: A ≈ A[:, J] @ X
    # From eq 5.5: X = [I | (R_11^-1) R_12] P^T
    # A simpler way is to solve the least squares problem A[:, J] @ X = A
    C = A[:, J]                          # Selected columns
    X, _, _, _ = np.linalg.lstsq(C, A, rcond=None)
    
    return J, X, C


# -----------------------------------------------------------
# CUR Decomposition (Section 2.1.4)
# A ≈ C @ U_small @ R
# C: random column subset, R: random row subset, U: linkage
# -----------------------------------------------------------
def cur_decomposition(A, k, p=5):
    """
    Randomized CUR Decomposition.
    A ≈ C @ U @ R where C, R are actual submatrices of A.

    Params:
        A: (m, n) ndarray
        k: target rank
        p: oversampling
    Returns:
        C: (m, k) — k columns of A
        U: (k, k) — linkage matrix
        R: (k, n) — k rows of A
    """
    m, n = A.shape
    rng = np.random.default_rng(0)
    
    # Sample k+p columns by leverage-score-like probabilities
    # Simple version: sample uniformly (more advanced: use SVD-based leverage scores)
    col_norms = np.linalg.norm(A, axis=0) ** 2
    col_probs = col_norms / col_norms.sum()
    col_idx = rng.choice(n, size=k, replace=False, p=col_probs)
    
    row_norms = np.linalg.norm(A, axis=1) ** 2
    row_probs = row_norms / row_norms.sum()
    row_idx = rng.choice(m, size=k, replace=False, p=row_probs)
    
    C = A[:, col_idx]           # m x k
    R = A[row_idx, :]           # k x n
    W = A[np.ix_(row_idx, col_idx)]  # k x k intersection submatrix
    
    # Linkage: U = W^+ (pseudoinverse)
    U = np.linalg.pinv(W)       # k x k
    
    return C, U, R, col_idx, row_idx


# -----------------------------------------------------------
# Benchmarking: Gaussian vs SRFT speed/accuracy
# -----------------------------------------------------------
def benchmark_srft_vs_gaussian(m=2000, n=1500, k=50, p=10):
    rng = np.random.default_rng(1)
    U_t = rng.standard_normal((m, k))
    V_t = rng.standard_normal((n, k))
    A = U_t @ V_t.T + 0.01 * rng.standard_normal((m, n))
    l = k + p
    
    # Gaussian
    t0 = time.time()
    Omega_g = rng.standard_normal((n, l))
    Y_g = A @ Omega_g
    Q_g, _ = qr(Y_g, mode='economic')
    t_gauss = time.time() - t0
    err_g = np.linalg.norm(A - Q_g @ Q_g.T @ A) / np.linalg.norm(A)
    
    # SRFT
    t0 = time.time()
    Q_s = srft_range_finder(A, l)
    t_srft = time.time() - t0
    err_s = np.linalg.norm(A - Q_s @ Q_s.T @ A) / np.linalg.norm(A)
    
    print(f"Matrix {m}x{n}, target rank l={l}")
    print(f"Gaussian Range Finder | Time: {t_gauss:.4f}s | Error: {err_g:.4e}")
    print(f"SRFT Range Finder     | Time: {t_srft:.4f}s | Error: {err_s:.4e}")

    # Test ID
    J, X, C = interpolative_decomposition(A, k=k, p=p)
    err_id = np.linalg.norm(A - C @ X) / np.linalg.norm(A)
    print(f"Interpolative Decomp  | Error: {err_id:.4e} | Cols used: {len(J)}")
    
    # Test CUR
    C_cur, U_cur, R_cur, _, _ = cur_decomposition(A, k=k)
    A_cur = C_cur @ U_cur @ R_cur
    err_cur = np.linalg.norm(A - A_cur) / np.linalg.norm(A)
    print(f"CUR Decomposition     | Error: {err_cur:.4e}")


if __name__ == "__main__":
    benchmark_srft_vs_gaussian()
