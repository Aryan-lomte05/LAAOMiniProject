# ============================================================
# ARYAN's Module: randomized_svd_core.py
# Implements: Algorithms 4.1, 4.2, 4.3/4.4, 5.1
# Paper: Halko, Martinsson, Tropp (2009)
# ============================================================

import numpy as np
from scipy.linalg import qr, svd
import time

# -----------------------------------------------------------
# Algorithm 4.1: Basic Randomized Range Finder
# -----------------------------------------------------------
def randomized_range_finder(A, l, random_state=None):
    """
    Stage A — Basic randomized range finder.
    Given m x n matrix A and sample count l = k + p,
    returns m x l orthonormal matrix Q s.t. A ≈ QQ^T A.

    Params:
        A: (m, n) ndarray
        l: int — target rank k + oversampling p (e.g. k=50, p=10 → l=60)
    Returns:
        Q: (m, l) orthonormal matrix
    """
    rng = np.random.default_rng(random_state)
    m, n = A.shape
    Omega = rng.standard_normal((n, l))   # Step 1: Draw Gaussian test matrix
    Y = A @ Omega                          # Step 2: Sample matrix Y = A * Omega
    Q, _ = qr(Y, mode='economic')         # Step 3: Orthonormalize
    return Q


# -----------------------------------------------------------
# Algorithm 4.3/4.4: Power Iteration Range Finder (for flat spectra)
# -----------------------------------------------------------
def randomized_range_finder_power(A, l, q=2, random_state=None):
    """
    Stage A — Range finder with q steps of power iteration.
    Improves accuracy when singular values decay slowly.
    Uses stabilized orthogonalization (Algorithm 4.4) to prevent
    floating-point collapse.

    Params:
        A: (m, n) ndarray
        l: int — target samples (k + oversampling)
        q: int — number of power iterations (q=1 or 2 usually sufficient)
    Returns:
        Q: (m, l) orthonormal matrix
    """
    rng = np.random.default_rng(random_state)
    m, n = A.shape
    Omega = rng.standard_normal((n, l))
    
    # Build (A A^T)^q A * Omega with stabilized QR between each step
    Y = A @ Omega
    for _ in range(q):
        Q_temp, _ = qr(Y, mode='economic')         # Stabilize (Alg 4.4)
        Z = A.T @ Q_temp
        Q_temp2, _ = qr(Z, mode='economic')
        Y = A @ Q_temp2
    
    Q, _ = qr(Y, mode='economic')
    return Q


# -----------------------------------------------------------
# Algorithm 4.2: Adaptive Randomized Range Finder
# (Unknown rank — stops when error < tolerance)
# -----------------------------------------------------------
def adaptive_range_finder(A, tol=1e-6, r=10, max_rank=None):
    """
    Stage A — Adaptive range finder for fixed-precision problem.
    Grows Q incrementally until ||( I - QQ^T )A|| < tol.

    Params:
        A: (m, n) ndarray
        tol: float — target tolerance epsilon
        r: int — block size for error estimation (r=10 recommended)
    Returns:
        Q: (m, k) orthonormal matrix
    """
    m, n = A.shape
    max_rank = max_rank or min(m, n)
    rng = np.random.default_rng(42)
    
    # Initial r Gaussian samples for error estimation
    Omega = rng.standard_normal((n, r))
    Y = A @ Omega  # (m, r)
    
    Q = np.zeros((m, 0))   # empty basis
    
    while True:
        # Error estimate: max ||( I - QQ^T ) y_i||
        if Q.shape[1] > 0:
            Y_res = Y - Q @ (Q.T @ Y)
        else:
            Y_res = Y
        
        err = np.max(np.linalg.norm(Y_res, axis=0))
        if err <= 10 * tol * np.sqrt(2 / np.pi):
            break
        if Q.shape[1] >= max_rank:
            break
        
        # Pick the column with largest residual, orthogonalize, add to Q
        j = np.argmax(np.linalg.norm(Y_res, axis=0))
        q_new = Y_res[:, j]
        q_new /= np.linalg.norm(q_new)
        Q = np.column_stack([Q, q_new]) if Q.shape[1] > 0 else q_new.reshape(-1, 1)
        
        # Add a new Gaussian sample to maintain the pool
        omega_new = rng.standard_normal((n, 1))
        y_new = A @ omega_new
        # Add to sample pool
        Y = np.hstack([Y, y_new])

        # Cap pool to prevent unbounded growth (Fix 1: memory leak)
        MAX_POOL = r + max_rank
        if Y.shape[1] > MAX_POOL:
            Y = Y[:, -MAX_POOL:]
    
    return Q


# -----------------------------------------------------------
# Algorithm 5.1: Stage B — Approximate SVD from Q
# -----------------------------------------------------------
def stage_b_svd(A, Q):
    """
    Stage B — Given orthonormal Q from Stage A,
    compute approximate SVD: A ≈ U Σ V^T.

    Steps:
        1. B = Q^T A          (small k x n matrix)
        2. SVD of B: B = Û Σ V^T
        3. U = Q Û
    Returns:
        U, S, Vt: approximate SVD factors
    """
    B = Q.T @ A             # Step 1: k x n
    U_hat, S, Vt = svd(B, full_matrices=False)  # Step 2: SVD of small matrix
    U = Q @ U_hat           # Step 3: lift back
    return U, S, Vt


# -----------------------------------------------------------
# Full Pipeline: Randomized SVD
# -----------------------------------------------------------
def randomized_svd(A, k, p=10, q=2, random_state=42):
    """
    Full Randomized SVD pipeline.
    
    Params:
        A: input matrix (m x n)
        k: target rank
        p: oversampling (default 10)
        q: power iteration steps (default 2)
    Returns:
        U[:, :k], S[:k], Vt[:k, :] — truncated approximate SVD
    """
    l = k + p
    Q = randomized_range_finder_power(A, l=l, q=q, random_state=random_state)
    U, S, Vt = stage_b_svd(A, Q)
    return U[:, :k], S[:k], Vt[:k, :]


# -----------------------------------------------------------
# Probabilistic Error Estimator (Section 4.3)
# -----------------------------------------------------------
def error_estimator(A, Q, r=10, random_state=0):
    """
    Estimate || (I - QQ^T) A || using r random Gaussian vectors.
    With prob >= 1 - 10^{-r}, true error <= 10 * sqrt(2/pi) * estimate.
    """
    rng = np.random.default_rng(random_state)
    m, n = A.shape
    Omega = rng.standard_normal((n, r))
    residuals = A @ Omega - Q @ (Q.T @ (A @ Omega))
    return np.max(np.linalg.norm(residuals, axis=0))


# -----------------------------------------------------------
# Benchmarking
# -----------------------------------------------------------
def benchmark(m=1000, n=800, k=50, q=2, p=10):
    rng = np.random.default_rng(0)
    # Create a low-rank matrix + noise
    U_true = rng.standard_normal((m, k))
    V_true = rng.standard_normal((n, k))
    A = U_true @ V_true.T + 0.01 * rng.standard_normal((m, n))

    # Classical SVD
    t0 = time.time()
    U_c, S_c, Vt_c = svd(A, full_matrices=False)
    t_classical = time.time() - t0
    A_approx_c = (U_c[:, :k] * S_c[:k]) @ Vt_c[:k, :]
    err_c = np.linalg.norm(A - A_approx_c) / np.linalg.norm(A)

    # Randomized SVD
    t0 = time.time()
    U_r, S_r, Vt_r = randomized_svd(A, k=k, p=p, q=q)
    t_rand = time.time() - t0
    A_approx_r = (U_r * S_r) @ Vt_r
    err_r = np.linalg.norm(A - A_approx_r) / np.linalg.norm(A)

    print(f"Matrix size: {m}x{n}, Target rank k={k}")
    print(f"Classical SVD  | Time: {t_classical:.4f}s | Rel. error: {err_c:.2e}")
    print(f"Randomized SVD | Time: {t_rand:.4f}s | Rel. error: {err_r:.2e}")
    
    Q_p = randomized_range_finder_power(A, k+p, q=q)
    est = error_estimator(A, Q_p)
    print(f"Error Estimator (Section 4.3): {est:.4e}")

if __name__ == "__main__":
    benchmark()
