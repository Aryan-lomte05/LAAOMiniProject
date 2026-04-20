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
    Stage A , Basic randomized range finder.
    Given m x n matrix A and sample count l = k + p,
    returns m x l orthonormal matrix Q s.t. A ≈ QQ^T A.
    """
    rng = np.random.default_rng(random_state)
    m, n = A.shape

    # pehli baar ye padhke laga ki random matrix kyun use kar rahe hain
    # but basically humari A matrix bohot badi hoti hai
    # agar hum isko directly process karein toh O(mn^2) time lagega
    # toh idea ye hai ki ek choti random Gaussian matrix Omega se multiply karo
    # ye Omega basically "test directions" hain , kuch hi directions me A project hoti hai
    Omega = rng.standard_normal((n, l))

    # Y = A * Omega -> ye ek nayi matrix hai jo A ke important column directions ko capture karti hai
    # socho jaise ki A ek 3D shape hai, aur hum uske ऊपर random angle se light daala
    # jo shadow aaya (Y) wahi uski approximate structure hai
    Y = A @ Omega

    # ab Y ke columns ko orthonormal banana hai (mutually perpendicular unit vectors)
    # QR decomposition exactly yahi karta hai
    # mode='economic' isliye use kiya kyunki full mode m x m Q deta, jo memory waste hai
    # hume sirf m x l chahiye (l = k+p, usually ≪ m)
    Q, _ = qr(Y, mode='economic')

    return Q


# -----------------------------------------------------------
# Algorithm 4.3/4.4: Power Iteration Range Finder
# (ye actually zyada important hai jab singular values flat hoti hain)
# -----------------------------------------------------------
def randomized_range_finder_power(A, l, q=2, random_state=None):
    """
    Stage A with power iteration , improves accuracy for flat spectra.
    Alg 4.4 ka stabilized version hai (intermediate QR steps).
    """
    rng = np.random.default_rng(random_state)
    m, n = A.shape
    Omega = rng.standard_normal((n, l))

    # basic version se start karo
    Y = A @ Omega

    # power iteration ka concept samajhne me time laga:
    # agar singular values bohot similar hain (flat spectrum), toh random directions
    # sirf "dominant" direction ko focus nahi kar paate, noise mix ho jata hai
    # solution: multiply (A A^T)^q times , isse dominant singular values exponentially
    # badi ho jaati hain aur baaki dab jaati hain
    # mathematically: sigma_i^(2q+1) , toh agar sigma_1 = 1.0 aur sigma_2 = 0.9,
    # q=2 pe ye 1.0^5 = 1.0 aur 0.9^5 = 0.59 ho jaata hai , gap bohot zyada ho jaata hai!
    for _ in range(q):
        # seedha (A A^T)^q compute karte toh floating point overflow hoti
        # isliye beech beech mein QR lagaate hain (Algorithm 4.4 ka trick)
        # ye "stabilized" version hai , bina numerical collapse ke
        Q_temp, _ = qr(Y, mode='economic')

        # A^T se multiply , backward pass
        Z = A.T @ Q_temp

        # yahan bhi QR lagao taaki columns degenerate na ho jaayein
        Q_temp2, _ = qr(Z, mode='economic')

        # A se wapas forward , ek full iteration complete
        Y = A @ Q_temp2

    # final orthonormal basis
    Q, _ = qr(Y, mode='economic')
    return Q


# -----------------------------------------------------------
# Algorithm 4.2: Adaptive Randomized Range Finder
# (jab rank pata nahi hota , khud decide karta hai kab rukna hai)
# -----------------------------------------------------------
def adaptive_range_finder(A, tol=1e-6, r=10, max_rank=None):
    """
    Stage A , target tolerance ke basis pe automatically rank decide karta hai.
    Fixed-precision problem ke liye , jab hume exactly k pata nahi hota.
    """
    m, n = A.shape
    max_rank = max_rank or min(m, n)
    rng = np.random.default_rng(42)

    # pehle r random vectors se error check karte hain
    # ye r vectors "test probe" hain , inse pata chalta hai kitna error bacha hai
    Omega = rng.standard_normal((n, r))
    Y = A @ Omega

    # Q empty se start hota hai , koi basis nahi initially
    Q = np.zeros((m, 0))

    while True:
        # residual calculate karo , i.e., A ka woh hissa jo abhi Q mein capture nahi hua
        # agar Q already accha hai toh Y_res ~ 0 hoga
        if Q.shape[1] > 0:
            Y_res = Y - Q @ (Q.T @ Y)  # projection nikal do, jo bacha woh residual
        else:
            Y_res = Y  # pehli iteration mein koi basis nahi toh full Y hi residual hai

        # maximum residual norm check karo , ye humara error estimate hai
        err = np.max(np.linalg.norm(Y_res, axis=0))

        # stopping condition paper se directly liya (Section 4.3 probabilistic bound)
        # 10 * sqrt(2/pi) ek mathematical factor hai jo ensure karta hai ke
        # probability >= 1 - 10^(-r) ke saath actual error <= tol
        if err <= 10 * tol * np.sqrt(2 / np.pi):
            break
        if Q.shape[1] >= max_rank:
            break

        # jo direction mein sabse zyada error hai, woh add karo Q mein
        j = np.argmax(np.linalg.norm(Y_res, axis=0))
        q_new = Y_res[:, j]
        q_new /= np.linalg.norm(q_new)  # unit vector bana do

        Q = np.column_stack([Q, q_new]) if Q.shape[1] > 0 else q_new.reshape(-1, 1)

        # ek naya random sample bhi pool mein add karo taaki agle iteration mein
        # kuch fresh vectors ho check karne ke liye
        omega_new = rng.standard_normal((n, 1))
        y_new = A @ omega_new
        Y = np.hstack([Y, y_new])

        # pehle yahan memory leak tha , Y infinitely grow karta tha
        # fix: pool size cap karo, purane samples hata do
        MAX_POOL = r + max_rank
        if Y.shape[1] > MAX_POOL:
            Y = Y[:, -MAX_POOL:]  # sirf latest samples rakh

    return Q


# -----------------------------------------------------------
# Algorithm 5.1: Stage B , Stage A ke baad actual SVD nikalna
# -----------------------------------------------------------
def stage_b_svd(A, Q):
    """
    Stage B , Q (from Stage A) given hain, isse A ka approx SVD nikalte hain.
    Key insight: A ko directly decompose nahi karte, Q ke through chhotaa banana pehle.
    """
    # B = Q^T A , ye ek bohot choti matrix hai (k x n instead of m x n)
    # humne basically massive A ko tiny B mein compress kar diya Q ke through
    B = Q.T @ A

    # ab SVD B pe chalaao , B chota hai toh ye seconds mein chalega
    # yahi randomized SVD ka main speedup hai , bade matrix ki jagah chote pe kaam
    U_hat, S, Vt = svd(B, full_matrices=False)

    # U_hat choti space mein hai (k x k), isse wapas full space mein laana hai
    # Q @ U_hat se U (m x k) milta hai , "lift back" operation
    U = Q @ U_hat

    # ab U, S, Vt milkar A ka approximate SVD dete hain: A ≈ U Σ V^T
    return U, S, Vt


# -----------------------------------------------------------
# Full Pipeline: Randomized SVD (Stage A + Stage B combined)
# -----------------------------------------------------------
def randomized_svd(A, k, p=10, q=2, random_state=42):
    """
    Complete randomized SVD: Stage A (range finder) + Stage B (factorization).
    p = oversampling parameter, q = power iteration steps.
    """
    # l = total vectors we sample. k is what we want, p is buffer
    # pehle confusion tha ki p kyun add karte hain Gpt sir ne clear kiya:
    # random vectors guaranteed nahi hote ki exactly k directions cover karein
    # thoda extra (p=10) le lo, error dramatically kam ho jaata hai (Theorem 1.1)
    l = k + p

    # Stage A: Q find karo jo A ke column space ko represent kare
    Q = randomized_range_finder_power(A, l=l, q=q, random_state=random_state)

    # Stage B: Q use karke A ka SVD nikalo
    U, S, Vt = stage_b_svd(A, Q)

    # return only top k , p extra vectors ka kaam ho gaya, ab discard
    return U[:, :k], S[:k], Vt[:k, :]


# -----------------------------------------------------------
# Probabilistic Error Estimator (Section 4.3)
# -----------------------------------------------------------
def error_estimator(A, Q, r=10, random_state=0):
    """
    Q kitna accurate hai wo estimate karta hai bina exact computation ke.
    r random Gaussian vectors se error bound calculate hota hai.
    """
    rng = np.random.default_rng(random_state)
    m, n = A.shape
    Omega = rng.standard_normal((n, r))

    # residual = A*Omega - Q*(Q^T * A*Omega)
    # agar Q accha hai toh Q*(Q^T * A*Omega) ≈ A*Omega, toh residual ~ 0
    residuals = A @ Omega - Q @ (Q.T @ (A @ Omega))

    # maximum column norm , yahi our "estimated error" hai
    # paper bolta hai: with prob >= 1 - 10^(-r), actual error <= 10*sqrt(2/pi) * this value
    return np.max(np.linalg.norm(residuals, axis=0))


# -----------------------------------------------------------
# Benchmarking , Classical vs Randomized SVD compare karta hai
# -----------------------------------------------------------
def benchmark(m=1000, n=800, k=50, q=2, p=10):
    rng = np.random.default_rng(0)

    # synthetically low-rank matrix banao + thoda noise
    # U_true @ V_true.T ek exactly rank-k matrix hai
    # 0.01 * noise add kiya taaki perfectly clean na ho (realistic scenario)
    U_true = rng.standard_normal((m, k))
    V_true = rng.standard_normal((n, k))
    A = U_true @ V_true.T + 0.01 * rng.standard_normal((m, n))

    # Classical SVD , ye puri matrix compute karta hai (slow for large A)
    t0 = time.time()
    U_c, S_c, Vt_c = svd(A, full_matrices=False)
    t_classical = time.time() - t0
    A_approx_c = (U_c[:, :k] * S_c[:k]) @ Vt_c[:k, :]
    err_c = np.linalg.norm(A - A_approx_c) / np.linalg.norm(A)

    # Randomized SVD , humara implementation
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
