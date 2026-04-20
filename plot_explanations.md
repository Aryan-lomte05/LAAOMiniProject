# Understanding Our Experiment Plots
### LAAO Mini Project — Halko, Martinsson, Tropp (2009)

---

## Plot 1: Image Compression (`experiment_image_compression.png`)

![Image Compression](e:\LAAOMiniProject\scripts\experiment_image_compression.png)

### What is this?
We took a 256×256 synthetic grayscale matrix (outer products of sine/cosine waves — simulates smooth texture like a real image) and compressed it using Randomized SVD at different values of `k` (rank).

### What each panel is:
| Panel | What it shows |
|---|---|
| **Original** | The uncompressed matrix plotted as a grayscale image |
| **k=5** | Only 5 singular vectors used — very blurry, 3.91% of original data stored |
| **k=20** | 20 vectors — structure is visible, 15.62% stored |
| **k=50** | Noticeably cleaner, nearly recognizable, 39% stored |
| **k=100** | Very close to original, 78% stored |

### What `err` and `CR` mean:
- **`err`** = Frobenius norm of `(A - A_approx) / A` — relative error. Closer to 0 = better reconstruction
- **`CR` (Compression Ratio)** = `k*(m+n) / m*n` — fraction of numbers we actually stored compared to the original. k=5 stores just 3.91% of the original!

### What this proves:
Even storing **under 40% of the data**, the image is visually very close to original. This is the power of low-rank approximation — most real-world data lives in a very small dimensional subspace.

---

## Plot 2: Oversampling Effect (`experiment_oversampling.png`)

### What is this?
This verifies **Theorem 1.1** of the paper. We asked: how does the error of Stage A (just `Q`, not even full SVD) behave as we increase the oversampling parameter `p`?

### Reading the axes:
- **X-axis** = Oversampling `p` (from 0 to 50, step 5). Total vectors sampled = `l = k + p = 20 + p`
- **Y-axis** = `||A - QQ^T A||_2` (spectral norm of the residual — how much of `A` is NOT captured by `Q`)
- **Blue band** = mean ± 1 std across 25 random trials

### What the lines mean:
- **Blue curve (Mean error)** = actual error of our random range finder at each `p`. It's DECREASING as `p` increases.
- **Red dashed line** = `σ_{k+1} = 0.0476` — the **theoretical lower bound** (Eckart-Young theorem). No rank-k approximation can EVER go below this, no matter what algorithm you use.
- **Orange dotted line** = `σ_k = 0.0500` — the last singular value we "keep"

### The key insight:
At `p=0` (no oversampling), error ≈ 0.054 — above even `σ_k`! With just `p=10`, we're already well below `σ_k` and approaching the theoretical floor. **This empirically proves Theorem 1.1** — oversampling rapidly saturates the quality of the approximation.

The shaded band getting thinner as `p` increases means: more oversampling also makes the algorithm more **consistent** (less variance between runs).

---

## Plot 3: Power Iteration Effect (`experiment_power_iteration.png`)

### What is this?
This verifies **Theorem 1.2**. We tested the same range finder with `q = 0, 1, 2, 3` power iterations on THREE types of matrices:
- **Flat spectrum** (`decay ∝ 1/√i`): singular values drop very slowly — worst case for random SVD
- **Moderate spectrum** (`decay ∝ 1/i²`): moderate drop
- **Fast spectrum** (`decay ∝ 1/i⁵`): singular values drop very quickly — best case

### What each bar means:
Each group of colored bars = one spectrum type. Each bar = error at that `q` value. Red dashed line = theoretical optimal `σ_{k+1}`.

### What the three panels show:

**Panel 1 — Flat Spectrum (worst case):**
- `q=0`: Error = 0.402 → extremely bad, almost 2× the `σ_{k+1}` line!
- `q=1`: Drops to 0.227
- `q=2`: Down to 0.205 — now below the red line!
- `q=3`: 0.196 — basically optimal

This is the most important panel — it PROVES that without power iteration, random SVD completely fails on flat spectra. Just `q=2` fixes everything.

**Panel 2 — Moderate Spectrum:**
- Even `q=0` overshoots the bound significantly (0.003 vs 0.0023 optimal)
- `q=1` brings it to 0.001 — already well below the floor
- Power iteration still helps, but the gains are smaller since decay is faster

**Panel 3 — Fast Spectrum (best case):**
- All bars (even `q=0`) are essentially at the theoretical minimum (1e-7 scale)
- Power iteration makes no difference when spectrum decays fast
- This confirms: power iteration is only needed for flat spectra

---

## Plot 4: PCA via Randomized SVD (`experiment_pca.png`)

### What is this?
We simulated a typical machine learning scenario: a 500×2000 data matrix with true rank 10 (only 10 underlying patterns), plus noise. Then we ran PCA using both Classical SVD and our Randomized SVD and compared results.

### Left Panel — Explained Variance Curve:
- **Y-axis** = cumulative explained variance (how much % of total information is in the first `n` components)
- Both lines are **almost identical** — randomized SVD captures variance just as well
- **The elbow at `True rank=10`** is the key observation — after index 10, adding more components adds almost no new information (because the remaining 'components' are just noise)
- Classical SVD: 0.29s | Randomized SVD: **0.015s** → **~19× faster** for same result!

### Right Panel — Singular Value Spectrum (log scale):
- **Y-axis is log scale** — the singular values span multiple orders of magnitude
- Blue (classical) and red (randomized) lines **completely overlap** for the first 10 singular values
- After index 10, there's a dramatic drop by ~2 orders of magnitude — noise floor
- This confirms the matrix truly has rank 10 AND our algorithm accurately finds those top 10 values

### What this proves:
Randomized SVD = Classical SVD in terms of accuracy, but much faster. This is exactly what sklearn uses internally in `TruncatedSVD` and `PCA(svd_solver='randomized')`.

---

## Plot 5: Final Benchmark Dashboard (`experiment_dashboard.png`)

This is 6 subplots combined into one dashboard. Matrix sizes tested: 200×150, 400×350, ..., 1200×1150.

### (Top Left) Time vs Matrix Size:
- **Blue (Classical SVD)** grows steeply — it scales roughly as matrix size squared
- **Green (Randomized SVD)** is nearly flat — stays under 0.03s even at 1200×1150
- **Red (SRFT)** is similarly flat but slightly above Randomized for small matrices
- At size 1200: Classical=0.518s, Randomized=0.028s, SRFT=0.034s

### (Top Middle) Relative Error vs Matrix Size (log scale):
- Classical SVD error (blue) ≈ 1.5-1.8×10⁻³ — this is the "floor" for this matrix type
- Randomized SVD matches exactly — same error as classical
- SRFT is ~2.5× higher error consistently — SRFT trades some accuracy for speed

### (Top Right) Speedup vs Classical SVD:
- Y-axis = `time_classical / time_method` — higher = faster than classical
- At size 1200: Randomized = ~18× speedup, SRFT = ~15× speedup
- The dashed line at 1.0 = equal to classical. Anything above = faster.
- Speedup GROWS with matrix size — exactly what the theory predicts (O(mn²) vs O(mnk))

### (Bottom Left) Singular Value Accuracy vs k:
- Tests how accurately `σ₁` (top singular value) is recovered as we vary rank `k`
- Randomized SVD (green): error ~10⁻¹⁵ — essentially machine precision
- SRFT (red): error ~10⁻⁵ — still very accurate, 10 orders of magnitude better than needed
- Both improve as `k` grows — more vectors = better approximation

### (Bottom Middle) Summary Bar Chart at size 1200×1150:
- Left Y-axis (bars) = time in seconds
- Right Y-axis (triangles + dashed line) = relative error
- Key: Classical takes 0.518s, Randomized 0.078s, SRFT 0.034s
- Error (triangles): Classical and Randomized nearly equal, SRFT slightly higher
- This is the "headline result" — same accuracy, 15-18× faster

### (Bottom Right) Singular Value Spectra:
- Plots the first 40 singular values of the fixed test matrix on log scale
- Blue (classical) is the ground truth
- Green (randomized) and red (SRFT) both trace it almost perfectly
- Minor divergence only at the very end (small singular values) — expected and acceptable

---

## Summary Table — What Each Plot Proves

| Plot | Paper Section | Core Point Proven |
|---|---|---|
| Image Compression | Section 7 | Low-rank approx works in practice. k=50 recovers structure at 39% storage |
| Oversampling Effect | Theorem 1.1 | p=10 is enough. Error converges to theoretical floor σ_{k+1} |
| Power Iteration | Theorem 1.2 | q=2 fixes flat-spectrum failures. Fully validates Algorithm 4.3/4.4 |
| PCA | Section 7.1/7.3 | Randomized SVD = Classical for PCA, ~19× faster |
| Dashboard | Section 7 (full) | End-to-end: 15-18× speedup at large scale, error parity with classical |
