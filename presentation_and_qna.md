# Linear Algebra & Optimization (LAAO) Mini Project 2
## Halko, Martinsson, Tropp (2009) — Randomized Matrix Decompositions
**Complete Presentation Guide & Defense Q&A**

---

## 1. End-To-End Core Theory (The "Elevator Pitch")

### The Problem
Classic Singular Value Decomposition (SVD) requires $O(mn \min(m,n))$ floating-point operations. For massive matrices in modern machine learning (e.g., millions of rows/columns), standard deterministic SVD becomes computationally infeasible and suffers from heavy RAM limitations.

### The Innovation: Randomized SVD
The HMT-2009 paper introduces a **two-stage randomized framework** that computes approximate low-rank decompositions exponentially faster while maintaining near-optimal accuracy. It reduces the interaction with the massive matrix to just a few matrix-vector products.

*   **Stage A (Find the Subspace):** We want bounds on a smaller subspace that accurately "captures" the crucial action (range) of the large matrix $A$. We do this by sketching $A$ using a random test matrix $\Omega$. 
    *   Initialize a random Gaussian matrix $\Omega$ of size $n \times (k+p)$ (where $k$ is target rank, $p$ is slight oversampling).
    *   Form the sample matrix $Y = A\Omega$. By projecting $A$ onto random vectors, we naturally highlight the directions of maximum variance/significance.
    *   Compute an orthonormal basis $Q$ for the columns of $Y$ using a QR factorization ($Y = QR$). $Q$ now forms our approximate orthogonal basis.
*   **Stage B (Deterministic Factorization):** 
    *   Project the massive matrix $A$ down to our small subspace: $B = Q^T A$. Resulting matrix $B$ is tiny ($k \times n$).
    *   Compute the standard SVD of this tiny matrix: $B = \hat{U} \Sigma V^T$.
    *   Lift the left singular vectors back up to the original dimension: $U = Q\hat{U}$.
    *   **Result:** $A \approx U \Sigma V^T$.

### Why does it work? 
It builds on the **Johnson-Lindenstrauss Lemma** and random projection theory. In a high-dimensional space, random vectors are highly likely to be orthogonal to each other and safely capture the dominant dimensions of $A$'s range without needing exactly aligned eigenvectors.

---

## 2. Presentation Script (10 Minutes)

> [!TIP]
> Coordinate your transitions. Do not read off a paper. Bring up the visual plots Aditya generated whenever mentioning experiments!

### Speaker 1: Aryan (0:00 - 3:00) — Introduction & Core Algorithm
**[Slide 1: Intro & Problem]**
"Good Morning / Afternoon Sir. Our mini-project implements the landmark 2009 Halko-Martinsson-Tropp paper on Randomized Matrix Decompositions. 
Standard SVD is the backbone of ML, PCA, and Data Compression, but computing it for massive datasets is incredibly slow—scaling at $O(mn^2)$. Our goal was to implement their randomized approach, which drastically reduces this computation by exploiting random sampling."

**[Slide 2: Stage A & Stage B]**
"I focused on the core algorithmic engine. The math relies on two stages. In **Stage A**, instead of working with the huge matrix $A$, we map it against a random Gaussian matrix to 'sketch' it. We multiply $A$ by random vectors, capturing the most important column space. We orthogonalize this to get a basis, $Q$. 
In **Stage B**, we project $A$ into this tiny subspace $Q$, do a standard SVD on the tiny matrix, and lift the results back. The computational heavy lifting is completely bypassed."

**[Slide 3: Power Iterations]**
"If a matrix's singular values drop off very quickly, standard randomization works perfectly. However, if the singular values decay slowly (a flat spectrum), the random samples get 'polluted' by noise from smaller singular values. To fix this, I implemented the **Power Iteration Scheme** (Algorithm 4.3). We multiply $A$ by its transpose iteratively: $(AA^T)^q A \Omega$. This mathematically suppresses the smaller eigenvalues and forces the basis to align with the dominant ones. I also added intermediate QR normalizations to avoid floating-point overflow."

*Transition:* "Now Naman will explain how we accelerated this even further using structured randomness, and how we extract actual dataset columns instead of abstract mathematical vectors."

### Speaker 2: Naman (3:00 - 6:00) — SRFT & ID/CUR
**[Slide 4: The SRFT Optimization]**
"Thank you, Aryan. While Gaussian matrices give highly accurate random projections, multiplying $A \times \Omega$ still takes $O(mnk)$ time. I implemented **Section 4.6**: the Subsampled Random Fourier Transform (SRFT).
Instead of a fully dense Gaussian matrix, SRFT constructs $\Omega$ using a random sign diagonal, a Discrete Fourier Transform (FFT), and a random row selector. Because it uses FFT, the matrix multiplication drops from $O(mnk)$ down to $O(mn \log k)$. Our benchmarks prove SRFT provides nearly identical accuracy to Gaussian but scales exponentially better."

**[Slide 5: Interpolative Decomposition (ID)]**
"I also implemented Interpolative Decomposition (ID). Unlike standard SVD, which gives abstract mathematical vectors, ID selects *actual literal columns* from our original dataset. Mathematically, it expresses $A \approx C X$, where $C$ is a subset of exactly $k$ columns of $A$. I used our randomized range finder to get $Q$, then used a **Pivoted QR factorization** on $Q^T$ to figure out which columns are linearly independent and bear the most information."

**[Slide 6: CUR Decomposition]**
"For extreme interpretability, I extended this to CUR Decomposition, approximating $A$ using actual columns ($C$) and actual rows ($R$), glued by a linkage matrix $U$. This implies we don't just compress data—we summarize it using real-world data points."

*Transition:* "Finally, Aditya will demonstrate our empirical validations, error bounds, and real-world applications of these methods."

### Speaker 3: Aditya (6:00 - 9:00) — Experiments & Applications
**[Slide 7: Image Compression & PCA]**
"Thanks, Naman. My module brings the math into real-world applications (Section 7 of the paper). First, we tested image compression. By taking the SVD up to $k=50$ components, we retain the structural integrity of the image while aggressively saving memory, calculating it at a fraction of the time of standard routines. We also proved our randomized SVD perfectly parallels classical PCA outputs for variance capture."

**[Slide 8: Error Bounds Verification (Theorems 1.1 & 1.2)]**
"The most rigorous part was empirically proving the paper's main error bounds. 
For **Theorem 1.1**, I plotted expected error against 'oversampling' ($p$). The paper claims adding just 5-10 extra random sample vectors allows the error to plummet optimally to $\sigma_{k+1}$. Our plots beautifully confirm this exact asymptote constraint.
For **Theorem 1.2**, I verified the Power Iteration. On a matrix with slow decay, $q=0$ fails horribly. But bumping $q=2$ immediately slashes the error margin mathematically, fully validating Aryan's power-iteration code pipeline."

**[Slide 9: Dashboards & Conclusion]**
"Our final dashboard compares all three algorithms. As matrix sizes scale to $1000 \times 1000$, standard SVD begins stalling. Our Randomized SVD runs up to $10\times$ faster, and SRFT pushes this even further for massive densities. We successfully recreated the core proofs of the paper entirely from scratch in Python. Thank you."

---

## 3. In-Depth Technical Defense (Line-by-Line Q&A)

> [!WARNING]
> The evaluator "Sir" will ask you specific syntax and conceptual questions to test if you wrote/understand the code. Memorize your sections below!

### 🟢 ARYAN'S SECTION: Core Randomized SVD

**Q: Explain line `Q, _ = qr(Y, mode='economic')` in your code. Why economic?**
**Sir's expectation:** Why use economic QR instead of full?
**Aryan's Answer:** "The matrix $Y$ is of size $m \times (k+p)$. A 'full' QR would yield a completely unneeded $Q$ matrix of size $m \times m$ (zeros added to square it off), which defeats the entire purpose of dimension reduction and would crash our memory limits. `mode='economic'` only computes the first $(k+p)$ columns of $Q$, yielding an $m \times (k+p)$ orthonormal basis that efficiently spans the exact range of $Y$."

**Q: In power iteration, why do you calculate $Z = A^T Q_{temp}$ and then $Y = A Q_{temp2}$ instead of just doing $(A A^T)^q$ directly?**
**Aryan's Answer:** "The theoretical power iteration calculates $(AA^T)^q A \Omega$. But multiplying matrices directly destroys floating-point precision; the singular values decay exponentially, making the columns linearly dependent numerically (as they all converge blindly to the top singular vector). By doing a QR decomposition between *every* application of $A^T$ and $A$, we actively re-orthogonalize the vectors, completely preventing floating-point overflow and preserving the trailing singular vectors."

**Q: What is oversampling ($p$), and why do we set $p=10$?**
**Aryan's Answer:** "Oversampling means if we want a rank $k=50$ subspace, we draw $l=60$ random Gaussian vectors instead. Gaussian random vectors are highly independent, but in lower dimensions, they might not span perfectly. Adding $p=5$ or $p=10$ provides enough a probabilistic 'buffer' to guarantee—with immense mathematical likelihood—that we capture the high-variance subspace without missing essential data structures. Aditya's visual graphs rigorously prove $p=10$ guarantees convergence to the absolute minimum."

**Q: In your Adaptive Range finder (`adaptive_range_finder`), how does your exact memory leak fix work conceptually? (`if Y.shape[1] > MAX_POOL: Y = Y[:, -MAX_POOL:]`)**
**Aryan's Answer:** "In Algorithm 4.2, we add new random vectors dynamically in a `while` loop until we hit an explicit error tolerance. However, if a matrix doesn't converge quickly, the `Y` pool arrays grow infinitely, burning RAM in $O(n^2)$. My fix adds a cap: `MAX_POOL = r + max_rank`. We retain the dimension constraint, and mathematically we prune the 'oldest' standard nominal distributions from the left slice of multidimensional arrays because the newest right-slice distributions fully handle the unreached residuals."

---

### 🔵 NAMAN'S SECTION: SRFT & Interpolative Decomposition (ID)

**Q: What exactly is happening in your SRFT implementation with the FFT? `fft(AD, axis=1)`?**
**Sir's expectation:** Explain the pieces of SRFT.
**Naman's Answer:** "SRFT builds $\Omega = \sqrt{n/l} \cdot D \cdot F \cdot R$. First, $D$ randomly flips the signs of $A$'s columns (`AD = A * signs`), which spreads out localized data across the spectrum so no column spikes disproportionately. Then $F$, the Fast Fourier Transform (`fft`), perfectly mixes all the data uniformly across frequency spectrums in $O(n \log n)$ time avoiding dense standard multiplication. Finally, we randomly subsample columns (`idx`). I specifically ensured I apply the full complex FFT *before* projecting to `np.real()` to prevent destroying sequence magnitudes."

**Q: Why do you need `pivoted QR` (`qr(B, pivoting=True)`) in Interpolative Decomposition?**
**Naman's Answer:** "In Interpolative Decomposition, our specific goal isn't abstract orthonormalization—it's mathematically selecting *actual columns* of $A$. We compute $B = Q^T A$. By doing a *Column-Pivoted* QR on $B$, the algorithm actively computes vector norms and swaps the columns dynamically at runtime to pull the most linearly independent, highest-norm vectors into the extreme left of the matrices (`B P = Q_2 R_2`). The permutation array `piv` basically explicitly lists the index numbers: *'Columns [14, 82, 3...] of matrix A are your most important, defining columns!'*"

**Q: How are you computing $X$ in the equation $A \approx C X$? And why can $X$ recreate an entire identity block?**
**Naman's Answer:** "Since $C = A[:, J]$ are literal isolated columns strictly extracted from $A$, the coefficient matrix $X$ needs to map $C$ back precisely to approximate $A$. For the original indices $J$, $X$ will naturally format itself as an Identity matrix $I_k$ because a column perfectly reconstructs itself natively. For the rest of the columns, $X$ distributes the coefficients. I bypassed error-prone manual inversions (`[I | (R_11^-1) R_12] P^T`) and instead used Python's robust least squares solver `np.linalg.lstsq(C, A)` to compute the theoretically tightest $X$ without floating point division breakdowns."

---

### 🔴 ADITYA'S SECTION: Experiments & The Math Behind the Plots

**Q: In your Dashboard Experiment, how do you mathematically verify relative error across algorithms?**
**Aditya's Answer:** "After factoring the massive matrix, I explicitely re-multiplied the decomposition factors $A_{approx} = U \Sigma V^T$. Then I calculate the raw Frobenius norm (which sums the absolute matrix structural variance) of the overall difference matrix: `np.linalg.norm(A - A_approx)`. Finally, I divide this scalar by the absolute scalar norm of the parent test-bed `np.linalg.norm(A)` to yield a strict 0.0 to 1.0 fraction (relative error). Our randomized approaches effortlessly locked in identical relative errors ($\sim 1.33 \cdot 10^{-3}$) to deterministic routines."

**Q: In your code for Oversampling (`experiment_oversampling_effect`), you explicitly plot a red dashed line at `sigma_k1`. Why?**
**Sir's expectation:** Eckart-Young-Mirsky Theorem correlation.
**Aditya's Answer:** "This relies on the definitive Eckart-Young-Mirsky theorem, which states the absolute maximum strict accuracy any rank-$k$ approximation can computationally achieve is limited exactly to $\sigma_{k+1}$ (the singular value immediately following our cutoff index). In my Theorem 1.1 visualization, I anchored the red 'perfect limit' line directly to the test-matrix's generated $\sigma_{k+1}$. We visually prove that adding just 5 oversampling vectors drops our randomized SVD error directly onto this mathematically absolute optimal limit without generating an immense $m \times m$ standard SVD."

**Q: Explain how you generated matrices with a 'slow/flat' vs 'fast' singular value decay to trick/test Aryan's Power Iteration?**
**Aditya's Answer:** "In my helper code `_make_matrix()`, I explicitly bypassed standard python RNG wrappers to build deterministic custom diagonal bases. I populated the explicit diagonal matrices $\Sigma$ with fractional scalars using the exponent scale `1 / (i+1)**decay_exp`. If decay limit is $0.5$ (Flat), singular values remain numerically almost identical (e.g., $1.0, 0.707, 0.577...$), meaning columns are near-linear combinations mathematically causing classic SRFT and randomized methods to absorb surrounding noise. My charts definitively prove Aryan's $q=2$ iteration forces numerical divergence over flat noise to reconstruct ideal vectors."

---

### ⚡ Generic Catch-All Questions for Anyone

**Q: "I notice you randomly generate test matrices. Does PRNG (pseudo-randomness) vs True Randomness ruin the theoretical boundaries defined in the paper?"**
"No sir, the paper's deterministic boundaries are predicated on Gaussian distribution uniformity—not quantum unpredictability. By using `np.random.default_rng(seed)`, we generated uniformly unbiased standard deviations which structurally exceed standard statistical multidimensional randomness thresholds. The fundamental bounds operate natively via the Johnson-Lindenstrauss lemma; dimensional orthogonality is essentially guaranteed as long as independent vectors are normally standard distributions."

**Q: What is the most obvious takeaway or 'so what?' from this entire project?**
"Dimensionality limit bypass. SVD powers compression and models. The traditional $O(mn^2)$ algorithm completely halts large model operations. Using probability to reduce spatial queries reduces operational RAM limits massively, allowing calculations with a 10$\times$ proven velocity speedup locally without compromising analytical bounds."
