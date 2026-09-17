---
title: "03 — Linear Algebra for Graphs"
date: 2026-08-12T17:40:00+03:00
draft: false
params:
  math: true
---

# 03 — Linear Algebra for Graphs
*The bridge from Ch 01 to spectral GNNs: eigenvalues of the Laplacian and adjacency matrix, the graph Fourier transform, spectral graph convolution, and random-walk mixing.*

This chapter is the most mathematical in the cheatsheet. The point is to develop a working intuition for what the eigenstructure of a graph "means," and to show how every spectral GNN (ChebNet, spectral GCN, CayleyNets, heat-kernel GNNs) is the same idea with a different filter.

## 3.1 The Laplacian eigendecomposition

The Laplacian $L = D - A$ is real symmetric positive semi-definite, so it admits a spectral decomposition

$$L = U \Lambda U^\top = \sum_{i=1}^{n} \lambda_i u_i u_i^\top,$$

where $U$ is orthogonal ($U^\top U = I$) and $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$ with $0 = \lambda_1 \le \lambda_2 \le \cdots \le \lambda_n$. The vectors $u_i$ are the **Laplacian eigenvectors**; the scalars $\lambda_i$ are the **Laplacian eigenvalues** (sometimes called the **spectrum** of $L$).

A few key facts, each with a graph-level meaning:

| Fact | Meaning |
|---|---|
| $\lambda_1 = 0$, $u_1 = \mathbf{1}/\sqrt{n}$ (if connected) | $L$ has a constant null space |
| Multiplicity of $0$ = number of connected components | $\text{null}(L) = \text{span of indicator vectors of components}$ |
| $\lambda_n \le 2 \max_v \deg(v) = 2 \Delta$ | Spectral radius bound |
| $\lambda_n \ge (n/(n-1)) \bar d$ for connected | At least one big eigenvalue |
| $\sum_i \lambda_i = \text{tr}(L) = 2m$ | Sum of eigenvalues = sum of degrees |

**The quadratic form, restated.** For any $x \in \mathbb{R}^n$:

$$x^\top L x = \sum_i \lambda_i (u_i^\top x)^2 = \frac{1}{2} \sum_{(u,v) \in E} (x_u - x_v)^2.$$

This is the bridge to **graph signal processing**: a signal $x$ on the graph has total variation $x^\top L x$; small if $x$ is "smooth" with respect to the graph (neighbors have similar values). The eigenvalues of $L$ are the **frequencies** of this signal space; small eigenvalues are "low frequency" (smooth) signals, large eigenvalues are "high frequency" (oscillatory) signals.

> **Worked example.** Path graph $P_4$: vertices $\{1,2,3,4\}$, edges $\{(1,2),(2,3),(3,4)\}$. $L = \begin{pmatrix} 1 & -1 & 0 & 0 \\ -1 & 2 & -1 & 0 \\ 0 & -1 & 2 & -1 \\ 0 & 0 & -1 & 1 \end{pmatrix}$. Eigenvalues: $2 - 2\cos(k\pi/4)$ for $k = 0,1,2,3$, so $\{0, 2 - \sqrt{2}, 2, 2 + \sqrt{2}\} \approx \{0, 0.586, 2, 3.414\}$. Eigenvector for $\lambda_1 = 0$: $u_1 = (1,1,1,1)/2$ (constant). Eigenvector for $\lambda_4 = 2 + \sqrt{2}$: $u_4 \propto (1, -1, 1, -1)$ (alternating sign — the highest-frequency signal on the path).

## 3.2 The adjacency spectrum and how it differs

$A$ is also symmetric for undirected graphs, so it has a real spectrum. The difference from $L$ is the meaning: $L$ measures "smoothness," $A$ measures "co-occurrence."

The relationship: if $\lambda^A_i$ is an eigenvalue of $A$, then $\lambda^L_i = d_v - \lambda^A_i$ is an eigenvalue of $L$ **only if the graph is regular** ($d_v$ constant for all $v$). In general, the spectra of $A$ and $L$ are not related by a simple shift because $A$ and $D$ don't share eigenvectors.

For a $d$-regular graph, $A$ and $L$ commute ($L = dI - A$, $AL = (dI - A)L = LA$ trivially), and $L$ has eigenvalues $d - \lambda^A_i$. This is why many GNN theory papers assume $d$-regularity: it makes spectral analysis tractable.

**Eigenvalues of $A$ and regularity** — the rule of thumb: for a $d$-regular graph, $A$ has largest eigenvalue exactly $d$ (with eigenvector $\mathbf{1}$), and all other eigenvalues are bounded in $[-d, d]$.

> **Worked example.** $K_4$ (complete graph on 4 vertices): regular of degree 3. Eigenvalues of $A$: $3$ (with eigenvector $\mathbf{1}$), and $-1$ with multiplicity 3. So $L = 3I - A$ has eigenvalues $0$ (mult. 1) and $4$ (mult. 3). ✓ (matches §1.7 of Ch 01).

## 3.3 The graph Fourier transform

Define the **graph Fourier transform** of a signal $x \in \mathbb{R}^n$:

$$\hat x = U^\top x, \quad \text{so} \quad \hat x_i = u_i^\top x = \langle u_i, x \rangle.$$

And the **inverse graph Fourier transform**:

$$x = U \hat x = \sum_i \hat x_i u_i.$$

The $u_i$ are the "Fourier basis" of the graph. The signal $x$ is decomposed into $n$ modes; mode $i$ has amplitude $\hat x_i$ and lives in the direction of $u_i$.

**Why $u_i$ are frequencies.** The Laplacian $L$ is the "graph Laplacian operator" — analogous to $-\nabla^2$ in continuous space. Its eigenfunctions are sinusoids (in 1D, 2D, etc.). The "smoothness" measure $x^\top L x = \sum \lambda_i \hat x_i^2$ says: the total variation of $x$ is the squared Fourier coefficients weighted by their eigenvalues. Small eigenvalues = low frequency = smooth signals; large eigenvalues = high frequency = oscillatory signals.

**Filtering.** A "filter" in the graph Fourier domain is a function $g: \mathbb{R}_{\ge 0} \to \mathbb{R}$ applied to each eigenvalue:

$$y = U g(\Lambda) U^\top x, \quad \text{where } g(\Lambda) = \text{diag}(g(\lambda_1), \ldots, g(\lambda_n)).$$

This is **spectral graph convolution**. The output $y$ at vertex $v$ is a linear combination of the input $x$, where the weights depend on the spectrum.

> **Worked example.** Take $P_4$ from §3.1, $L$ eigenvalues $\{0, 0.586, 2, 3.414\}$. Define $g(\lambda) = 1 / (1 + \lambda)$ (a low-pass filter). The filter values are $\{1, 0.631, 0.333, 0.226\}$. For a signal $x = (1, 0, 0, 0)^\top$ (delta at vertex 1), the output is the graph Fourier transform of this delta, filtered, then inverted: $y = U g(\Lambda) U^\top x = g(\Lambda) u_1 / \|u_1\| = (0.5, 0.5, 0.5, 0.5)^\top$ (the first column of $U$ scaled by $g(\lambda_1)=1$ and normalised). Hmm, that example was degenerate because $\lambda_1=0$ has a unit vector. Let me redo with $x = (1, -1, 1, -1)^\top$ (the $\lambda_4$ eigenvector). Then $\hat x = (0,0,0,\|x\|)$ and $y = U g(\Lambda) \hat x = g(\lambda_4) u_4 \|x\| = 0.226 \cdot u_4 \cdot 2 = 0.452 \cdot u_4$. The high-frequency signal is attenuated by the low-pass filter. ✓

## 3.4 Polynomial filters and ChebNet

Computing $U$ for an $n \times n$ matrix is $O(n^3)$. You cannot do this for large graphs. The standard fix: **polynomial filters**. Approximate $g$ with a polynomial of degree $K$:

$$g(\lambda) \approx \sum_{k=0}^{K} \theta_k \lambda^k.$$

Then

$$y = U g(\Lambda) U^\top x = \sum_k \theta_k L^k x.$$

The result: a $K$-degree polynomial filter is computed by $K$ matrix-vector products, no eigendecomposition. This is the **graph convolution by polynomial** idea, and it's the basis of **ChebNet** ([`1606.09375`](https://arxiv.org/abs/1606.09375), Defferrard, Bresson, Vandergheynst 2016) and **Graph Convolutional Network** ([`1609.02907`](https://arxiv.org/abs/1609.02907), Kipf & Welling 2017).

**Chebyshev polynomials** $T_k$ are used because they are stable on $[-1, 1]$ and orthogonal, so they give a clean basis. Define $\tilde L = 2 L / \lambda_n - I$ (rescale $L$ to $[-1, 1]$). Then a Chebyshev filter of degree $K$ is

$$y = \sum_{k=0}^{K} \theta_k T_k(\tilde L) x,$$

computed by the recurrence $T_0(x) = x$, $T_1(x) = x$, $T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)$. Each $T_k(\tilde L) x$ costs one matrix-vector product, so total cost is $O(K \cdot m)$.

**Kipf–Welling GCN** is the $K = 1$ ChebNet with one extra trick: add self-loops (so $A \to A + I$) and renormalise by degree. The result is the famous propagation rule

$$H^{(\ell+1)} = \sigma\left( \hat A H^{(\ell)} W^{(\ell)} \right), \quad \hat A = D^{-1/2}(A + I) D^{-1/2}.$$

This is **not** an exact spectral filter — it's a first-order Taylor approximation of one. Many papers have noted that the "spectral" interpretation of GCN is loose; the practical success of GCN comes from the propagation rule, not from the spectral theory.

> **Worked example.** ChebNet on $P_4$, $K = 2$, $\theta_0 = 0.5, \theta_1 = -0.3, \theta_2 = 0.1$, $x = (1, 0, 0, 0)^\top$. $\tilde L = 2L/\lambda_4 - I = 2L/3.414 - I$. Compute $T_0(\tilde L) x = x$. $T_1(\tilde L) x = \tilde L x$. $T_2(\tilde L) x = 2 \tilde L^2 x - x$. Sum weighted by $\theta$: $y = 0.5 \cdot x - 0.3 \cdot \tilde L x + 0.1 \cdot (2 \tilde L^2 x - x) = (0.4) x - 0.3 \tilde L x + 0.2 \tilde L^2 x$. Compute $\tilde L x$ = first column of $\tilde L$ = $\frac{2}{3.414}(1, -1, 0, 0)^\top - (1, 0, 0, 0)^\top \approx (-0.414, -0.586, 0, 0)^\top$. Compute $\tilde L^2 x$ = first column of $\tilde L^2$. (You'll need to compute $\tilde L^2$ explicitly.) The output $y$ is a 4-vector; the second element (vertex 2) is non-zero because of the propagation.

## 3.5 Spectral vs. spatial

There are two views of GNN convolution:

**Spectral view.** The filter is a function $g$ of the eigenvalues; the propagation is $U g(\Lambda) U^\top X$. Defined for undirected graphs with explicit eigen-decomposition; doesn't generalise to directed or signed graphs.

**Spatial view.** The filter is "applied locally": each vertex aggregates features from its neighbors, then a transform, then a non-linearity. This is the **message-passing** view (Ch 05), and it works on any graph you can write down.

The two views are **equivalent** for polynomial filters of bounded degree, by the argument in §3.4: a $K$-local spatial filter is a $K$-polynomial spectral filter. They diverge for non-polynomial spectral filters, which are uncomputable for large $n$.

**Theorem (spectral = spatial for polynomials).** For any polynomial $p$ of degree $K$, there is a $K$-hop local aggregation that computes $U p(\Lambda) U^\top X$, and vice versa. So in practice, all message-passing GNNs are polynomial spectral filters.

> **Worked example.** GCN propagation $\hat A X W$ with $\hat A = D^{-1/2}(A+I)D^{-1/2}$. Spectral view: $\hat A$ is not symmetric (unless you mean a specific normalisation) — but you can symmetrise. The point is that $\hat A H^{(\ell)} W$ can be written as $U p(\Lambda) U^\top H^{(\ell)} W$ for some polynomial $p$ of degree 1 (in fact, $p(\lambda) = (1 - \lambda_{\text{eff}}) \cdot \text{const}$). The non-linearity $\sigma$ breaks the spectral interpretation.

## 3.6 The normalised Laplacian

The two normalised Laplacians from Ch 01 have spectra in $[0, 2]$:

**Symmetric** $L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}$. Eigenvalues in $[0, 2]$; the constant vector has eigenvalue $0$. The eigenvectors form an orthonormal basis even for irregular graphs.

**Random walk** $L_{\text{rw}} = I - D^{-1} A$. Eigenvalues in $[0, 2]$; the right eigenvector of $0$ is $\mathbf{1}$, but the left eigenvector is $\pi = D \mathbf{1} / 2m$ (the stationary distribution). It is not symmetric, so the eigenvectors are not orthogonal. It is a "right" eigenproblem: $L_{\text{rw}} v = \lambda v$ for $v$, but $v^\top L_{\text{rw}} = \lambda v^\top$ uses $\pi$ as the inner product.

**Why two normalisations?** $L_{\text{sym}}$ gives a unitary transform ($U^\top U = I$); $L_{\text{rw}}$ gives a stochastic operator ($P = I - L_{\text{rw}}$ is row-stochastic). For GNN propagation, $L_{\text{sym}}$ is the default (because you can multiply by $W$ on either side without changing norms). For random-walk interpretations (DeepWalk, APPNP), $L_{\text{rw}}$ is the right object.

> **Worked example.** Triangle $K_3$ (3-regular): $L_{\text{sym}} = L_{\text{rw}} = I - (1/2) A$. Eigenvalues: $0$ (constant), $3/2$ with multiplicity 2. Both normalisations agree because the graph is regular.

## 3.7 Random walks and spectral mixing

The **random-walk transition matrix** is $P = D^{-1} A$. The probability of being at vertex $v$ after $t$ steps starting from $u$ is $(P^t)_{uv}$. The **mixing time** is the smallest $t$ such that for all $u, v$, $|(P^t)_{uv} - \pi_v| < \epsilon$, where $\pi$ is the stationary distribution.

**Spectral interpretation.** $P$ is not symmetric, but it has the same eigenvalues as $L_{\text{rw}} = I - P$ (since the spectrum is preserved under the transformation $L_{\text{rw}} = I - P$). If $L_{\text{rw}}$ has eigenvalues $0 = \mu_1 \le \mu_2 \le \cdots \le \mu_n$ (in $[0, 2]$), then $P$ has eigenvalues $1 \ge 1 - \mu_2 \ge \cdots \ge 1 - \mu_n \ge -1$.

The **spectral gap** is $\mu_2 = 1 - \lambda_2^{(P)}$, the second-smallest eigenvalue of $L_{\text{rw}}$. The mixing time is bounded by

$$t_{\text{mix}} \le \frac{1}{\mu_2} \log \frac{n}{\epsilon \pi_{\min}}.$$

**The "expressive" spectral gap.** When $\mu_2$ is small (close to 0), the graph has **bottleneck structure** — the walk is slow to mix, and there is a meaningful bipartition or community split. When $\mu_2$ is large, the graph is "expander-like" — every cut is roughly balanced, and the walk mixes fast.

**Connection to GNNs.** Graph Transformers (Ch 08) and structural pretraining (Ch 09) use **spectral features** (eigenvectors of $L$) as positional encodings precisely because the eigenvectors capture this structural information. The first non-trivial eigenvector $u_2$ is the **Fiedler vector** — it gives a 1D embedding of the vertices that respects the graph's "slowest" mode.

> **Worked example.** Bar graph: $K_5 \cup K_5$ joined by a single edge. The two cliques are "tight" communities; the bridge edge is a bottleneck. $\mu_2$ is small (close to 0); the Fiedler vector is roughly $+1$ on one clique, $-1$ on the other, and ~$0$ on the bridge. This is the spectral signature of community structure.

## 3.8 Cheeger's inequality and conductance

For a graph $G$ and a partition $(S, V \setminus S)$, the **Cheeger ratio** (conductance) is

$$h(S) = \frac{|\partial S|}{\min(\text{vol}(S), \text{vol}(V \setminus S))}, \quad \text{where } \partial S = \{e \in E : e \text{ has one endpoint in } S\}, \quad \text{vol}(S) = \sum_{v \in S} \deg(v).$$

The **Cheeger constant** is $h(G) = \min_S h(S)$.

**Cheeger's inequality** (Alon, Milman 1985; Alon 1986):

$$\frac{\mu_2}{2} \le h(G) \le \sqrt{2 \mu_2}.$$

This says: a small spectral gap is **necessary and sufficient** for the existence of a "bottleneck" cut, up to a quadratic factor. The Fiedler vector $u_2$ gives an **approximately optimal** bipartition (sweep cut: sort vertices by $u_2(v)$, try all cuts along the sorted order, pick the best).

**Why this matters for GFMs.** Community / cluster structure is a major source of useful structure in real-world graphs. The Cheeger bound says you can find it by computing one Laplacian eigenvector. A GFM that produces embeddings aligned with the Fiedler vector will naturally "see" communities.

> **Worked example.** Cycle $C_n$: $\mu_2 = 1 - \cos(2\pi/n) \approx 2\pi^2/n^2$. The Cheeger constant is $h(C_n) = 2/n$ (the cut $\{1, 2\}, \{3, \ldots, n\}$ has 2 boundary edges and $\min(\text{vol}) = 2 \cdot 2 = 4$, so $h = 2/4 = 0.5$ — wait, that's $2/n$? Let me redo: $|\partial\{1\}| = 2$ (two edges of vertex 1), $\text{vol}(\{1\}) = 2$, so $h(\{1\}) = 2/2 = 1$ — so the Cheeger constant is at most 1. But the actual minimiser is a contiguous block: $h(\{1, 2, \ldots, k\}) = 2 / \min(2k, 2(n-k))$, minimised at $k = n/2$ giving $h = 2/n$. ✓

## 3.9 Spectral methods in modern GNNs

Spectral ideas are not just historical. They show up in:

| Method | Spectral idea | Where |
|---|---|---|
| ChebNet | Chebyshev polynomial filter on $L$ | [`1606.09375`](https://arxiv.org/abs/1606.09375) |
| GCN | First-order ChebNet with normalisation | [`1609.02907`](https://arxiv.org/abs/1609.02907) |
| CayleyNet | Cayley transform (rational filter on $L$) | Levie et al. 2018 |
| Heat-kernel GNN | $g(\lambda) = e^{-t\lambda}$ | Xu et al. 2018 |
| ARMA filter | Auto-regressive moving average on $L$ | Bianchi et al. 2021 |
| Spectral attention (SAN) | $A \odot M$ where $M$ is a learnable spectral mask | [`2106.03893`](https://arxiv.org/abs/2106.03893) |
| GraphGPS | Local message passing + global attention, both informed by spectral PE | [`2205.12454`](https://arxiv.org/abs/2205.12454) |
| SIGN | Scalable inception-style GNN using $A^k X$ powers | Frasca et al. 2020 |
| Position encodings (LapPE, SignNet) | First $k$ non-trivial eigenvectors of $L$ | Dwivedi & Bresson 2020; [`2102.12798`](https://arxiv.org/abs/2102.12798) |

**Spectral position encodings (LapPE).** Take the first $k$ non-trivial eigenvectors of $L$, $U_k = [u_2, u_3, \ldots, u_{k+1}] \in \mathbb{R}^{n \times k}$, and append $U_k$ as node features. This breaks the permutation symmetry of message passing: after LapPE, vertex 1 and vertex 2 are distinguishable even if they have the same degree. LapPE is the dominant PE in Graph Transformers (Ch 08).

**SignNet** ([`2102.12798`](https://arxiv.org/abs/2102.12798)) learns a function on the *signs* of the eigenvectors to handle the $U \to -U$ sign ambiguity (the spectrum determines $u$ only up to a sign per coordinate).

> **Worked example.** Path $P_4$, LapPE with $k=2$. Eigenvectors: $u_2 \approx (0.65, 0.27, -0.27, -0.65)$ and $u_3 \approx (0.27, -0.65, -0.65, 0.27)$ (for eigenvalues 0.586 and 2). Append these as features. Now vertices 1 and 2 are distinguishable: $(0.65, 0.27)$ vs $(0.27, -0.65)$. Without LapPE, both are degree-1 in a path — only their position in the path distinguishes them, and message passing cannot recover position information.

## 3.10 Worked example: spectral clustering

Putting it all together. To cluster a graph into 2 clusters:
1. Compute $L_{\text{sym}}$.
2. Compute the Fiedler vector $u_2$ (the eigenvector for $\lambda_2$).
3. Cluster by sign of $u_2$: $S = \{v : u_2(v) > 0\}$, $T = \{v : u_2(v) \le 0\}$.

This is **spectral clustering**, and by Cheeger's inequality, the cut $(S, T)$ is near-optimal in the Cheeger sense. For $k > 2$ clusters, use the first $k$ non-trivial eigenvectors and run k-means in the resulting $\mathbb{R}^k$ space.

**Spectral clustering is the ancestor of every graph-pooling method** that uses the Fiedler vector or its analogues. It is also the inspiration for **MinCut pooling** ([`1903.03894`](https://arxiv.org/abs/1903.03894) cites it), **DMoN pooling**, and several graph-classification pretraining objectives.

## Exercises

1. **Spectrum of $K_n$.** Compute the eigenvalues and eigenvectors of $L$ for $K_n$. Verify that $\lambda_1 = 0$ (constant vector), $\lambda_i = n$ for $i = 2, \ldots, n$.

2. **Spectrum of $K_{3,3}$.** Compute the eigenvalues of $L$ for the complete bipartite graph $K_{3,3}$. Verify the formula in §1.7: $\{0, 6, 6, 3, 3, 3, 3\}$? Or is it $\{0, 3, 3, 3, 3, 6\}$? (Check the multiplicities from Ch 01.)

3. **Quadratic form on a real graph.** Take the Zachary's Karate Club graph (built into NetworkX as `nx.karate_club_graph()`). Compute the Fiedler vector and split the 34 members into two groups. How many are in the "wrong" community (compared to the actual split that occurred in 1977)? (The actual split is into 2 groups of 16 and 18; Fiedler-cut typically has 0 or 1 misclassified members.)

4. **ChebNet propagation by hand.** For $P_4$ with $L$ as in §3.1, compute the output of a 2-degree Chebyshev filter with $\theta_0 = 1, \theta_1 = 0, \theta_2 = -1$ on the signal $x = (1, 2, 3, 4)^\top$. Use $\tilde L = 2L/3.414 - I$. Compare to the original $x$ — is the output "smoother" (less variation between neighbors)?

5. **Random walk mixing on a bipartite graph.** Walk on $C_4$ (bipartite). Compute $P^t \mathbf{e}_1$ for $t = 0, 1, 2, 3, \ldots$ by hand. Show that the distribution does **not** converge; instead, it alternates between two distributions. Why?

6. **PageRank power iteration.** Implement power iteration for PageRank in numpy on a 10-vertex graph. Verify convergence to a stationary distribution. Plot the L1 distance between consecutive iterates vs. iteration number on a log scale. The slope should be $-O(\log(\alpha^{-1}))$ where $\alpha$ is the teleport probability.

7. **Fiedler vector as a positional encoding.** Take a random $G(20, 0.3)$ graph. Compute the Fiedler vector. Now pick a pair of vertices with the same degree; show that their Fiedler-vector coordinates are different. This demonstrates the "LapPE breaks degree symmetry" property.

8. **Polynomial filter equivalence.** For the cycle $C_6$ with $L$ known analytically, write a degree-2 polynomial filter and an equivalent 2-hop message-passing rule. Confirm the equivalence: the polynomial filter $g(L) = a I + b L + c L^2$ is the same as applying the message passing rule $(I + \alpha \hat A + \beta \hat A^2) X W$ for appropriate $a, b, c, \alpha, \beta$ (depending on the normalisation).

9. **Spectrum of the path.** Prove the formula $\lambda_k = 2 - 2 \cos(k \pi / (n-1))$ for the path graph $P_n$. (Hint: assume $u_k(v) = \sin(k v \pi / n)$ and verify it is an eigenvector with the claimed eigenvalue.)

10. **Laplacian eigenvalues from a paper.** Pick a recent GFM paper (e.g., [`2505.15116`](https://arxiv.org/abs/2505.15116), [`2205.12454`](https://arxiv.org/abs/2205.12454), or [`2106.05234`](https://arxiv.org/abs/2106.05234)). Identify which spectral ideas it uses. Sketch how a spectral filter would be applied in its architecture.

## Further Reading

- **Chung, *Spectral Graph Theory*** (CBMS Lecture Notes 1997, freely available). The mathematical foundation. Read Ch 1–3 for §3.1–3.7 of this chapter.
- **Spielman, *Spectral and Algebraic Graph Theory*** (online draft, Yale 2024). Cleaner, more modern treatment of the same material.
- **Tremblay, Loukas, *Fractional graph Laplacians...*** (not directly related but the modern generalisation direction).
- **[`1606.09375`](https://arxiv.org/abs/1606.09375)** — Defferrard, Bresson, Vandergheynst, *ChebNet: CNNs on Graphs with Fast Localized Spectral Filtering* (NeurIPS 2016). The classic spectral GNN.
- **[`1609.02907`](https://arxiv.org/abs/1609.02907)** — Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks* (ICLR 2017). The most-cited GNN paper; the propagation rule is a first-order ChebNet.
- **[`2106.03893`](https://arxiv.org/abs/2106.03893)** — Kreuzer, Beaini, Hamilton, Blondel, Lio, *Rethinking Graph Transformers with Spectral Attention* (SAN, NeurIPS 2021). The cleanest "spectral attention" paper; the spectral mask is a learnable function of the eigenvalues.
- **[`2205.12454`](https://arxiv.org/abs/2205.12454)** — Rampášek et al., *Recipe for a General, Powerful, Scalable Graph Transformer* (GPS, NeurIPS 2022). Combines message passing + spectral global attention.
- **[`2102.12798`](https://arxiv.org/abs/2102.12798)** — Lim, Robinson, Zhao, Smidt, Sra, Ceci, Maron, *SignNet* (ICML 2022). Sign-equivariant positional encoding using Laplacian eigenvectors.
- **[`1903.03894`](https://arxiv.org/abs/1903.03894)** — Ying et al., *GNNExplainer* (NeurIPS 2019). Uses Fiedler-vector-style ideas to identify important subgraphs.
- **Von Luxburg, *A Tutorial on Spectral Clustering*** (Statistics and Computing 2007). The 50-page tutorial that everyone cites; the most useful single paper on spectral methods in ML.
- **Shi & Malik, *Normalized Cuts and Image Segmentation*** (TPAMI 2000). The original normalised-cut / spectral-clustering paper for image segmentation.
- **Luxburg, Belkin, Bousquet, *Consistency of Spectral Clustering*** (Annals of Statistics 2008). The consistency theory: as $n \to \infty$, does spectral clustering recover the true community structure?
- **Anderson, Morley, *Eigenvalues of the Laplacian of a Graph*** (Linear and Multilinear Algebra 1985). The early eigenvalue-inequality paper.
- **For Cheeger's inequality in detail**: Cheeger, *A Lower Bound for the Smallest Eigenvalue of the Laplacian* (Problems in Analysis 1970); Alon & Milman, *$\lambda_1$, Isoperimetric Inequalities for Graphs and Superconcentrators* (JCTB 1985).
