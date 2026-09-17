---
title: "08 — Graph Transformers"
date: 2026-08-12T17:40:00+03:00
draft: false
params:
  math: true
---

# 08 — Graph Transformers
*The transformer as a graph model: positional encodings, spectral attention, global vs. local mixing, and the architectures that dominate OGB leaderboards.*

Graph Transformers (GTs) are the dominant architecture in the modern GFM literature. The core idea: take the transformer (Ch 04 §4.6) and adapt it for graph data. The "adaptation" is mostly about handling the permutation symmetry that the graph imposes, and providing a positional encoding that breaks the symmetry.

## 8.1 The fundamental challenge: permutation symmetry

A standard transformer (Ch 04 §4.6) is **permutation-equivariant** on a sequence of tokens. On a graph, the "tokens" are vertices, and the graph has no inherent order. So the transformer is naturally suited for graph data — but two things are needed:

1. **A positional encoding (PE)** that breaks the symmetry. Without it, two vertices with the same features and same neighborhood structure get the same embedding.

2. **A graph-aware attention mask** (or its absence). Standard attention is fully connected (every token attends to every other). On a graph, you may want to:
   - **Restrict attention to local neighbors** (local attention, like GAT).
   - **Mix local and global** (local GNN + global attention, like GPS).
   - **Use full attention** (treating the graph as a complete graph with edge features, like Graphormer).

> **Worked example.** A 4-cycle with features $X = \begin{pmatrix} 1 \\ 0 \\ 1 \\ 0 \end{pmatrix}$ (vertex 1 and 3 have feature 1, vertex 2 and 4 have feature 0). A standard transformer without PE: tokens 1 and 3 produce the same output (by symmetry of the input and the model). A transformer with LapPE: token 1 has $(u_2, u_3) = (0.65, 0.27)$ and token 3 has $(u_2, u_3) = (-0.27, -0.65)$ — distinguishable.

## 8.2 Positional encodings for graphs

This subsection surveys the main PE families used in Graph Transformers.

### 8.2.1 Laplacian PE (LapPE)

Take the first $k$ non-trivial eigenvectors of the symmetric normalised Laplacian $L_{\text{sym}}$:

$$U_k = [u_2, u_3, \ldots, u_{k+1}] \in \mathbb{R}^{n \times k}.$$

Use $U_k$ as node features. This is the **LapPE** used in Graphormer ([`2106.05234`](https://arxiv.org/abs/2106.05234), Ying et al. 2021) and many subsequent works.

**Issue: the $U \to -U$ sign ambiguity.** Each eigenvector is determined only up to a sign. SignNet ([`2102.12798`](https://arxiv.org/abs/2102.12798), Lim et al. 2022) addresses this by learning a sign-equivariant function on the eigenvectors.

**Cost.** Eigendecomposition is $O(n^3)$. For large graphs, use Lanczos iteration to get the first $k$ eigenvectors in $O(km)$ time. Memory: $O(kn)$.

### 8.2.2 Random walk PE (RWPE)

For each vertex $v$ and each $k = 1, \ldots, K$:

$$\text{RWPE}_k(v) = (A^k \mathbf{e}_v)_v = (A^k)_{v, v}.$$

This is the probability of returning to $v$ after a $k$-step random walk. Captures local graph structure without eigendecomposition. Cost: $O(Km)$ for all vertices and all $K$.

**vs LapPE.** RWPE is cheaper but less expressive. LapPE captures global structure; RWPE captures local random-walk probabilities. Often combined.

### 8.2.3 SignNet

Learn a function on the *signs* of the Laplacian eigenvectors. Handles the $U \to -U$ ambiguity by aggregating the values of $f(u)$ and $f(-u)$ across all eigenvectors. More expensive but provably sign-equivariant.

### 8.2.4 Distance / shortest-path PE

Pick a set of $K$ anchor vertices. For each vertex $v$, the PE is the vector of shortest-path distances to the anchors: $(d(v, a_1), d(v, a_2), \ldots, d(v, a_K))$. Anchors can be chosen by random sampling, by degree (highest degree first), or by learnable selection.

### 8.2.5 Heat kernel PE

$\text{HKPE}_t(v, u) = \exp(-t L)_{v, u}$ for some diffusion time $t$. Similar in spirit to LapPE but uses a specific kernel. Not widely adopted in practice.

> **Worked example.** A path $P_6$, LapPE with $k = 2$. Eigenvectors: $u_2 \approx (0.23, 0.42, 0.5, 0.5, 0.42, 0.23)$ and $u_3 \approx (0.5, 0.5, 0, -0.5, -0.5, 0)$. The first is "smooth" (peak in the middle); the second is "anti-symmetric" (positive on one half, negative on the other). Combining, vertex 1 has $\approx (0.23, 0.5)$, vertex 3 has $\approx (0.5, 0)$, vertex 6 has $\approx (0.23, -0.5)$ — all distinguishable.

## 8.3 The Graphormer

**Paper.** [`2106.05234`](https://arxiv.org/abs/2106.05234) — Ying et al., *Do Transformers Really Perform Bad for Graph Representation?* (NeurIPS 2021).

**Architecture.**
1. Embed each vertex's features plus a LapPE.
2. Add a learned "centrality encoding" based on the vertex's in/out degree.
3. Add a learned "spatial encoding" based on the shortest-path distance between vertex pairs.
4. Add a learned "edge encoding" — applied inside the attention based on the edge feature.
5. Run a stack of standard transformer layers.

**Spatial encoding.** For each pair $(v, u)$, compute the shortest-path distance $d(v, u)$ and look up a learned embedding $b_{d(v, u)}$. Add this to the attention score:

$$\alpha_{vu} = \frac{(W_Q h_v)^\top (W_K h_u) + b_{d(v, u)}}{\sqrt{d}}.$$

This biases attention toward "close" vertices.

**Edge encoding in attention.** For each edge $(v, u)$ with edge feature $e_{vu}$, look up a learned embedding $c_{e_{vu}}$ and add to the attention:

$$\alpha_{vu} = \frac{(W_Q h_v)^\top (W_K h_u) + b_{d(v, u)} + c_{e_{vu}}}{\sqrt{d}}.$$

For non-edges, $c_{e_{vu}} = 0$ (or treated as a special "no edge" value).

**Properties:**
- Treats the graph as a complete graph with edge features for non-edges set to a special value. This makes the model very expressive (full attention) but expensive ($O(n^2 d)$ per attention layer).
- The spatial and edge encodings inject graph-specific inductive bias into the otherwise domain-agnostic transformer.

> **Worked example.** OGB-LSC PCQM4Mv2 (molecular property prediction, ~4M molecules, regression to HOMO-LUMO gap). Graphormer is the OGB-LSC winner. The attention is over all atom pairs in a molecule (~30 atoms on average), so $O(n^2) = O(900)$ is manageable.

## 8.4 The GPS recipe: local + global

**Paper.** [`2205.12454`](https://arxiv.org/abs/2205.12454) — Rampášek et al., *Recipe for a General, Powerful, Scalable Graph Transformer* (GPS, NeurIPS 2022).

**Architecture.** Each GPS layer has **two** sublayers:
1. **Local message passing**: a 1-WL-bounded MPNN (e.g., GAT, GINE).
2. **Global attention**: a standard multi-head self-attention layer, applied to all vertices in the graph.

The two are combined with a residual connection and a feed-forward layer. The local MPNN handles nearby structure; the global attention handles long-range.

**Positional/structural encodings.** GPS uses both LapPE and RWPE, learnable random-walk structural encoding (RWSE, the diagonal of $A^k$ for $k = 1, \ldots, K$), and (optionally) SignNet.

**Key insight.** The MPNN already captures local structure; the attention layer adds the global view. The combination is strictly more expressive than either alone.

**Cost.** $O(m d)$ for the MPNN sublayer + $O(n^2 d)$ for the attention sublayer. For large graphs, the $O(n^2)$ term dominates, so GPS uses **sparse attention** variants (BigBird, Performer-style, etc.).

**Why GPS is the most-cited GT.** It is the most practical: explicit recipe, works out-of-the-box, scales reasonably. The GPS paper is the closest thing to a "standard" GT architecture.

> **Worked example.** OGB-LSC PCQM4Mv2 leaderboard (as of 2024): GPS variants in the top-10. Specifically, GPS with LapPE + RWSE + GINE-style local MPNN achieves MAE around 0.085 eV on the HOMO-LUMO gap task — the winning non-ensemble model.

## 8.5 SAN: Spectral Attention Networks

**Paper.** [`2106.03893`](https://arxiv.org/abs/2106.03893) — Kreuzer, Beaini, Hamilton, Blondel, Lio, *Rethinking Graph Transformers with Spectral Attention* (NeurIPS 2021).

**Architecture.** SAN replaces the standard attention with a **learnable spectral filter** on the Laplacian. The attention coefficients are

$$\alpha_{vu} = \frac{(W_Q h_v)^\top (W_K h_u)}{\sqrt{d}} \cdot \sigma\left( f(\lambda_v) \cdot f(\lambda_u) \right),$$

where $f$ is a learnable function of the Laplacian eigenvalues at the vertices (or of the Laplacian eigenvalues of the full graph, applied positionally). The "spectral mask" $\sigma(f(\lambda_v) f(\lambda_u))$ biases the attention based on the spectral structure.

**Properties:**
- The spectral mask is a learnable filter on the graph's frequency content.
- Can be thought of as "soft spectral convolution" + attention.

> **Worked example.** On a barbell graph, the Fiedler vector $u_2$ has eigenvalue $\lambda_2 \approx 0$ (small spectral gap). A spectral mask $f(\lambda) = 1 / (1 + \lambda)$ would upweight the Fiedler mode — i.e., the model pays more attention to vertices that are "diametrically opposed" in the Fiedler sense (across the bottleneck).

## 8.6 TokenGT, Graphormer, and the "graph as tokens" family

**TokenGT** ([`2207.02505`](https://arxiv.org/abs/2207.02505), Kim et al. NeurIPS 2022). Treats the graph as a sequence of tokens, where each vertex and each edge is a token. Uses standard transformer attention over all tokens. The "edges" are added as additional tokens that attend to the adjacent vertex tokens and are attended by them.

**Graph Inductive Bias (GIB)** — many of these models are studied under the name "graph inductive bias." The taxonomy:
- **Implicit graph bias** (TokenGT, some Graphormer variants): the graph is encoded in the PE and edge embeddings, but the architecture is fully attention.
- **Explicit graph bias** (GPS, SAN): the architecture has separate sublayers for graph operations.

> **Worked example.** TokenGT on the 4-cycle: 4 vertex tokens and 4 edge tokens, total 8 tokens. The vertex tokens attend to each other (with the same attention as in a sequence). The edge tokens are inserted between vertex tokens. Standard transformer attention over 8 tokens is $O(64 d)$ per head per layer.

## 8.7 NAGphormer, NodeFormer, SGFormer

These three are designed for **scalable** graph transformers on large graphs.

**NAGphormer** ([`2206.04910`](https://arxiv.org/abs/2206.04910), Chen et al. NeurIPS 2022). Aggregates multi-hop neighborhood information into a sequence of tokens per vertex. The transformer is over the sequence of "hop representations."

**NodeFormer** ([`2306.08385`](https://arxiv.org/abs/2306.08385), Wu et al. NeurIPS 2022). Uses a **kernelized Gumbel-Softmax** attention that approximates the full $O(n^2)$ attention in $O(n)$ time. Allows training on very large graphs.

**SGFormer** (Wu et al. NeurIPS 2023). Combines a single-layer global attention with a local GNN. Achieves linear scaling by attending only to a subset of nodes + a learnable "global token" that aggregates everything.

| Method | Complexity | Best for |
|---|---|---|
| Graphormer | $O(n^2 d)$ | Small graphs, full expressivity |
| GPS | $O(n^2 d + m d)$ | Medium graphs, strong baselines |
| TokenGT | $O((n+m)^2 d)$ | Medium graphs, sequence-style |
| SAN | $O(n^2 d)$ | Small graphs with strong spectral structure |
| NAGphormer | $O(n d)$ | Large node-classification |
| NodeFormer | $O(n d)$ | Very large node-classification |
| SGFormer | $O(n d)$ | Million-scale node-classification |

> **Worked example.** On OGB-Papers100M (111M nodes, 1.6B edges), a vanilla transformer is infeasible. SGFormer can train in reasonable time on a single GPU (with appropriate sampling). Test accuracy: ~67% — competitive with SOTA GNN baselines.

## 8.8 PE ablation: what works

From the Graphormer, GPS, and SAN papers, the practical PE ranking:

1. **LapPE** with the first 3-10 non-trivial eigenvectors: significant improvement on most tasks.
2. **RWSE / RWPE** with the first 5-10 random-walk return probabilities: improvement on most tasks.
3. **SignNet** for handling the sign ambiguity: required if you use LapPE and want full expressivity.
4. **Distance encoding** (shortest-path distance to anchors): helpful on large graphs where LapPE is expensive.
5. **Centrality encoding** (degree-based): small improvement on most tasks.

The combination of LapPE + RWSE + SignNet is the most commonly used "full" PE stack in modern GTs.

> **Worked example.** A 4-layer GAT on OGB-ArXiv without PE: ~73% test accuracy. With LapPE: ~75%. With LapPE + RWSE + SignNet: ~76%. Each PE adds ~0.5-1.5 points.

## 8.9 The relationship between GTs and MPNNs

A Graph Transformer is **strictly more expressive** than an MPNN with the same depth, in the following sense:

- An MPNN is 1-WL-bounded.
- A transformer with full attention and LapPE can distinguish any two non-isomorphic graphs (modulo the sign ambiguity, which SignNet resolves).

**Theorem (informal).** A transformer with $L$ layers, full attention, and a sufficient positional encoding (e.g., LapPE with all eigenvectors) is as powerful as the $L$-step WL test, in the limit.

**Practical caveat.** This expressivity is rarely reached in practice, because:
- Only a finite number of LapPE eigenvectors are used (cost of eigendecomposition).
- The transformer may not be deep enough to refine the WL colors.
- The training data may not be sufficient to learn the distinguishing functions.

**For GFM design.** A GFM built on a GT backbone is potentially more expressive than one built on an MPNN. The trade-off: GTs are more expensive, and the expressivity gain may not help on tasks that are well-handled by 1-WL.

> **Worked example.** On OGB-MolHIV, GIN achieves ~77% test ROC-AUC. A GPS-style GT achieves ~78-79%. The 1-2 point gain is from the additional expressivity; the larger gains come from the inductive bias (PE, edge features) rather than from the expressivity per se.

## 8.10 Graph Transformers in the GFM era

Modern GFMs use GTs as the backbone because:
1. **Pretrainability**: GTs are the same architecture as LLMs, so pretraining techniques transfer (masked autoencoding, contrastive, etc.).
2. **Scalability**: linear-attention GTs scale to large graphs.
3. **Expressivity**: the LapPE + SignNet stack makes them strictly more powerful than 1-WL.

The GFM recipes in Ch 11 are mostly GT recipes with additional pretraining objectives.

## 8.11 The "spectral vs spatial" attention debate

**Spatial attention.** Standard transformer attention, treating the graph as a complete graph with edge features. Used in Graphormer, GPS global sublayer. Pro: simple, expressive, well-understood. Con: $O(n^2)$.

**Spectral attention.** Use the Laplacian eigenvectors as a basis; learn a function on the eigenvalues; use this as the attention mask. Used in SAN. Pro: theoretically grounded, can be $O(n \log n)$ with FFT-style tricks. Con: requires eigendecomposition (or approximation), may not generalise to directed graphs.

**Hybrid.** Use both: spectral PE in the input, spatial attention in the layers. Used in GPS, the most common in practice.

> **Worked example.** For a 1000-vertex graph: spatial attention is $10^6$ per layer per head. Spectral attention with the first 10 eigenvectors is $10^4$ per layer per head. The spectral is 100x faster but loses the full-attention expressivity.

## Exercises

1. **LapPE on a real graph.** Take a 4-vertex graph: edges $\{(1,2),(2,3),(3,4),(1,4)\}$ (a 4-cycle with one extra chord). Compute $L_{\text{sym}}$, find the first 3 non-trivial eigenvectors, plot the 2D PE. How do the four vertices' PEs compare?

2. **SignNet intuition.** Take the first non-trivial eigenvector of $L_{\text{sym}}$ for the 4-cycle. Note that $u_2$ can be replaced with $-u_2$ (both are eigenvectors). A function $f(h_v, u_2(v))$ may be sensitive to the sign choice. Show how SignNet averages $f(h, u)$ and $f(h, -u)$ to get a sign-invariant representation.

3. **GPS on a small graph.** Implement a 1-layer GPS: a GINE local MPNN + a 2-head attention sublayer. Train on Cora for node classification. Compare to a 2-layer GIN. The accuracy should be similar (within 1-2 points), but the GPS model is more expensive.

4. **Graphormer attention.** Implement the Graphormer spatial encoding: for each pair of vertices, look up the shortest-path distance, add a learned embedding to the attention score. On a small molecular dataset, show that the encoding improves over standard attention.

5. **Scalable attention.** Implement NodeFormer's kernelized attention on a 100K-vertex graph. Verify the runtime is closer to $O(n)$ than $O(n^2)$. Compare test accuracy to a GCN baseline.

6. **PE ablation.** Train a 4-layer GT on OGB-ArXiv with: (a) no PE, (b) LapPE only, (c) RWSE only, (d) both. Plot test accuracy for each. The "both" should be the best.

7. **Read GPS.** Read [`2205.12454`](https://arxiv.org/abs/2205.12454) (Rampášek et al.). Identify the three "input encoding" tricks (LapPE, RWSE, sign-augmented LapPE). Identify the three "layer" components (local MPNN, global attention, FFN). Note the "dim", "n_layers", "attn_dropout" hyperparameters.

8. **Read the original Transformer paper.** Read Vaswani et al. 2017 (search title). Identify the structural choices that Graphormer, GPS, and SAN inherit (residual stream, layer norm, FFN sublayer) and the ones they modify (positional encoding, attention bias).

9. **TokenGT.** Implement a minimal TokenGT on a 4-cycle: 4 vertex tokens, 4 edge tokens, transformer over all 8 tokens. Train to distinguish the 4-cycle from a path $P_4$. TokenGT should be able to.

10. **Spectral vs spatial.** Compare a SAN-style model and a Graphormer-style model on a small graph. Both should reach similar accuracy; SAN may be more interpretable (you can look at the learned spectral mask).

## Further Reading

- **[`2106.05234`](https://arxiv.org/abs/2106.05234)** — Ying et al., *Do Transformers Really Perform Bad for Graph Representation?* (Graphormer, NeurIPS 2021). Read this first.
- **[`2205.12454`](https://arxiv.org/abs/2205.12454)** — Rampášek et al., *Recipe for a General, Powerful, Scalable Graph Transformer* (GPS, NeurIPS 2022). The most-cited practical GT.
- **[`2106.03893`](https://arxiv.org/abs/2106.03893)** — Kreuzer et al., *Rethinking Graph Transformers with Spectral Attention* (SAN, NeurIPS 2021). The spectral attention paper.
- **[`2207.02505`](https://arxiv.org/abs/2207.02505)** — Kim et al., *Pure Transformers are Powerful Graph Learners* (TokenGT, NeurIPS 2022). The "graph as tokens" approach.
- **[`2102.12798`](https://arxiv.org/abs/2102.12798)** — Lim et al., *Sign and Basis Invariant Networks for Spectral Graph Representation Learning* (SignNet, ICML 2022).
- **[`2206.04910`](https://arxiv.org/abs/2206.04910)** — Chen et al., *NAGphormer* (NeurIPS 2022). Multi-hop aggregation for scalable GTs.
- **[`2306.08385`](https://arxiv.org/abs/2306.08385)** — Wu et al., *NodeFormer* (NeurIPS 2022). Kernelized Gumbel-Softmax attention.
- **[`2301.09474`](https://arxiv.org/abs/2301.09474)** — Wu et al., *DIFFormer* (ICML 2023). Energy-constrained diffusion-based attention.
- **[`2302.04181`](https://arxiv.org/abs/2302.04181)** — Müller et al., *Attending to Graph Transformers* (TMLR 2023). The most comprehensive GT survey.
- **[`2407.09777`](https://arxiv.org/abs/2407.09777)** — Shehzad et al., *Graph Transformers: A Survey* (2024).
- **[`2502.16533`](https://arxiv.org/abs/2502.16533)** — Yuan et al., *A Survey of Graph Transformers: Architectures, Theories and Applications* (2025).
- **[`2401.16176`](https://arxiv.org/abs/2401.16176)** — Hoang & Lee, *A Survey on Structure-Preserving Graph Transformers* (2024).
- **For the connection between GTs and GFMs**: see the GFM survey [`2505.15116`](https://arxiv.org/abs/2505.15116), §4.2.
- **Min et al., *Graph Transformer Neural Networks for Chemistry*** (the original Chemistry GT; not strictly required but historical).
- **Dwivedi & Bresson, *A Generalization of Transformer Networks to Graphs*** (2020). The early GT paper that introduced LapPE.
- **Choromanski et al.** — the **Performer** ([`2009.14794`](https://arxiv.org/abs/2009.14794)) is the linear-attention backbone used in many scalable GTs.
