---
title: "07 — Expressive Power & Weisfeiler-Leman"
date: 2026-08-12T17:40:00+03:00
draft: false
params:
  math: true
---

# 07 — Expressive Power & Weisfeiler-Leman
*What graphs can a GNN tell apart? The Weisfeiler-Leman test, its relation to message passing, and the path to higher-order and subgraph-based architectures.*

This chapter answers the most theoretical question in the cheatsheet: *what are the limits of GNNs?* The answer is surprisingly clean — the 1-Weisfeiler-Leman (1-WL) test is the upper bound for MPNNs, and there are well-defined ways to exceed 1-WL. Every GFM and Graph Transformer is, in some sense, an attempt to escape 1-WL.

## 7.1 The graph isomorphism problem

Two graphs $G, H$ are **isomorphic** if there is a bijection $\phi: V(G) \to V(H)$ such that $(u, v) \in E(G) \iff (\phi(u), \phi(v)) \in E(H)$. The **graph isomorphism problem** is to decide if two given graphs are isomorphic.

**Status:** GI is in **NP** but not known to be NP-complete or in P. In 2015, Babai gave a quasi-polynomial-time algorithm. Practical algorithms (nauty, Traces, bliss) are very fast in practice but exponential in the worst case for general graphs.

**The Weisfeiler-Leman (WL) test** is a hierarchy of polynomial-time heuristics for GI. The $k$-WL test operates on $k$-tuples of vertices and refines them iteratively. The 1-WL test is also called "color refinement" or "naive vertex classification."

## 7.2 The 1-Weisfeiler-Leman test

**Algorithm.** Assign to each vertex a color (a hash of its initial label). Iteratively, update the color of each vertex $v$ by hashing the multiset of colors of its neighbors. If two vertices get different colors in some iteration, they are distinguished. If the entire color distribution of $G$ and $H$ differs, they are distinguished.

```
1-WL(G, H):
    color[v] = init_label(v) for v in V(G) ∪ V(H)
    repeat:
        for v in V(G) ∪ V(H):
            new_color[v] = hash( (color[v], sorted([color[u] for u in N(v)]) ) )
        if all colors stabilised: break
    return multiset(color) for G equals multiset(color) for H
```

**Convergence.** The number of colors is finite (bounded by $|V|$), so the process must terminate. Termination is in $O(n)$ iterations (and often constant in practice).

**What 1-WL can and cannot do:**
- **Can** distinguish: most "non-pathological" non-isomorphic pairs.
- **Cannot** distinguish: regular graphs of the same degree, certain graph families (e.g., $C_6$ and $2K_3$ — the 6-cycle and two disjoint triangles are both 2-regular on 6 vertices; 1-WL colors all vertices with the same color, so it cannot distinguish them).

**Connection to GNNs.** A 1-layer MPNN with sum aggregation is *exactly* a differentiable relaxation of 1-WL, where the "hash" function is replaced by a learnable MLP. The number of MPNN layers = number of 1-WL iterations. So:
- A 1-layer GIN is as expressive as 1-WL.
- A 2-layer GIN is as expressive as 2 iterations of 1-WL.
- An $L$-layer GIN is bounded by 1-WL in its distinguishing power (but not always, see below).

> **Worked example.** $C_6$ vs $2K_3$. Initial colors: all 0. Iteration 1: for each vertex, multiset of neighbor colors = {0}. All vertices get color hash(0, {0}). Both graphs have all vertices with the same color. The color distribution is the same. The test concludes "isomorphic" — but they are not! This is the classic failure of 1-WL.

## 7.3 The GIN expressivity theorem

**Theorem (Xu et al. 2019, [`1810.00826`](https://arxiv.org/abs/1810.00826)).** Let $\mathcal{A}$ be the family of GNNs that, at each layer, compute

$$h_v^{(\ell+1)} = \text{UPD}\left( h_v^{(\ell)}, \text{AGG}\left( \{ h_u^{(\ell)} : u \in \mathcal{N}(v) \} \right) \right),$$

where AGG is sum, mean, or max, and UPD is an MLP. Then:

1. For any choice of (AGG, UPD) in $\mathcal{A}$, the GNN's distinguishing power is bounded by 1-WL.

2. Conversely, for any graph pair $(G, H)$ that 1-WL can distinguish, there is a GIN that can distinguish them.

**Interpretation.** GIN is *the* most-expressive MPNN in the standard family. Sumpooling, sum aggregation, MLP update — that's it. Other aggregations (mean, max) are strictly weaker.

**Why?** The proof shows:
- Sum is injective on multisets (Ch 05 §5.3) — the only aggregator with this property.
- MLP is a universal function approximator.
- So GIN can, in principle, learn any function on the neighborhood multiset.
- 1-WL is bounded by what can be expressed on the neighborhood multiset (up to multiset isomorphism).

> **Worked example.** Consider two 3-vertex graphs: $G_1$ = path $1-2-3$, $G_2$ = triangle $1-2-3-1$. Same vertex set, vertex 1 in $G_1$ has degree 1, in $G_2$ degree 2. 1-WL distinguishes (different degrees → different colors). A 1-layer GIN with sum aggregation also distinguishes: for $G_1$, $h_1^{(1)} = \text{MLP}((1+\epsilon) h_1 + h_2)$; for $G_2$, $h_1^{(1)} = \text{MLP}((1+\epsilon) h_1 + h_2 + h_3)$. The MLP can be trained to make these different.

## 7.4 Higher-order WL and $k$-GNNs

The $k$-WL test operates on $k$-tuples of vertices. Two $k$-tuples are "neighbors" if they differ in exactly one coordinate and the corresponding vertices are adjacent. The refinement proceeds analogously to 1-WL.

**Hierarchy.** $1\text{-WL} \subsetneq 2\text{-WL} \subsetneq \cdots \subsetneq k\text{-WL} \subsetneq (k+1)\text{-WL} \subsetneq \cdots$

Each $k$-WL is strictly more expressive than $(k-1)$-WL. In the limit, the $\infty$-WL test is GI-complete (i.e., it can solve graph isomorphism).

**Computational cost.** $k$-WL operates on $n^k$ tuples. So even $k = 2$ is expensive for large $n$ (OGB-ArXiv has $n^k \approx 3 \times 10^{10}$ for $k = 2$).

**$k$-GNNs.** Maron, Ben-Hamu, Serviansky, Lipman (2019) showed that $k$-GNNs (MPNNs that aggregate over $k$-tuples) can be as expressive as $k$-WL. Morris et al. (2019) independently proposed a similar idea. The cost is $O(n^k)$, which limits practical $k$ to 2 or 3.

**Provably Powerful GNNs** (Maron et al. 2019, $k = 2$): use a sum over 2-tuples with a 2-tuple "message" function. The cost is $O(n^2 d^2)$ in the message function alone, but can be reduced.

> **Worked example.** A 2-GNN maintains a hidden state for each ordered pair $(u, v)$. The 2-tuple $(u, v)$'s neighbors in the 2-tuple graph are all $(u', v)$ with $(u, u') \in E$, all $(u, v')$ with $(v, v') \in E$, and all $(u, v')$ with $u = v'$. The "2-tuple graph" has $n^2$ vertices and is much richer than the original. A 2-GNN can distinguish $C_6$ and $2K_3$ (where 1-WL cannot).

## 7.5 Subgraph GNNs: lifting to higher order

A different approach: replace the original graph with a collection of subgraphs, run a 1-WL-bounded GNN on each, then aggregate. This is the **subgraph GNN** family.

**ID-GNN** (You, Gomes, Hoffman, Gupta 2021): each layer, augment the input graph by marking one vertex as "distinguished" per connected component. The marked vertex's hidden state is updated differently, breaking the 1-WL symmetry.

**ESAN** (Bevilacqua et al. 2022): sample subgraphs around each vertex, run a 1-WL-bounded GNN on each, aggregate the per-subgraph predictions.

**NGNN** (Zhang & Li 2021): nest the MPNN computation — each vertex's state is updated based on the MPNN output of a small subgraph centered on it.

**Subgraph MPNN** (in the GFM survey, [`2505.15116`](https://arxiv.org/abs/2505.15116)): general framework.

**Cost.** Subgraph GNNs are typically $K$ times more expensive than the base MPNN (for $K$ subgraphs per vertex). Empirically $K = 8$ to $32$ is enough to recover much of the higher-order expressivity.

> **Worked example.** $C_6$ vs $2K_3$. Sample subgraphs: 3-edge subgraphs (path of length 3). In $C_6$, the only 3-edge subgraph is a path $P_4$. In $2K_3$, the 3-edge subgraph is a triangle $K_3$. A 1-WL test on the subgraph collection distinguishes.

## 7.6 Local vs global expressivity

A 1-WL-bounded MPNN has another limitation beyond graph-level distinguishability: **local expressivity**. Even within a single graph, two vertices can have the same neighborhood structure at all $k$ hops, but differ in their global position.

**Example.** Two vertices in different positions in a long path. Both have degree 2, both have neighbors with degree 2, etc. — 1-WL assigns the same color. A 1-WL-bounded MPNN assigns the same embedding. The two vertices are "indistinguishable" by the MPNN even though they are at different positions.

**Solution: positional encodings.** Add to each vertex a feature that depends on its position in the graph. The standard options:
- **Laplacian PE** (LapPE, Dwivedi & Bresson 2020): the first $k$ non-trivial eigenvectors of $L$. $O(kn)$ memory; the eigendecomposition is $O(n^3)$ but can be approximated.
- **Random walk PE** (RWPE, Dwivedi et al. 2022): $A^k \mathbf{e}_v$ for the first few $k$ powers. $O(km)$ memory.
- **SignNet** ([`2102.12798`](https://arxiv.org/abs/2102.12798), Lim et al. 2022): a learned function on the *signs* of the eigenvectors, handling the $\pm u_i$ ambiguity.
- **Distance PE**: shortest-path distance to a set of anchor vertices.
- **Heat kernel PE**: $\exp(-t L)$ at some diffusion time $t$.

**Why this matters for GFMs.** A GFM that uses 1-WL-bounded message passing + LapPE is strictly more expressive than 1-WL alone. The combination is the basis of every competitive Graph Transformer (Ch 08).

> **Worked example.** A path $P_8$, vertices $1, 2, \ldots, 8$. A 1-WL test colors them all the same. LapPE with the first 3 non-trivial eigenvectors: $u_2 \approx (0.46, 0.35, 0.16, -0.07, -0.27, -0.39, -0.39, -0.46)$. Vertex 1 has $u_2 \approx 0.46$, vertex 8 has $u_2 \approx -0.46$. Different.

## 7.7 Over-squashing and the WL test

**Over-squashing** (Alon & Yahav 2021, [`2006.05205`](https://arxiv.org/abs/2006.05205)) is the phenomenon where a deep GNN compresses too much information into a fixed-size vector, losing long-range dependencies. It is *related to* but not *identical to* expressivity.

**Formal connection.** The "bottleneck ratio" of an edge $(u, v)$ in a graph is the size of the smallest cut separating $u$ from $v$. If the bottleneck ratio is small, the edge is a "bottleneck." In a deep MPNN, the message that must pass through this bottleneck contains information about everything beyond the bottleneck — but the message is a fixed-size vector, so the information is compressed.

**Bottleneck graphs.** Trees have bottleneck ratio 1 (a single edge between two halves). Grids have bottleneck ratio $O(\sqrt n)$. Random graphs have bottleneck ratio $\Theta(\log n / n)$.

**WL over-smoothing.** As $k$ increases, the $k$-WL test becomes more powerful but also more expensive. A practical question: how many MPNN layers do you need? The answer depends on the graph's "WL-diameter" — the number of 1-WL iterations needed to stabilise the colors.

**For a GFM**, the "receptive field" of the model must match the WL-diameter of the target graphs. Graph Transformers (Ch 08) have effectively infinite receptive field (because of global attention), so this is not a concern for them.

> **Worked example.** The 4-cycle. Bottleneck ratio: between vertex 1 and vertex 3, the cut is $\{1, 2\}$ vs $\{3, 4\}$, with 2 boundary edges. The 4-cycle has bottleneck ratio 2. A 2-layer GCN has receptive field 2 hops — sufficient to "see" the whole cycle. A 1-layer GCN: receptive field 1 hop, can't distinguish vertex 1 from vertex 3 by structure. LapPE helps: the Fiedler vector distinguishes them.

## 7.8 Practical implications

**For graph classification.** If your task is distinguishing non-isomorphic graphs (or graphs that 1-WL can distinguish), GIN is the right baseline. If you need to distinguish graphs that 1-WL cannot (rare in practice), use a 2-GNN or subgraph GNN.

**For node classification.** Most node classification tasks are about labeling vertices with distinct local features. GCN, GAT, GATv2 are all sufficient; GIN is overkill. LapPE helps a lot for distinguishing distant vertices.

**For graph regression.** Same as graph classification.

**For link prediction.** Often requires a global view (two vertices far apart in the graph). GCN + LapPE, or a Graph Transformer.

**For molecular property prediction.** GIN with edge features (GINE) is the standard. Adding 3D positions requires an equivariant architecture (SchNet, DimeNet, Equiformer, NequIP).

> **Worked example.** The OGB-MolPCBA dataset (molecular property prediction, 400+ tasks). Top-3 leaderboard models (as of 2024): GIN with virtual node, GINE with virtual node, Graph Transformer with LapPE. The virtual node is the key trick: it makes the receptive field "whole graph" with a shallow architecture.

## 7.9 The k-WL zoo: what each $k$ gives you

| $k$ | WL name | Cost | What it distinguishes |
|---|---|---|---|
| 1 | Color refinement | $O(n)$ | Most "obvious" non-isomorphisms; fails on regular graphs |
| 2 | 2-tuple WL | $O(n^2)$ | Some regular graph pairs, but not all |
| 3 | 3-tuple WL | $O(n^3)$ | More pairs, including some $k=2$ misses |
| $k$ | $k$-tuple WL | $O(n^k)$ | Increasingly powerful |
| $\infty$ | Full GI | $O(n^{\log n})$ quasi-poly | Decides GI |

**2-WL is *not* the same as 2-FWL (folklore WL).** 2-FWL is more powerful than 2-WL (strictly). The folklore variant considers unordered pairs; the standard considers ordered pairs. The 2-FWL hierarchy is the "right" one for GNN expressivity analysis.

**Connection to GFM design.** A GFM that uses a 2-FWL-bounded architecture (e.g., a subgraph GNN, an ID-GNN, or a $k$-GNN) is strictly more expressive than a 1-WL-bounded GFM. The cost: $K$ times more compute. The benefit: captures higher-order substructures that a 1-WL GFM cannot.

> **Worked example.** Shrikhande graph vs. Rook's graph (both 16-vertex 6-regular strongly regular graphs with the same parameters). 1-WL cannot distinguish (both are 6-regular on 16 vertices). 2-WL: still cannot (they have the same number of triangles, 4-cycles, etc.). 3-WL: distinguishes. So a 3-FWL-bounded GNN would distinguish them.

## 7.10 Theoretical open questions

**Q1.** Is there a $k$-GNN with $O(n \text{poly}(k))$ cost that is as expressive as $k$-WL? Currently $k$-GNNs cost $O(n^k)$ (exponential in $k$).

**Q2.** What is the relationship between the WL hierarchy and the practical tasks (like molecular property prediction)? Are 1-WL-bounded GNNs sufficient for chemistry?

**Q3.** Can transformers (with appropriate positional encodings) be made as expressive as $k$-WL? Some recent results (e.g., "transformers are universal approximators for graph functions" — Sanford et al. 2023, `"expressivity"`) suggest yes.

**Q4.** What is the relationship between WL expressivity and adversarial robustness? More expressive GNNs may be more vulnerable to adversarial perturbations.

These are active research areas. A GFM researcher should at least know the questions.

## Exercises

1. **1-WL by hand.** $C_6$ (vertices 1, 2, 3, 4, 5, 6 in a cycle) and $2K_3$ (two triangles: 1, 2, 3 and 4, 5, 6). Run 1-WL on both. Show that after iteration 1, all vertices in both graphs have the same color. The test says "isomorphic" — but they are not. (This is the classic 1-WL failure case.)

2. **GIN vs GCN distinguishability.** Train a 1-layer GIN and a 1-layer GCN on a synthetic dataset of 100 graphs that includes $C_6$ and $2K_3$ pairs. GIN should be able to classify them; GCN cannot. (Hint: GIN can use sum aggregation to distinguish the graphs because the sums differ — but actually, both $C_6$ and $2K_3$ have 2 neighbors per vertex with the same initial labels, so 1-layer GIN may also fail. The issue is *initial labels*. If you use degree as initial label, both have degree 2 everywhere. If you use random initial features, GIN can learn to distinguish — but this is feature-dependent, not structural.)

3. **2-FWL distinction.** Take a strongly regular graph pair (e.g., the 4 × 4 rook's graph and the Shrikhande graph, both on 16 vertices, both 6-regular with the same SRG parameters). A 1-WL test cannot distinguish. A 2-WL test cannot distinguish. A 3-WL test can. (You don't need to implement 3-WL — just read about it and write a paragraph explaining why 3-WL can distinguish them.)

4. **LapPE distinguishability.** Generate a path $P_8$, compute the Laplacian PE for all 8 vertices. Show that the embeddings are different (using cosine distance). Repeat for a 4-cycle and a 8-cycle. The cycle embeddings should be "rotationally symmetric" — vertex $i$ has the same PE as vertex $i + 4$ in the 8-cycle (by the cycle's symmetry).

5. **Expressivity in the MPNN family.** Implement sum, mean, and max aggregation in a 1-layer MPNN. On a synthetic graph dataset where the graphs differ in multiset structure, evaluate which aggregation distinguishes them. Sum should be best, mean second, max worst (for expressivity).

6. **Bottleneck ratio.** For the barbell graph $K_n \cup K_n$ joined by a single edge: compute the bottleneck ratio of the joining edge. (It is 1 — that single edge is the only path between the two cliques.) Compare to a random $G(n, p)$ with $p$ such that the average degree matches: the bottleneck ratio is $O(\log n)$.

7. **Read the GIN paper.** Read [`1810.00826`](https://arxiv.org/abs/1810.00826) (Xu et al.). Identify the exact statement of Theorem 3 (the expressivity theorem). What conditions are placed on the MLP for GIN to be 1-WL-equivalent?

8. **Read the WL survey.** Read the relevant section of [`2003.04078`](https://arxiv.org/abs/2003.04078) (Sato's expressivity survey). Identify the relationships between MPNN expressivity, $k$-WL, $k$-FWL, and the subgraph GNN family.

9. **Higher-order GNN at small scale.** Implement a simple 2-GNN on the MUTAG dataset (small molecules). Compare to a 1-GNN. The 2-GNN should be strictly more expressive but $K$ times slower.

10. **Positional encoding ablation.** Take a 4-layer GAT on OGB-ArXiv. Add LapPE. Compare test accuracy. The improvement is usually 1-3 points (significant for OGB). Try removing LapPE: accuracy drops.

## Further Reading

- **Weisfeiler & Lehman, *A reduction of a graph to a canonical form and an algebra arising during this reduction*** (Nauchno-Technicheskaya Informatsiya 1968). The original WL paper (in Russian; English translation available).
- **Babai, *Graph Isomorphism in Quasipolynomial Time*** (STOC 2016). The quasi-poly GI algorithm.
- **Cai, Fürer & Immerman, *An optimal lower bound on the number of variables for graph identification*** (STOC 1989). The classic WL hierarchy paper.
- **[`1810.00826`](https://arxiv.org/abs/1810.00826)** — Xu, Hu, Leskovec, Jegelka, *How Powerful are GNNs?* (GIN, ICLR 2019). The expressivity paper for the MPNN family.
- **[`2003.04078`](https://arxiv.org/abs/2003.04078)** — Sato, *A Survey on The Expressive Power of GNNs*. The most comprehensive modern survey on this topic.
- **Maron, Ben-Hamu, Serviansky, Lipman, *Provably Powerful Graph Networks*** (NeurIPS 2019). The 2-GNN paper.
- **Morris et al., *Weisfeiler and Leman Go Neural*** (NeurIPS 2019). The other 2-GNN paper.
- **Chen, Chen, Villar, Bruna, *Can Graph Neural Networks Go "Online"?*** (ICML 2020). Identifies the issue with 1-WL for distinguishing certain graphs.
- **You, Gomes, Hoffman, Gupta, *Identity-aware Graph Neural Networks*** (ID-GNN, AAAI 2021). The marked-vertex approach to higher-order.
- **Bevilacqua, Frasca, Lin, Bronstein, Morris, *Equivariant Subgraph Aggregation Networks*** (ESAN, ICLR 2022). Subgraph GNN family.
- **Dwivedi, Joshi, Laurent, Bengio, Bresson, *Benchmarking Graph Neural Networks*** (2020). The benchmark paper that introduced LapPE and RWPE.
- **[`2102.12798`](https://arxiv.org/abs/2102.12798)** — Lim, Robinson, Zhao, Smidt, Sra, Ceci, Maron, *SignNet* (ICML 2022). Sign-equivariant PE.
- **For the deeper theory**: see the lecture notes of Christopher Morris, *Graph Neural Networks: A Review of Methods and Applications* (2024). Available on his homepage.
- **For the connection to GFMs**: see [`2505.15116`](https://arxiv.org/abs/2505.15116) §3.2, which discusses higher-order GFM architectures.
- **For the 1-WL vs 2-WL connection to GNNs**: see the appendix of Maron et al. 2019 for the most precise statement.
