---
title: "01 — Graph Theory Foundations"
date: 2026-08-12T17:40:00+03:00
draft: false
params:
  math: true
---

# 01 — Graph Theory Foundations
*The language every GFM researcher must own: vertices, edges, paths, walks, connectivity, Laplacians, and the matrix representations that GNNs operate on.*

This chapter is intentionally rigorous because everything in the rest of the cheatsheet assumes fluency here. If you can draw, label, and operate on the objects in this chapter without hesitation, the GNN chapters will read as natural extensions rather than mysterious architectures.

## 1.1 The basic objects

A **graph** $G = (V, E)$ is a pair of sets: a set $V$ of $n = |V|$ **vertices** (or **nodes**), and a set $E \subseteq V \times V$ of $m = |E|$ **edges** (or **links**). Most graphs in ML are **simple** (no self-loops, no multi-edges); many are also **undirected** ($E$ is symmetric).

| Variant | Edge set | Notes |
|---|---|---|
| Undirected | $(u,v) \in E \iff (v,u) \in E$ | Most social, molecular, citation graphs |
| Directed (digraph) | $(u,v) \in E$ does not imply $(v,u) \in E$ | Web graph, knowledge graph, citation graph |
| Weighted | $E \mapsto \mathbb{R}_{>0}$ | Distances, traffic, transactions |
| Multi-relational | Each edge has a type | Knowledge graphs (TransE, R-GCN) |
| Bipartite | $V = U \cup W$, $E \subseteq U \times W$ | User–item, document–word |
| Heterogeneous | Vertex/edge types drawn from sets | Academic networks (author, paper, venue) |
| Temporal | Edges have timestamps | Transactions, communication logs |
| Hypergraph | Edges can connect $\geq 2$ vertices | Co-authorship, multi-entity relations |

A graph is **sparse** if $m = O(n)$ and **dense** if $m = \Theta(n^2)$. Most real-world graphs are sparse: the average degree $\bar{d} = 2m/n$ is small (e.g., $\bar{d} \approx 5$ in web graphs, $\approx 2$ in molecular graphs). GNNs are designed for the sparse regime; dense graph algorithms are typically $O(n^3)$.

**Degree** of vertex $v$ in an undirected graph is $\deg(v) = |\{u : (v,u) \in E\}|$. In a directed graph, you have in-degree and out-degree separately.

> **Worked example.** A 4-vertex cycle: $V = \{1,2,3,4\}$, $E = \{(1,2),(2,3),(3,4),(4,1)\}$. Then $n=4$, $m=4$, $\deg(v)=2$ for all $v$, $\bar d = 2$. Sparse. Every vertex is symmetric under rotation; this symmetry is what GNNs are designed to respect.

## 1.2 Matrix representations

The same graph admits three matrix representations; choosing the right one is the first design decision when you start writing code.

**Adjacency matrix** $A \in \{0,1\}^{n \times n}$ with $A_{uv} = 1$ iff $(u,v) \in E$. For undirected graphs, $A$ is symmetric, so diagonalizable by an orthogonal matrix. For weighted graphs, $A_{uv}$ is the weight. The diagonal is zero for simple graphs. Memory: $O(n^2)$ — only feasible for $n \lesssim 10^4$.

**Degree matrix** $D \in \mathbb{R}^{n \times n}$ is diagonal with $D_{uu} = \deg(u)$. (For directed graphs, use the in-degree or out-degree depending on convention.)

**Incidence matrix** $B \in \mathbb{R}^{n \times m}$ with $B_{ve} = 1$ if vertex $v$ is incident to edge $e$ (and $-1$ for the other endpoint, in a signed variant). Memory: $O(nm)$. Used in flow problems and in the derivation of the Laplacian.

For sparse graphs in practice, you almost never materialize $A$. You use an **edge list** $(u_i, v_i, w_i)_{i=1..m}$ or a **CSR (compressed sparse row) adjacency** — `nnz` nonzero entries stored row by row, plus an index pointer per row. This is the format that PyTorch Geometric, DGL, and OGB use.

> **Worked example.** The 4-vertex cycle from §1.1. With vertices ordered $1,2,3,4$:
>
> $$A = \begin{pmatrix} 0 & 1 & 0 & 1 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 0 & 1 & 0 \end{pmatrix}, \quad D = 2I, \quad B = \begin{pmatrix} 1 & 0 & 0 & 1 \\ 1 & 1 & 0 & 0 \\ 0 & 1 & 1 & 0 \\ 0 & 0 & 1 & 1 \end{pmatrix}$$
>
> Edge list: $E = [(1,2),(2,3),(3,4),(4,1)]$ with weights all 1.

## 1.3 The Laplacian and its normalised variants

The **combinatorial Laplacian** is

$$L = D - A.$$

It is symmetric positive semi-definite for undirected graphs, so it admits a spectral decomposition. The smallest eigenvalue is $0$ with multiplicity equal to the number of connected components; the eigenvector is the all-ones vector. The number of zero eigenvalues **is** the connected-component count, a fact that reappears throughout spectral graph theory and in GCN theory (the "over-smoothing" phenomenon: as $k \to \infty$, repeated application of $(I - \alpha L)$ drives everything to the 0-eigenspace, i.e., to a single point).

**Two normalised variants** are used in modern GNNs:

**Symmetric normalised Laplacian**

$$L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}.$$

**Random-walk normalised Laplacian**

$$L_{\text{rw}} = I - D^{-1} A.$$

$L_{\text{rw}}$ is the transition matrix of the **random walk** on the graph; the second term $D^{-1}A$ gives the probability of stepping from a vertex to one of its neighbors in one step. $L_{\text{sym}}$ is the one used in **Kipf & Welling's GCN** (Ch 06).

A fourth quantity, $\hat{A} = D^{-1/2}(A + I)D^{-1/2}$, is the "renormalisation trick" that adds self-loops before normalising; it appears in the GCN propagation rule.

**Quadratic-form identity** that drives much of the theory: for any $x \in \mathbb{R}^n$,

$$x^\top L x = \frac{1}{2} \sum_{(u,v)\in E} (x_u - x_v)^2.$$

This says $L$ is a discrete second-difference operator and is the entry point for spectral graph theory (Ch 03). It also has a clean interpretation in **graph signal processing**: $x^\top L x$ measures the smoothness of a signal $x$ on the graph, small if neighbors have similar values.

> **Worked example.** Same 4-cycle. $A$ as above, $D = 2I$, $L = D - A = \begin{pmatrix} 2 & -1 & 0 & -1 \\ -1 & 2 & -1 & 0 \\ 0 & -1 & 2 & -1 \\ -1 & 0 & -1 & 2 \end{pmatrix}$. Take $x = (1, 0, 1, 0)^\top$. Then $x^\top L x = 2 \cdot 1^2 + 2 \cdot 1^2 + 2 \cdot 0^2 + 2 \cdot 0^2 = 4$ (we count each edge twice; the factor 1/2 is implicit). Or: $(1-0)^2 + (0-1)^2 + (1-0)^2 + (0-1)^2 = 4$. ✓

## 1.4 Paths, walks, connectivity

A **walk** is a sequence $v_0, v_1, \ldots, v_k$ with $(v_i, v_{i+1}) \in E$. A **path** is a walk with no repeated vertices. A **cycle** is a path with $v_0 = v_k$ and no other repeats. The **length** of a walk/path/cycle is its number of edges.

The **distance** $d(u,v)$ between two vertices is the length of the shortest path. The **diameter** of a connected graph is $\max_{u,v} d(u,v)$.

A graph is **connected** if every pair of vertices is joined by a path; equivalently, the Laplacian has exactly one zero eigenvalue. The **connected components** are the maximal connected subgraphs.

A graph is **$k$-regular** if every vertex has degree $k$. The 4-cycle is 2-regular; the Petersen graph is 3-regular.

The **adjacency matrix powers count walks**: $(A^k)_{uv}$ is the number of walks from $u$ to $v$ of length exactly $k$. This is the bridge from the adjacency matrix to the random-walk interpretation in §1.6 and to the spectral theory in Ch 03.

> **Worked example.** In the 4-cycle, $(A^2)_{11} = 2$ because there are two length-2 walks from vertex 1 back to itself: $1 \to 2 \to 1$ and $1 \to 4 \to 1$. $(A^2)_{12} = 0$ because there is no length-2 walk from 1 to 2: any walk of length 2 starts at 1, visits a neighbor, then ends at 2 — but the neighbors of 1 are 2 and 4, and only 4 is not 2, but then from 4 we can only reach 1 or 3, neither of which is 2. So 0. ✓

## 1.5 Graph properties used in GFM work

The following structural descriptors appear repeatedly in the GFM literature. They are the **statistics a "generalist" GFM should be invariant to** (Ch 11).

| Property | Definition | Why it matters |
|---|---|---|
| Homophily | Fraction of edges where endpoints have the same label | Determines whether neighbour aggregation is useful |
| Heterophily | Complement of homophily | Drives designs like Geom-GCN (2002.05287) |
| Clustering coefficient | $C_v = \frac{2 \cdot (\text{triangles at }v)}{\deg(v)(\deg(v)-1)}$ | Local density; distinguish social from web |
| Degree distribution | $P(\deg = k)$ | Power-law vs. exponential |
| Small-worldness | High $C$ and low diameter | Many real graphs (Watts–Strogatz) |
| Modularity | $Q = \frac{1}{2m} \sum_{uv} \left[A_{uv} - \frac{\deg(u)\deg(v)}{2m}\right] \delta(c_u, c_v)$ | Community structure |
| Sparsity | $m / n^2$ | Determines algorithm choice |
| Diameter | $\max d(u,v)$ | Affects receptive field of GNN |

**Homophily vs. heterophily** is the single most practically important distinction for designing a GFM. Datasets split roughly:
- **Homophilic** (Cora, Citeseer, PubMed): neighbors are usually the same class — vanilla GCN works.
- **Heterophilic** (many molecular, fraud, dating graphs): neighbors are usually different class — you need Geom-GCN-style geometric methods, signed attention, or a transformer with positional encodings.

## 1.6 Random walks

A **random walk** on $G$ is a stochastic process: from vertex $u$, pick a neighbor $v$ uniformly at random and move there. The **transition matrix** is $P = D^{-1} A = I - L_{\text{rw}}$.

The **stationary distribution** $\pi$ satisfies $\pi^\top P = \pi^\top$, giving $\pi_u = \deg(u) / (2m)$. Intuitively: you spend time at a vertex proportional to its degree. This is the basis of **node2vec** (Grover & Leskovec, 2016, `1607.00653`) and **DeepWalk** (Perozzi, Al-Rfou, Skiena, 2014, `1403.6652`) — the random walk traces a sentence, and a Skip-gram gives you node embeddings.

**Mixing time** is the number of steps $t$ until the walk's distribution is within $\epsilon$ of $\pi$, regardless of starting vertex. The mixing time is roughly $O(1/\lambda_2)$ where $\lambda_2$ is the **spectral gap** — the second-smallest eigenvalue of $L$ (Ch 03). This is the connection between graph structure, random walks, and GNN message passing: a GCN's "receptive field" after $k$ layers is roughly the $k$-step random-walk distribution.

> **Worked example.** In the 4-cycle, $P = (1/2) A$ (every degree is 2). Starting at vertex 1: step 0 = (1,0,0,0), step 1 = (0, 1/2, 0, 1/2), step 2 = (1/2, 0, 1/2, 0), step 3 = (0, 1/2, 0, 1/2), and so on. The walk oscillates between even and odd vertices forever; it does **not** mix. Why? The 4-cycle is **bipartite**, so it has no stationary distribution concentrated on individual vertices — only the average. The Laplacian has eigenvalues $0, 2, 2, 4$ (you can verify), and $\lambda_2 = 2$, so mixing is slow on this graph.

## 1.7 Special graph classes

These are the *named* graphs whose properties you should be able to recite cold; they appear in every GNN benchmark and in many GFM evaluation protocols.

| Class | Definition | Eigenvalues of $L$ | Why it matters |
|---|---|---|---|
| Complete $K_n$ | Every pair of vertices connected | $0$ (once), $n$ (with mult. $n-1$) | Limit case; often used as a sanity test |
| Cycle $C_n$ | Vertices on a circle, edges between neighbors | $2 - 2\cos(2\pi k / n)$ | Symmetric, has nice closed-form Fourier basis |
| Path $P_n$ | A line | $2 - 2\cos(\pi k / (n-1))$ | 1D analogue, the "Cartesian" base case |
| Complete bipartite $K_{a,b}$ | Two halves, all cross-edges | $0, a+b, a+b, a, a, b, b$ (with multiplicities) | Bipartite graphs have spectrum symmetric about $\lfloor n/2 \rfloor$ |
| Star $K_{1,n-1}$ | One center, $n-1$ leaves | $0, 1, 1, \ldots, 1, n$ | Bottleneck graphs (over-squashing, Ch 10) |
| Grid $P_a \square P_b$ | Cartesian product of two paths | Sums of path eigenvalues | Image graphs |
| Hypercube $Q_d$ | $d$-dim, $n = 2^d$ vertices | $0, 2, 2, \ldots, 2d$ | Used in equivariant GNN proofs |
| Random $G(n,p)$ | Erdős–Rényi: each edge with prob $p$ | Wigner semicircle in the limit | Null model for benchmarks |

A graph is **$k$-partite** if its vertex set can be partitioned into $k$ independent sets. Bipartite = 2-partite.

> **Worked example.** $K_3$ (triangle): $A = \begin{pmatrix} 0&1&1 \\ 1&0&1 \\ 1&1&0 \end{pmatrix}$, $D = 2I$, $L = \begin{pmatrix} 2&-1&-1 \\ -1&2&-1 \\ -1&-1&2 \end{pmatrix}$. Eigenvalues: $0$ (constant vector), $3$ (with multiplicity 2). ✓ (matches the table: $n=3$, so the non-zero eigenvalue is $n=3$ with multiplicity $n-1=2$).

## 1.8 The bridge to GNNs

Why does this matter? Every GNN operates on the matrix representations above. The three big design choices in a GNN are all visible at the linear-algebra level:

1. **Aggregation** is $A X W$ (or some normalised variant) — a matrix product with a transform.
2. **Convolution** is $U g(\Lambda) U^\top X$ where $U, \Lambda$ are the eigendecomposition of $L$ — the spectral view (Ch 03).
3. **Positional encoding** is some function of the Laplacian eigenvectors — what makes a transformer permutation-aware (Ch 08).

If you can derive the propagation rule of Kipf's GCN in three lines from $L_{\text{sym}}$, you understand what GCNs are doing. If you can write the random-walk update for DeepWalk, you understand what node embeddings are doing. The rest is engineering.

## Exercises

1. **Edge list to matrix.** Given the directed graph $V = \{1,2,3,4\}$, $E = \{(1,2), (2,3), (3,1), (2,4)\}$: write down $A$, $D_{\text{in}}$, $D_{\text{out}}$, and $L_{\text{sym}}$. Identify the connected components.

2. **Laplacian quadratic form.** For the 4-vertex cycle from §1.3, take $x = (1, 1, -1, -1)^\top$. Compute $x^\top L x$ in two ways: (a) directly via the formula, (b) by summing $(x_u - x_v)^2$ over all edges. Confirm they agree.

3. **Walk counting.** For the 4-cycle, list all length-3 walks starting at vertex 1. How many end at vertex 1, how many at vertex 3? Verify against $(A^3)_{1,*}$.

4. **Degree regular graph.** Construct a 3-regular graph on 6 vertices. (Hint: try the **utility graph** $K_{3,3}$ or the **triangular prism**.) Compute its Laplacian and its eigenvalues.

5. **Sparsity check.** The Open Graph Benchmark's `ogbn-arxiv` dataset has $n \approx 170K$ nodes and $m \approx 1.16M$ edges. The arXiv citation graph. Compute the sparsity $m / n^2$ and the average degree. Is the graph sparse enough that an adjacency-matrix implementation is infeasible on a single GPU? (A modern GPU has ~80 GB memory, a float32 adjacency takes $4n^2$ bytes.)

6. **Power-law degree.** For a graph drawn from a power-law degree distribution $P(d) \propto d^{-\alpha}$ with $\alpha = 2.5$, estimate (a) the expected maximum degree, (b) the expected number of degree-1 vertices, (c) which eigenvalue of the normalised Laplacian will be "close to 0" (the **spectral signature** of community structure). The reference is Chung, Lu & Vu, *Spectra of random graphs with given expected degrees* (PNAS 2003).

7. **Bipartite structure check.** A bipartite graph has a Laplacian spectrum that is symmetric about $n/2$. Verify this on the 4-cycle. (Eigenvalues of $L$ for $C_4$: $0, 2, 2, 4$. Note symmetry about $n/2 = 2$.)

8. **Laplacian of a complete graph.** Show that for $K_n$, the Laplacian is $L = nI - \mathbf{1}\mathbf{1}^\top + I$ (with a sign convention check), and has eigenvalues $0$ (multiplicity 1) and $n$ (multiplicity $n-1$).

## Further Reading

- **Diestel, *Graph Theory*** (5th ed., Springer 2017). The standard graduate reference. Chapters 1–3 cover the material in this chapter with proofs.
- **Bollobás, *Modern Graph Theory*** (Springer 1998). Cleaner for the asymptotic / random-graph parts.
- **Chung, *Spectral Graph Theory*** (CBMS 1997, freely available). The mathematical backbone of Ch 03.
- **Godsil & Royle, *Algebraic Graph Theory*** (Springer 2001). The bridge from §1.3 to spectral methods.
- **[`1901.00596`](https://arxiv.org/abs/1901.00596)** — Wu et al., *A Comprehensive Survey on Graph Neural Networks* (TNNLS 2020). §II is a compact recap of graph theory for ML readers.
- **[`1806.01261`](https://arxiv.org/abs/1806.01261)** — Battaglia et al., *Relational inductive biases, deep learning, and graph networks*. The "graph networks" manifesto. Read this before any GNN paper.
- **[`2003.04078`](https://arxiv.org/abs/2003.04078)** — Sato, *A Survey on The Expressive Power of GNNs*. §2 has the cleanest formal graph-theory recap I have seen in an ML survey.
- **[`2103.09430`](https://arxiv.org/abs/2103.09430)** — Hu et al., *OGB-LSC: A Large-Scale Challenge for ML on Graphs*. The "real-world graph sizes" reference for the sparsity discussion in §1.2.
- **Newman, *Networks: An Introduction*** (Oxford 2010). The single best general book on real-world graph structure; covers the degree distribution, clustering, and small-worldness material in §1.5 in much more depth.
- **Erdős & Rényi, *On the evolution of random graphs*** (Publ. Math. Inst. Hungar. Acad. Sci. 1960). The original random graph paper. Short and beautiful.
- **Watts & Strogatz, *Collective dynamics of small-world networks*** (Nature 1998). The small-worldness paper that launched network science.
- **Barabási & Albert, *Emergence of scaling in random networks*** (Science 1999). The power-law-degree paper.
- For a worked derivation of $x^\top L x = \frac{1}{2} \sum (x_u - x_v)^2$ that you can verify by hand on a 4-vertex graph, see Strang, *Linear Algebra and Learning from Data* (Wellesley 2019), §III.2.
