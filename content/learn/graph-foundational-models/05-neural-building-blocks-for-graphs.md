---
title: "05 — Neural Building Blocks for Graphs"
date: 2026-08-12T17:40:00+03:00
draft: false
params:
  math: true
---

# 05 — Neural Building Blocks for Graphs
*Message passing, aggregation, permutation symmetry, and the deep toolbox of tricks (residual, normalisation, attention pooling) that GNNs and GFMs compose.*

This chapter is the bridge between "vanilla deep learning" (Ch 04) and "graph neural networks" (Ch 06). You will see the *primitive operations* that every GNN is made of, the *symmetry arguments* that constrain how those operations must look, and the *engineering tricks* (residuals, normalisation, dropout on graphs) that make them trainable in practice.

## 5.1 The Message Passing Neural Network (MPNN) framework

Gilmer et al. 2017 ([`1704.01212`](https://arxiv.org/abs/1704.01212)) unified the then-existing GNN literature into a single framework. Every GNN in the "message passing" family fits this template:

**For each layer $\ell = 0, \ldots, L-1$:**

1. **Message.** Each edge $(u, v)$ computes a message $m_{uv}^{(\ell)} = \text{MSG}^{(\ell)}(h_u^{(\ell)}, h_{v}^{(\ell)}, e_{uv})$, where $h_v^{(\ell)}$ is the hidden state of vertex $v$ at layer $\ell$, and $e_{uv}$ is an edge feature (optional).

2. **Aggregate.** Each vertex aggregates incoming messages: $a_v^{(\ell)} = \text{AGG}^{(\ell)}(\{m_{uv}^{(\ell)} : u \in \mathcal{N}(v)\})$.

3. **Update.** Vertex update: $h_v^{(\ell+1)} = \text{UPD}^{(\ell)}(h_v^{(\ell)}, a_v^{(\ell)})$.

4. (Optional) **Readout.** After $L$ layers, a graph-level representation: $h_G = \text{READOUT}(\{h_v^{(L)} : v \in V\})$.

**Canonical choices:**

| Function | Options | Used by |
|---|---|---|
| MSG | $W h_u$ (linear of source) | GCN, GraphSAGE |
| MSG | $W [h_u \| h_v]$ (concat) | GAT |
| MSG | $\text{MLP}(h_u \| h_v \| e_{uv})$ | MPNN, GINE |
| AGG | sum, mean, max | most GNNs |
| AGG | LSTM / GRU over neighbors | GraphSAGE with LSTM aggregator |
| UPD | $h_v + a_v$ (residual) | GCN with residual |
| UPD | $\sigma(W [h_v \| a_v])$ | GraphSAGE |
| READOUT | sum, mean | GIN |

**Key insight:** MPNN is just "neural network" with two extra constraints:
1. The operation must be **permutation-equivariant** (changing the order of vertices in $V$ permutes the outputs in the same way).
2. The operation must be **local** (each vertex's output depends only on its $k$-hop neighborhood, where $k$ is the number of layers).

> **Worked example.** A 1-layer MPNN on the 4-cycle. Initial $h_v^{(0)} = x_v$ (some scalar feature). For each vertex $v$, compute messages $m_{uv} = W h_u$ for each neighbor $u$, aggregate by sum: $a_v = \sum_{u \in \mathcal{N}(v)} W h_u = W \sum_{u \in \mathcal{N}(v)} h_u$, update: $h_v^{(1)} = \sigma(h_v^{(0)} + a_v) = \sigma(x_v + W \sum_{u \in \mathcal{N}(v)} x_u)$. For the 4-cycle with all $x_v = 1$ and $W = 1$, $a_v = 2$ for all $v$, so $h_v^{(1)} = \sigma(1 + 2) = \sigma(3)$ — constant. (Same issue as Ch 04 §4.2: the 4-cycle is fully symmetric.) If you change $x_1 = 0$ and the rest are 1, then $h_1^{(1)} = \sigma(0 + 2) = \sigma(2)$, $h_2^{(1)} = h_4^{(1)} = \sigma(1 + 1) = \sigma(2)$, $h_3^{(1)} = \sigma(1 + 2) = \sigma(3)$ — vertex 1 is now distinguishable from vertex 3.

## 5.2 Permutation equivariance and invariance

A function $f: \mathcal{G}_n \to \mathbb{R}^{n \times d}$ on graphs with $n$ vertices is **permutation equivariant** if for any permutation $P$ of the vertices,

$$f(P \cdot G) = P \cdot f(G).$$

If $f: \mathcal{G}_n \to \mathbb{R}^d$ produces a single graph-level vector, it is **permutation invariant**:

$$f(P \cdot G) = f(G).$$

**Theorem (characterisation).** A function is permutation equivariant iff it is a sum over local terms; the only permutation-equivariant linear maps on $h \in \mathbb{R}^{n \times d}$ are the identity and the sum-aggregation $\mathbf{1} \mathbf{1}^\top$ (with appropriate projection). The full set of equivariant linear layers is parameterised by the **convolutions** on the group, but in the graph case the standard result is that any permutation-equivariant linear layer can be written as

$$h' = A h W_1 + \mathbf{1} \mathbf{1}^\top h W_2,$$

i.e., a combination of "neighbor aggregation" and "self-transformation." This is the "linear-GNN" result.

**For GFM-level reasoning.** A GFM must produce:
- **Node embeddings** that are permutation-equivariant.
- **Graph embeddings** that are permutation-invariant.
- **Predictions per node** that are equivariant (e.g., node classification).
- **Predictions per graph** that are invariant (e.g., graph classification).

All of these are achievable within the MPNN framework (the readout step provides the invariance for graph-level tasks).

> **Worked example.** A permutation matrix $P$ that swaps vertices 1 and 2 in the 4-cycle: $P = \begin{pmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$. Apply $P$ to the adjacency: $P A P^\top$ swaps rows 1 and 2 of $A$ and then swaps columns 1 and 2 — the result is the same adjacency matrix up to relabeling. Apply $P$ to the 1-layer MPNN output: it should also just permute the rows. Verify: $P h^{(1)} = P \sigma(x + W A x) = \sigma(P x + W P A P^\top P x) = \sigma(P x + W P A x)$ — using $P A P^\top = A$ in the 4-cycle (the cycle is symmetric). So $P h^{(1)} = \sigma(P(x + W A x))$, and $h^{(1)}(P \cdot G) = P h^{(1)}(G)$ if $\sigma$ is applied elementwise. ✓

## 5.3 Aggregation functions

The aggregator $\text{AGG}$ in MPNN is critical. The standard options are:

| Aggregator | Pros | Cons |
|---|---|---|
| Sum | Most expressive (GIN, §6.4) | Sensitive to degree |
| Mean | Degree-normalised, smooth | Not injective on multisets |
| Max | Picks the "extreme" neighbor | Loses average information |
| LSTM / GRU | Models "set as sequence" | Not permutation-invariant in general — needs a canonical order |
| Attention-weighted sum | Learnable, can focus on important neighbors | Extra parameters, can overfit |
| Sort + MLP | Provably expressive (k-set pooling) | Expensive |
| Top-k | Discards low-importance neighbors | Information loss |
| Set transformer | Pooling by attention | Heavy |

**The injectivity question.** When is an aggregator a sufficient statistic for a multiset? Sum is injective when the multiset is bounded (in particular, when the number of elements is fixed and values are in a discrete set). Mean is not injective in general (e.g., $\{\{1, 1\}\}$ vs $\{\{2, 0\}\}$ both mean to 1). Max is even less injective.

**Why sum dominates in practice.** Sum is the most expressive aggregator that is also cheap, and it has a clean gradient signal. GIN (Ch 06 §6.4) proves that sum + MLP is the most expressive aggregation within the MPNN framework.

> **Worked example.** Multisets $\{1, 1, 2, 3\}$ and $\{1, 2, 2, 2\}$ both have sum 7, mean 1.75. Max gives 3 and 2 respectively (different). Only sum + max together distinguish them. For graph learning, sum is usually enough because the multiset is "neighbors" of a vertex and the graph topology provides additional context.

## 5.4 Edge features and multi-relational graphs

Many graphs have edge features: bond type in molecules, relation type in knowledge graphs, traffic flow in road networks, edge weight in weighted graphs.

**How to handle them in MPNN:**

**Option 1: Concatenate.** $m_{uv} = \text{MLP}([h_u \| h_v \| e_{uv}])$. Used in **GINE** (Hu et al. 2020), which is the edge-feature-aware variant of GIN.

**Option 2: Edge-conditioned message.** $m_{uv} = \text{MLP}_{e_{uv}}([h_u \| h_v])$ — different MLP per edge type. Used in **R-GCN** (Schlichtkrull et al. 2018, [`1703.06103`](https://arxiv.org/abs/1703.06103)) for knowledge graphs.

**Option 3: Multiplication.** $m_{uv} = h_u \odot W_{e_{uv}} h_v$ — element-wise product with edge-typed matrix. Used in some R-GCN variants.

**Knowledge graph embedding** (Ch 09 supplement) uses these ideas for link prediction:
- **TransE** ([`1412.6575`](https://arxiv.org/abs/1412.6575), Bordes et al. 2013): $h_r + h_u \approx h_v$ for $(u, r, v)$ triples.
- **RotatE** ([`1902.10197`](https://arxiv.org/abs/1902.10197), Sun et al. 2019): $h_v = h_u \circ h_r$ where $\circ$ is rotation in complex space.
- **ComplEx** ([`1707.01475`](https://arxiv.org/abs/1707.01475), Trouillon et al. 2016): complex-valued embeddings, dot-product scoring.

> **Worked example.** Methane (CH4) as a graph. 5 atoms: 1 carbon, 4 hydrogens. Edges: 4 C–H bonds, each with type "single bond" and feature vector (1, 0, 0, 0). GINE message: $m_{C-H} = \text{MLP}([h_C \| h_H \| (1,0,0,0)])$. With 4 such messages, aggregate at carbon, update. With edge features, you can distinguish the molecular geometry — a 1-layer MPNN on CH4 sees 4 messages at the carbon, all the same; without edge features, it's even more uniform.

## 5.5 Residual connections, normalisation, and dropout on graphs

**Residual connections.** Add the input to the output of each layer: $h_v^{(\ell+1)} = h_v^{(\ell)} + f^{(\ell)}(h_v^{(\ell)}, \mathcal{N}(v))$. Critical for deep GNNs (more than ~4 layers) to train. Used in GCNII (Chen et al. 2020), GATv2 with residuals, etc.

**Initial residual.** Even at the first layer, add the input: $h^{(1)} = h^{(0)} + \text{MPNN}^{(0)}(h^{(0)})$. This is the GCNII-style "initial residual."

**Dense connections.** Each layer takes the concatenation of all previous layers' outputs as input. Used in **JK-Net** ([`1806.03536`](https://arxiv.org/abs/1806.03536), Xu et al. 2018).

**Normalisation.** Layer norm applied per-vertex: $\text{LayerNorm}(h_v^{(\ell)})$, where the normalisation is across the feature dimension. Batch norm across vertices is also possible but interacts poorly with graph size variability. For Graph Transformers, **pair norm** (Zhao & Akoglu 2020) is a graph-aware norm that prevents feature collapse.

**Dropout.**
- **Standard dropout** on the vertex features: zero out a random fraction of features per vertex.
- **DropEdge** (Rong et al. 2020): at each training step, randomly drop a fraction $p$ of edges. The graph is sparsified, providing a regularisation effect similar to dropout.
- **DropNode**: drop a fraction of vertices (and their edges). Similar to standard dropout but on the vertex set.
- **DropMessage** (Feng et al. 2020): drop a fraction of messages.

The dropout rate that works in practice is graph-dependent. For dense graphs (molecular), DropEdge with $p \approx 0.1$ is standard. For sparse social graphs, $p$ can be higher (0.3–0.5).

> **Worked example.** 2-layer GCN on Cora with and without DropEdge($p=0.5$). Expected: DropEdge improves test accuracy by 1–2 points and reduces overfitting (smaller gap between train and test). Run: train both, plot the train/test curves. The version with DropEdge has a smaller train-test gap.

## 5.6 Virtual nodes and skip connections

**Virtual node** (or master node): a single extra node connected to every other node in the graph. Acts as a "global scratchpad" — every node's information can reach every other node in 2 hops through the virtual node. Used in GIN with virtual node, OGB winners, and many Graph Transformers.

Effect: a 2-layer GCN with virtual node is roughly equivalent to a 4-layer GCN without, in terms of receptive field. But the depth/over-smoothing trade-off is much better.

**Skip connections** beyond residuals: skip from layer $i$ directly to layer $L$ via concatenation, additive, or gating. JK-Net and Graph Transformers use this heavily.

> **Worked example.** 2-layer GCN with virtual node on a 10-vertex cycle. After 1 layer, each vertex sees its 2 neighbors + the virtual node. After 2 layers, each vertex sees its 4 neighbors + the virtual node + the virtual node's other connections (which is everything). So the receptive field at depth 2 is the whole graph — compare to depth 2 without virtual node, which is only 2-hop neighborhood.

## 5.7 Graph-level readout

For graph-level tasks (molecular property prediction, graph classification), you need to combine the per-vertex embeddings into a single graph embedding. The standard readouts:

| Readout | Formula | Properties |
|---|---|---|
| Sum | $h_G = \sum_v h_v^{(L)}$ | Permutation invariant, expressive |
| Mean | $h_G = (1/n) \sum_v h_v^{(L)}$ | Permutation invariant, smooth |
| Max | $h_G = \max_v h_v^{(L)}$ (per-feature) | Permutation invariant, selective |
| Sort + MLP | Sort features, apply MLP | Provably more expressive (k-set pooling) |
| Attention pooling | $h_G = \sum_v \alpha_v h_v^{(L)}$, $\alpha$ from attention | Learnable weights |
| Set transformer | Pooling by Set Transformer | Heavy but powerful |

**Sum vs mean.** Sum preserves more information (it's injective on multisets), but it grows with $n$. Mean normalises by size. For variable-size graphs, sum followed by an MLP that learns to normalise is often best.

**Jumping Knowledge.** Concatenate (or sum) the outputs of all layers: $h_G = \text{READOUT}([h^{(1)} \| h^{(2)} \| \ldots \| h^{(L)}])$. This is the standard "JK" idea from JK-Net. The intuition: a deep GNN may over-smooth, but a shallow GNN may not have enough receptive field; JK combines the best of both.

> **Worked example.** Molecule "caffeine" as a graph. 14 atoms, 14 bonds. 3-layer GIN with sum readout. After 3 layers, each atom's embedding is a 64-dim vector. Sum readout produces a 64-dim vector that captures the whole molecule. Train an MLP head on top to predict solubility.

## 5.8 Pooling operations

**Graph pooling** reduces the size of a graph by clustering vertices. The two main families:

**Spectral / cut-based.** Pool vertices by solving a cut problem. Examples:
- **DiffPool** (Ying et al. 2018): learn an assignment matrix $S \in \mathbb{R}^{n \times n'}$ that maps $n$ vertices to $n'$ clusters. Trained end-to-end. The coarsened adjacency is $A' = S^\top A S$ and the new features are $X' = S^\top X$. Loss includes a **link prediction** term encouraging $S$ to be cluster-friendly.
- **MinCut pooling** (Bianchi, Grattarola, Alippi 2020): a continuous relaxation of the normalised min-cut problem. Learns a soft cluster assignment.
- **DMoN** (Tsitsulin et al. 2020): spectral clustering with a soft NMI objective.

**Top-k selection.** Pick the top-$k$ vertices by some learned score, drop the rest. Examples:
- **gPool** (Gao & Ji 2019): score is $y = X p / \|p\|$ for a learned vector $p$, take the top $k$, project.
- **SAGPool** (Lee, Lee, Kang 2019): score from a GCN layer, top-$k$.
- **TopKPool** (Cangea et al. 2018): similar, used in the GIN paper's baselines.

**Connection to GFM.** A GFM should learn pooling operations implicitly (e.g., via the readout) or use them as components. Recent GFMs (e.g., [`2502.03251`](https://arxiv.org/abs/2502.03251) RiemannGFM) include pooling in their design.

> **Worked example.** DiffPool on a 4-vertex cycle with $n' = 2$. Assignment $S \in \mathbb{R}^{4 \times 2}$, $S_{ij} \in [0, 1]$, $\sum_j S_{ij} = 1$. If the model learns to map $S$ such that vertices 1, 3 are assigned to cluster 1 and 2, 4 to cluster 2, the coarsened graph is $K_2$ (the two clusters are fully connected because in the cycle, vertex 1 is adjacent to vertex 2). Coarsened adjacency: $A' = S^\top A S = \begin{pmatrix} 0 & 2 \\ 2 & 0 \end{pmatrix}$ (the 2 indicates 2 edges in the original graph that connect the two clusters — i.e., 1-2 and 3-4).

## 5.9 The expressive power of aggregators

A formal statement: among $\{ \text{sum}, \text{mean}, \text{max} \}$, only **sum** is injective on multisets. Mean and max are not. This is the basis of GIN (Ch 06 §6.4) being "the most expressive MPNN."

**Theorem (Xu et al. 2019, GIN, [`1810.00826`](https://arxiv.org/abs/1810.00826)).** A sum-based MPNN with a sufficiently expressive MLP is at most as powerful as the **1-Weisfeiler-Leman** test for distinguishing non-isomorphic graphs. And conversely, the 1-WL test can be approximated by a sum-MPNN.

This means: any MPNN is bounded in distinguishability by 1-WL, and 1-WL-bounded MPNNs can be built. To go beyond 1-WL, you need higher-order GNNs (Ch 07).

**Connection to GFM design.** A GFM that uses only 1-WL-bounded operations will be limited in what structural information it can encode. The "structural perspective" GFMs (e.g., [`2407.19941`](https://arxiv.org/abs/2407.19941)) explicitly try to capture motifs / substructures beyond 1-WL by augmenting the MPNN with structural features.

> **Worked example.** Consider two non-isomorphic graphs $G_1 = $ a 6-cycle $C_6$ and $G_2 = $ two disjoint triangles $2 K_3$. Both have 6 vertices and 6 edges, same degree sequence (all 2s). A 1-WL test cannot distinguish them (in fact, 1-WL terminates with the same coloring on all vertices of both graphs). A sum-based MPNN has the same problem. To distinguish, you need a higher-order method — e.g., count triangles, or use a 3-WL-style architecture.

## 5.10 The GCN propagation rule, derived

Putting it all together, Kipf & Welling's GCN ([`1609.02907`](https://arxiv.org/abs/1609.02907)) is:

1. **Add self-loops:** $\tilde A = A + I$.
2. **Symmetric normalise:** $\hat A = \tilde D^{-1/2} \tilde A \tilde D^{-1/2}$, where $\tilde D$ is the degree matrix of $\tilde A$.
3. **One round of message passing:** $H^{(\ell+1)} = \sigma(\hat A H^{(\ell)} W^{(\ell)})$.

The matrix $\hat A$ can be read as: $H^{(\ell+1)}_v = \sigma\left( W^{(\ell)\top} \left[ \frac{1}{\sqrt{\deg(v) + 1}} h_v^{(\ell)} + \sum_{u \in \mathcal{N}(v)} \frac{1}{\sqrt{\deg(v) + 1} \sqrt{\deg(u) + 1}} h_u^{(\ell)} \right] \right)$.

Interpretation: each vertex's new state is a weighted sum of its own state (with self-loop) and the normalised states of its neighbors, then projected through $W^{(\ell)}$, then through a non-linearity.

**Why the symmetric normalisation?** It balances contributions from high-degree and low-degree vertices. Without it, a vertex with 1000 neighbors would dominate a vertex with 10. The symmetric normalisation $\hat A$ is a "graph-aware" version of standard batch normalisation.

**Connection to Laplacian filtering.** $\hat A$ is approximately $I - L_{\text{sym}} / 2$ (for graphs with uniform degree), so the propagation is approximately applying a low-pass filter in the spectral domain. This is the spectral "explanation" of GCN.

> **Worked example.** The 4-cycle, $\hat A = D^{-1/2}(A + I)D^{-1/2}$ (here $D$ becomes $3I$ because of self-loops, so $D^{-1/2} = I / \sqrt 3$). So $\hat A = (A + I) / 3 = \begin{pmatrix} 1/3 & 1/3 & 0 & 1/3 \\ 1/3 & 1/3 & 1/3 & 0 \\ 0 & 1/3 & 1/3 & 1/3 \\ 1/3 & 0 & 1/3 & 1/3 \end{pmatrix}$. For $H^{(0)} = X = \begin{pmatrix} 1 \\ 0 \\ 1 \\ 0 \end{pmatrix}$, $H^{(1)} = \sigma(\hat A X) = \sigma((1/3, 1/3, 1/3, 1/3)^\top) = (\sigma(1/3), \sigma(1/3), \sigma(1/3), \sigma(1/3))$ — every vertex has the same hidden state after 1 layer. (Same over-smoothing problem.) With $W = \text{diag}(1, 2)$, $H^{(1)} = \sigma(\hat A X W) = \sigma(\hat A \cdot (1, 0, 1, 0)^\top) = $ same as before (the weights are absorbed in $X$). Try with $X = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 0 \end{pmatrix}$ (4 vertices, 2 features each). Then $\hat A X = \begin{pmatrix} 2/3 \\ 2/3 \\ 2/3 \\ 2/3 \end{pmatrix} \otimes$ (averages of each column). The two features become mixed: each row of $\hat A X$ is the average of $X$ weighted by the symmetric degree. Hmm, this needs a full numeric example. Just remember: the propagation mixes the feature of each vertex with the average of its neighbors' features, scaled by $1/\sqrt{\deg}$.

## Exercises

1. **MPNN by hand.** A 5-vertex path $1 - 2 - 3 - 4 - 5$, initial features $h_v^{(0)} = v$ (i.e., $h_1^{(0)} = 1, \ldots, h_5^{(0)} = 5$), $W = I$, sum aggregation, no non-linearity. Compute $h_v^{(1)}$ and $h_v^{(2)}$ for all $v$. Notice the boundary effect: $h_1^{(1)} = h_1^{(0)} + h_2^{(0)} = 1 + 2 = 3$, etc.

2. **Permutation equivariance proof.** Let $f: \mathbb{R}^{n \times d} \to \mathbb{R}^{n \times d}$ be a 1-layer MPNN with sum aggregation. Show that $f$ is permutation equivariant iff the message function is $m(h_u, h_v) = W h_u$ (or symmetric in $u, v$).

3. **Sum vs mean vs max.** For a graph with 3 vertices, features $h_1 = (1, 0), h_2 = (0, 1), h_3 = (1, 1)$, compute (a) sum, (b) mean, (c) max readout. Which is most informative? Which is most affected by $n$?

4. **Virtual node effect.** A 2-layer GCN on a 4-cycle without virtual node has receptive field 2 hops. With virtual node, what is the receptive field? (Answer: the whole graph.) Why?

5. **Edge features with GINE.** Implement GINE on a small molecule (e.g., methane). Compare to a vanilla GIN (no edge features) on the same task. The edge features are usually necessary for chemistry tasks.

6. **DropEdge.** Implement a 3-layer GCN on Cora. Compare test accuracy with DropEdge($p = 0, 0.1, 0.3, 0.5$). The sweet spot is usually around $p = 0.2$.

7. **JK-Net on Cora.** Implement 4-layer GCN with Jumping Knowledge (concat) and 4-layer GCN without. JK-Net should have higher test accuracy (less over-smoothing).

8. **Injectivity of sum.** Show that sum is injective on multisets of bounded size. Construct two multisets $M_1, M_2$ of real numbers where $\text{sum}(M_1) = \text{sum}(M_2)$ but $M_1 \neq M_2$. (Hint: trivial — many multisets have the same sum. The claim is that the multiset can be *recovered* from the sum for bounded support, not that no other multiset shares the sum.)

9. **GNN vs MLP on graph tasks.** Take a 4-cycle, label vertices $\{1, 3\}$ as class 0, $\{2, 4\}$ as class 1. Train (a) an MLP that takes only the degree (always 2) as input, (b) a 1-layer GCN with the degree as input. The MLP cannot do better than 50% accuracy. The GCN should reach 100% after a few iterations. Why?

10. **Reading.** Read [`1704.01212`](https://arxiv.org/abs/1704.01212) (Gilmer et al., MPNN). Identify which of the GNNs it surveys correspond to which choices in the MSG/AGG/UPD/READOUT table. Also note the QM9 results in §4.2.

## Further Reading

- **[`1704.01212`](https://arxiv.org/abs/1704.01212)** — Gilmer, Schoenholz, Riley, Vinyals, Dahl, *Neural Message Passing for Quantum Chemistry* (ICML 2017). The MPNN framework. The single most important GNN paper to read first.
- **[`1806.01261`](https://arxiv.org/abs/1806.01261)** — Battaglia et al., *Relational inductive biases, deep learning, and graph networks*. The "graph networks" generalisation including edge and graph-level state. Read this alongside Gilmer.
- **[`1810.00826`](https://arxiv.org/abs/1810.00826)** — Xu, Hu, Leskovec, Jegelka, *How Powerful are Graph Neural Networks?* (GIN, ICLR 2019). The injectivity analysis of aggregators.
- **[`1901.00596`](https://arxiv.org/abs/1901.00596)** — Wu et al., *A Comprehensive Survey on GNNs*. §III has the cleanest taxonomy of GNN building blocks.
- **[`2003.04078`](https://arxiv.org/abs/2003.04078)** — Sato, *A Survey on The Expressive Power of GNNs*. The theoretical analysis of what GNNs can and cannot distinguish.
- **[`1703.06103`](https://arxiv.org/abs/1703.06103)** — Schlichtkrull et al., *R-GCN* (ESWC 2018). The multi-relational GNN.
- **[`1806.03536`](https://arxiv.org/abs/1806.03536)** — Xu et al., *Jumping Knowledge Networks* (ICML 2018). Dense connections for GNNs.
- **[`2106.03058`](https://arxiv.org/abs/2106.03058)** — Klicpera, Bojchevski, Günnemann, *APPNP* (ICLR 2019). The personalised PageRank-based propagation rule.
- **[`2105.14491`](https://arxiv.org/abs/2105.14491)** — Brody, Alon, Yahav, *How Attentive are Graph Attention Networks?* (GATv2, ICLR 2022). The improved attention mechanism for graphs.
- **For pool operations**: Ying et al., *Hierarchical Graph Representation Learning with Differentiable Pooling* (DiffPool, NeurIPS 2018); Bianchi, Grattarola, Alippi, *MinCut Pooling* (ICML 2020).
- **For DropEdge**: Rong, Huang, Xu, Huang, *DropEdge: Towards Deep Graph Convolutional Networks on Node Classification* (ICLR 2020).
- **For GINE**: Hu, Liu, Gomes, Zitnik, Liang, Leskovec, *Strategies for Pre-training Graph Neural Networks* (ICLR 2020). The pre-training paper that introduced GINE.
- **The PyTorch Geometric docs** — excellent API reference for MPNN layers, aggregators, and pool operations. See `torch_geometric.nn` and `torch_geometric.nn.pool`.
