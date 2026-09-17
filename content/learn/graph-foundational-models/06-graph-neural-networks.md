---
title: "06 — Graph Neural Networks"
date: 2026-08-12T17:40:00+03:00
draft: false
params:
  math: true
---

# 06 — Graph Neural Networks
*GCN, GraphSAGE, GAT, GIN — the four canonical architectures you must be able to derive, implement, and critique from memory.*

This is the technical heart of the cheatsheet. The four architectures below are not just historical — they remain the building blocks of every modern Graph Transformer and GFM. If you understand them deeply, you can read any newer paper and identify which ideas it borrows from which ancestor.

## 6.1 GCN: Kipf & Welling (2017)

**Paper.** [`1609.02907`](https://arxiv.org/abs/1609.02907) — Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks* (ICLR 2017). The most-cited GNN paper.

**Derivation.** Start with the spectral view (§3.4): a graph convolutional filter of degree $K$ is $\sum_{k=0}^{K} W_k T_k(\tilde L) X$, where $\tilde L$ is the rescaled Laplacian. Restrict to $K = 1$:

$$H^{(1)} = W_0 X + W_1 (I - A) X = W_0 X + W_1 X - W_1 A X.$$

Apply two simplifying assumptions:
1. $W_0 = W_1 = W$ (so the filter is constrained).
2. Add self-loops: $A \to \tilde A = A + I$.

Then $H^{(1)} = W X + W \tilde A X = W (I + \tilde A) X$.

Normalise by degree: $I + \tilde A \to \tilde D^{-1/2} (I + \tilde A) \tilde D^{-1/2} = \hat A$. The full propagation rule:

$$\boxed{H^{(\ell+1)} = \sigma\left( \hat A H^{(\ell)} W^{(\ell)} \right)}, \quad \hat A = \tilde D^{-1/2} (A + I) \tilde D^{-1/2}.$$

**Properties:**
- Permutation equivariant.
- Spectral interpretation: first-order approximation of a Chebyshev filter.
- Effective receptive field: $K$ layers = $K$-hop neighborhood.
- Over-smoothing: with too many layers, all node embeddings converge to a fixed point.
- Per-layer parameter count: $O(d_\ell d_{\ell+1})$, where $d_\ell$ is the hidden dim.

**Limitations:**
- Inherently transductive (the graph is fixed at training time).
- Memory: $O(L d^2 + L |\hat A| d)$ for $L$ layers.
- Static weights: every edge is treated the same.

> **Worked example.** 4-cycle, 1-layer GCN. $\hat A = (A + I)/3$ (from §5.10). $H^{(0)} = X \in \mathbb{R}^{4 \times 2}$. $H^{(1)} = \sigma(\hat A X W)$. With $W = I$ and $\sigma = $ identity: $H^{(1)} = \hat A X$. Each row of $H^{(1)}$ is the average of the corresponding row of $X$ with its two neighbors (and itself). Boundary vertices (1, 3) have the same average; same for (2, 4). The symmetry of the cycle is broken only if $X$ breaks it.

## 6.2 GraphSAGE: Hamilton, Ying, Leskovec (2017)

**Paper.** [`1706.02216`](https://arxiv.org/abs/1706.02216) — Hamilton, Ying, Leskovec, *Inductive Representation Learning on Large Graphs* (NeurIPS 2017).

**Architecture.** The "SAGE" stands for "Sample and Aggregate." The key idea: instead of using the full graph at each layer, **sample** a fixed-size neighborhood, then **aggregate** features.

**Forward pass:**
$$h_{\mathcal{N}(v)}^{(\ell)} = \text{AGG}^{(\ell)}\left( \{h_u^{(\ell)} : u \in \mathcal{N}(v)\} \right), \quad h_v^{(\ell+1)} = \sigma\left( W^{(\ell)} \cdot \text{CONCAT}\left( h_v^{(\ell)}, h_{\mathcal{N}(v)}^{(\ell)} \right) \right).$$

Normalisation: $h_v^{(\ell+1)} \leftarrow h_v^{(\ell+1)} / \|h_v^{(\ell+1)}\|_2$ (L2 normalise). This stabilises training.

**Three aggregators originally proposed:**
1. **Mean**: $h_{\mathcal{N}(v)}^{(\ell)} = \text{mean}(\{h_u^{(\ell)}\})$. (Equivalent to GCN's symmetric-normalised aggregation, with a different normalisation.)
2. **LSTM**: apply LSTM to a random permutation of neighbors. *Not* permutation invariant in general; mitigated by shuffling.
3. **Pooling**: $h_{\mathcal{N}(v)}^{(\ell)} = \max(\{\sigma(W_{\text{pool}} h_u^{(\ell)} + b) : u \in \mathcal{N}(v)\})$ — element-wise max over a learned transformation of neighbors.

**Properties:**
- **Inductive**: can generate embeddings for new nodes (as long as they connect to the training graph).
- **Sampling** allows it to scale to large graphs (fixed per-vertex compute per layer).
- The concat-and-project aggregation is more expressive than GCN's average: it allows the model to separately use the node's own features and its neighbors'.

**Inductive learning on graphs.** This is the key contribution: at test time, you can apply the trained SAGE to a new graph (or a new part of the graph), without re-training. The transductive GCN cannot do this — it's tied to the specific adjacency matrix at training time.

> **Worked example.** 5-vertex path, 1-layer SAGE-mean. $h_v^{(0)} = v$ (scalar). For each $v$, $\text{AGG} = $ mean of neighbors. $h_1^{(0)} = 1$, neighbors = {2}, mean = 2, concat = (1, 2), $W = (1, 0)^\top$ (a 2x1 matrix), so $h_1^{(1)} = \sigma(1) = \sigma(1)$ (or with sigmoid $\approx 0.73$). For interior $v = 3$, neighbors = {2, 4}, mean = 3, concat = (3, 3), $h_3^{(1)} = \sigma(6 W_1 + 3 W_2)$ — depends on $W$.

## 6.3 GAT: Graph Attention Networks (2018)

**Paper.** [`1710.10903`](https://arxiv.org/abs/1710.10903) — Veličković, Cucurull, Casanova, Romero, Liò, Bengio, *Graph Attention Networks* (ICLR 2018).

**Architecture.** Replace the fixed weights of GCN with **learnable, edge-dependent** attention coefficients:

For each edge $(u, v)$ at layer $\ell$:
$$e_{uv}^{(\ell)} = \text{LeakyReLU}\left( \mathbf{a}^\top [W^{(\ell)} h_u^{(\ell)} \| W^{(\ell)} h_v^{(\ell)}] \right),$$
$$\alpha_{uv}^{(\ell)} = \frac{\exp(e_{uv}^{(\ell)})}{\sum_{w \in \mathcal{N}(u)} \exp(e_{uw}^{(\ell)})}.$$

The new state:
$$h_u^{(\ell+1)} = \sigma\left( \sum_{v \in \mathcal{N}(u)} \alpha_{uv}^{(\ell)} W^{(\ell)} h_v^{(\ell)} \right).$$

**Multi-head attention** (transformer-style): $h_u^{(\ell+1)} = \text{CONCAT}_k \sigma\left( \sum_v \alpha_{uv,k}^{(\ell)} W_k^{(\ell)} h_v^{(\ell)} \right)$ for $K$ heads; or average at the last layer.

**Properties:**
- **Implicit learnable edge weights**: the attention coefficient $\alpha_{uv}$ can focus on important neighbors.
- **Inductive**: same as SAGE.
- **No spectral interpretation**: the attention is not a polynomial filter in $L$.
- **More parameters than GCN**: $O(d^2 + 2 d)$ per head per layer.

**GATv2** ([`2105.14491`](https://arxiv.org/abs/2105.14491), Brody, Alon, Yahav 2022): fixes a "static attention" issue with the original GAT — the original GAT computes $\alpha_{uv}$ such that any node $u$ with the same neighborhood (after the $W$ projection) gets the same attention distribution. GATv2 swaps the order: it computes $e_{uv} = \mathbf{a}^\top \text{LeakyReLU}(W [h_u \| h_v])$, which makes the attention strictly more expressive. **GATv2 is strictly better than GAT on every benchmark I have seen.**

> **Worked example.** A 3-vertex star: center $c$ connected to leaves $L_1, L_2$. For the center, $h_c^{(1)} = \sigma(\alpha_{c,L_1} W h_{L_1} + \alpha_{c,L_2} W h_{L_2})$. If $h_{L_1} = h_{L_2}$ (symmetric leaves), GAT computes $\alpha_{c, L_1} = \alpha_{c, L_2} = 0.5$ (by symmetry of the $e$ formula). GATv2 also computes the same. To make them differ, change $h_{L_1}$: the attention should shift toward $L_1$.

## 6.4 GIN: Graph Isomorphism Network (2019)

**Paper.** [`1810.00826`](https://arxiv.org/abs/1810.00826) — Xu, Hu, Leskovec, Jegelka, *How Powerful are Graph Neural Networks?* (ICLR 2019).

**Architecture.** The simplest MPNN that achieves the maximum expressivity bounded by 1-WL:

$$h_v^{(\ell+1)} = \text{MLP}^{(\ell)}\left( (1 + \epsilon^{(\ell)}) h_v^{(\ell)} + \sum_{u \in \mathcal{N}(v)} h_u^{(\ell)} \right).$$

**Key choices:**
- **Sum aggregation** (most expressive).
- **MLP** instead of a linear projection (MLPs are universal approximators for the inner function).
- The $(1 + \epsilon)$ term: a learnable or fixed scalar that gives the node its own "weight" relative to the neighborhood. Often set to $0$ for simplicity.

**Properties:**
- **Maximally expressive** within the MPNN family: as powerful as 1-WL in distinguishing non-isomorphic graphs.
- **Injective** on multisets of bounded size: the sum + MLP is an injective map (under mild conditions on the MLP).
- Often the **strongest baseline** for graph classification.

**GIN with edge features (GINE).** Replace the aggregation: $h_v^{(\ell+1)} = \text{MLP}^{(\ell)}\left( (1 + \epsilon) h_v^{(\ell)} + \sum_{u \in \mathcal{N}(v)} \text{MLP}_e(e_{uv}) \cdot h_u^{(\ell)} \right)$. Used in the molecular property prediction benchmarks.

> **Worked example.** Two graphs: $G_1 = $ path $1-2-3$, $G_2 = $ triangle $1-2-3-1$. A 1-WL test cannot distinguish them. A 1-layer GIN: $h_v^{(1)} = \text{MLP}((1+\epsilon) h_v^{(0)} + \sum h_u^{(0)})$. For vertex 1 in $G_1$: $\text{MLP}((1+\epsilon) h_1 + h_2)$. For vertex 1 in $G_2$: $\text{MLP}((1+\epsilon) h_1 + h_2 + h_3)$. These differ if the features are non-trivial. But the multiset of GIN outputs over all vertices in $G_1$ vs $G_2$ — for a 1-layer GIN, both graphs give the same multiset if the initial features are the same. So a 1-layer GIN cannot distinguish them. A 2-layer GIN: now each vertex's output depends on the 2-hop neighborhood. The triangle has more 2-hop structure than the path. A 2-layer GIN **can** distinguish them.

## 6.5 Comparison table

| | GCN | GraphSAGE | GAT | GIN |
|---|---|---|---|---|
| Aggregation | Sum + symmetric norm | Mean / LSTM / Pool | Attention-weighted sum | Sum |
| Update | Linear + ReLU | Concat + Linear + ReLU | Attention-weighted + ReLU | MLP (universal) |
| Self-loop | Yes ($A + I$) | Optional | Optional | Optional |
| Edge features | No (by default) | No (by default) | No (by default) | Yes (GINE) |
| Inductive | No | Yes | Yes | Yes |
| Expressivity | 1-WL | 1-WL (mean) | 1-WL (GAT) / less (GATv1) | **Max 1-WL** |
| Spectral interpretation | Yes (1st-order Cheb) | No | No | No |
| Per-layer params | $O(d^2)$ | $O(d^2)$ | $O(d^2 + 2d)$ | $O(d^2)$ per MLP layer |
| Memory per layer | $O(\|A\| d)$ | $O(\|A\| d)$ (no sampling) | $O(\|A\| d)$ | $O(\|A\| d)$ |
| OGB winners? | Rarely | Sometimes | Rarely | **Frequently** |

**Why GIN dominates on graph classification.** Sum + MLP is the most expressive aggregation in the MPNN family. With proper readout (sum), GIN matches or exceeds all other GNNs on standard benchmarks. The OGB leaderboards show GIN variants in the top tier.

**Why GCN dominates on node classification.** GCN's spectral interpretation gives a clean theoretical foundation, and the simple propagation rule is fast. On node classification (homophilic graphs), GCN is competitive with much more complex models.

**Why GAT/GATv2 dominate on attention-heavy tasks.** When edges have variable importance (e.g., in citation networks where some citations are more relevant), attention helps.

## 6.6 Modern variants and extensions

This section is a quick survey of important variants, with pointers.

**SGC** (Wu et al. 2019): simplify GCN by removing the non-linearity. $H^{(L)} = \hat A^L X W$. A single linear layer. Surprisingly competitive on many benchmarks.

**APPNP** ([`2106.03058`](https://arxiv.org/abs/2106.03058), Klicpera, Bojchevski, Günnemann 2019): decouple feature transformation from propagation. $H^{(0)} = \text{MLP}(X)$, $H^{(k+1)} = (1 - \alpha) \hat A H^{(k)} + \alpha H^{(0)}$. Allows very deep "propagation depth" without over-smoothing.

**GCNII** (Chen et al. 2020): two residual connections — initial residual and identity residual. $H^{(\ell+1)} = \sigma( ((1 - \alpha_\ell) \hat A + \alpha_\ell I) ((1 - \beta_\ell) H^{(\ell)} + \beta_\ell H^{(0)}) W^{(\ell)} )$. Allows training very deep GCNs (up to 64 layers) without over-smoothing.

**PairNorm** (Zhao & Akoglu 2020): a normalisation that prevents total collapse of node embeddings in deep GNNs.

**Gated GCN** (Bresson & Laurent 2018): use a GRU-style gate to combine current and propagated states.

**GATv2** ([`2105.14491`](https://arxiv.org/abs/2105.14491)): strictly more expressive attention than GAT (see §6.3).

**Gated Graph Neural Networks** (Li et al. 2016): use GRUs as the update function for an MPNN. Historical predecessor.

**Directed GNNs** (e.g., MagNet, DirGNN): explicitly handle directed graphs. Standard GCN/GraphSAGE treat the graph as undirected; these variants use in/out-degree separate normalisations.

> **Worked example.** A 2-layer GCN on Cora for node classification. Standard benchmark: ~81% test accuracy. A 2-layer GAT: ~83% (slight improvement). A 2-layer GATv2: ~84%. A 2-layer GIN on Cora: not typically used (GIN shines on graph classification). A GCNII (64 layers): ~84% with initial + identity residual.

## 6.7 Heterophily and what to do about it

Many real graphs are **heterophilic**: neighbors of a vertex tend to have different labels. Examples: dating networks, fraud detection, molecular property graphs.

**Symptoms in GNNs:**
- Deep GCNs fail (they over-smooth, mixing different-class neighbors).
- The performance gap between GCN and MLP can be small or even reverse.

**Solutions:**
- **H2GCN** (Zhu et al. 2020): use 3 ego/neighbor separation: aggregate over neighbors-of-neighbors, not just neighbors.
- **GPR-GNN** (Chien et al. 2021): learn the PageRank-style propagation weights (GPR = Generalized PageRank).
- **Geom-GCN** ([`2002.05287`](https://arxiv.org/abs/2002.05287), Pei et al. 2020): use geometric relationships (in the latent space) to define the aggregation.
- **FAGCN** (Bo et al. 2021): low-frequency + high-frequency filters, learned.
- **GloGNN** (Li et al. 2022): global + local neighbors, learned coefficient.

**Connection to GFMs.** A generalist GFM should be able to handle both homophilic and heterophilic graphs. This is one of the open challenges discussed in Ch 12.

> **Worked example.** A heterophilic graph: two clusters A and B, all edges are between A and B (a complete bipartite graph $K_{n/2, n/2}$). Node labels: A = class 0, B = class 1. A 1-layer GCN produces the same embedding for all A vertices and the same (different) embedding for all B vertices — so it can still classify them (perfect accuracy on this synthetic case). A 2-layer GCN: now each A vertex aggregates from B's, which are the same; the GCN can no longer distinguish. GCN with 1 layer works on this heterophilic graph; 2 layers fail. The opposite of a homophilic graph.

## 6.8 Benchmarks and evaluation

**Node classification.**
- **Cora, Citeseer, PubMed** (the "Planetoid" datasets). Small, ~3K nodes. Homophilic. The original GNN benchmarks. 7-class classification on Cora, 6 on Citeseer, 3 on PubMed. Random splits are common; some papers use the **public split** (Sen et al. 2008) for fair comparison.
- **OGB-ArXiv** ([`2005.00687`](https://arxiv.org/abs/2005.00687), Hu et al. 2020): ~170K nodes, 40-class subject classification of arXiv papers. The OGB leaderboard is the modern standard for node classification.
- **OGB-Products**: ~2.4M nodes, 47-class product category. Much larger scale.
- **OGB-Papers100M**: ~111M nodes, the largest open dataset.

**Graph classification.**
- **ZINC** (subset of ZINC database, ~250K molecules, regression to "penalized logP"): 12K training molecules, 1K val, 1K test. Used in the GIN paper and most follow-up work.
- **MoleculeNet** ([`1703.00564`](https://arxiv.org/abs/1703.00564), Wu et al. 2018): the molecular ML benchmark suite (BBBP, Tox21, SIDER, etc.).
- **TUDatasets**: a collection of small graph-classification datasets (MUTAG, PROTEINS, NCI1, etc.). The Weisfeiler-Leman kernel and early GNN benchmark.
- **OGB-MolPCBA, OGB-MolHIV**: molecular property prediction.

**Link prediction.**
- **OGB-DDI**: drug-drug interaction prediction.
- **OGB-Collab**: collaboration prediction in a citation network.
- **WN18, FB15k**: knowledge-graph link prediction (the original TransE/ComplEx benchmarks).

**Standard evaluation protocol:**
- OGB: fixed train/val/test split, 10 runs, report mean and std.
- Cora: 10 random splits, 100 random initialisations, report mean.
- **Report:** mean ± std over runs. **Critical:** do not tune on the test set.

> **Worked example.** A typical OGB-ArXiv experiment: train a 3-layer GCN for 500 epochs, learning rate 0.01, hidden dim 256, dropout 0.5. Expected test accuracy: ~72-73%. A GAT with 4 attention heads: ~73-74%. A GraphSAGE: ~71-72%. The differences are small (1-2 points), so benchmark reporting requires careful statistical analysis.

## 6.9 The "what to use when" decision tree

| Task | Default model | Why |
|---|---|---|
| Node classification, small graph | 2-layer GCN | Fast, simple, good baseline |
| Node classification, large graph | GraphSAGE with neighbor sampling | Memory-bounded |
| Node classification, heterophilic | GPR-GNN or H2GCN | Need higher-order info |
| Graph classification, small molecules | GIN with sum readout | Most expressive MPNN |
| Graph classification, large graphs | GIN with virtual node | Better long-range |
| Link prediction | VGAE or SEAL | Graph-autoencoder or subgraph methods |
| Knowledge graph completion | TransE / RotatE / ComplEx | Relational embeddings |
| Very deep GNN | GCNII or APPNP | Avoid over-smoothing |
| Attention over neighbors | GATv2 | Strictly better than GAT |
| With edge features | GINE | Edge-feature-aware aggregation |
| 3D molecular | SchNet / DimeNet / Equiformer | Equivariance to rotations |

**Connection to GFMs.** A GFM must support a wide range of tasks. The natural way is to pretrain a flexible architecture (e.g., a Graph Transformer) on a large corpus of graphs, then finetune for each task. The "what to use when" question becomes "what is the best finetuning recipe" — see Ch 11.

## Exercises

1. **GCN by hand.** A 4-vertex cycle with self-loops, $X = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 0 \end{pmatrix}$, 1-layer GCN with $W = I$ and no bias. Compute $H^{(1)}$.

2. **GraphSAGE sampling.** For a vertex with degree 100, GraphSAGE with $S_1 = 25$ first-hop samples, $S_2 = 10$ second-hop samples: compute the total number of vertices visited in a 2-layer forward pass. (Answer: $1 + 25 + 25 \cdot 10 = 276$.)

3. **GAT attention weights.** A 3-vertex star with center $c$ and leaves $L_1, L_2$. Initial $h = (1, 0, 0)^\top$ for $c, L_1, L_2$. Compute the GAT attention coefficients $\alpha_{c, L_1}, \alpha_{c, L_2}$ with $W = I$, attention vector $\mathbf{a} = (1, 1)^\top$, LeakyReLU slope 0.2. (You should get $\alpha_{c, L_1} = \alpha_{c, L_2} = 0.5$ because the leaves are symmetric.)

4. **GIN expressivity.** Two graphs: $G_1$ = a 4-cycle, $G_2$ = two disjoint edges. Same number of vertices (4) and edges (4) and degree sequence (2, 2, 2, 2). A 1-layer GIN with MLP = identity: do the per-vertex embeddings distinguish them? (Answer: yes — in $G_1$, the 1-hop sums are (2, 2, 2, 2); in $G_2$, the 1-hop sums are (1, 1, 1, 1). So GIN distinguishes them. GCN with mean aggregation: in $G_1$, mean of neighbors is (1, 1, 1, 1) for all; in $G_2$, mean is (0.5, 0.5, 0.5, 0.5). Different values but the same multiset — GCN distinguishes the values, not the structures. Wait, actually different *values* of means do distinguish the graphs.)

5. **GAT vs GATv2.** Implement both on a synthetic graph. On a graph where leaves are identical (by feature), both GAT and GATv2 assign equal attention. On a graph where leaves have different features, GATv2 should assign different attention; GAT may or may not, depending on the weights.

6. **SGC vs GCN.** A 2-layer SGC: $H = \hat A^2 X W$ (one linear layer). Compare test accuracy to a 2-layer GCN on Cora. Often SGC is within 0.5 points of GCN, and is much faster.

7. **APPNP propagation.** Implement APPNP with $\alpha = 0.1$ and $K = 10$ propagation steps on Cora. Compare to a 10-layer GCN. APPNP should have higher test accuracy (less over-smoothing).

8. **GCNII for depth.** Implement GCNII with 64 layers on Cora. With proper hyperparameters ($\alpha = 0.1, \lambda = 0.5$), it should reach ~84% test accuracy — better than a 2-layer GCN.

9. **Read the GIN paper.** Read [`1810.00826`](https://arxiv.org/abs/1810.00826) (Xu et al.). Identify the proof that GIN is at most as powerful as 1-WL (Theorem 3 in the paper). Identify the proof that any 1-WL-bounded GNN can be approximated by a GIN.

10. **Reproduce Cora results.** Implement a 2-layer GCN from scratch in PyTorch. Train on Cora with the public split. Target: ~81% test accuracy. Bonus: add DropEdge (p=0.5) and see if test accuracy improves.

## Further Reading

- **[`1609.02907`](https://arxiv.org/abs/1609.02907)** — Kipf & Welling, *Semi-Supervised Classification with GCNs* (ICLR 2017). Read this first.
- **[`1706.02216`](https://arxiv.org/abs/1706.02216)** — Hamilton, Ying, Leskovec, *GraphSAGE* (NeurIPS 2017). The inductive learning paper.
- **[`1710.10903`](https://arxiv.org/abs/1710.10903)** — Veličković et al., *GAT* (ICLR 2018). The attention paper.
- **[`1810.00826`](https://arxiv.org/abs/1810.00826)** — Xu et al., *GIN* (ICLR 2019). The expressivity paper.
- **[`2105.14491`](https://arxiv.org/abs/2105.14491)** — Brody, Alon, Yahav, *GATv2* (ICLR 2022). Strictly better than GAT.
- **[`1901.00596`](https://arxiv.org/abs/1901.00596)** — Wu et al., *A Comprehensive Survey on GNNs*. §III-V cover the four architectures with derivations.
- **[`2003.04078`](https://arxiv.org/abs/2003.04078)** — Sato, *A Survey on The Expressive Power of GNNs*. The formal expressivity analysis.
- **[`2106.03058`](https://arxiv.org/abs/2106.03058)** — Klicpera, Bojchevski, Günnemann, *APPNP* (ICLR 2019). PageRank propagation.
- **[`2005.00687`](https://arxiv.org/abs/2005.00687)** — Hu et al., *Open Graph Benchmark* (NeurIPS 2020). The benchmark paper.
- **Gilmer et al. MPNN** ([`1704.01212`](https://arxiv.org/abs/1704.01212)) — the unifying framework.
- **OGB Leaderboard** (`ogb.stanford.edu`) — the live leaderboard. Look at the winning models: most are GIN with virtual node, or Graph Transformers.
- **PyTorch Geometric docs** — implementation reference.
- **DGL** — the other major GNN library; has its own taxonomy and a different API.
- **For more on expressivity**: see Ch 07 of this cheatsheet.
- **For spectral intuition**: see Ch 03 of this cheatsheet.
