---
title: "10 — Open Problems in GNNs"
date: 2026-08-12T17:40:00+03:00
draft: false
params:
  math: true
---

# 10 — Open Problems in GNNs
*Over-smoothing, over-squashing, scalability, distribution shift, robustness, and the open research questions every GFM must confront.*

This chapter is a "state of the field" review. Each section is a research problem that has been recognised for 2-5 years and remains (as of 2026) only partially solved. GFMs are, in part, attempts to address these problems at scale.

## 10.1 Over-smoothing

**The phenomenon.** In a deep GNN, after many message-passing layers, all node embeddings converge to a fixed point. The model becomes unable to distinguish vertices. Test accuracy on node classification collapses.

**Formal statement.** Let $H^{(\ell)} = \sigma(\hat A H^{(\ell-1)} W^{(\ell)})$ be a GCN. Under mild conditions on $W$ (e.g., bounded norm), $H^{(\ell)}$ converges as $\ell \to \infty$ to a rank-1 matrix $H^{(\infty)} = \mathbf{1} v^\top$ for some $v$. The "over-smoothing" is the collapse to a single embedding for all nodes.

**Why it happens.** $\hat A$ is a stochastic-like matrix (rows sum to 1). Repeated application of $\hat A$ drives any signal to the top eigenvector, which is the constant vector. The non-linearity $\sigma$ doesn't help; the spectral contraction wins.

**Solutions (in order of popularity):**
1. **Residual connections** (GCNII, ResGCN).
2. **Initial residual** (GCNII): $H^{(\ell)} = \sigma(((1-\alpha)\hat A + \alpha I) ((1-\beta) H^{(\ell-1)} + \beta H^{(0)}) W^{(\ell)})$.
3. **Identity residual**: $(1-\alpha)\hat A + \alpha I$ adds a self-loop weight to the propagation.
4. **APPNP** ([`2106.03058`](https://arxiv.org/abs/2106.03058)): decouple transformation from propagation. Use an MLP for transformation, then $K$ propagation steps with restart.
5. **PairNorm** (Zhao & Akoglu 2020): normalise so that the total pairwise distance is preserved.
6. **DropEdge** (Rong et al. 2020): randomly drop edges during training. The "sparsified" graph has higher spectral gap, slower mixing.
7. **Virtual node** (Gilmer et al. 2017): add a master node connected to all. Effectively 2-layer model with global receptive field.

**Why GFM helps.** A Graph Transformer with full attention doesn't suffer from over-smoothing in the same way, because attention is not a Markov chain — the same query can attend to all keys, not just neighbors. So a GT can be very deep without over-smoothing.

> **Worked example.** A 4-layer GCN on Cora: test accuracy ~81%. A 16-layer GCN without mitigation: ~63% (over-smoothed). A 16-layer GCN with initial + identity residual (GCNII-style): ~84%. The residual is the difference.

## 10.2 Over-squashing

**The phenomenon.** In a deep GNN, long-range information must be compressed into a fixed-size message vector. When the graph has a "bottleneck" (a small cut separating two regions), the message through the bottleneck is information-limited. Far-apart vertices' embeddings become "squashed" together, regardless of how deep the network is.

**Formal statement.** ([`2006.05205`](https://arxiv.org/abs/2006.05205), Alon & Yahav 2021). Let $\partial S$ be a cut separating $S$ from $V \setminus S$ in graph $G$. The number of bits that can flow through the cut per layer is $O(|\partial S| d)$ where $d$ is the hidden dim. If the cut is small, the message is bounded and the network can't propagate long-range information.

**Why it matters.** Graph Transformers solve over-squashing by attending globally — the message is no longer constrained by the cut size. So GTs are the practical answer to over-squashing.

**Other solutions (for MPNN):**
1. **FA layers** (Alon, Yahav 2021): add a fully-connected layer at the end, which "broadcasts" information globally.
2. **Graph Transformers**: replace the MPNN with a transformer.
3. **PairNorm**: reduces the squashing by normalising.
4. **Virtual node**: makes the cut size 1, so no bottleneck.
5. **Tree decompositions**: explicitly handle the treewidth.

**For GFM design.** Over-squashing is the main argument for using Graph Transformers as the GFM backbone. A GFM based on a deep MPNN would have over-squashing issues; a GFM based on a GT does not.

> **Worked example.** A 5-layer GCN on a 100-vertex barbell graph. The two cliques (50 vertices each) are joined by a single edge. A 5-layer GCN has 5-hop receptive field; in a clique, 5 hops = the whole clique. So each clique's vertices can communicate freely. But to communicate from one clique to the other, the message must pass through the single bridge edge. The bridge is a bottleneck. The bridge's message vector has fixed size; the 50 vertices in one clique must compress their information into this one vector. Test accuracy on a cross-clique task: low. With a virtual node, the bottleneck is the same size, but the virtual node can "aggregate" from all vertices and "broadcast" to all — so the cross-clique communication is improved.

## 10.3 Scalability

**The problem.** Most GNN training is on graphs with $n \le 10^6$ vertices. State-of-the-art graphs (e.g., OGB-Papers100M with 111M vertices, 1.6B edges) are 100x larger.

**Computational bottlenecks.**
1. **Neighborhood explosion**: a $K$-layer GNN has receptive field $K$ hops; in a power-law graph, the $K$-hop neighborhood is $O(n)$ for $K \ge 2$.
2. **Attention $O(n^2)$**: full attention in a GT is $O(n^2 d)$ per layer.
3. **Memory**: storing embeddings for all $n$ vertices is $O(nd)$.

**Solutions for MPNN:**
1. **Neighbor sampling** (GraphSAGE): sample a fixed-size neighborhood at each layer. Cost: $O(s^K d)$ where $s$ is the sample size. Independent of $n$.
2. **Cluster sampling** (ClusterGCN): partition the graph into clusters, train on batches of clusters. Cost: $O(\text{cluster size})$.
3. **Subgraph sampling with sharing** (GraphSAINT): sample subgraphs, share computation across epochs.

**Solutions for GT:**
1. **Sparse attention** (BigBird, Longformer): restrict attention to local + a few global tokens. Cost: $O(n \sqrt n d)$ or $O(n d)$.
2. **Linear attention** (Performer, [`2009.14794`](https://arxiv.org/abs/2009.14794)): kernelize attention to $O(n d^2)$. Cost: linear in $n$.
3. **NodeFormer** ([`2306.08385`](https://arxiv.org/abs/2306.08385)): kernelized Gumbel-Softmax attention. Linear in $n$.
4. **SGFormer** (Wu et al. 2023): one global attention layer + many local GNN layers. Linear in $n$.

**For GFM scale.** The state-of-the-art GFMs combine:
- A linear-attention GT backbone.
- A large pretraining corpus (1M+ graphs).
- A small finetuning corpus (1K-10K samples).

> **Worked example.** OGB-Papers100M: 111M vertices. A 3-layer GCN with full-batch training is infeasible (one epoch = ~1 day on a single GPU). With neighbor sampling ($s = 10$), one epoch is ~30 minutes. With SGFormer, ~1 hour per epoch. The GFM pretraining for this scale requires days of GPU time even with efficient sampling.

## 10.4 Distribution shift and OOD generalisation

**The problem.** GNNs trained on a graph $G$ may not generalise to a different graph $G'$. Even within the same graph, the test-time distribution may differ from training (e.g., new node types, new communities).

**Forms of distribution shift on graphs:**
1. **Size shift**: train on 100-vertex graphs, test on 1000-vertex graphs.
2. **Structural shift**: train on homophilic, test on heterophilic.
3. **Feature shift**: train on one feature distribution, test on another.
4. **Temporal shift**: train on past graph, test on future graph (the graph changes).

**Solutions:**
1. **GFM-style pretraining**: pretrain on a diverse corpus, so the model has seen many graph types.
2. **Domain adaptation**: align train and test feature distributions (DANN-style).
3. **Invariant learning**: find features that are invariant to the shift (IRM-style).
4. **Test-time training**: adapt the model at test time.

**Connection to GFM.** The GFM literature is, in part, motivated by OOD generalisation: a model pretrained on many graphs should be more robust to shift than a model trained on a single graph. The empirical evidence is mixed — see [`2412.17609`](https://arxiv.org/abs/2412.17609) (Frasca et al.) for a critical analysis.

> **Worked example.** Train a 2-layer GCN on Cora. Test on the "new" Cora (with 50% of nodes deleted at test time). Test accuracy drops from 81% to ~70% — the model overfits the original graph structure. A GFM pretrained on a corpus of 1000 graphs and finetuned on Cora: test accuracy drops less, because the pretrained features are more general.

## 10.5 Robustness and adversarial attacks

**The problem.** Small perturbations to the graph (added/removed edges, perturbed features) can drastically change a GNN's prediction. The classic attacks are:
- **Nettack** (Zügner, Akbarnejad, Günnemann 2018): targeted attack on node classification.
- **Metattack** (Zügner, Günnemann 2019): a meta-learning attack that finds globally optimal perturbations.
- **Feature attack**: perturb node features.

**Defenses:**
1. **GNNGuard** (Zhang & Zitnik 2020): use attention to downweight suspicious edges.
2. **Robust GCN** (Zhu et al. 2019): Gaussian-based aggregation.
3. **RGCN** (Zhang & Zitnik 2020): robust aggregation via median.
4. **Pre-training**: pretrain on a clean graph, then finetune with adversarial examples.

**Open questions:**
- Is there a fundamental robustness limit for GNNs?
- Can we train GNNs that are certifiably robust (similar to randomised smoothing in vision)?

**For GFM.** A GFM should be robust by design, because the pretraining corpus may contain adversarial graphs. This is an active research area.

> **Worked example.** Cora + Nettack with 5% edge perturbation: 2-layer GCN test accuracy drops from 81% to 50%. A 2-layer GCN with GNNGuard: drops to 70%. The GCN is sensitive to small graph changes.

## 10.6 Heterophily

**The problem.** Many real graphs have heterophilic structure: connected vertices tend to have different labels. Examples: dating networks, fraud detection, molecular property graphs (where connected atoms may have very different properties).

**Why GNNs struggle.** GCN aggregates neighbor features. In heterophilic graphs, the neighbor features are different from the node's own feature. So the aggregation moves the node's embedding *away* from its correct label.

**Solutions:**
1. **Higher-order aggregation**: H2GCN, GPR-GNN — use 2-hop or learnable-weight PageRank propagation.
2. **Signed attention**: FAGCN — learn positive and negative attention weights.
3. **Geometric methods**: Geom-GCN ([`2002.05287`](https://arxiv.org/abs/2002.05287)) — use latent-space geometry.
4. **GNN+MLP hybrid**: use a GNN to find "similar" vertices, ignore the dissimilar ones.
5. **GFM with structural PE**: a GT with LapPE can capture the global structure that drives heterophily.

> **Worked example.** A heterophilic graph: 100 vertices, labels 0/1, all edges connect label 0 to label 1. A 2-layer GCN: each vertex's embedding is the average of its neighbors' embeddings. The label-0 vertex's embedding becomes the average of label-1 vertices' embeddings — the opposite of its label. Test accuracy: 0%. A GPR-GNN with learnable weights: can downweight the neighbors, so the label-0 vertex's embedding is mostly its own. Test accuracy: 95%.

## 10.7 Long-range dependencies

**The problem.** Some tasks require reasoning about vertices that are far apart in the graph. Examples: knowledge graph completion, question answering on knowledge graphs, route planning on a road network.

**Why GNNs struggle.** A $K$-layer MPNN has $K$-hop receptive field. Far-apart vertices may need $K = $ graph-diameter hops. In a path graph, the diameter is $n$, so $K = n$ for full receptive field. With $K = n$ layers, the MPNN over-smooths.

**Solutions:**
1. **Graph Transformers**: full attention → $O(1)$ hops to any vertex.
2. **Virtual node**: one hop to all vertices.
3. **Memory networks**: external memory of past states.
4. **Lifting to higher-order**: $k$-GNNs with $K$ layers can propagate $k$-tuple information.

**For GFM.** Graph Transformers are the de-facto answer to long-range dependencies in GFMs.

> **Worked example.** A 10-vertex path $P_{10}$. A 1-layer GCN: each vertex sees only its 2 neighbors. To predict a property of vertex 1 from vertex 10, the GCN needs 5 layers (diameter = 5). A 5-layer GCN: works but at the edge of over-smoothing. A 1-layer Graph Transformer with full attention: vertex 1 directly attends to vertex 10. Much better.

## 10.8 Benchmark and evaluation issues

**The problem.** The community uses different evaluation protocols, splits, and metrics. This makes it hard to compare results across papers.

**Standard practice (OGB):**
- Fixed train/val/test split.
- Multiple runs (typically 10).
- Report mean and std of test metric.
- Use the official evaluator (no hand-rolled evaluation).

**Issues:**
- **"Public" splits on Cora**: many papers report on Cora using different splits, making numbers incomparable.
- **"Data leakage" in molecular benchmarks**: TUDatasets has duplicate molecules in train/test. OGB-MolPCBA fixed this.
- **Adversarial splits**: the "Geometric Shape" benchmark uses structurally different test graphs. This is the only true OOD benchmark for graph classification.

**For GFM evaluation.** The community needs:
- A standard pretraining corpus.
- A standard finetuning protocol.
- A diverse set of downstream tasks.
- OOD evaluation in addition to IID.

> **Worked example.** TUDatasets MUTAG: 188 molecules. A standard split: 80/10/10. A model that memorises the dataset can get 95% accuracy. The same model with 5-fold cross-validation: ~85%. The same model with "scaffold split" (train on one chemical scaffold, test on another): ~70%. The same model with OOD evaluation: ~50%. The number varies dramatically with the evaluation protocol.

## 10.9 Foundation-model-specific challenges

**Challenge 1: tokenisation.** How do you "tokenise" a graph into a sequence for an LLM? Options: SMILES (for molecules), DFS traversal, BFS tokens. Each has trade-offs.

**Challenge 2: cross-domain pretraining.** A GFM should transfer across domains. But molecules, social networks, and citation networks have very different structures. Pretraining on all may not help any.

**Challenge 3: scale of pretraining corpus.** NLP has essentially unlimited text. The "graph corpus" is much smaller — there are only ~10⁹ known molecular structures, ~10⁶ known proteins, etc. Pretraining corpus is limited.

**Challenge 4: evaluation.** There is no "GLUE" or "MMLU" for graphs — a single benchmark that tests many tasks. OGB is the closest, but it's a collection of separate benchmarks.

**Challenge 5: interpretability.** A GFM should be interpretable. But a 100M-parameter GFM is hard to interpret. The graph version of "mechanistic interpretability" is in its infancy.

> **Worked example.** LLM-on-graph: a transformer pretrained on text and then finetuned on SMILES strings. The "graph" is the molecular structure; the "tokens" are the SMILES characters. This works surprisingly well — the LLM can predict molecular properties from SMILES. But the model has no explicit graph structure; the graph is implicit in the SMILES syntax.

## 10.10 The 2026 open research agenda

Based on recent GFM papers (e.g., the 2025 GFM survey [`2505.15116`](https://arxiv.org/abs/2505.15116)), the open questions in 2026 are:

1. **What is the right pretraining objective for graphs?** Contrastive, generative, hybrid? (See Ch 09.)
2. **How to handle cross-domain pretraining?** Pretrain on molecules and test on social networks? (Currently infeasible.)
3. **How to scale pretraining to billions of graphs?** Memory-efficient training, model parallelism, etc.
4. **How to evaluate GFMs?** Need a GLUE-like benchmark for graphs.
5. **How to handle out-of-distribution graphs?** Pretraining + test-time training?
6. **What is the right architecture?** GTs dominate, but MPNN-based GFMs are competitive.
7. **How to interpret GFMs?** Mechanistic interpretability for graphs.
8. **What are the limits of the GFM approach?** Can a single GFM handle all graph tasks, or is task-specific finetuning always needed?

These are the questions a GFM researcher in 2026 should be working on.

## Exercises

1. **Over-smoothing measurement.** Train a 64-layer GCN on Cora. Plot the test accuracy vs. layer count. Verify the over-smoothing phenomenon (accuracy drops after ~5 layers). Now add initial + identity residual (GCNII-style) and re-run. Verify the residual prevents over-smoothing.

2. **Over-squashing on a bottleneck graph.** Generate a 100-vertex barbell graph. Train a 5-layer GCN to classify the two cliques. Test accuracy: how many of the "correct" cliques are predicted correctly? Compare to a 1-layer GCN (no cross-clique info) and a 5-layer GAT with full attention. The GT should solve the bottleneck; the MPNN should not.

3. **Neighbor sampling scaling.** Implement GraphSAGE with neighbor sampling on OGB-Products (2.4M nodes). Compare training time to full-batch. The neighbor-sampled version is ~10x faster per epoch with similar accuracy.

4. **Heterophilic graph.** Take the "Roman-empire" dataset from the heterophilic benchmark (a citation-like graph with heterophilic labels). Train a 2-layer GCN: ~50% accuracy. Train a 2-layer H2GCN: ~75%. Train a 2-layer GPR-GNN: ~80%. Compare.

5. **Adversarial robustness.** Take Cora, apply Nettack with 5% edge perturbation. Test a 2-layer GCN: ~50% accuracy. Apply GNNGuard defense: ~70% accuracy.

6. **OOD evaluation.** Pretrain a GIN on a corpus of 1000 molecules with GraphMAE. Finetune on a target task with 100 molecules. Compare to a from-scratch GIN. The pretrained model should generalise better, especially with limited data.

7. **Scaffold split.** Take the ZINC dataset, apply a scaffold split (train on one chemical scaffold, test on another). Train a GIN from scratch. Train a GIN pretrained on PubChem (10M molecules). Compare test MAE.

8. **Read Alon & Yahav 2021.** Read [`2006.05205`](https://arxiv.org/abs/2006.05205) (On the Bottleneck of GNNs). Identify the formal definition of over-squashing. Identify the proposed solution (FA layers).

9. **Read the over-squashing survey.** Read [`2308.15568`](https://arxiv.org/abs/2308.15568) (Singh, 2023). Identify the seven major mitigation strategies.

10. **Read the GFM survey.** Read [`2505.15116`](https://arxiv.org/abs/2505.15116) (Wang et al. 2025). Identify the section on open problems and the proposed research directions.

## Further Reading

- **Li, Han, Wu, Cui, *A Comprehensive Survey on Graph Neural Networks*** ([`1901.00596`](https://arxiv.org/abs/1901.00596)). The general GNN survey.
- **Sato, *A Survey on The Expressive Power of GNNs*** ([`2003.04078`](https://arxiv.org/abs/2003.04078)). The expressivity survey.
- **[`2006.05205`](https://arxiv.org/abs/2006.05205)** — Alon & Yahav, *On the Bottleneck of GNNs and its Practical Implications* (ICLR 2021). The over-squashing paper.
- **[`2308.15568`](https://arxiv.org/abs/2308.15568)** — Singh, *Over-Squashing in GNNs: A Comprehensive Survey* (2023). The over-squashing survey.
- **Chen et al., *GCNII: Simple and Deep Graph Convolutional Networks*** (ICML 2020). The over-smoothing solution.
- **Klicpera, Bojchevski, Günnemann, *Predict then Propagate* (APPNP)** ([`2106.03058`](https://arxiv.org/abs/2106.03058), ICLR 2019). The over-smoothing solution via personalised PageRank.
- **Rong et al., *DropEdge*** (ICLR 2020). The over-smoothing solution via edge dropout.
- **Zhao & Akoglu, *PairNorm*** (ICML 2020). The over-smoothing solution via normalisation.
- **Zügner, Akbarnejad, Günnemann, *Adversarial Attacks on Neural Networks for Graph Data* (Nettack)** (KDD 2018).
- **Zügner, Günnemann, *Adversarial Attacks on Graph Neural Networks via Meta Learning* (Metattack)** (ICLR 2019).
- **Pei et al., *Geom-GCN*** ([`2002.05287`](https://arxiv.org/abs/2002.05287), ICLR 2020). The geometric GNN for heterophily.
- **Zhu et al., *H2GCN*** (NeurIPS 2020). The heterophily MPNN.
- **Chien et al., *GPR-GNN*** (NeurIPS 2021). The generalised PageRank GNN.
- **Frasca et al., *Towards Foundation Models on Graphs*** ([`2412.17609`](https://arxiv.org/abs/2412.17609), 2024). The critical analysis of GFM transfer.
- **[`2505.15116`](https://arxiv.org/abs/2505.15116)** — Wang et al., *Graph Foundation Models: A Comprehensive Survey* (2025). The GFM open-problems section.
- **For benchmarking**: see the OGB paper ([`2005.00687`](https://arxiv.org/abs/2005.00687)) and the OGB-LSC paper ([`2103.09430`](https://arxiv.org/abs/2103.09430)).
- **For distribution shift**: see the "Geometric Shapes" benchmark by Bourke et al., and the "GOOD" benchmark by Gui et al. (2022).
