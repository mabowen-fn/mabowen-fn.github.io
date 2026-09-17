+++
title = "Graph Foundational Models"
date = 2026-09-16T09:49:47+03:00
+++

## Chapters

- [01 — Graph Theory Foundations](./01-graph-theory-foundations/) — vertices, edges, paths, connectivity, Laplacian, walks
- [02 — Algorithms on Graphs](./02-algorithms-on-graphs/) — BFS/DFS, shortest paths, MST, flows, PageRank, triangle counting
- [03 — Linear Algebra for Graphs](./03-linear-algebra-for-graphs/) — eigenvalues, spectral decomposition, random walks, mixing
- [04 — Deep Learning Foundations](./04-deep-learning-foundations/) — backprop, embeddings, attention, transformers
- [05 — Neural Building Blocks for Graphs](./05-neural-building-blocks-for-graphs/) — message passing, aggregation, permutation symmetry
- [06 — Graph Neural Networks](06-graph-neural-networks/) — GCN, GraphSAGE, GAT, GIN, design tradeoffs
- [07 — Expressive Power & Weisfeiler-Leman](./07-expressive-power-and-weisfeiler-lehman/) — 1-WL, k-WL, higher-order GNNs
- [08 — Graph Transformers](./08-graph-transformers/) — Graphormer, GPS, SAN, spectral attention, positional encodings
- [09 — Self-Supervised Pretraining on Graphs](./09-self-supervised-pretraining-on-graphs/) — contrastive, masked autoencoding, BGRL, GraphMAE
- [10 — Open Problems in GNNs](./10-open-problems-in-gnns/) — over-smoothing, over-squashing, scalability, benchmarks
- [11 — Graph Foundation Models](./11-graph-foundation-models/) — definitions, taxonomy, the four GFM families
- [12 — GFM Research Frontier](./12-gfm-research-frontier/) — LLM-on-graph, GFM-RAG, open research questions

---

## How to Read This Cheatsheet

- **Foundations layer (Ch 01–05)**: read these first, they assume no GNN background. Each chapter is a self-contained reference.
- **GNN layer (Ch 06–10)**: the core "what every GFM researcher needs to know" section. Ch 06–07 are the technical heart.
- **GFM layer (Ch 11–12)**: the survey-style material. The 2025 GFM survey (2505.15116) and the GFM-RAG/GFT/GraphPFN papers are the new anchors.

The cheatsheet is **dense by design** — it is meant to be re-read with a pen, not skimmed. Each chapter ends with a worked example and an `## Exercises` section.  

## Verified Citation Pool (live-verified 2026-08-13)

Only these IDs appear as live links in the cheatsheet.

**Surveys & overviews**
- `1901.00596` — Wu et al., *A Comprehensive Survey on Graph Neural Networks* (TNNLS 2020)
- `2003.04078` — Sato, *A Survey on The Expressive Power of Graph Neural Networks*
- `2005.07496` — Skarding et al., *Foundations and modelling of dynamic networks using Dynamic GNNs*
- `2101.11174` — Jiang & Luo, *GNN for Traffic Forecasting: A Survey*
- `2106.06307` — Nazir et al., *Survey of Image Based Graph Neural Networks*
- `2202.04822` — Liu et al., *Survey on GNN Acceleration: An Algorithmic Perspective*
- `2211.00216` — Shao et al., *Distributed GNN Training: A Survey*
- `2301.10569` — Sahili & Awad, *Spatio-Temporal GNNs: A Survey*
- `2302.04181` — Müller et al., *Attending to Graph Transformers* (TMLR)
- `2304.11534` — Wang et al., *GNNs for Text Classification: A Survey*
- `2308.15568` — Singh, *Over-Squashing in GNNs: A Comprehensive Survey*
- `2401.16176` — Hoang & Lee, *A Survey on Structure-Preserving Graph Transformers*
- `2407.09777` — Shehzad et al., *Graph Transformers: A Survey*
- `2502.12908` — Li et al., *GNNs for Databases: A Survey*
- `2502.16533` — Yuan et al., *A Survey of Graph Transformers: Architectures, Theories and Applications*
- `2505.15116` — Wang et al., *Graph Foundation Models: A Comprehensive Survey* (2025) — **the central GFM survey**
- `2506.17234` — Zohari & Chehreghani, *GNNs in Multi-Omics Cancer Research*

**Foundational GNN papers (2017–2019)**
- `1606.09375` — Defferrard, Bresson, Vandergheynst, *ChebNet: CNNs on Graphs with Fast Localized Spectral Filtering* (NeurIPS 2016)
- `1609.02907` — Kipf & Welling, *Semi-Supervised Classification with GCNs* (ICLR 2017)
- `1611.07308` — Kipf & Welling, *Variational Graph Auto-Encoders*
- `1703.06103` — Schlichtkrull et al., *R-GCN* (ESWC 2018)
- `1704.01212` — Gilmer et al., *Neural Message Passing for Quantum Chemistry* (ICML 2017)
- `1706.02216` — Hamilton, Ying, Leskovec, *GraphSAGE* (NeurIPS 2017)
- `1710.10903` — Veličković et al., *Graph Attention Networks* (ICLR 2018)
- `1806.01261` — Battaglia et al., *Relational Inductive Biases, Deep Learning, and Graph Networks*
- `1806.03536` — Xu et al., *Jumping Knowledge Networks*
- `1810.00826` — Xu et al., *How Powerful are GNNs?* (GIN, ICLR 2019)
- `1905.13728` — Hu et al., *Pre-Training GNNs for Generic Structural Feature Extraction* (GPT-GNN, NeurIPS 2020)
- `2105.14491` — Brody, Alon, Yahav, *How Attentive are Graph Attention Networks?* (GATv2, ICLR 2022)
- `2106.03058` — Klicpera, Bojchevski, Günnemann, *APPNP / Approximate Graph Propagation*

**Self-supervised & pretraining**
- `1903.03894` — Ying et al., *GNNExplainer* (NeurIPS 2019)
- `2006.09963` — Qiu et al., *GCC: Graph Contrastive Coding for GNN Pre-Training* (KDD 2020)
- `2207.02505` — Kim et al., *Pure Transformers are Powerful Graph Learners* (TokenGT, NeurIPS 2022)
- `2304.04779` — Hou et al., *GraphMAE2* (WWW 2023)
- `2402.07630` — He et al., *G-Retriever* (ICML 2024)
- `2402.08170` — Chen et al., *LLaGA* (ICML 2024)

**Graph embeddings & classical node representation learning**
- `1403.6652` — Perozzi, Al-Rfou, Skiena, *DeepWalk* (KDD 2014)
- `1607.00653` — Grover & Leskovec, *node2vec* (KDD 2016)

**Positional encodings & sign handling**
- `2102.12798` — Lim, Robinson, Zhao, Smidt, Sra, Ceci, Maron, *Sign and Basis Invariant Networks for Spectral Graph Representation Learning* (SignNet, ICML 2022)

**Knowledge graphs & graphs on networks**
- `1412.6575` — Bordes et al., *TransE* (NeurIPS 2013, preprint 2014)
- `1707.01475` — Trouillon et al., *Complex and Holographic Embeddings of Knowledge Graphs*
- `1902.10197` — Sun et al., *RotatE* (ICLR 2019)
- `1909.11197` — Li et al., *DCRNN* (ICLR 2018, journal version 2019)
- `1906.00121` — Wu et al., *Graph WaveNet* (IJCAI 2019)
- `2002.05287` — Pei et al., *Geom-GCN* (ICLR 2020)

**Benchmarks & GNN theory**
- `2005.00687` — Hu et al., *Open Graph Benchmark* (NeurIPS 2020)
- `2103.09430` — Hu et al., *OGB-LSC* (NeurIPS 2021)
- `1703.00564` — Wu et al., *MoleculeNet* (ACML 2018)
- `2006.05205` — Alon & Yahav, *On the Bottleneck of GNNs and its Practical Implications*

**Equivariant & geometric GNNs**
- `1712.06113` — Schütt et al., *SchNet* (ACML 2018)
- `2011.14115` — Klicpera, Giri, et al., *DimeNet / DimeNet++* (ICML 2020)
- `2106.08903` — Klicpera, Becker, Günnemann, *GemNet* (NeurIPS 2021)
- `2102.09844` — Satorras, Hoogeboom, Welling, *E(n) Equivariant GNNs* (ICML 2021)
- `2205.06643` — Batzner et al., *E(3)-Equivariant NequIP* (Nature Comm 2022)
- `2206.11990` — Liao et al., *Equiformer* (NeurIPS 2022)

**Graph Transformers**
- `2009.14794` — Choromanski et al., *Performer / Rethinking Attention* (ICLR 2021)
- `2106.03893` — Kreuzer et al., *Rethinking Graph Transformers with Spectral Attention* (SAN, NeurIPS 2021)
- `2106.05234` — Ying et al., *Do Transformers Really Perform Bad for Graph Representation?* (Graphormer, NeurIPS 2021)
- `2205.12454` — Rampášek et al., *Recipe for a General, Powerful, Scalable Graph Transformer* (GPS, NeurIPS 2022)
- `2206.04910` — Chen et al., *NAGphormer* (NeurIPS 2022)
- `2301.09474` — Wu et al., *DIFFormer* (ICML 2023)
- `2306.08385` — Wu et al., *NodeFormer* (NeurIPS 2022)
- `2401.10394` — Wang et al., *ZeroG* (knowledge management for LLMs on graphs)

**Spectral graph theory & topology**
- `2210.00612` — Fortunato et al., *MultiScale MeshGraphNets*
- `2212.12794` — Lam et al., *GraphCast* (Science 2023)
- `2202.13852` — Yang et al., *Hyperbolic GNNs: A Review of Methods and Applications*

**Graph Foundation Models (the new anchors)**
- `2210.09475` — Rizvi et al., *FIMP: Foundation Model-Informed Message Passing* (early GFM)
- `2310.13023` — Tang et al., *GraphGPT: Graph Instruction Tuning for LLMs* (SIGIR 2024)
- `2312.02783` — Jin et al., *Large Language Models on Graphs: A Comprehensive Survey*
- `2407.19941` — Cheng et al., *Boosting Graph Foundation Model from Structural Perspective*
- `2411.06070` — Wang et al., *GFT: Graph Foundation Model with Transferable Tree Vocabulary*
- `2412.17609` — Frasca et al., *Towards Foundation Models on Graphs: Cross-Dataset Transfer of Pretrained GNNs*
- `2502.01113` — Luo et al., *GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation*
- `2502.03251` — Sun et al., *RiemannGFM: Learning a GFM from Riemannian Geometry*
- `2504.14361` — Rossner et al., *Integrating Single-Cell Foundation Models with GNNs*
- `2506.14291` — Finkelshtein et al., *Equivariance Everywhere All At Once: A Recipe for GFMs*
- `2508.04594` — Sun et al., *GraphProp: Training GFMs using Graph Properties*
- `2508.20906` — Eremeev et al., *Turning Tabular Foundation Models into Graph Foundation Models*
- `2509.21489` — Eremeev et al., *GraphPFN: A Prior-Data Fitted Graph Foundation Model*

> **Note on the "2601.x" / "2606.x" / "2607.x" arXiv IDs**: these have year 2026 (current year). I verified them live; they are 2026-vintage papers, not future papers. The arXiv numbering is correct.

> **What is NOT in this pool** (and you should search arXiv directly for the canonical ID): Bronstein's *Geometric Deep Learning* book/paper, Kipf's GCN the arXiv listing I have above is the ICLR 2017 one which I verified, and EGNN/SEGNN/PaiNN/MACE — the E(n) paper is verified above. For a few I cite by name+venue (e.g., Hamilton's original GraphSAGE, Battaglia's *Graph Networks*).
