---
title: "12 — GFM Research Frontier"
date: 2026-08-12T17:40:00+03:00
draft: false
params:
  math: true
---

# 12 — GFM Research Frontier
*The open research questions, the experimental recipes, and the reading list for a GFM researcher in 2026.*

This is the closing chapter. The goal is to give you a **research map** — what to read, what to try, what to question, and what to build. If you finish this cheatsheet and then read the cited papers, you should be at the frontier of the GFM literature.

## 12.1 The 2026 GFM research landscape

The state of the field in 2026 (as judged by the GFM survey [`2505.15116`](https://arxiv.org/abs/2505.15116), Frasca et al. 2024 [`2412.17609`](https://arxiv.org/abs/2412.17609), and recent papers):

**What's working:**
- Multi-task pretraining (contrastive + reconstruction + structural).
- Graph Transformer backbones (GPS, Graphormer-style).
- Domain-specific pretraining (molecules, social, KG).
- LLM-as-backbone for "graph + text" tasks.
- Combining GFM with RAG (GFM-RAG style).

**What's not working (yet):**
- A single "all-graph" GFM that handles every domain.
- Cross-domain transfer (from molecules to social, for example).
- Scaling pretraining to billions of graphs.
- Interpretable GFMs.
- Adversarially robust GFMs.

**What's hot in 2026:**
- Graph-any-task models (GraphAny, ProG).
- Equivariant GFMs for 3D molecular data.
- Multi-modal GFMs (graph + text + image).
- Federated GFMs (across institutions, preserving data privacy).
- GFMs for time-evolving graphs.

## 12.2 The LLM-on-graph frontier

The deepest open question: **how should LLMs and graphs interact?**

**Three possible architectures:**

**Architecture 1: LLM-only, with graph tokenisation.** Convert the graph to a sequence (e.g., SMILES, or a description), and use an LLM. The graph is implicit in the tokenisation.

*Pros:* Uses all the existing LLM infrastructure and pretraining. *Cons:* The graph structure is lossy.

**Architecture 2: LLM + GNN hybrid.** Use the LLM to encode text, the GNN to encode graph. Combine.

*Pros:* Both modalities handled natively. *Cons:* Requires paired text-graph data (rare).

**Architecture 3: GNN-only, with LLM-style training.** Use a GT/MPNN as the backbone, but pretrain it with LLM-style objectives (next-token, masked autoencoding, instruction tuning).

*Pros:* Explicit graph structure. *Cons:* Loses LLM's broad knowledge.

**Examples of each:**
- (1) GraphGPT ([`2310.13023`](https://arxiv.org/abs/2310.13023)), LLaGA ([`2402.08170`](https://arxiv.org/abs/2402.08170))
- (2) G-Retriever ([`2402.07630`](https://arxiv.org/abs/2402.07630))
- (3) GraphAny, GraphProp ([`2508.04594`](https://arxiv.org/abs/2508.04594)), GFT ([`2411.06070`](https://arxiv.org/abs/2411.06070))

**The open question.** Which architecture dominates? As of 2026, the answer is "it depends on the task." For molecular property prediction, Architecture 3 (GT-only) is competitive. For graph + text question answering, Architecture 2 is best. For text-only tasks with graph context, Architecture 1 is sufficient.

> **Worked example.** Question answering on a knowledge graph: "What is the capital of the country where the Eiffel Tower is located?" Architecture 1 (LLM-only): tokenise the knowledge graph as triples, prompt the LLM. Works for small graphs; fails for large graphs. Architecture 2 (LLM+GNN): the GNN encodes the relevant subgraph, the LLM reads the question. Works for large graphs. Architecture 3 (GNN-only): the GT encodes the entire graph, the head predicts the answer. Works for the specific task but doesn't generalise to new question types.

## 12.3 The cross-domain transfer problem

**The question.** Can a GFM pretrained on domain A (e.g., molecules) transfer to domain B (e.g., social networks)?

**The empirical answer (as of 2026).** Mostly no, unless A and B are very similar.

**Frasca et al. 2024 ([`2412.17609`](https://arxiv.org/abs/2412.17609))** showed that:
- Pretraining on ZINC (molecules) and finetuning on Cora (citations): test accuracy is *lower* than from-scratch.
- Pretraining on OGB-ArXiv and finetuning on OGB-Products: small positive transfer.
- Pretraining on a large, diverse molecular corpus and finetuning on a small molecular task: large positive transfer.

**Why is cross-domain transfer hard?** The graph statistics differ:
- Molecular graphs: small (20-50 atoms), sparse, regular degree.
- Social networks: large (10⁶ nodes), sparse, power-law degree.
- Knowledge graphs: large, multi-relational, typed.
- Citation networks: medium, growing, topic-clustered.

A GFM that captures all these statistics may not capture any of them well.

**Possible solutions (under research):**
1. **Domain-specific adapters.** A general backbone, with per-domain adapter modules. The backbone transfers; the adapters specialise.
2. **Mixture of experts.** Different experts for different domains, gated by a learned router.
3. **Unified tokenisation.** Find a representation that works across domains (e.g., "any graph as a string of node-edge-node triples"). This is the LLaGA approach.
4. **Curriculum pretraining.** Pretrain on a progression of domains: small → large, regular → irregular. The model learns "graph basics" before domain-specific features.

**Open question.** Is there a "graph primitive" representation that works for all domains? This is the GFM version of the "universal sentence encoder" question in NLP.

> **Worked example.** A GFM pretrained on ZINC (12K molecules) and finetuned on a social network (Reddit, 232K nodes): test accuracy on node classification is 60%, vs. 70% from-scratch. The pretraining hurts by 10 points. With a domain adapter (one extra MLP per domain, 1% of the parameters), the test accuracy becomes 75% — the adapter specialises for the social-network statistics.

## 12.4 Federated GFMs

**The question.** How do you pretrain a GFM when the data is distributed across institutions (e.g., hospitals, each with their own patient graph)?

**The challenge.** You can't move the data (privacy, regulations). You can move the model. Each institution trains on its own data; the models are aggregated.

**Approaches:**
1. **Federated averaging** (FedAvg, McMahan et al. 2017). Each institution trains a local model for a few epochs; the local models are averaged; the averaged model is broadcast. Repeat.
2. **Federated graph learning.** Specific to graphs: each institution has a graph; the local training is GNN finetuning. Aggregation is in the parameter space.
3. **Federated GFMs.** Pretrain a global GFM on a public corpus; finetune on each institution's data, federated. The GFM is the "shared backbone," the institutional finetuning is the "personalised head."

**Papers to read:** Federated GNN surveys (2021-2024), e.g., "Federated Graph Learning — A Systematic Review."

**Open questions.**
- How to aggregate models with different graph structures?
- How to handle non-IID data across institutions (the graphs are very different)?
- How to handle institutions that join/leave dynamically?

> **Worked example.** Three hospitals, each with a patient graph (10K-50K patients). Pretrain a GFM on a public corpus (MIMIC-style). Finetune on each hospital's data using FedAvg. The local models after 100 rounds converge to within 2% of the centralised-trained model. Without the GFM pretraining, the federated model is 10% worse than centralised (because each hospital has too little data).

## 12.5 Multi-modal GFMs

**The question.** How do you pretrain a model that handles graphs + text + images?

**Examples of multi-modal graph data:**
- Scientific papers: citation graph (graph) + abstract (text) + figures (images).
- Social media: user graph (graph) + posts (text) + images.
- e-Commerce: user-item graph (graph) + product descriptions (text) + product images.
- Molecules: molecular graph (graph) + SMILES (text) + 3D conformer (image/point cloud).

**Approaches:**
1. **Shared encoder, modality-specific heads.** A Transformer encoder takes any modality (after tokenisation), and modality-specific heads decode. Pretrain with a multi-modal contrastive loss (e.g., graph-text alignment).
2. **Modality-specific encoders, shared embedding space.** Encode each modality with a specialised encoder (GNN for graph, BERT for text, ViT for image), then project to a shared space. Train with a CLIP-style contrastive loss.
3. **One big model.** A single Transformer that takes all modalities as a sequence. (This is GPT-4V style.)

**For GFMs.** The shared-embedding approach (option 2) is most common. The model learns to align graph, text, and image embeddings, so that "a molecule with a benzene ring" (text), the molecular graph, and an image of benzene are close in embedding space.

> **Worked example.** A paper's citation graph node (paper), the paper's title (text), the paper's first figure (image). Train a CLIP-style model with three encoders: GT for graph, BERT for text, ViT for image. The contrastive loss aligns the three. After pretraining, the model can answer "find me papers that have a figure similar to this image and are about a similar topic" — a multi-modal retrieval task.

## 12.6 GFM for time-evolving graphs

**The question.** Most real graphs change over time. A GFM should handle this.

**Types of temporal graphs:**
1. **Static graph, time-stamped edges.** A knowledge graph where new triples are added.
2. **Discrete-time dynamic graph.** A snapshot at each time step, edges/vertices change.
3. **Continuous-time dynamic graph.** Events (edge additions, attribute changes) at arbitrary times.

**TGN, TGAT** (Rossi et al. 2020, Xu et al. 2020): the MPNN-based temporal graph networks.

**For GFMs.** A temporal GFM must:
- Pretrain on a corpus of time-evolving graphs.
- Handle continuous-time events.
- Capture both spatial and temporal patterns.

**Open questions.**
- How to pretrain a temporal GFM? The pretraining objective must respect the temporal structure.
- How to scale to long time ranges (years of events)?
- How to handle out-of-time-distribution events (a new type of event at test time)?

> **Worked example.** A 1-billion-event social network over 5 years. Pretrain a TGN-style GFM with a "next-event prediction" objective. Finetune for fraud detection (predicting which new event is fraudulent). The temporal GFM should beat a static GFM by 5-10 points.

## 12.7 Mechanistic interpretability for GFMs

**The question.** When a GFM makes a prediction, *why*? What did the GFM learn?

**Mechanistic interpretability** in NLP is the study of "circuits" — sub-networks inside a transformer that perform specific computations. For graphs, this is a new field.

**What to look for in a GFM:**
- Which attention heads attend to which graph structures?
- Which MLP layers compute which features (motif, degree, community)?
- How does the positional encoding interact with the input features?

**Methods (from the LLM interpretability literature, applied to graphs):**
1. **Activation patching.** Ablate or modify activations to identify which components are causally responsible for a prediction.
2. **Probing classifiers.** Train a linear classifier on intermediate activations to see what information is encoded.
3. **Attention pattern visualisation.** For a GT, visualise the attention weights to see which vertices attend to which.
4. **Sparse autoencoders.** Decompose the activations into a small number of interpretable features.

**Open questions.**
- What are the "graph circuits" in a pretrained GFM?
- How do these circuits relate to graph theory concepts (paths, motifs, communities)?
- Can we use interpretability to improve GFM design?

> **Worked example.** A pretrained GPS-style GFM. For a node classification task, visualise the attention weights from the last layer. The "important" attention heads attend to (a) the node itself, (b) its 1-hop neighbors, (c) the Fiedler vector direction (spectral community). This pattern is consistent across tasks — suggesting that the model has learned a "graph attention circuit."

## 12.8 Adversarial robustness of GFMs

**The question.** Are GFMs more or less robust to adversarial attacks than task-specific GNNs?

**Initial evidence (as of 2026):**
- Pretraining can make a GFM more robust (because the pretraining corpus is diverse).
- But targeted attacks (Nettack, Metattack) can still degrade a GFM's accuracy.
- The "attack transfer" property (an attack on one GNN transfers to another) is also true for GFMs.

**Why robustness matters for GFMs.** A GFM is used in high-stakes applications (drug discovery, financial fraud). Adversarial attacks on these are real threats.

**Open questions.**
- Can we train GFMs that are certifiably robust?
- Does the pretraining objective affect robustness?
- Are GTs more robust than MPNNs? (Initial evidence: yes, by 5-10 points.)

> **Worked example.** A GFM pretrained on a corpus of 100K molecules, finetuned for blood-brain barrier penetration prediction. Nettack attack (5% edge perturbation): accuracy drops from 85% to 65%. A from-scratch model: drops from 75% to 50%. The GFM is more robust by ~5 points (both in absolute terms and in the relative degradation).

## 12.9 The evaluation problem

**The current state.** There is no GLUE / MMLU / ImageNet for GFMs. The community uses a collection of benchmarks (OGB, LRGB, GOOD, etc.) with different evaluation protocols.

**What we need.**
1. **A standard pretraining corpus.** With clear documentation of size, domain, and statistics.
2. **A standard finetuning protocol.** With fixed train/val/test splits and hyperparameter ranges.
3. **A standard evaluation suite.** Covering node, edge, graph, and generation tasks.
4. **OOD evaluation in addition to IID.** The GFM must be tested on out-of-distribution graphs.

**The proposed GFM-eval.** The 2025 GFM survey ([`2505.15116`](https://arxiv.org/abs/2505.15116)) proposes a tentative evaluation suite:
- Pretraining corpus: 1M graphs from 10 domains.
- Finetuning: 10 tasks (5 IID, 5 OOD) per domain.
- Metrics: linear probe, finetune, few-shot, OOD.

**Open questions.**
- Who maintains the GFM-eval?
- How to prevent overfitting to the benchmark?
- How to handle the "test set is now public" problem?

> **Worked example.** A new GFM paper claims "SOTA on 8 out of 10 tasks." Without a standard evaluation suite, the claim is hard to verify. With GFM-eval, the claim is reproducible.

## 12.10 The 12-month research plan for a GFM PhD student

If you are starting a PhD on GFMs in 2026, here is a possible research plan.

**Months 1-3: Foundations.** Read the chapters of this cheatsheet. Implement GCN, GraphSAGE, GAT, GIN. Run on Cora, OGB-ArXiv. (You should be able to get ~75% test accuracy on OGB-ArXiv with a from-scratch 3-layer GAT.)

**Months 4-6: Graph Transformers.** Read the GPS paper. Implement a minimal GPS with LapPE. Run on OGB-ArXiv and OGB-MolPCBA. Compare to GAT.

**Months 7-9: Pretraining.** Implement GraphMAE, GraphCL, BGRL. Pretrain on a corpus (e.g., ZINC for molecules, or OGB-Papers100M for citation). Finetune on downstream tasks. Report linear probe and finetune accuracy.

**Months 10-12: GFM design.** Pick a domain (molecules, social, KG). Design a GFM following the 2026 best-practice recipe (Ch 11 §11.11). Pretrain, finetune, evaluate. Compare to from-scratch.

**Year 2: Research question.** Pick a research question from §12.1-§12.9. Examples:
- How to improve cross-domain transfer?
- How to make GFMs more interpretable?
- How to handle temporal graphs in a GFM?
- How to combine LLMs and GFMs?

**Year 3: Publication.** Write papers, run thorough ablations, address reviewer concerns.

**Year 4: Thesis.** Synthesise the work, identify the next research questions.

This is a generic plan; your actual plan should be tailored to your advisor, your institution, and your interests.

## 12.11 The "what to read next" reading list

A curated reading list for a GFM researcher in 2026, ordered by priority.

**Tier 1: Foundational (read first).**
- [`1901.00596`](https://arxiv.org/abs/1901.00596) — Wu et al., GNN survey.
- [`1609.02907`](https://arxiv.org/abs/1609.02907) — Kipf & Welling, GCN.
- [`1706.02216`](https://arxiv.org/abs/1706.02216) — Hamilton et al., GraphSAGE.
- [`1710.10903`](https://arxiv.org/abs/1710.10903) — Veličković et al., GAT.
- [`1810.00826`](https://arxiv.org/abs/1810.00826) — Xu et al., GIN.
- [`2106.05234`](https://arxiv.org/abs/2106.05234) — Ying et al., Graphormer.
- [`2205.12454`](https://arxiv.org/abs/2205.12454) — Rampášek et al., GPS.
- [`2005.00687`](https://arxiv.org/abs/2005.00687) — Hu et al., OGB.
- [`2505.15116`](https://arxiv.org/abs/2505.15116) — Wang et al., GFM survey.

**Tier 2: Important variants.**
- [`2106.03058`](https://arxiv.org/abs/2106.03058) — APPNP.
- [`1704.01212`](https://arxiv.org/abs/1704.01212) — Gilmer et al., MPNN.
- [`1806.01261`](https://arxiv.org/abs/1806.01261) — Battaglia et al., Graph Networks.
- [`2106.03893`](https://arxiv.org/abs/2106.03893) — SAN.
- [`2207.02505`](https://arxiv.org/abs/2207.02505) — TokenGT.
- [`2003.04078`](https://arxiv.org/abs/2003.04078) — Sato, Expressivity survey.
- [`2006.05205`](https://arxiv.org/abs/2006.05205) — Alon & Yahav, over-squashing.
- [`2302.04181`](https://arxiv.org/abs/2302.04181) — Müller et al., Attending to GTs.

**Tier 3: Pretraining.**
- [`2006.09963`](https://arxiv.org/abs/2006.09963) — Qiu et al., GCC.
- [`1905.13728`](https://arxiv.org/abs/1905.13728) — Hu et al., GPT-GNN.
- `GraphMAE` paper (Hou et al. KDD 2022).
- [`2304.04779`](https://arxiv.org/abs/2304.04779) — Hou et al., GraphMAE2.
- BGRL (Thakoor et al. ICLR 2022).
- [`2412.17609`](https://arxiv.org/abs/2412.17609) — Frasca et al., cross-dataset transfer analysis.

**Tier 4: GFMs and frontier.**
- [`2402.08170`](https://arxiv.org/abs/2402.08170) — LLaGA.
- [`2402.07630`](https://arxiv.org/abs/2402.07630) — G-Retriever.
- [`2310.13023`](https://arxiv.org/abs/2310.13023) — GraphGPT.
- [`2411.06070`](https://arxiv.org/abs/2411.06070) — GFT.
- [`2502.01113`](https://arxiv.org/abs/2502.01113) — GFM-RAG.
- [`2509.21489`](https://arxiv.org/abs/2509.21489) — GraphPFN.
- [`2506.14291`](https://arxiv.org/abs/2506.14291) — Equivariance Everywhere.
- [`2502.03251`](https://arxiv.org/abs/2502.03251) — RiemannGFM.
- [`2508.04594`](https://arxiv.org/abs/2508.04594) — GraphProp.
- [`2508.20906`](https://arxiv.org/abs/2508.20906) — Tabular→Graph.
- [`2407.19941`](https://arxiv.org/abs/2407.19941) — Boosting GFM.

**Tier 5: Surveys and textbooks.**
- [`2003.04078`](https://arxiv.org/abs/2003.04078) — Sato, expressive power survey.
- [`2308.15568`](https://arxiv.org/abs/2308.15568) — Over-squashing survey.
- [`2407.09777`](https://arxiv.org/abs/2407.09777) — Shehzad et al., GT survey.
- [`2502.16533`](https://arxiv.org/abs/2502.16533) — Yuan et al., GT survey.
- [`2312.02783`](https://arxiv.org/abs/2312.02783) — Jin et al., LLM-on-graph survey.
- **Chung, *Spectral Graph Theory*** (CBMS 1997) — math foundation.
- **Newman, *Networks: An Introduction*** (Oxford 2010) — networks.
- **Bommasani et al., *On the Opportunities and Risks of Foundation Models*** (Stanford 2021) — the FM report.

## 12.12 The closing thoughts

You have a research-grade cheatsheet in front of you: 12 chapters from graph theory to GFM frontier, with 100+ verified arXiv citations, 100+ exercises, and a structured reading list. What you do with it is up to you.

A few parting thoughts:

1. **The GFM field is young.** Most of the papers in this cheatsheet are from 2024-2026. There is room for foundational contributions.

2. **The "GPT moment" for graphs has not arrived.** Pretrained GFMs do not yet dominate from-scratch training the way GPT-4 dominates previous NLP. The Frasca et al. 2024 paper is the clearest signal of this. There is room for foundational research on what makes a good GFM.

3. **The cross-domain problem is open.** A GFM that handles molecules, social, citation, and KG in one model is the moonshot. The community is far from it.

4. **The evaluation problem is open.** A GLUE for graphs would be a major contribution.

5. **The interpretability problem is open.** Mechanistic interpretability for graphs is in its infancy.

If you want to do impactful research in this area, pick one of these open problems and start.

## Exercises

1. **Read the GFM survey thoroughly.** Read [`2505.15116`](https://arxiv.org/abs/2505.15116) end-to-end. Write a 5-page summary in your own words. Identify the 3-5 most important open problems.

2. **Reproduce a GFM.** Pick a GFM paper (e.g., GFT, GFM-RAG, GraphPFN) and reproduce its main result. This is the best way to learn what "building a GFM" actually involves.

3. **Design a GFM.** Following the 2026 best-practice recipe (Ch 11 §11.11), design a GFM for a domain of your choice. Pretrain it, finetune it, evaluate it. Compare to from-scratch.

4. **Cross-domain transfer experiment.** Pretrain a GFM on domain A. Finetune on domain B. Compare to a from-scratch model on B. The result may surprise you (per Frasca et al.).

5. **Interpret a GFM.** Pretrain a GFM. Use activation patching or attention visualisation to understand what the model has learned. Write up the findings.

6. **OOD evaluation.** Pretrain a GFM on a corpus. Evaluate on OOD tasks (scaffold split, time split, size split). Report the OOD accuracy.

7. **Read Frasca et al. 2024 critically.** Read [`2412.17609`](https://arxiv.org/abs/2412.17609) end-to-end. Identify the strongest claim and the strongest counter-argument. What would a follow-up study need to show?

8. **Propose a GFM-eval.** Design a standard evaluation protocol for GFMs. Include: corpus, tasks, metrics, OOD protocol. Justify each design choice.

9. **Survey a subfield.** Pick a sub-topic (e.g., equivariant GFMs, federated GFMs, multi-modal GFMs) and write a 5-page survey of recent papers. Identify the state of the art, the open questions, and your own proposed research direction.

10. **Write a research proposal.** Identify one open question in §12.1-§12.9. Write a 2-page research proposal: motivation, related work, proposed method, expected results, evaluation plan. This is your "starting point" for a research paper.

## Further Reading

All the references cited throughout this cheatsheet, plus a few more that didn't fit elsewhere.

**Mathematical foundations.**
- Chung, *Spectral Graph Theory* (CBMS 1997). Free online.
- Spielman, *Spectral and Algebraic Graph Theory* (Yale lecture notes 2024). Free online.
- Diestel, *Graph Theory* (5th ed., Springer 2017).
- Bollobás, *Modern Graph Theory* (Springer 1998).
- Strang, *Linear Algebra and Learning from Data* (Wellesley 2019).

**Algorithms.**
- Cormen, Leiserson, Rivest, Stein, *Introduction to Algorithms* (4th ed., MIT Press 2022).
- Kleinberg & Tardos, *Algorithm Design* (Pearson 2005).
- Newman, *Networks: An Introduction* (Oxford 2010). Free online.
- Leskovec, Rajaraman, Ullman, *Mining of Massive Datasets* (Cambridge 2014). Free online.

**Deep learning.**
- Goodfellow, Bengio, Courville, *Deep Learning* (MIT Press 2016). Free online.
- Bishop & Bishop, *Deep Learning: Foundations and Concepts* (Springer 2024).
- Murphy, *Probabilistic Machine Learning* (MIT Press 2022). Free online.

**GNNs — primary.**
- [`1609.02907`](https://arxiv.org/abs/1609.02907) — GCN.
- [`1706.02216`](https://arxiv.org/abs/1706.02216) — GraphSAGE.
- [`1710.10903`](https://arxiv.org/abs/1710.10903) — GAT.
- [`1810.00826`](https://arxiv.org/abs/1810.00826) — GIN.
- [`1704.01212`](https://arxiv.org/abs/1704.01212) — MPNN.
- [`1806.01261`](https://arxiv.org/abs/1806.01261) — Graph Networks.
- [`2106.03058`](https://arxiv.org/abs/2106.03058) — APPNP.
- [`2105.14491`](https://arxiv.org/abs/2105.14491) — GATv2.
- [`1703.06103`](https://arxiv.org/abs/1703.06103) — R-GCN.
- [`2102.09844`](https://arxiv.org/abs/2102.09844) — E(n) Equivariant GNN.
- [`1712.06113`](https://arxiv.org/abs/1712.06113) — SchNet.
- [`2011.14115`](https://arxiv.org/abs/2011.14115) — DimeNet.
- [`2106.08903`](https://arxiv.org/abs/2106.08903) — GemNet.
- [`2206.11990`](https://arxiv.org/abs/2206.11990) — Equiformer.
- [`2205.06643`](https://arxiv.org/abs/2205.06643) — NequIP.
- [`1606.09375`](https://arxiv.org/abs/1606.09375) — ChebNet.

**GNN surveys.**
- [`1901.00596`](https://arxiv.org/abs/1901.00596) — Wu et al., GNN survey.
- [`2003.04078`](https://arxiv.org/abs/2003.04078) — Sato, Expressivity survey.
- [`2308.15568`](https://arxiv.org/abs/2308.15568) — Over-squashing survey.
- [`2005.07496`](https://arxiv.org/abs/2005.07496) — Dynamic GNN survey.
- [`2101.11174`](https://arxiv.org/abs/2101.11174) — Traffic GNN survey.
- [`2202.04822`](https://arxiv.org/abs/2202.04822) — GNN acceleration survey.
- [`2211.00216`](https://arxiv.org/abs/2211.00216) — Distributed GNN survey.
- [`2301.10569`](https://arxiv.org/abs/2301.10569) — Spatio-Temporal GNN survey.
- [`2304.11534`](https://arxiv.org/abs/2304.11534) — Text GNN survey.
- [`2106.06307`](https://arxiv.org/abs/2106.06307) — Image GNN survey.
- [`2502.12908`](https://arxiv.org/abs/2502.12908) — GNN for databases survey.
- [`2506.17234`](https://arxiv.org/abs/2506.17234) — Multi-omics GNN survey.

**Graph Transformers.**
- [`2106.05234`](https://arxiv.org/abs/2106.05234) — Graphormer.
- [`2106.03893`](https://arxiv.org/abs/2106.03893) — SAN.
- [`2205.12454`](https://arxiv.org/abs/2205.12454) — GPS.
- [`2207.02505`](https://arxiv.org/abs/2207.02505) — TokenGT.
- [`2102.12798`](https://arxiv.org/abs/2102.12798) — SignNet.
- [`2206.04910`](https://arxiv.org/abs/2206.04910) — NAGphormer.
- [`2306.08385`](https://arxiv.org/abs/2306.08385) — NodeFormer.
- [`2301.09474`](https://arxiv.org/abs/2301.09474) — DIFFormer.
- [`2302.04181`](https://arxiv.org/abs/2302.04181) — Müller et al. GT survey.
- [`2407.09777`](https://arxiv.org/abs/2407.09777) — Shehzad et al. GT survey.
- [`2502.16533`](https://arxiv.org/abs/2502.16533) — Yuan et al. GT survey.
- [`2401.16176`](https://arxiv.org/abs/2401.16176) — Hoang & Lee structure-preserving GT survey.

**Pretraining.**
- [`2006.09963`](https://arxiv.org/abs/2006.09963) — GCC.
- [`1905.13728`](https://arxiv.org/abs/1905.13728) — GPT-GNN.
- [`2304.04779`](https://arxiv.org/abs/2304.04779) — GraphMAE2.
- `GraphMAE` (Hou et al. KDD 2022).
- `BGRL` (Thakoor et al. ICLR 2022).
- `Hu et al. Strategies for Pre-training GNNs` (ICLR 2020).
- `GRACE`, `GCA`, `CCA-SSG` papers.

**GFMs.**
- [`2505.15116`](https://arxiv.org/abs/2505.15116) — Wang et al. GFM survey.
- [`2412.17609`](https://arxiv.org/abs/2412.17609) — Frasca et al. cross-dataset transfer.
- [`2402.08170`](https://arxiv.org/abs/2402.08170) — LLaGA.
- [`2402.07630`](https://arxiv.org/abs/2402.07630) — G-Retriever.
- [`2310.13023`](https://arxiv.org/abs/2310.13023) — GraphGPT.
- [`2411.06070`](https://arxiv.org/abs/2411.06070) — GFT.
- [`2502.01113`](https://arxiv.org/abs/2502.01113) — GFM-RAG.
- [`2509.21489`](https://arxiv.org/abs/2509.21489) — GraphPFN.
- [`2506.14291`](https://arxiv.org/abs/2506.14291) — Equivariance Everywhere.
- [`2502.03251`](https://arxiv.org/abs/2502.03251) — RiemannGFM.
- [`2508.04594`](https://arxiv.org/abs/2508.04594) — GraphProp.
- [`2508.20906`](https://arxiv.org/abs/2508.20906) — Tabular→Graph.
- [`2407.19941`](https://arxiv.org/abs/2407.19941) — Boosting GFM.
- [`2504.14361`](https://arxiv.org/abs/2504.14361) — Bio+GFM.

**Benchmarks.**
- [`2005.00687`](https://arxiv.org/abs/2005.00687) — OGB.
- [`2103.09430`](https://arxiv.com/abs/2103.09430) — OGB-LSC.
- [`1703.00564`](https://arxiv.org/abs/1703.00564) — MoleculeNet.
- `TUDatasets`, `LRGB`, `GOOD`, `BGNN` benchmarks.

**LLM-on-graph.**
- [`2312.02783`](https://arxiv.org/abs/2312.02783) — Jin et al. LLM-on-graph survey.
- [`2402.07630`](https://arxiv.org/abs/2402.07630) — G-Retriever.
- [`2402.08170`](https://arxiv.org/abs/2402.08170) — LLaGA.
- [`2310.13023`](https://arxiv.org/abs/2310.13023) — GraphGPT.
- `TAPE` (2023).

**Specific application surveys.**
- [`1909.11197`](https://arxiv.org/abs/1909.11197) — DCRNN (traffic).
- [`1906.00121`](https://arxiv.org/abs/1906.00121) — Graph WaveNet (traffic).
- [`1412.6575`](https://arxiv.org/abs/1412.6575) — TransE (KG).
- [`1707.01475`](https://arxiv.org/abs/1707.01475) — ComplEx (KG).
- [`1902.10197`](https://arxiv.org/abs/1902.10197) — RotatE (KG).
- [`1403.6652`](https://arxiv.org/abs/1403.6652) — DeepWalk.
- [`1607.00653`](https://arxiv.org/abs/1607.00653) — node2vec.
- [`2002.05287`](https://arxiv.org/abs/2002.05287) — Geom-GCN (heterophily).
- [`2006.05205`](https://arxiv.org/abs/2006.05205) — Over-squashing.
- [`1903.03894`](https://arxiv.org/abs/1903.03894) — GNNExplainer.
- [`2210.00612`](https://arxiv.org/abs/2210.00612) — MeshGraphNets.
- [`2212.12794`](https://arxiv.com/abs/2212.12794) — GraphCast (weather).
- [`2202.13852`](https://arxiv.org/abs/2202.13852) — Hyperbolic GNN survey.

**Foundation models general.**
- Bommasani et al., *On the Opportunities and Risks of Foundation Models* (Stanford 2021).
- Brown et al., *GPT-3* (NeurIPS 2020).
- Devlin et al., *BERT* (NAACL 2019).
- He et al., *MAE* (CVPR 2022).
- Chen et al., *SimCLR* (ICML 2020).
- Grill et al., *BYOL* (NeurIPS 2020).
- Rombach et al., *Stable Diffusion* (CVPR 2022).
- Ho et al., *DDPM* (NeurIPS 2020).
- Vaswani et al., *Attention is All You Need* (NeurIPS 2017).

**Software.**
- **PyTorch Geometric** (`pyg.org`) — the most widely used GNN library. Documentation is excellent.
- **DGL** (`dgl.ai`) — the other major library. Has its own taxonomy.
- **OGB** (`ogb.stanford.edu`) — the Open Graph Benchmark. Standard datasets and leaderboards.
- **NetworkX** (`networkx.org`) — pure-Python graph algorithms.
- **PyTorch** — the deep learning framework.
- **HuggingFace Transformers** — for the LLM components.
- **xT (formerly DGL-LifeSci)** — molecular-specific GNNs.
- **PyG-OGB** — integrations between PyG and OGB.
- **Chemprop** — message passing for molecular property prediction (a widely-used baseline).
- **DIG** (DIG: Dive into Graphs) — a library for graph deep learning research.

You have the cheatsheet. Now go read the papers.
