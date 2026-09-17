---
title: "11 — Graph Foundation Models"
date: 2026-08-12T17:40:00+03:00
draft: false
params:
  math: true
---

# 11 — Graph Foundation Models
*Definitions, taxonomy, the central survey, and the four GFM families — a research-grade reference for the GFM era.*

This is the chapter you've been building toward. Everything in Ch 01-10 is the prerequisite; this chapter is the synthesis. A "Graph Foundation Model" is a graph model pretrained on a large corpus of graphs with a self-supervised objective, finetuned for downstream tasks, and ideally generalises to new graphs and tasks with minimal additional data.

## 11.1 What is a foundation model?

**Definition (Bommasani et al. 2021, Stanford CRFM).** A foundation model is any model that is:
1. Trained on broad data (generally using self-supervision at scale).
2. Adaptable to a wide range of downstream tasks.

The term emerged from NLP (BERT, GPT) and vision (CLIP, DINO). The same idea has now been imported to graphs.

**Key properties of a foundation model:**
- **Pretrained** on a large, diverse corpus.
- **Self-supervised** objectives (no labels at pretraining time).
- **Transferable** to many tasks with minimal finetuning.
- **Scale**: typically 100M-100B parameters, trained on large data.

For graphs, the same idea applies, but with the extra constraints:
- Graphs are structured, not "IID samples."
- The corpus is a set of graphs of different sizes, topologies, and feature distributions.
- A GFM must respect permutation symmetry (Ch 05).

## 11.2 What is a Graph Foundation Model (GFM)?

**Definition (informal, from the 2025 GFM survey [`2505.15116`](https://arxiv.org/abs/2505.15116) and others).** A Graph Foundation Model is a model that:
1. Is **pretrained on a large corpus of graphs** with a self-supervised objective.
2. Can be **finetuned (or prompted) to perform diverse downstream tasks** on graphs of varying sizes and structures.
3. **Generalises across domains** (e.g., molecular, social, citation) and across task types (node classification, link prediction, graph classification, graph generation).

**What is NOT a GFM:**
- A GCN trained on a single graph (Cora) for node classification. (Not pretrained, not transferable.)
- A GT pretrained on a single domain (e.g., OGB-MolPCBA) for one task. (Single domain, single task.)
- A general-purpose LLM (GPT-4) that can answer questions about graphs but was not pretrained on graph data. (Not a graph model.)

**The gray zone.** "How large is large?" "How diverse is diverse?" The community has not converged on precise definitions. Most GFM papers are "soft" — they claim GFM status but the model is in fact a strong pretrained GT/MPNN with multi-domain finetuning.

## 11.3 The 2025 GFM survey

**Paper.** [`2505.15116`](https://arxiv.org/abs/2505.15116) — Wang et al., *Graph Foundation Models: A Comprehensive Survey* (2025). The central reference for this chapter.

**Structure of the survey:**
1. **Background**: GNN recap, the GFM motivation.
2. **Architectures**: GT-based GFMs, MPNN-based GFMs, LLM-based GFMs.
3. **Pretraining objectives**: contrastive, generative, hybrid.
4. **Adaptation**: finetuning, prompting, adapters.
5. **Applications**: molecular, social, recommendation, knowledge graphs.
6. **Open problems**: scale, transfer, evaluation, interpretability.

**Key insight from the survey:** the GFM literature has converged on a few design choices:
- **Backbone**: Graph Transformer (GPS-style) is the most common.
- **Pretraining**: multi-task (contrastive + generative + structural).
- **Adaptation**: full finetuning or low-rank adapters (LoRA-style).

## 11.4 The four GFM families

Based on the 2025 GFM survey and subsequent papers, there are four main families of GFMs.

### 11.4.1 LLM-as-backbone GFMs (linearise the graph)

**Approach.** Convert the graph to a sequence (e.g., SMILES for molecules, or a BFS/DFS token sequence) and apply a pretrained LLM. The "GFM" is the LLM with appropriate tokenisation.

**Examples:**
- **TAPE** (Transformer Encoder for Property Estimation, 2023): pretrained LLM on SMILES, finetuned for molecular properties.
- **LLaGA** ([`2402.08170`](https://arxiv.org/abs/2402.08170), Chen et al. ICML 2024): LLaMA2 with graph "templates" — turns the graph into a structured prompt.
- **G-Retriever** ([`2402.07630`](https://arxiv.org/abs/2402.07630), He et al. ICML 2024): RAG-style — converts the graph to a text representation, retrieves relevant subgraphs, queries an LLM.
- **GraphGPT** ([`2310.13023`](https://arxiv.org/abs/2310.13023), Tang et al. SIGIR 2024): instruction-tuning an LLM with graph data.

**Pros:**
- Leverages the existing massive pretraining of LLMs.
- The LLM's reasoning capabilities transfer to graph tasks (sort of).
- Easy to implement (just tokenise and use HuggingFace).

**Cons:**
- The graph structure is implicit; the LLM has no explicit graph reasoning.
- Tokenisation is lossy (a graph has more structure than a sequence).
- Opaque — hard to interpret what the LLM is doing.

### 11.4.2 GT-backbone GFMs (explicit graph model)

**Approach.** Use a Graph Transformer as the backbone. Pretrain with a multi-task self-supervised objective. Finetune for downstream tasks.

**Examples:**
- **GraphAny** (2024): GT backbone, contrastive + structural pretraining.
- **GFT** ([`2411.06070`](https://arxiv.org/abs/2411.06070), Wang et al. 2024): Graph Foundation Model with **Transferable Tree Vocabulary**. The "tree vocabulary" is a discrete set of local tree patterns; the model learns to recognise and reason over these.
- **GFM-RAG** ([`2502.01113`](https://arxiv.org/abs/2502.01113), Luo et al. 2025): a GT-based GFM with retrieval-augmented generation. Given a query (e.g., "what is the property of this molecule?"), retrieve relevant subgraphs from a knowledge base, and use the GT to answer.
- **GraphPFN** ([`2509.21489`](https://arxiv.org/abs/2509.21489), Eremeev et al. 2025): a prior-data-fitted GFM. Pretrained on synthetic graph distributions, evaluates on real graphs in a TabPFN-style way.
- **RiemannGFM** ([`2502.03251`](https://arxiv.org/abs/2502.03251), Sun et al. 2025): a GFM trained on Riemannian geometry — handles non-Euclidean graph embeddings.
- **GraphProp** ([`2508.04594`](https://arxiv.org/abs/2508.04594), Sun et al. 2025): trains GFMs using graph properties as the pretraining signal.

**Pros:**
- Explicit graph reasoning.
- Strictly more expressive than MPNNs (Ch 08).
- Works for node-, edge-, and graph-level tasks naturally.

**Cons:**
- More expensive than LLM-as-backbone.
- Requires graph-specific pretraining corpus.

### 11.4.3 MPNN-backbone GFMs

**Approach.** Use a strong MPNN (e.g., GIN with virtual node, or GATv2) as the backbone. Pretrain with self-supervision. The MPNN is bounded by 1-WL, so the GFM inherits that limit.

**Examples:**
- **OGB-style GIN** (Hu et al. 2020 "Strategies"): pretrain GIN with attribute masking + context prediction. Finetune on OGB tasks.
- **GCC** ([`2006.09963`](https://arxiv.org/abs/2006.09963), Qiu et al. 2020): ego-network contrastive pretraining of a GIN. The classic GFM-era pretraining recipe.
- **Boosting GFM from Structural Perspective** ([`2407.19941`](https://arxiv.org/abs/2407.19941), Cheng et al. 2024): a GFM that uses structural features (motifs, graphlets) to augment the MPNN, escaping 1-WL.

**Pros:**
- Mature, well-understood architectures.
- Cheaper than GTs.
- Strong on graph classification.

**Cons:**
- 1-WL bounded — cannot distinguish certain graph pairs.
- Over-smoothing limits depth.

### 11.4.4 Equivariant / geometric GFMs

**Approach.** A GFM that respects physical symmetries (rotation, translation, permutation). Used for 3D molecular data.

**Examples:**
- **Equivariance Everywhere All At Once** ([`2506.14291`](https://arxiv.org/abs/2506.14291), Finkelshtein et al. 2025): a recipe for equivariant GFMs.
- **Integrating Single-Cell Foundation Models with GNNs** ([`2504.14361`](https://arxiv.org/abs/2504.14361), Rossner et al. 2025): combines a pretrained single-cell model (Geneformer-style) with a GNN. The GFM has both molecular and graph structure.

**Pros:**
- Respects physics.
- Best for 3D molecular property prediction.

**Cons:**
- Niche — only useful for 3D molecular tasks.

## 11.5 Pretraining objectives (recap and GFM-specific)

The GFM literature uses four main pretraining objectives, often combined.

**1. Contrastive (InfoNCE, BGRL).** Two views of a graph, pull embeddings together. Standard recipe.

**2. Generative (masked reconstruction).** Mask some nodes/edges, reconstruct. The "BERT for graphs" approach (GraphMAE, GraphMAE2).

**3. Structural (predict graph properties).** Predict the degree of a masked node, the local clustering coefficient, the local motif counts. This forces the model to learn structural information beyond 1-WL.

**4. Multi-task (combination).** Most GFMs use a weighted sum:
$$\mathcal{L} = \alpha \mathcal{L}_{\text{contrastive}} + \beta \mathcal{L}_{\text{reconstruction}} + \gamma \mathcal{L}_{\text{structural}} + \delta \mathcal{L}_{\text{domain-specific}}.$$

The weights $\alpha, \beta, \gamma, \delta$ are hyperparameters; the literature has not converged on a "right" choice.

> **Worked example.** The GFT model ([`2411.06070`](https://arxiv.org/abs/2411.06070)) pretraining loss:
> $$\mathcal{L} = \mathcal{L}_{\text{tree reconstruction}} + \mathcal{L}_{\text{tree contrastive}} + \mathcal{L}_{\text{node prediction}}.$$
> The "tree reconstruction" is to reconstruct a spanning tree of the graph; the "tree contrastive" is to distinguish the graph's tree from random trees; the "node prediction" is to predict the node's role in the tree.

## 11.6 Adaptation: finetuning, prompting, adapters

After pretraining, a GFM is adapted to a downstream task. The options:

**1. Full finetuning.** Continue training the entire model on the downstream task with a small learning rate. Most common, most expensive.

**2. Linear probe.** Freeze the backbone, train a linear head on the embeddings. Cheapest, but may underfit.

**3. LoRA / adapters.** Add a small number of trainable parameters to the backbone (e.g., low-rank matrices in attention). The backbone is frozen. This is the standard "parameter-efficient finetuning" used in NLP.

**4. Prompt tuning.** Add a learnable "prompt" to the input. The GFM is frozen. Common in vision (CoOp, MaPLe).

**5. In-context learning.** No parameter update. The GFM is given a few examples in the input and asked to perform the task. This is the GPT-3 style. In graphs, this is very early-stage research.

**For GFMs.** Most current GFM papers do full finetuning. LoRA and prompt tuning are emerging. In-context learning on graphs is a research frontier.

> **Worked example.** GFM-RAG ([`2502.01113`](https://arxiv.org/abs/2502.01113)) combines (1) graph-level finetuning of the GT backbone, (2) a retrieval-augmented inference step (the model retrieves relevant subgraphs from a knowledge base). The retrieval step is "in-context" — the retrieved subgraphs are appended to the input.

## 11.7 Benchmarks for GFM evaluation

There is no "GLUE for graphs" yet. The closest analogues:

**1. OGB (Open Graph Benchmark).** A collection of graph tasks (OGB-ArXiv, OGB-Products, OGB-MolPCBA, OGB-MolHIV, OGB-PPA, OGB-Code2). The most standard benchmark suite for GNNs. Also has a leaderboard.

**2. OGB-LSC (Large-Scale Challenge).** The large-scale version: PCQM4Mv2 (4M molecules), MAG240M (240M nodes), WikiKG90M (90M entities). The "ImageNet" for graphs.

**3. Long Range Graph Benchmark (LRGB).** A collection of tasks requiring long-range dependencies. Used to evaluate GTs.

**4. GOOD (Graph OOD).** A benchmark for OOD generalisation. Splits graphs by structural properties (size, scaffold, domain).

**5. TUDatasets.** The classic small-scale graph classification benchmark.

**6. Heterophilic benchmarks.** Datasets like Roman-empire, Minesweeper, Tolokers, etc. Used to evaluate heterophily-aware GNNs.

**For GFM-specific evaluation.** The GFM survey recommends:
- Pretrain on a diverse corpus.
- Finetune on a held-out task.
- Report both IID and OOD performance.
- Compare to (a) from-scratch training, (b) supervised pretraining on the same corpus.

> **Worked example.** GFMPerf (a hypothetical GFM):
> - Pretrain on 1M molecules (PubChem).
> - Finetune on BBBP (blood-brain barrier penetration) — 2000 molecules.
> - Test on (a) BBBP test set (IID), (b) a different BBBP-like dataset (OOD).
> - From-scratch GIN: 70% IID, 60% OOD.
> - GFMPerf: 80% IID, 75% OOD.
> - The GFM improvement is 10 points IID, 15 points OOD.

## 11.8 The "all-graph" GFM question

The vision-LLM community has CLIP, which is a single model that handles image and text. The GFM community has not produced an equivalent.

**The challenge.** A "CLIP for graphs" would need:
- A single architecture that handles molecular, social, citation, and knowledge graphs.
- A pretraining corpus that covers all domains.
- A finetuning protocol that works across all tasks.

**The current state.** Most GFMs are domain-specific (e.g., pretrained on molecules only). The few "cross-domain" GFMs (e.g., GraphAny) are still narrow.

**Why this is hard.**
1. **Scale of corpus.** A GFM trained on 1M molecules is impressive; a GFM trained on 1M graphs from each of 10 domains is 10M graphs — much more compute.
2. **Domain heterogeneity.** The structural statistics of a molecular graph and a social network are very different. A GFM that captures both may underfit each.
3. **Task heterogeneity.** Node classification, link prediction, graph classification — different tasks may need different finetuning heads.

**Open question.** Is the "all-graph" GFM possible, or is domain-specific pretraining always better?

> **Worked example.** The GFM survey ([`2505.15116`](https://arxiv.org/abs/2505.15116)) notes that the average GFM is evaluated on 3-5 tasks, almost all in the same domain. Cross-domain evaluation is rare.

## 11.9 Critical analysis: are GFMs working?

A 2024 paper by Frasca et al. ([`2412.17609`](https://arxiv.org/abs/2412.17609), "Towards Foundation Models on Graphs: An Analysis on Cross-Dataset Transfer of Pretrained GNNs") raised a critical question: **do pretrained GNNs actually transfer across datasets?**

**Their findings.**
- Pretraining on a corpus of graphs does help for finetuning on related tasks.
- But the benefit is small (1-3% accuracy) and the pretraining corpus must be carefully chosen.
- For tasks with very different graph structure (e.g., from molecules to social networks), pretraining can hurt (negative transfer).
- The GFM literature's claims of "foundation model" status are sometimes overstated.

**Implication.** As of 2026, the GFM field is in its "ImageNet moment" — the architectures and pretraining objectives are maturing, but the "GPT moment" (where a single pretrained model dominates all downstream tasks) has not arrived.

**For a GFM researcher in 2026.** The honest position:
- GFMs are a promising direction.
- The empirical results are mixed.
- There is room for foundational research on pretraining objectives, transfer mechanisms, and evaluation.
- The "all-graph" GFM is a moonshot, not a near-term reality.

> **Worked example.** From Frasca et al. ([`2412.17609`](https://arxiv.org/abs/2412.17609)): a GIN pretrained on ZINC (molecules) and finetuned on Cora (citation network) — test accuracy is *lower* than a from-scratch GIN. The pretraining hurts. A GIN pretrained on Cora and finetuned on Cora: test accuracy is similar to from-scratch (no benefit). A GIN pretrained on a diverse molecular corpus and finetuned on a small molecule dataset: test accuracy is *higher* than from-scratch. The GFM helps only when the pretraining domain matches the finetuning domain.

## 11.10 GFM taxonomy (consolidated)

A single-table summary of the major GFM papers by family:

| Family | Paper | Architecture | Pretraining | Key innovation |
|---|---|---|---|---|
| LLM-as-backbone | TAPE | LLM on SMILES | MLM on SMILES | Linearise molecular graph |
| LLM-as-backbone | LLaGA ([`2402.08170`](https://arxiv.org/abs/2402.08170)) | LLaMA2 | Instruction tuning | Graph templates |
| LLM-as-backbone | G-Retriever ([`2402.07630`](https://arxiv.org/abs/2402.07630)) | LLM + RAG | RAG over subgraphs | Subgraph retrieval |
| LLM-as-backbone | GraphGPT ([`2310.13023`](https://arxiv.org/abs/2310.13023)) | LLM | Instruction tuning | Graph instruction tuning |
| GT-backbone | GFT ([`2411.06070`](https://arxiv.org/abs/2411.06070)) | GT | Tree reconstruction | Tree vocabulary |
| GT-backbone | GFM-RAG ([`2502.01113`](https://arxiv.org/abs/2502.01113)) | GT | Contrastive + retrieval | RAG-style finetuning |
| GT-backbone | GraphPFN ([`2509.21489`](https://arxiv.org/abs/2509.21489)) | GT | Synthetic-data prior | TabPFN for graphs |
| GT-backbone | GraphProp ([`2508.04594`](https://arxiv.org/abs/2508.04594)) | GT | Property prediction | Graph properties as signal |
| GT-backbone | RiemannGFM ([`2502.03251`](https://arxiv.org/abs/2502.03251)) | GT | Riemannian pretraining | Non-Euclidean embeddings |
| MPNN-backbone | GCC ([`2006.09963`](https://arxiv.org/abs/2006.09963)) | GIN | Ego-network contrastive | Structural pretraining |
| MPNN-backbone | Boosting GFM ([`2407.19941`](https://arxiv.org/abs/2407.19941)) | GIN | Structural augmentation | Motif-aware |
| MPNN-backbone | Tabular→Graph ([`2508.20906`](https://arxiv.org/abs/2508.20906)) | Tabular→GIN | Distill from tabular FM | Cross-domain pretraining |
| Equivariant | Equivariance Everywhere ([`2506.14291`](https://arxiv.org/abs/2506.14291)) | Equivariant GT | E(3)-equivariant pretraining | Recipe for equivariant GFMs |
| Equivariant | Bio+GFM ([`2504.14361`](https://arxiv.org/abs/2504.14361)) | Bio+GT | Bio + graph | Multi-modal pretraining |

## 11.11 Practical GFM recipe (2026 best practice)

Based on the GFM survey and recent papers, the practical recipe for building a GFM in 2026 is:

1. **Choose the domain.** Molecular, social, knowledge graph, etc. Don't try to be "all-graph" — the field is not ready.

2. **Choose the architecture.** GPS-style GT with LapPE + RWSE + SignNet. GIN with virtual node for graph classification tasks.

3. **Choose the pretraining corpus.** As large as possible within the domain. For molecules: PubChem, ZINC, ChEMBL. For social: Reddit, MAG. For knowledge graphs: Wikidata, ConceptNet.

4. **Choose the pretraining objective.** Multi-task: contrastive + reconstruction + structural. Weight the loss terms empirically.

5. **Pretrain for many epochs.** 100-1000 epochs, depending on corpus size. Use large batch sizes (1K-10K graphs per batch).

6. **Evaluate on a held-out set of tasks.** IID (same domain, different graphs) and OOD (different domain, different distribution).

7. **Report carefully.** Linear probe accuracy, finetune accuracy, few-shot accuracy, OOD accuracy. With error bars over multiple runs.

8. **Compare to from-scratch.** The GFM should beat from-scratch training on the same downstream task, especially in the few-shot regime.

**If your GFM does not beat from-scratch by a meaningful margin (5%+ on linear probe, 3%+ on finetune), it is not really a GFM.** This is a useful rule of thumb.

## Exercises

1. **Reproduce GCC.** Implement GCC (Qiu et al. 2020) on a corpus of 10K ego-networks. Pretrain a 2-layer GIN. Finetune on OGB-ArXiv. Compare to a from-scratch GIN.

2. **Pretrain a GraphMAE on ZINC.** Use the official ZINC subset (12K molecules). Pretrain a 5-layer GIN with GraphMAE for 100 epochs. Finetune on the supervised ZINC task. Compare to from-scratch.

3. **Compare GFM approaches.** Implement three GFMs: (a) LLM-as-backbone (G-Retriever style), (b) GT-backbone (GraphMAE on a GT), (c) MPNN-backbone (GraphMAE on GIN). Evaluate on the same downstream task. Compare.

4. **Linear probe vs finetune.** Pretrain a GIN with any method. Report (a) linear probe accuracy, (b) finetune accuracy on the downstream task. The gap measures the "finetunability" of the pretrained features.

5. **Cross-domain transfer.** Pretrain a GIN on a corpus of molecules. Finetune on a citation network. Compare to a from-scratch GIN on the citation network. The transfer should hurt, per Frasca et al.

6. **Read the GFM survey.** Read [`2505.15116`](https://arxiv.org/abs/2505.15116) (Wang et al. 2025). Identify the proposed taxonomy, the key open questions, and the evaluation protocol.

7. **Read Frasca et al. 2024.** Read [`2412.17609`](https://arxiv.org/abs/2412.17609) (Frasca et al. 2024). Identify the key empirical findings and the implications for GFM design.

8. **Read GFT.** Read [`2411.06070`](https://arxiv.org/abs/2411.06070) (Wang et al. 2024, GFT). Identify the "tree vocabulary" and how it is used in pretraining.

9. **Read GFM-RAG.** Read [`2502.01113`](https://arxiv.org/abs/2502.01113) (Luo et al. 2025, GFM-RAG). Identify the retrieval step and how it integrates with the GT.

10. **Read GraphPFN.** Read [`2509.21489`](https://arxiv.org/abs/2509.21489) (Eremeev et al. 2025, GraphPFN). Identify how the prior-data-fitted approach works for graphs.

## Further Reading

- **[`2505.15116`](https://arxiv.org/abs/2505.15116)** — Wang et al., *Graph Foundation Models: A Comprehensive Survey* (2025). The central reference.
- **Bommasani et al., *On the Opportunities and Risks of Foundation Models*** (Stanford CRFM 2021). The original foundation-model report.
- **Brown et al., *Language Models are Few-Shot Learners* (GPT-3)** (NeurIPS 2020). The paper that popularised "foundation model."
- **[`2412.17609`](https://arxiv.org/abs/2412.17609)** — Frasca et al., *Towards Foundation Models on Graphs: Cross-Dataset Transfer of Pretrained GNNs* (2024). The critical analysis.
- **[`2402.08170`](https://arxiv.org/abs/2402.08170)** — Chen et al., *LLaGA: Large Language and Graph Assistant* (ICML 2024). The LLM-on-graph recipe.
- **[`2402.07630`](https://arxiv.org/abs/2402.07630)** — He et al., *G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding* (ICML 2024). The RAG-for-graphs recipe.
- **[`2310.13023`](https://arxiv.org/abs/2310.13023)** — Tang et al., *GraphGPT: Graph Instruction Tuning for Large Language Models* (SIGIR 2024). The instruction-tuning recipe.
- **[`2411.06070`](https://arxiv.org/abs/2411.06070)** — Wang et al., *GFT: Graph Foundation Model with Transferable Tree Vocabulary* (2024).
- **[`2502.01113`](https://arxiv.org/abs/2502.01113)** — Luo et al., *GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation* (2025).
- **[`2509.21489`](https://arxiv.org/abs/2509.21489)** — Eremeev et al., *GraphPFN: A Prior-Data Fitted Graph Foundation Model* (2025).
- **[`2407.19941`](https://arxiv.org/abs/2407.19941)** — Cheng et al., *Boosting Graph Foundation Model from Structural Perspective* (2024).
- **[`2502.03251`](https://arxiv.org/abs/2502.03251)** — Sun et al., *RiemannGFM: Learning a Graph Foundation Model from Riemannian Geometry* (2025).
- **[`2506.14291`](https://arxiv.org/abs/2506.14291)** — Finkelshtein et al., *Equivariance Everywhere All At Once: A Recipe for Graph Foundation Models* (2025).
- **[`2508.04594`](https://arxiv.org/abs/2508.04594)** — Sun et al., *GraphProp: Training the Graph Foundation Models using Graph Properties* (2025).
- **[`2508.20906`](https://arxiv.org/abs/2508.20906)** — Eremeev et al., *Turning Tabular Foundation Models into Graph Foundation Models* (2025).
- **[`2006.09963`](https://arxiv.org/abs/2006.09963)** — Qiu et al., *GCC: Graph Contrastive Coding for GNN Pre-Training* (KDD 2020). The classic GFM-era pretraining.
- **[`1905.13728`](https://arxiv.org/abs/1905.13728)** — Hu et al., *GPT-GNN* (NeurIPS 2020). The original generative pretraining.
- **Bommasani et al., *Picking on the Same Person: A Critical Re-evaluation of Foundation Model Evaluations*** (2024). The critical analysis of how FM evaluation is done.
