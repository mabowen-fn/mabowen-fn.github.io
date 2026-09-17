---
title: "04 — Deep Learning Foundations"
date: 2026-08-12T17:40:00+03:00
draft: false
params:
  math: true
---

# 04 — Deep Learning Foundations
*Backpropagation, embeddings, attention, and the transformer — the four ideas you must own before any GNN or GFM architecture makes sense.*

This chapter is the "no graphs" interlude: the deep-learning machinery that GNNs and GFMs are built on. If you are comfortable with these, the next chapters are straightforward combinations. If you are not, the rest of the cheatsheet will read as one mystery after another.

## 4.1 The supervised learning pipeline

**Inputs** $X \in \mathbb{R}^{n \times d}$ (a batch of $n$ vectors in $d$ dimensions), **labels** $y \in \{1, \ldots, K\}^n$ (for classification) or $y \in \mathbb{R}^n$ (for regression), **model** $f_\theta : \mathbb{R}^d \to \mathbb{R}^K$ parametrised by $\theta$, **loss** $\ell(f_\theta(x), y)$ measuring the discrepancy.

**Per-example cross-entropy loss** for classification:

$$\ell(f, y) = -\log \frac{\exp(f_y)}{\sum_{k} \exp(f_k)} = -f_y + \log \sum_k \exp(f_k).$$

**Mean squared error** for regression:

$$\ell(f, y) = \frac{1}{2} (f - y)^2.$$

**Training** minimises the empirical risk $L(\theta) = \frac{1}{n} \sum_{i=1}^{n} \ell(f_\theta(x_i), y_i)$ over a training set of size $n$, by **stochastic gradient descent** or a variant (Adam, AdamW, RMSprop).

## 4.2 The multilayer perceptron

An **MLP** is a stack of affine maps interleaved with elementwise non-linearities:

$$h^{(0)} = x, \quad h^{(\ell+1)} = \sigma(W^{(\ell)} h^{(\ell)} + b^{(\ell)}), \quad \ell = 0, 1, \ldots, L-1,$$

where $W^{(\ell)} \in \mathbb{R}^{d_{\ell+1} \times d_\ell}$, $b^{(\ell)} \in \mathbb{R}^{d_{\ell+1}}$, and $\sigma$ is a non-linear activation.

**Common activations:**

| Activation | Formula | Derivative | Use |
|---|---|---|---|
| ReLU | $\max(0, x)$ | $\mathbb{1}[x > 0]$ | Default for hidden layers |
| GELU | $x \cdot \Phi(x)$ | $\Phi(x) + x \phi(x)$ | Transformers, BERT/GPT |
| SiLU / Swish | $x \cdot \sigma(x)$ | $\sigma(x) + x \sigma(x)(1-\sigma(x))$ | Modern alternatives |
| Tanh | $\tanh(x)$ | $1 - \tanh^2(x)$ | Recurrent nets, output |
| Sigmoid | $1/(1+e^{-x})$ | $\sigma(x)(1-\sigma(x))$ | Binary output |
| Softmax | $e^{x_i} / \sum_j e^{x_j}$ | — | Multiclass output |

**Universal approximation Theorem (Cybenko 1989, Hornik 1991).** A 1-hidden-layer MLP with sufficiently many hidden units and a non-polynomial activation can approximate any continuous function on a compact set to arbitrary accuracy. This is the *existence* theorem, not a constructive one — depth buys expressivity, but in practice depth + width + good optimisation are all needed.

> **Worked example.** 2-layer MLP on the 4-cycle (Ch 01). Input: degree of each vertex (so $d=1$). $h^{(1)} = \sigma(W^{(0)} x + b^{(0)})$, where $W^{(0)} \in \mathbb{R}^{4 \times 1}$ maps degree to a 4-dim hidden representation. For the 4-cycle, $x = (2, 2, 2, 2)^\top$, so $W^{(0)} x + b^{(0)} = 2 W^{(0)} + b^{(0)}$ — every vertex gets the same hidden state. **This is the fundamental failure of vanilla MLPs on graphs**: they cannot distinguish vertices that have the same local features. Message passing (Ch 05) is the fix.

## 4.3 Backpropagation

Backprop is just the chain rule applied to a computation graph. The forward pass computes the loss; the backward pass computes gradients of the loss with respect to every parameter, in the same overall complexity as the forward pass (up to a constant).

**Reverse-mode autodiff.** For a computation graph $x \to h_1 \to h_2 \to \cdots \to h_L = \ell$:
- Forward: compute each $h_i$ in order, store intermediates.
- Backward: starting from $\partial \ell / \partial h_L = 1$, propagate $\partial \ell / \partial h_{i-1} = (\partial h_i / \partial h_{i-1})^\top (\partial \ell / \partial h_i)$ in reverse order.

**Why the gradient cost is the same as the forward cost.** Each local Jacobian $\partial h_i / \partial h_{i-1}$ is computed as part of the forward pass (or in a single extra pass). For an MLP layer, the forward is $O(d_\ell d_{\ell+1})$ and the backward is $O(d_\ell d_{\ell+1})$ — same order.

> **Worked example.** A 1-layer network $f(x) = \sigma(w^\top x)$, $\ell = (f - y)^2 / 2$. Forward: $f = \sigma(s)$ where $s = w^\top x$, $\ell = (f - y)^2 / 2$. Backward: $\partial \ell / \partial f = f - y$, $\partial f / \partial s = \sigma'(s) = f (1 - f)$ (for sigmoid), $\partial s / \partial w = x$, so $\partial \ell / \partial w = (f - y) f (1 - f) x$. Update: $w \leftarrow w - \eta \cdot \partial \ell / \partial w$.

## 4.4 Embeddings

An **embedding** is a learned map from a discrete object (word, vertex, edge) to a dense vector. The "lookup table" view: an embedding is a matrix $E \in \mathbb{R}^{|V| \times d}$, where $E_v \in \mathbb{R}^d$ is the embedding of vertex $v$. The forward pass is a row lookup; the backward pass accumulates gradients into the looked-up row.

**Properties of good embeddings:**
- Similar objects have similar embeddings (in cosine or Euclidean).
- Embeddings capture multiple semantic axes (one direction for sentiment, another for tense, etc.).
- Embeddings are stable across contexts (a word's embedding doesn't change much with context — this is a property of static embeddings like word2vec, not of contextual ones like BERT).

**Learning.** Embeddings are learned by **backprop from a downstream task** (e.g., node classification) or by **self-supervision** (e.g., word2vec, node2vec, DeepWalk). The self-supervised case is the bridge to Chapter 09.

**Connection to graphs.** Node embeddings (DeepWalk, node2vec, LINE) are the predecessors of GNN-based embeddings. They learn a $d$-dim vector per node such that nodes that appear close in random walks have similar vectors. The same idea underlies GFM pretraining.

> **Worked example.** word2vec Skip-gram with negative sampling (Mikolov 2013). Given a target word $w$ and a context word $c$, the model scores the pair as $\sigma(v_w^\top u_c)$ where $v_w, u_c \in \mathbb{R}^d$ are the target and context embeddings. The loss for a positive sample is $-\log \sigma(v_w^\top u_c)$ and for $k$ negative samples $-\log \sigma(-v_w^\top u_{c_i})$. This is the template for all embedding learning, including graph embedding.

## 4.5 Attention

**Attention** is a soft lookup: given a **query** $q$, a set of **keys** $\{k_1, \ldots, k_n\}$, and a set of **values** $\{v_1, \ldots, v_n\}$ (often $v_i = k_i$ or a projection of it), the output is a weighted average of values with weights determined by query-key compatibility.

**Scaled dot-product attention** (Vaswani et al. 2017, "Attention is All You Need"):

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V,$$

where $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{n \times d_k}$, $V \in \mathbb{R}^{n \times d_v}$. The scaling by $\sqrt{d_k}$ keeps the softmax inputs at a reasonable scale; without it, dot products grow like $d_k$ and softmax saturates.

**Multi-head attention** runs $h$ parallel attention heads, each with their own projections, then concatenates and projects:

$$\text{MHA}(Q, K, V) = [O_1; O_2; \ldots; O_h] W^O, \quad O_i = \text{Attn}(Q W_i^Q, K W_i^K, V W_i^V).$$

**Self-attention** sets $Q = K = V = X W$ (linear projections of the same input). This is the core of the transformer.

**Properties of attention:**
- **Permutation-equivariant**: $\text{Attn}(P Q, P K, P V) = P \text{Attn}(Q, K, V)$ for any permutation $P$.
- **Quadratic in sequence length**: $O(n^2 d)$ for $n$ tokens.
- **No positional information**: without a positional encoding, "the cat sat on the mat" and "the mat sat on the cat" produce the same output.

> **Worked example.** 4 tokens, $d = 2$. $X = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 0 \end{pmatrix}$, $W^Q = W^K = W^V = I$, $d_k = 2$. Compute $Q K^\top = X X^\top = \begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 0 \\ 1 & 1 & 2 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$. Divide by $\sqrt 2$, softmax each row to get attention weights, then multiply by $V = X$. The result is a 4-vector per row, with the first row's weights: $\text{softmax}((0.707, -1.41, 1.41, -\infty)) \approx (0.13, 0.015, 0.85, 0)$ — token 1 attends most to token 3 (its closest in dot product).

## 4.6 The transformer block

A standard **pre-norm transformer block**:

$$x' = x + \text{MHA}(\text{LayerNorm}(x)), \quad y = x' + \text{FFN}(\text{LayerNorm}(x')),$$

where FFN is a 2-layer MLP with hidden dim $4 d$ and a non-linearity (GELU typically). The residual connections are critical: without them, deep transformers do not train.

**Architectural details worth knowing:**
- **Pre-norm vs post-norm**: pre-norm (normalise before attention/FFN) is more stable for deep stacks; post-norm (normalise after, around the residual) is the original Vaswani design.
- **Dropout**: applied to attention weights, residual stream, and FFN hidden. Modern LLMs sometimes use 0 dropout in the residual stream.
- **GQA / MQA**: Grouped-Query Attention and Multi-Query Attention share $K, V$ heads across $Q$ heads, reducing KV-cache memory.
- **RoPE** (rotary position embedding): rotates $Q, K$ by a position-dependent angle, making attention position-aware.

**Permutation equivariance.** Pure self-attention is permutation-equivariant. The only way it can break the symmetry is via positional encodings, the attention mask, or the causal mask. In language models, the causal mask (lower triangular) makes the transformer a left-to-right processor.

> **Worked example.** A 1-layer transformer on a 4-token sequence, no positional encoding. Take the output of attention: $H = \text{softmax}(X X^\top / \sqrt 2) X$. Then $H' = H + \text{FFN}(\text{LayerNorm}(H))$. Because of permutation equivariance, shuffling the input rows shuffles the output rows in the same way. **This is exactly the same issue as MLPs on graphs (§4.2)**: the transformer cannot distinguish "the cat" from "the mat" without position information.

## 4.7 Optimisation

**SGD:** $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$. The simplest optimiser.

**SGD with momentum:** $\theta_{t+1} = \theta_t - \eta v_t$, where $v_t = \beta v_{t-1} + \nabla L(\theta_t)$. This damps oscillations and accelerates convergence.

**AdamW** (Loshchilov & Hutter 2019, the de-facto LLM optimiser):
- $m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(\theta_t)$ (first moment)
- $v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla L(\theta_t))^2$ (second moment)
- $\hat m_t = m_t / (1 - \beta_1^t)$, $\hat v_t = v_t / (1 - \beta_2^t)$ (bias correction)
- $\theta_{t+1} = \theta_t - \eta \hat m_t / (\sqrt{\hat v_t} + \epsilon)$ (update)
- Then "decoupled" weight decay: $\theta_{t+1} = \theta_{t+1} - \eta \lambda \theta_t$.

The $\epsilon$ (e.g., $10^{-8}$) is a numerical stability constant. The standard hyperparameters are $\beta_1 = 0.9, \beta_2 = 0.999, \lambda = 0.1, \eta \approx 10^{-4}$ for LLM training, $\eta \approx 10^{-3}$ for GNN training.

**Learning rate schedules.** Cosine schedule, linear warmup then decay, or constant after warmup. Most LLMs use linear warmup over the first 1–5% of training followed by cosine decay to ~10% of the peak LR.

**Gradient clipping.** Clip the gradient norm to a maximum value (e.g., 1.0) to prevent exploding gradients. This is essential for training transformers and many GNNs.

## 4.8 Regularisation

**Weight decay** ($\ell_2$ regularisation): add $\lambda \|\theta\|^2$ to the loss. Equivalent in AdamW to the decoupled weight decay term.

**Dropout** (Srivastava et al. 2014): at training time, zero out each activation with probability $p$. At test time, multiply by $(1 - p)$ (or use the "inverted dropout" convention where you scale at training time).

**Batch normalisation** (Ioffe & Szegedy 2015): normalise activations across the batch, with learned scale and shift. Less common in modern transformers.

**Layer normalisation** (Ba, Kiros, Hinton 2016): normalise across features, not the batch. Standard in transformers. $\text{LayerNorm}(x) = (x - \mu) / \sigma \odot \gamma + \beta$, where $\mu, \sigma$ are computed across the feature dimension of $x$.

**Label smoothing**: replace hard targets $y \in \{0, 1\}$ with $(1 - \epsilon) y + \epsilon / K$. Improves generalisation and calibration.

**Early stopping**: stop training when validation loss stops improving. Essential for hyperparameter selection.

## 4.9 The pretraining / finetuning paradigm

The **pretrain-then-finetune** workflow:
1. Pretrain a large model on a large, generic dataset (or many tasks) using self-supervised objectives.
2. Finetune on a smaller, task-specific dataset by continuing training with a task-specific head.

**Why it works:** pretrained models capture general-purpose features (in language: syntax, world knowledge; in graphs: local substructure, common motifs, functional groups). Finetuning adapts these features to the specific task.

**In the GFM world**, the pretraining objective is the key design choice. Options include:
- Contrastive: positive and negative pairs, InfoNCE loss.
- Generative: masked reconstruction, as in BERT/MAE.
- Hybrid: a combination of both.

**Connection to LLMs.** Pretrained LLMs (BERT, GPT, LLaMA) are the model class from which "foundation models" get their name. The transfer to graphs (Ch 11) is a natural extension: pretrain on a large corpus of graphs, then finetune on a downstream graph task.

## 4.10 Loss functions for graph tasks

| Task | Loss | Notes |
|---|---|---|
| Node classification | Cross-entropy | Standard classification |
| Graph classification | Cross-entropy or BCE | Often a sum pooling + MLP head |
| Link prediction | BCE on $z_u^\top z_v$ | Or negative-sampling loss |
| Graph regression | MSE | e.g., molecular property prediction |
| Self-supervised contrastive | InfoNCE | $\ell_{ij} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_k \exp(\text{sim}(z_i, z_k) / \tau)}$ |
| Self-supervised generative | Reconstruction | e.g., masked feature / masked edge |
| Graph generation | MLE or adversarial | Cross-entropy on edges, GAN/VGAE |

The InfoNCE loss is the workhorse of self-supervised learning on graphs. The temperature $\tau$ is critical: smaller $\tau$ makes the loss focus more on the hardest negatives.

> **Worked example.** Contrastive loss with $n=2$ positive pairs and 4 negatives per positive, $\tau = 0.1$. For one positive pair $(z_1, z_2)$ with $\text{sim}(z_1, z_2) = 1.0$ and the four negatives with similarities $-0.5, -0.3, 0.1, 0.2$: $\ell = -\log \frac{e^{10}}{e^{10} + e^{5} + e^{3} + e^{-1} + e^{-2}} = -\log \frac{e^{10}}{e^{10} + e^{5} + e^{3} + e^{-1} + e^{-2}}$. Numerator: $e^{10} \approx 22026$. Denominator dominated by $e^{10}$: $\ell \approx -\log(1 / (1 + e^{-5} + e^{-7} + e^{-11} + e^{-12})) \approx 0$ because the positive is much more similar than any negative. With $\tau = 1$ instead, the difference compresses, and $\ell$ becomes larger.

## Exercises

1. **Backprop by hand.** For a 2-layer MLP $f(x) = W^{(1)} \sigma(W^{(0)} x + b^{(0)}) + b^{(1)}$ with sigmoid $\sigma$ and MSE loss $\ell = (f - y)^2 / 2$, derive $\partial \ell / \partial W^{(0)}$ and $\partial \ell / \partial W^{(1)}$ in terms of the forward-pass intermediates.

2. **Attention is permutation-equivariant.** Show that for any permutation matrix $P$, $\text{softmax}(P X (P X)^\top) (P X) = P \text{softmax}(X X^\top) X$ when $X$ is the input. (Hint: $P$ is orthogonal and $P P^\top = I$, so $P X (P X)^\top = P X X^\top P^\top$, and $P$ commutes with row-wise softmax if you argue per row.)

3. **Layer norm vs. batch norm.** For a 4-row input $X = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \\ 7 & 8 \end{pmatrix}$, compute (a) batch norm with $\gamma = 1, \beta = 0$ across the batch, (b) layer norm with $\gamma = 1, \beta = 0$ per row. Show the difference: batch norm normalises the columns (across rows), layer norm normalises the rows (across columns).

4. **Adam update by hand.** Initialise $\theta = 1$, $m_0 = v_0 = 0$, $\beta_1 = 0.9, \beta_2 = 0.999, \eta = 0.1, \epsilon = 10^{-8}$. Gradient at this step: $\nabla L = 0.5$. Compute $\theta_1$. (After enough steps the bias-corrected moments will be roughly $0.5$.)

5. **InfoNCE temperature.** Implement InfoNCE for a 4-sample batch with positive pair at index 0 and labels indicating which other sample is the positive. Compare losses at $\tau = 0.1, 0.5, 1.0$. At low $\tau$, the loss focuses on the hardest negative; verify by computing the per-negative contribution.

6. **Transformer without PE.** Take a 6-token sequence, embed each as a 4-dim random vector, run 2 transformer layers. Show that the output is permutation-equivariant. (i.e., shuffle the input and the output shuffles in the same way.) Then add a positional encoding (e.g., learnable embeddings) and show the output is now permutation-sensitive.

7. **MLP on the 4-cycle.** A 2-layer MLP takes the degree of each vertex (all 2) and tries to label them 0 or 1 based on some "ground truth" that depends on the position in the cycle. Show that the MLP cannot fit this — the loss is bounded below by a positive number. This is the formal statement of "MLP cannot capture graph structure."

8. **Read the original Transformer paper.** Read Vaswani et al. 2017 (cited extensively in this chapter — search the title). Identify (a) the motivation for multi-head attention, (b) the role of the FFN sublayer, (c) the role of positional encoding, (d) why "scaled" dot-product.

9. **Compare AdamW to SGD.** Train a 2-layer MLP on MNIST with SGD and AdamW, plot the loss curves. When does AdamW converge faster? When is SGD with momentum competitive? (You can use the PyTorch examples in the official tutorials.)

10. **Activation functions.** Compare ReLU, GELU, SiLU on a small regression task. Plot the loss curve and the final test error. (ReLU is fastest but plateaus earlier; GELU and SiLU often converge to slightly better minima.)

## Further Reading

- **Goodfellow, Bengio, Courville, *Deep Learning*** (MIT Press 2016). Ch 6–10 cover the material in this chapter. Free online.
- **Bishop, *Pattern Recognition and Machine Learning*** (Springer 2006). The classical reference; less deep-learning specific but the foundations are gold.
- **Murphy, *Probabilistic Machine Learning: An Introduction*** (MIT Press 2022). Free online; the most modern general ML textbook.
- **Bishop & Bishop, *Deep Learning: Foundations and Concepts*** (Springer 2024). The successor to the 1995 Neural Networks book; very clean treatment.
- **Vaswani et al., *Attention Is All You Need*** (NeurIPS 2017). The transformer paper.
- **Devlin et al., *BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding*** (NAACL 2019). The pretraining paradigm.
- **Brown et al., *Language Models are Few-Shot Learners*** (NeurIPS 2020, GPT-3). The scaling result that launched the foundation-model era.
- **Loshchilov & Hutter, *Decoupled Weight Decay Regularization*** (ICLR 2019). The AdamW paper.
- **He, Zhang, Ren, Sun, *Deep Residual Learning for Image Recognition*** (CVPR 2016). The ResNet paper — the residual connection is the key trick that made very deep networks trainable.
- **Ba, Kiros, Hinton, *Layer Normalization*** (NeurIPS 2016).
- **Ioffe & Szegedy, *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*** (ICML 2015).
- **Srivastava et al., *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*** (JMLR 2014).
- **Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality* (word2vec)** (NeurIPS 2013). The embedding paradigm.
- **For the connection to GNNs/GFMs**: see the GFM survey [`2505.15116`](https://arxiv.org/abs/2505.15116), which puts most of these ideas in the graph context.
