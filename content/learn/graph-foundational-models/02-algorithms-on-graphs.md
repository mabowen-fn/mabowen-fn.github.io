---
title: "02 — Algorithms on Graphs"
date: 2026-08-12T17:40:00+03:00
draft: false
params:
  math: true
---

# 02 — Algorithms on Graphs
*The classical algorithms a GFM researcher needs to know: BFS, DFS, shortest paths, MST, max flow, triangle counting, and PageRank. These are the primitives that motivated both graph kernels and modern message-passing GNNs.*

This chapter is intentionally algorithm-focused. Most of the GNN ideas in later chapters (over-smoothing, message passing, propagation) have direct analogues in classical graph algorithms; knowing the classical side first means the ML side feels natural, not magical.

## 2.1 Graph traversal: BFS and DFS

Both visit every vertex reachable from a source. The difference is the **frontier discipline**: BFS uses a FIFO queue and discovers vertices in non-decreasing distance from the source; DFS uses a LIFO stack (or recursion) and goes as deep as possible first.

**BFS pseudocode** (unweighted, returns distance from source $s$):
```
BFS(G, s):
    for u in V: dist[u] = inf, parent[u] = None
    dist[s] = 0
    Q = queue([s])
    while Q not empty:
        u = Q.dequeue()
        for v in Adj(u):
            if dist[v] == inf:
                dist[v] = dist[u] + 1
                parent[v] = u
                Q.enqueue(v)
```

**DFS pseudocode** (recursive):
```
DFS(G, u, visited, parent):
    visited[u] = true
    for v in Adj(u):
        if not visited[v]:
            parent[v] = u
            DFS(G, v, visited, parent)
```

**Complexities:** both are $O(n + m)$ — each vertex and each edge is touched $O(1)$ times. BFS in a *weighted* graph still finds shortest paths in number of edges (unweighted distance). For weighted graphs you need Dijkstra.

**Connection to GNNs.** A $k$-layer GCN's receptive field is exactly the $k$-hop BFS tree from each node. The "over-squashing" problem (Ch 10) is fundamentally the consequence of the BFS tree of a $k$-layer network being exponentially larger than $k$ in a tree, leading to too much information being compressed into a fixed-size hidden vector.

> **Worked example.** A 5-vertex graph: edges $\{(1,2),(1,3),(2,4),(3,4),(4,5)\}$. BFS from 1: order = $1, 2, 3, 4, 5$ (two valid orders because 2 and 3 are at the same level). Distances: $d(1) = 0$, $d(2) = d(3) = 1$, $d(4) = 2$, $d(5) = 3$.

## 2.2 Shortest paths

**Single-source shortest paths** in a graph with non-negative edge weights is solved by **Dijkstra** in $O((n+m) \log n)$ with a binary heap, or $O(n + m \log n)$ with a Fibonacci heap. The output is a **shortest-path tree** rooted at $s$.

```
Dijkstra(G, s, w):
    for u: dist[u] = inf; dist[s] = 0
    PQ = priority queue keyed by dist
    PQ.push(s, 0)
    while PQ not empty:
        u = PQ.pop_min()
        for v in Adj(u):
            if dist[u] + w(u,v) < dist[v]:
                dist[v] = dist[u] + w(u,v)
                parent[v] = u
                PQ.push_or_decrease(v, dist[v])
```

**All-pairs shortest paths** is **Floyd–Warshall** in $O(n^3)$ time, $O(n^2)$ memory, or repeated Dijkstra in $O(n(m + n \log n))$. For sparse graphs, repeated Dijkstra is the practical choice.

**Negative weights and negative cycles.** Dijkstra fails on negative weights. **Bellman–Ford** handles them in $O(nm)$ and detects negative cycles. The reason: Dijkstra assumes once a vertex is popped from the queue, its distance is final — only true if no future relaxation can improve it, which requires non-negative weights.

**Connection to GNNs.** Path information is central to many GFM tasks (path reasoning, link prediction). **Personalised PageRank** (a relaxation of random-walk-with-restart shortest path) is used in GNN propagation rules like **APPNP** ([`2106.03058`](https://arxiv.org/abs/2106.03058)). The **Johnson's algorithm** technique (reweighting with Bellman–Ford) has analogues in "graph reweighting" tricks in geometric GNNs.

> **Worked example.** Vertices $\{1, 2, 3, 4\}$, edges $1\to 2$ (weight 1), $1 \to 3$ (weight 4), $2 \to 3$ (weight 2), $2 \to 4$ (weight 5), $3 \to 4$ (weight 1). Shortest path from 1 to 4: $1 \to 2 \to 3 \to 4$, total weight $1 + 2 + 1 = 4$. (Not the direct $1 \to 3 \to 4$ which is $4 + 1 = 5$.)

## 2.3 Minimum spanning tree

For a connected, weighted, undirected graph, the **minimum spanning tree** (MST) is the spanning tree of minimum total weight. It has $n - 1$ edges.

**Kruskal's algorithm**: sort edges by weight, greedily add each that doesn't create a cycle. Uses a **union-find** data structure. $O(m \log m)$ time, $O(n)$ extra memory.

**Prim's algorithm**: grow the tree from any root; at each step add the cheapest edge leaving the tree. $O(m \log n)$ with a binary heap.

```
Prim(G, w):
    in_tree = {s}
    while |in_tree| < n:
        e = (u, v) with u in_tree, v not in_tree, w(u,v) minimum
        in_tree.add(v)
        tree.add(e)
```

**Cut property.** For any cut $(S, V \setminus S)$, the lightest edge crossing the cut is in every MST. This is the foundation of both algorithms and connects MST to **minimum-cut** (Ch 02.5) and to **max-flow min-cut** duality.

**Connection to GNNs.** MSTs are used in **graph pooling** (e.g., **gPool** pools nodes by selecting the top-k of some learned score; related to tree decompositions). They are also the connection point for **graph kernels** based on subtree patterns.

## 2.4 Topological order and DAG algorithms

A **topological order** is a linear ordering of vertices such that every edge goes from earlier to later. It exists iff the graph is a **DAG** (directed acyclic graph).

```
Kahn's algorithm:
    Compute in-degree of every vertex.
    Enqueue all vertices with in-degree 0.
    while queue not empty:
        u = dequeue
        append u to order
        for v in Adj(u):
            in_degree[v] -= 1
            if in_degree[v] == 0: enqueue(v)
```

If $|order| < n$, the graph has a cycle. $O(n + m)$.

**Connection to GNNs.** DAG-structured data includes scene graphs, parse trees, citation networks (mostly), and code ASTs. Specialised GNN architectures for DAGs (DAGNN, D-VAE) exploit this structure. Topological order also gives you a natural way to **propagate messages** — do it in topological order, no iterative fixed-point needed.

## 2.5 Network flow and min-cut

A **flow network** is a directed graph with source $s$, sink $t$, and capacity $c(e) > 0$ on each edge. A **flow** $f$ satisfies:
- **Capacity constraint**: $0 \le f(e) \le c(e)$.
- **Conservation**: $\sum_{e \text{ into } v} f(e) = \sum_{e \text{ out of } v} f(e)$ for all $v \neq s, t$.

The **max-flow problem** is to maximise $f$ from $s$ to $t$. The **min-$s$-$t$-cut** is the partition $(S, V \setminus S)$ with $s \in S$, $t \notin S$ minimising $\sum_{e \in \delta^+(S)} c(e)$.

**Max-flow min-cut theorem** (Ford & Fulkerson 1956): the maximum flow value equals the minimum cut capacity.

**Algorithms.** **Ford–Fulkerson** with BFS augmentation (**Edmonds–Karp**) is $O(n m^2)$. **Dinic's** algorithm is $O(\min(n^{2/3}, m^{1/2}) m)$ for unit capacities, $O(n^2 m)$ in general. **Push–relabel** achieves $O(n^3)$ worst case and is fastest in practice.

**Connection to GNNs.** Flow-based objectives appear in:
- **Graph cut losses** for unsupervised clustering GNNs.
- **Optimal-transport** style objectives for graph matching.
- **Min-cut pooling** (Bianchi, Grattarola, Alippi, ICML 2020) — a pooling operator that learns a soft cluster assignment by solving a continuous relaxation of normalized min-cut.
- **Spectral cut** (Cheeger cut) is connected to the Fiedler vector, the eigenvector of $L$ corresponding to $\lambda_2$.

> **Worked example.** A 4-vertex network $s=1, t=4$, edges $1 \to 2$ (cap 3), $1 \to 3$ (cap 2), $2 \to 4$ (cap 2), $3 \to 4$ (cap 2), $2 \to 3$ (cap 1). Augmenting paths: $1 \to 2 \to 4$ (bottleneck 2), $1 \to 3 \to 4$ (bottleneck 2). Total flow 4. Min cut: $\{1,2,3\} | \{4\}$ has capacity $2+2 = 4$ ✓.

## 2.6 Triangle counting and motif detection

A **triangle** is a 3-clique. Counting triangles is a primitive for **clustering coefficient** computation, **motif analysis**, and **subgraph GNN** architectures.

Three main techniques:
1. **Node-iterator**: for each vertex $v$, mark its neighbors, then for each pair of neighbors check if they are connected. $O\left(\sum_v \deg(v)^2\right) = O(m^{3/2})$ for many real graphs.
2. **Edge-iterator**: for each edge $(u, v)$, count the number of common neighbors. $O(m \cdot \Delta)$ where $\Delta$ is the max degree.
3. **Forward**: orient edges by degree ordering, count $v \to u$ with $u \to w$ and $v \to w$. $O(m^{3/2})$.

**Connection to GNNs.** Subgraph GNNs ([`2505.15116`](https://arxiv.org/abs/2505.15116) §3.2) and higher-order GNNs (Ch 07) lift the message-passing state to operate on $k$-tuples of vertices, which includes triangles. **Graph random features** (also called **subgraph counting** features) are the basis of pre-GNN kernel methods.

> **Worked example.** 4-vertex cycle: no triangles (the diagonal of $A^2$ counts walks of length 2, and the off-diagonal of $A^2$ gives common-neighbor counts — verify $(A^2)_{1,3} = 0$ because vertices 1 and 3 are at distance 2, not adjacent). Triangle count = 0. Add the edge $(1,3)$: now you have one triangle $\{1,2,3\}$.

## 2.7 PageRank and random walks

**PageRank** is the stationary distribution of a **random walk with restart**: from a vertex, with probability $\alpha$ follow a random edge, with probability $1 - \alpha$ jump back to a teleportation distribution. The stationary distribution $\pi$ satisfies

$$\pi = \alpha P^\top \pi + (1 - \alpha) v,$$

where $P = D^{-1}A$ is the random-walk transition matrix and $v$ is the teleportation vector (often uniform). Solving by power iteration:

```
x_0 = v
for t = 0, 1, 2, ...:
    x_{t+1} = alpha P^T x_t + (1 - alpha) v
until ||x_{t+1} - x_t||_1 < eps
```

Convergence rate: $O(1 / (1 - \alpha))$ iterations to within $\epsilon$ of the true PageRank, in the worst case. For typical $\alpha = 0.85$, ~50–100 iterations suffice.

**Personalised PageRank (PPR)** uses a non-uniform teleportation vector concentrated on a single source $s$; it is the basis of **APPNP** ([`2106.03058`](https://arxiv.org/abs/2106.03058)) and of scalable **local clustering**.

**Connection to GNNs.** PageRank is the "original" message passing: each vertex's value is a weighted average of its neighbors' values, plus a restart bias. The personalised version is exactly what **APPNP / GCN-with-Personalized-PageRank** computes:

$$H^{(k+1)} = (1 - \alpha) \hat A H^{(k)} + \alpha H^{(0)},$$

where $H^{(0)} = f(X)$ is the input MLP output and $\alpha$ is the teleport probability. This is **deeper** than a vanilla GCN without over-smoothing because the restart acts as a regulariser.

> **Worked example.** A "barbell" graph: two cliques $K_5$ joined by a single bridge vertex. Uniform PageRank: each vertex in the cliques has PageRank proportional to its degree (so all clique vertices are equal). The bridge vertex is the *unique* shortest path between the two cliques, so it has higher PageRank than typical clique vertices. The "personalised" PageRank from a single clique vertex concentrates mass within that clique and decays through the bridge.

## 2.8 Subgraph matching and graphlets

A **graphlet** is a small connected induced subgraph (3 vertices: paths $P_3$, triangle $K_3$; 4 vertices: $P_4$, $C_4$, $K_4$, paw, diamond; ...). **Graphlet degree vectors (GDV)** count, for each vertex $v$ and each isomorphism class $g$ of $k$-vertex graphlet, the number of graphlets isomorphic to $g$ that touch $v$.

GDVs were the dominant local structural feature before GNNs; they are still used in:
- **Graph kernels** (Weisfeiler–Leman graph kernels).
- **Structural node embeddings** in pretraining (e.g., GCC's motif features).
- **Position-aware GNNs** that augment features with GDV-based statistics.

**Connection to GFMs.** Many "structural" GFMs (GFT, Boosting GFMs from Structural Perspective, [`2407.19941`](https://arxiv.org/abs/2407.19941)) learn to produce embeddings that capture graphlet / motif statistics. See Ch 11.

## 2.9 Complexity summary

| Problem | Algorithm | Time | Space | Notes |
|---|---|---|---|---|
| BFS/DFS | queue/stack | $O(n+m)$ | $O(n)$ | |
| Dijkstra (SSSP) | heap | $O(m \log n)$ | $O(n)$ | non-negative weights |
| Bellman–Ford | dynamic programming | $O(nm)$ | $O(n)$ | handles negative |
| Floyd–Warshall | DP | $O(n^3)$ | $O(n^2)$ | all-pairs |
| Prim | heap | $O(m \log n)$ | $O(n)$ | MST |
| Kruskal | sort + union-find | $O(m \log m)$ | $O(n)$ | MST |
| Topological sort | Kahn | $O(n+m)$ | $O(n)$ | DAG only |
| Max flow (Dinic) | BFS-augmenting | $O(n^2 m)$ | $O(n+m)$ | practical |
| Triangle count | node-iterator | $O(m^{3/2})$ | $O(m)$ | |
| PageRank | power iteration | $O(k \cdot m)$ | $O(n)$ | $k$ ~ 50–100 |
| Subgraph isomorphism (3-vertex) | enumeration | $O(n^3)$ | $O(1)$ | $k$-vertex is NP-hard |

**Why this matters for GFMs.** Most GFM benchmarks are evaluated on **node classification** (Cora, OGB), **link prediction** (OGB), and **graph classification** (ZINC, MoleculeNet). Underneath all of these is a combination of these primitives. A "good" GFM must produce embeddings that make these tasks easy — i.e., the embeddings must capture local (triangle counts, motifs), mesoscale (clustering, communities), and global (PageRank, diameter) information.

## 2.10 Connection table

| Classical algorithm | Modern ML analog | Where to look |
|---|---|---|
| BFS / shortest paths | GNN receptive field, path reasoning | Ch 06, Ch 12 |
| Min-cut | Spectral clustering, MinCut pooling | §2.5 |
| MST | gPool, tree-decomposition | Bianconi & Dorogovtsev 2024 |
| Random walk / PageRank | DeepWalk, node2vec, APPNP | §2.7, Ch 06 |
| Topological order | DAG GNNs, autoregressive | Ch 06 |
| Triangle counting | Subgraph GNN, lifting | Ch 07 |
| Graphlets | Structural pretraining (GCC) | Ch 09 |
| Min-cost flow | Optimal-transport GNN pooling | Ch 10 |

## Exercises

1. **BFS vs DFS on a 6-vertex graph.** Draw the graph $V = \{1,\ldots,6\}$, $E = \{(1,2),(1,3),(2,4),(3,4),(4,5),(5,6)\}$. Run BFS and DFS from vertex 1, write down the order in which they discover vertices. Where do they agree, where do they differ?

2. **Dijkstra by hand.** Given the graph $V = \{1,2,3,4,5\}$, $E = \{(1,2,1),(1,3,4),(2,3,2),(2,4,5),(3,4,1),(3,5,8),(4,5,2)\}$ (format: $u,v,w$), find shortest paths from 1 to every other vertex. List each vertex's final distance and parent.

3. **Negative weight trap.** Graph: $1 \to 2$ (weight 1), $2 \to 3$ (weight -3), $1 \to 3$ (weight 5). Run Dijkstra from 1: what does it report? Why is it wrong? Run Bellman-Ford: what does it report?

4. **Kruskal's MST.** Vertices $\{1,2,3,4,5\}$, edges (with weights): $(1,2,1), (1,3,5), (2,3,2), (2,4,4), (3,4,3), (3,5,6), (4,5,7)$. Run Kruskal's algorithm. List the edges in MST order.

5. **Max flow.** Source $s=1$, sink $t=5$ in the same graph as exercise 4 (replace each undirected edge with two directed edges of the same capacity). Use Edmonds–Karp (BFS-augmenting). Find the max flow value, and a minimum cut.

6. **PageRank on a tiny graph.** 4 vertices, edges forming a "directed cycle" $1 \to 2 \to 3 \to 4 \to 1$ plus the edge $1 \to 3$ (so vertex 1 has an extra shortcut). Use teleportation $\alpha = 0.85$, $v = (1/4, 1/4, 1/4, 1/4)$. Run power iteration for 10 steps. Verify that the PageRank of vertex 1 is higher than that of vertex 4 (because 1 has an extra incoming shortcut).

7. **Triangle counting.** For the graph $K_4$ (complete on 4 vertices), how many triangles are there? Verify by direct counting and by computing $\text{tr}(A^3) / 6$.

8. **Subgraph matching.** For the 4-vertex graph with edges $\{(1,2),(2,3),(3,4),(1,4),(1,3)\}$: enumerate all 3-vertex graphlets (up to isomorphism) and count how many of each you see. Compare to the 4 graphlet classes for 3 vertices: empty (1), single edge (1), path $P_3$ (3), triangle $K_3$ (4).

9. **GNN receptive field.** A 4-layer GCN with 1-hop message passing has, for a node $v$, a receptive field that is the 4-hop BFS tree from $v$. In a $d$-regular tree (no cycles), this tree has $1 + d + d^2 + d^3 + d^4$ nodes. For $d = 3$, how many nodes is that? (This is the over-squashing math: the 4-hop neighborhood has 121 nodes but the node's final embedding is one vector.)

10. **Reading.** Read the introduction of [`1901.00596`](https://arxiv.org/abs/1901.00596) (Wu et al. GNN survey). Identify which classical graph algorithm the survey cites as inspiration for each major GNN category.

## Further Reading

- **Cormen, Leiserson, Rivest, Stein, *Introduction to Algorithms*** (4th ed., MIT Press 2022). Ch 20–26 cover everything in this chapter with proofs. The standard reference.
- **Kleinberg & Tardos, *Algorithm Design*** (Pearson 2005). More accessible than CLRS, with excellent motivation.
- **Ahuja, Magnanti, Orlin, *Network Flows: Theory, Algorithms, and Applications*** (Prentice Hall 1993). The book on max-flow / min-cut.
- **Newman, *Networks: An Introduction*** (Oxford 2010). §7 has the best intro to random walks and PageRank I have read.
- **[`1403.6652`](https://arxiv.org/abs/1403.6652)** — Perozzi, Al-Rfou, Skiena, *DeepWalk: Online Learning of Social Representations* (KDD 2014). The bridge from random walks to node embeddings.
- **[`1607.00653`](https://arxiv.org/abs/1607.00653)** — Grover & Leskovec, *node2vec: Scalable Feature Learning for Networks* (KDD 2016). The biased-random-walk refinement of DeepWalk.
- **[`2106.03058`](https://arxiv.org/abs/2106.03058)** — Klicpera, Bojchevski, Günnemann, *Predict then Propagate: Graph Neural Networks meet Personalized PageRank* (ICLR 2019, APPNP).
- **[`2002.05287`](https://arxiv.org/abs/2002.05287)** — Pei et al., *Geom-GCN: Geometric Graph Convolutional Networks* (ICLR 2020). Uses geometric/spectral ideas to handle heterophily.
- **[`1903.03894`](https://arxiv.org/abs/1903.03894)** — Ying et al., *GNNExplainer* (NeurIPS 2019). §2.1 has a clean recap of BFS/shortest-path as the "receptive field" intuition.
- **[`2302.04181`](https://arxiv.org/abs/2302.04181)** — Müller et al., *Attending to Graph Transformers* (TMLR 2023). §2.1 has a useful table connecting classical graph algorithms to modern graph-transformer operations.
- **Leskovec, Rajaraman, Ullman, *Mining of Massive Datasets*** (Cambridge 2014, free online). Ch 5–10 cover PageRank, spectral methods, and graph mining at scale.
- **Spielman, *Spectral Graph Theory*** (lecture notes, Yale). The modern, cleaner version of Chung's book. Especially Ch 3 (sparsification) and Ch 9 (Laplacian solvers).
- **For PageRank and its analysis**: Bryan & Leise, *The 25,000,000,000 Eigenvector: The Linear Algebra behind Google* (SIAM Review 2006). Readable and complete.
