---
title: "Graph Learning Exercises"
module: "13-graph-learning"
lesson: "graph-exercises"
doc_type: "exercise"
topic: "graphs-spectral-methods-message-passing-gnns"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
  - "13-graph-learning/graph-learning"
  - "13-graph-learning/spectral-graph-methods"
updated: "2026-04-13"
owner: "curriculum-team"
tags:
  - "graph-learning"
  - "gnn"
  - "message-passing"
  - "spectral-methods"
---

## Purpose

These exercises reinforce graph representations, Laplacians, message passing, spectral filtering, and the main architectural ideas behind GCN, GraphSAGE, and GAT.

## Exercise 1: adjacency, degree, and Laplacian

Consider the undirected graph on nodes $\{1,2,3,4\}$ with edges

$$
\{(1,2), (2,3), (2,4)\}.
$$

1. Write the adjacency matrix $A$.
2. Write the degree matrix $D$.
3. Compute the combinatorial Laplacian $L = D - A$.
4. Verify that each row of $L$ sums to zero.

## Exercise 2: Laplacian quadratic form

Let $f = [2, 1, 3, 0]^\top$ on the graph from Exercise 1.

1. Compute $f^\top L f$ directly using matrix multiplication.
2. Compute the same quantity using

$$
\frac{1}{2}\sum_{i,j} A_{ij}(f_i - f_j)^2.
$$

3. Explain in one short paragraph what this value says about the smoothness of $f$ on the graph.

## Exercise 3: message passing as a template

Suppose a graph layer is defined by

$$
m_i = \frac{1}{|\mathcal{N}(i)|}\sum_{j \in \mathcal{N}(i)} W_{\mathrm{msg}} h_j,
\qquad
h_i' = \phi(W_{\mathrm{self}} h_i + m_i).
$$

1. Identify the message, aggregation, and update functions.
2. Explain why the aggregation is permutation invariant.
3. State one change that would let edge features influence the messages.
4. State one reason to add self-loops explicitly.

## Exercise 4: permutation equivariance

Let $P$ be a permutation matrix that relabels the nodes of a graph.
Let $S$ be a propagation operator and consider the layer

$$
H' = \sigma(SHW).
$$

1. Write the relabeled adjacency matrix in terms of $P$ and $A$.
2. Write the relabeled feature matrix in terms of $P$ and $H$.
3. Show that if the propagation operator transforms as $S' = P S P^\top$, then the updated features satisfy

$$
H_{\mathrm{relabeled}}' = P H'.
$$

4. Explain why this property is desirable for graph models.

## Exercise 5: deriving the GCN operator

Starting from the normalized Laplacian

$$
L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2},
$$

derive the expression

$$
I + D^{-1/2} A D^{-1/2}
$$

from a first-order spectral approximation.

Then explain why GCN replaces this with

$$
\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2},
\qquad
\hat{A} = A + I.
$$

## Exercise 6: GraphSAGE versus GCN

Answer each part in two or three sentences.

1. Why is GraphSAGE often described as an inductive method?
2. Why can neighborhood sampling matter on large graphs?
3. What information is preserved by concatenating self and neighbor summaries before the update?
4. Give one setting where GraphSAGE may be preferable to a full-batch GCN.

## Exercise 7: attention on graphs

For a GAT layer with coefficients $\alpha_{ij}$:

1. State how the coefficients are normalized over a neighborhood.
2. Explain why GAT can distinguish two neighbors that have the same edge type but different features.
3. Give one computational drawback of attention compared with fixed normalized aggregation.
4. Give one setting where adaptive neighbor weighting is especially attractive.

## Exercise 8: oversmoothing and oversquashing

1. Define oversmoothing in graph neural networks.
2. Define oversquashing.
3. Explain why these are different problems.
4. Give one mitigation strategy for each.

## Exercise 9: graph transformers

Write a short paragraph answering all parts:

- Why does a graph transformer still need structural information even though it uses attention?
- What role can Laplacian eigenvectors or shortest-path features play?
- In what sense is a graph transformer a generalization of message passing rather than a completely unrelated model family?

## Exercise 10: node classification experiment design

You are given a citation graph with node features and class labels.

1. Define a train, validation, and test split that avoids obvious leakage mistakes.
2. State one baseline that ignores edges.
3. State one graph-based model you would compare against that baseline.
4. Name two evaluation risks that are specific to graph data.

## Exercise 11: relational structure and composition

Write one paragraph connecting graph learning to the idea of composing local relations into a global representation.
Then add one sentence explaining why this is a natural forward reference to Module 16 without relying on formal category-theory vocabulary yet.
