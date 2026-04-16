---
title: "Graph Learning"
module: "13-graph-learning"
lesson: "graph-learning"
doc_type: "notes"
topic: "graphs-message-passing-gnns-graph-transformers"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
  - "06-neural-networks/neural-networks-first-principles"
  - "07-deep-learning-systems/training-deep-networks"
  - "10-transformers-llms"
updated: "2026-04-13"
owner: "curriculum-team"
tags:
  - "graph-learning"
  - "gnn"
  - "message-passing"
  - "gcn"
  - "graphsage"
  - "gat"
  - "graph-transformers"
---

## Purpose

These notes introduce machine learning on graph-structured data.
The central question is how to build models that use not only the features attached to an object, but also the pattern of relations between objects.

We start with graph representations and a general message-passing framework.
Only after that general framework is in place do we study specific architectures such as GCN, GraphSAGE, and GAT.
The spectral viewpoint is developed in a companion note because it gives the cleanest derivation of GCN as an approximation to graph convolution.

## Learning objectives

After working through this note, you should be able to:

- represent graph data using adjacency matrices, degree matrices, and Laplacians;
- distinguish node-level, edge-level, and graph-level prediction tasks;
- formalize message passing as a permutation-equivariant update rule on a graph;
- explain the design ideas behind GCN, GraphSAGE, and GAT;
- describe why graph transformers extend rather than replace message passing; and
- connect graph learning to relational composition and a later categorical treatment in Module 16.

## 1. Why graph learning matters

Many ML datasets are not naturally IID tables.
They are collections of entities tied together by relations:

- molecules, where atoms are connected by bonds;
- citation networks, where papers cite papers;
- social networks, where users interact with users;
- knowledge graphs, where entities participate in typed relations; and
- program, scene, or traffic data, where structure is essential to prediction.

If the relation pattern matters, a plain tabular model throws away information.
Graph learning treats the relation structure as part of the input rather than as an afterthought.

## 2. Graph representations

Let

$$
G = (V, E)
$$

be a graph with node set $V = \{1,\dots,n\}$ and edge set $E$.
We will mostly use undirected graphs first, then mention directed and typed extensions.

### 2.1 Adjacency matrix

The adjacency matrix $A \in \mathbb{R}^{n \times n}$ is defined by

$$
A_{ij} =
\begin{cases}
1 & \text{if } (i,j) \in E, \\
0 & \text{otherwise.}
\end{cases}
$$

For weighted graphs, $A_{ij}$ can store the edge weight instead of a binary indicator.
For undirected graphs, $A$ is symmetric.

### 2.2 Degree matrix

The degree of node $i$ is

$$
d_i = \sum_{j=1}^n A_{ij}.
$$

The degree matrix is the diagonal matrix

$$
D = \operatorname{diag}(d_1,\dots,d_n).
$$

Degrees matter because they control how much neighborhood information each node receives.

### 2.3 Node, edge, and graph features

Graph data usually include features beyond topology:

- node features $x_i \in \mathbb{R}^{d_x}$;
- edge features $e_{ij} \in \mathbb{R}^{d_e}$ when relations are typed or weighted;
- graph-level labels or attributes when the whole graph is the prediction target.

Stack node features rowwise into

$$
X \in \mathbb{R}^{n \times d_x}.
$$

Most graph neural networks update this node-feature matrix layer by layer.

### 2.4 Laplacian matrices

The combinatorial Laplacian is

$$
L = D - A.
$$

The normalized Laplacian is

$$
L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2}.
$$

These matrices measure variation over the graph.
The quadratic form

$$
f^\top L f = \frac{1}{2}\sum_{(i,j)\in E} A_{ij}(f_i - f_j)^2
$$

is small when neighboring nodes take similar values.
That observation is the bridge to spectral graph methods.

## 3. Prediction tasks on graphs

Three task types appear repeatedly:

- **Node-level tasks.** Predict a label or value for each node, such as topic classification in a citation network.
- **Edge-level tasks.** Predict whether an edge exists or what type it has, such as link prediction or relation classification.
- **Graph-level tasks.** Predict a label for the whole graph, such as molecular property prediction.

The same message-passing layers can support all three tasks, but the final readout object differs.

## 4. Message passing as the general framework

### 4.1 Locality and shared parameters

A graph layer should respect two structural facts:

1. a node should update using its own state and nearby relational context;
2. the same rule should apply at every node rather than learning a separate parameter set for each node identity.

Those two principles lead to message passing.

### 4.2 General message-passing update

Let $h_i^{(k)} \in \mathbb{R}^{d_k}$ be the node representation at layer $k$.
A general message-passing layer has the form

$$
m_i^{(k+1)} =
\operatorname{AGG}^{(k)}
\left(
\left\{
\operatorname{MSG}^{(k)}\!\left(h_i^{(k)}, h_j^{(k)}, e_{ij}\right)
: j \in \mathcal{N}(i)
\right\}
\right),
$$

followed by

$$
h_i^{(k+1)} =
\operatorname{UPD}^{(k)}\!\left(h_i^{(k)}, m_i^{(k+1)}\right).
$$

Here:

- $\mathcal{N}(i)$ is the neighborhood of node $i$;
- `MSG` defines what information is sent across one edge;
- `AGG` combines incoming messages; and
- `UPD` produces the next node state.

Typical choices are:

- `AGG =` sum, mean, or max;
- `UPD =` affine map plus nonlinearity, or a gated recurrent update;
- self-loops so each node can keep its own information.

### 4.3 Why aggregation must be permutation invariant

Neighborhoods are sets, not ordered sequences.
If we relabel nodes or list neighbors in a different order, the model should not change its answer for purely bookkeeping reasons.
Therefore the aggregation operator must be permutation invariant.

Sums, means, and maxima satisfy this requirement.
This is one reason message-passing networks are often described as permutation-equivariant over node orderings.

### 4.4 Matrix form of neighborhood aggregation

If aggregation is linear, one layer often looks like

$$
H^{(k+1)} = \sigma\!\left(S H^{(k)} W^{(k)}\right),
$$

where:

- $H^{(k)} \in \mathbb{R}^{n \times d_k}$ stacks node states;
- $W^{(k)}$ is a learned weight matrix;
- $\sigma$ is an elementwise nonlinearity; and
- $S$ is a graph propagation operator built from $A$ and $D$.

Different architectures are largely distinguished by how they choose or learn $S$.

### 4.5 Readout for graph-level tasks

For graph classification, message passing is followed by a graph-level pooling step:

$$
h_G = \operatorname{READOUT}\!\left(\{h_i^{(K)} : i \in V\}\right).
$$

Again the readout must be permutation invariant.
Common choices are sum, mean, max, attention pooling, or hierarchical pooling.

## 5. GCN: a normalized message-passing layer

The graph convolutional network of Kipf and Welling can be written as

$$
H^{(k+1)} =
\sigma\!\left(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2} H^{(k)} W^{(k)}\right),
$$

where

$$
\hat{A} = A + I,
\qquad
\hat{D}_{ii} = \sum_j \hat{A}_{ij}.
$$

This layer:

- includes self-loops through $\hat{A}$;
- averages information from neighbors with degree normalization; and
- applies the same linear map at every node.

From the message-passing point of view, GCN uses:

- a linear message function;
- a weighted sum aggregator; and
- a simple nonlinear update.

From the spectral point of view, it approximates a low-order graph convolution.
That derivation appears in [spectral-graph-methods.md](./spectral-graph-methods.md).

### 5.1 Why normalization matters

Without normalization, high-degree nodes could dominate the aggregation simply because they have many neighbors.
The symmetric normalization

$$
\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}
$$

keeps the propagation operator better scaled and treats two endpoints more symmetrically than a purely row-normalized average.

### 5.2 Oversmoothing warning

Repeated multiplication by a smoothing operator makes neighboring node states increasingly similar.
If we stack too many layers, node representations can collapse toward a low-variation subspace.
This is the oversmoothing problem.

GCNs therefore work best with a moderate number of layers unless they are modified with residual, normalization, or architectural safeguards.

## 6. GraphSAGE: inductive aggregation by sampling neighborhoods

GraphSAGE was designed for inductive settings where we want to generalize to nodes not seen during training.
Instead of treating the graph as a fixed transductive object, it learns an aggregation rule over sampled neighborhoods.

A typical GraphSAGE layer computes

$$
\tilde{h}_i^{(k+1)} =
\operatorname{AGG}^{(k)}\!\left(\{h_j^{(k)} : j \in \mathcal{N}(i)\}\right),
$$

then

$$
h_i^{(k+1)} =
\sigma\!\left(
W^{(k)}
\begin{bmatrix}
h_i^{(k)} \\
\tilde{h}_i^{(k+1)}
\end{bmatrix}
\right).
$$

Common aggregators include mean, max-pooling, and LSTM-based aggregation.

The main ideas are:

- neighborhood sampling controls computation on large graphs;
- concatenating self and neighbor summaries preserves a clear self/neighbor distinction;
- the learned aggregator can be applied to unseen nodes or new graphs.

## 7. GAT: attention over neighbors

Graph attention networks replace fixed neighborhood weights with learned, data-dependent attention coefficients.
For one attention head,

$$
\alpha_{ij}
=
\operatorname{softmax}_{j \in \mathcal{N}(i) \cup \{i\}}
\left(
a^\top
\left[
W h_i \,\|\, W h_j
\right]
\right),
$$

and the update is

$$
h_i^{(k+1)}
=
\sigma\!\left(
\sum_{j \in \mathcal{N}(i)\cup\{i\}}
\alpha_{ij} W h_j
\right).
$$

The advantage is clear:
not every neighbor should contribute equally.
Attention lets the model weight neighbors according to their features and context.

Multi-head attention is often used to stabilize training and enrich the representation.

## 8. Graph transformers

Message passing is local by default.
Each layer communicates only across one edge, so $K$ layers reach only $K$ hops away.
Graph transformers relax that locality by using attention over larger parts of the graph.

A graph transformer usually adds some combination of:

- node features;
- structural encodings such as shortest-path distance, Laplacian eigenvector features, or centrality;
- edge bias terms that inform attention scores; and
- global attention, sparse attention, or virtual nodes for long-range communication.

This makes graph transformers attractive when:

- long-range dependencies matter;
- the graph is richly attributed;
- structural position is important in addition to local neighborhoods.

But they do not eliminate the need for graph structure.
They still require a mechanism for injecting topology into the attention computation.
In practice they should be viewed as extending graph message passing with transformer-style global interaction, not as abandoning the graph viewpoint.

## 9. Spatial versus spectral viewpoints

There are two complementary ways to think about graph learning layers:

- **Spatial viewpoint.** A node aggregates information from nearby nodes in the original graph.
- **Spectral viewpoint.** A graph signal is expanded in a Laplacian eigenbasis, filtered in frequency space, and transformed back.

Spatial formulations are usually easier to implement at scale.
Spectral formulations are mathematically cleaner for understanding smoothing, filtering, and the derivation of GCN.
Modern graph ML relies on both viewpoints.

## 10. Relational structure and composition

Graph learning is fundamentally about composing local interactions into global representations.
Each layer applies the same local update across edges, then composes those local transformations across depth.

That is one reason graph learning is a natural forward reference to Module 16.
Category theory will later provide a language for:

- compositional pipelines;
- invariance under relabeling;
- local-to-global construction; and
- typed relational structure.

For now, the important point is concrete:
a graph model is not just learning from values, but from values organized by a relation pattern and repeatedly composed through that pattern.

## 11. Common practical issues

- **Heterophily.** Neighboring nodes may have different labels, so pure smoothing can hurt.
- **Oversquashing.** Too much distant information is forced through narrow local bottlenecks.
- **Sampling bias.** Large graphs often require neighborhood sampling, which changes the estimator.
- **Leakage.** Splits on graph data can accidentally leak label information through edges.
- **Scalability.** Dense attention over large graphs can be expensive.

These are not side issues.
They are part of why graph learning remains an active research area.

## 12. Summary

Graphs represent data with explicit relational structure.
Message passing is the central abstraction:
messages are computed on edges, aggregated at nodes, and updated layer by layer.
GCN, GraphSAGE, and GAT are different choices inside that shared template.
Graph transformers extend the same agenda toward more global attention patterns.

The next note develops the spectral side in more detail and shows how GCN arises as a tractable approximation to graph convolution.

## References

- Bronstein, Bruna, Cohen, and Veličković, *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges*.
- Hamilton, *Graph Representation Learning*.
- Kipf and Welling, *Semi-Supervised Classification with Graph Convolutional Networks*.
- Hamilton, Ying, and Leskovec, *Inductive Representation Learning on Large Graphs*.
- Veličković et al., *Graph Attention Networks*.
