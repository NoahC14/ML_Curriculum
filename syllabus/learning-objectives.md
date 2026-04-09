# Learning Objectives

## Purpose
This document defines the shared learning-objective framework for the ML/AI curriculum and records module-level objectives for the full 18-module spine.

## Pedagogical stance
The curriculum uses a repo-native pedagogy built around three mutually reinforcing modes:
- `proof`: learners derive, justify, and critique mathematical results;
- `coding`: learners implement models, experiments, and tooling artifacts; and
- `intuition`: learners explain geometric, statistical, or systems-level behavior in plain language.

Each module should expose all three modes. As a default target, each module should include:
- at least `2` mathematically serious derivations or proof sketches;
- at least `2` coding exercises or labs;
- at least `1` intuition or interpretation exercise; and
- at least `1` synthesis artifact that connects mathematics, implementation, and empirical behavior.

Modules may document justified exceptions in their `README.md`. For example, reinforcement learning may substitute environment-building tasks for a notebook-centric lab sequence, and ethics or safety may emphasize case analysis over theorem-heavy derivations.

## Common module structure
Every module should default to the following internal structure:
- `README.md`
- `notes/`
- `derivations/`
- `notebooks/`
- `src/`
- `exercises/`
- `solutions/`
- `projects/`
- `references/`
- `unity/`

The module `README.md` should state:
- purpose;
- prerequisites;
- learning objectives;
- lesson breakdown;
- key definitions and results;
- computational labs;
- exercises and assessments;
- category theory insertion points; and
- Unity Theory insertion points.

## Course-level objectives
By the end of the full curriculum, a successful learner should be able to:
- explain the canonical machine learning pipeline from data representation through model selection, optimization, evaluation, and deployment-oriented systems concerns;
- derive core results in linear algebra, calculus, probability, statistics, optimization, and learning theory that support modern ML practice;
- implement standard ML and deep learning methods from first principles and with appropriate software abstractions;
- analyze empirical behavior using principled diagnostics, ablations, and error analysis;
- use category-theoretic language where it clarifies composition, invariance, abstraction, and transfer;
- distinguish canonical technical claims from interpretive or speculative Unity Theory commentary; and
- produce reusable repository artifacts such as notes, derivations, labs, code, projects, and references.

## Objective taxonomy
Module objectives should collectively cover:
- `formal mastery`: definitions, theorems, derivations, and assumptions;
- `computational fluency`: implementation, experimentation, and debugging;
- `interpretive judgment`: model behavior, limitations, and tradeoffs; and
- `structural transfer`: recognizing reusable patterns across modules.

## Module objectives

### Module 00: Math Toolkit
- use linear algebra, multivariable calculus, probability, and statistics notation fluently in ML settings;
- derive gradients, Jacobians, and matrix identities that recur throughout the curriculum;
- work with vector spaces, random variables, estimators, and asymptotic language at a level needed for later modules;
- interpret basic category-theory notions such as objects, morphisms, composition, products, and functors through concrete ML examples.

### Module 01: Optimization
- derive first-order and selected second-order optimization updates from objective functions and constraints;
- explain convergence-relevant concepts such as convexity, conditioning, step sizes, and stochastic approximation;
- implement gradient-based optimizers and compare their empirical behavior on representative loss landscapes;
- connect optimization choices to stability, efficiency, and generalization tradeoffs in later ML systems.

### Module 02: Statistical Learning Foundations
- define supervised learning, hypothesis classes, generalization, risk, empirical risk, and regularization with precision;
- explain bias-variance tradeoffs, capacity control, and train-validation-test methodology;
- derive and interpret core statistical-learning bounds and assumptions at an accessible level;
- evaluate model performance using sound experimental design and uncertainty-aware metrics.

### Module 03: Linear Models
- derive least squares, ridge, logistic regression, and related linear-model objectives and estimators;
- analyze the geometry and statistical assumptions behind linear decision rules;
- implement linear and generalized linear models with careful feature processing and evaluation;
- identify when linear models remain competitive due to interpretability, sample efficiency, or problem structure.

### Module 04: Kernel Methods
- explain feature maps, kernels, margin-based learning, and the representer-style logic behind kernelization;
- derive core support vector machine and kernel ridge formulations at the primal and dual level where appropriate;
- implement or inspect kernel methods on structured datasets and diagnose sensitivity to kernel and regularization choices;
- compare explicit feature engineering with implicit high-dimensional representations.

### Module 05: Probabilistic Modeling
- work fluently with likelihoods, priors, posteriors, latent variables, and graphical-model style factorizations;
- derive estimation or inference procedures for canonical probabilistic models;
- implement probabilistic models and evaluate calibration, uncertainty, and inference quality;
- explain when probabilistic framing offers advantages over purely discriminative methods.

### Module 06: Neural Networks
- derive forward and backward propagation for multilayer neural networks from first principles;
- explain activation functions, parameterization choices, and loss-function interactions;
- implement a small neural-network library or training loop without relying entirely on high-level abstractions;
- interpret optimization dynamics and failure modes such as vanishing gradients, saturation, and overfitting.

### Module 07: Deep Learning Systems
- explain the systems constraints that shape modern deep learning, including batching, memory, parallelism, and accelerator use;
- implement training pipelines with reproducibility, instrumentation, and modularity in mind;
- analyze performance bottlenecks and tradeoffs across hardware, software, and model design;
- relate systems decisions to research velocity, reliability, and deployment readiness.

### Module 08: CNN and Vision
- derive the core operations of convolutional architectures and explain translation-related inductive biases;
- implement and evaluate convolutional models for image tasks with appropriate augmentation and diagnostics;
- compare architectural choices such as pooling, normalization, residual connections, and receptive-field growth;
- interpret failure cases in vision systems using both qualitative and quantitative evidence.

### Module 09: Sequence Models
- explain recurrent, autoregressive, and sequence-to-sequence modeling assumptions and limitations;
- derive recurrence-based training mechanics, including backpropagation through time at an appropriate level;
- implement sequence models for text or time-series problems and evaluate them with task-appropriate metrics;
- compare recurrence-based approaches with attention-based alternatives in terms of memory, parallelism, and inductive bias.

### Module 10: Transformers and LLMs
- explain self-attention, positional information, masking, scaling behavior, and transformer block composition;
- derive the main tensor transformations involved in attention and training objectives used in language modeling;
- implement or meaningfully modify transformer components and training/evaluation workflows;
- analyze the capabilities and limits of large language models, including prompting, adaptation, and evaluation concerns.

### Module 11: Generative Models
- distinguish likelihood-based, adversarial, diffusion, and latent-variable generative modeling approaches;
- derive the main training logic for representative generative model families;
- implement generative experiments and evaluate sample quality, diversity, and failure modes;
- explain how objective choice shapes generation behavior and practical deployment risks.

### Module 12: Reinforcement Learning
- define Markov decision processes, value functions, policies, returns, and exploration-exploitation tradeoffs precisely;
- derive core dynamic-programming, value-based, and policy-gradient updates;
- implement agents in suitable environments and analyze learning stability, sample efficiency, and reward design;
- distinguish model-based and model-free reasoning and identify where RL assumptions break in practice.

### Module 13: Graph Learning
- explain graph representations, message passing, relational inductive biases, and graph-level versus node-level tasks;
- derive canonical update rules or objectives for graph-based learning methods;
- implement graph-learning pipelines and evaluate them with attention to topology, sparsity, and leakage risks;
- compare graph methods with tabular, sequence, or convolutional alternatives when relational structure matters.

### Module 14: Causality and Reasoning
- distinguish association, intervention, and counterfactual reasoning using precise causal language;
- work with structural causal models, graphical criteria, and identification logic at a practical level;
- implement or analyze causal estimation workflows and reasoning benchmarks;
- explain the limits of purely correlational ML in settings requiring intervention-aware reasoning.

### Module 15: Ethics, Safety, and Evaluation
- analyze fairness, robustness, reliability, misuse, and governance concerns using concrete technical cases;
- design evaluation protocols that go beyond aggregate benchmark performance;
- critique model claims in terms of measurement validity, uncertainty, and deployment context;
- produce responsible reporting artifacts that connect empirical results to stakeholder and safety considerations.

### Module 16: Category Theory for ML
- formalize the categorical ideas introduced earlier in terms appropriate for ML architecture and pipeline analysis;
- use categorical constructions to reason about composition, invariance, abstraction, and interoperability across ML systems;
- compare multiple ML workflows through diagrammatic and structural viewpoints without replacing standard mathematics;
- identify productive research or design questions that benefit from categorical reframing.

### Module 17: Unity Theory Perspectives
- distinguish canonical ML results from Unity Theory interpretations and speculative extensions;
- state any claimed correspondence between Unity-theoretic ideas and ML formalisms precisely and with scope limits;
- evaluate Unity Theory material as a companion lens for synthesis, research ideation, or philosophical reflection;
- produce disciplined companion notes that do not blur established results with exploratory conjecture.

## Implementation note
These objectives are the syllabus-level contract. Individual module `README.md` files may refine wording, add local objectives, or document exceptions, but they should not weaken the canonical-first posture or remove one of the three pedagogical modes without an explicit reason.
