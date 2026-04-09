# Work Cards

## Schema

Every card follows this structure:

| Field              | Description                                                    |
|--------------------|----------------------------------------------------------------|
| **Title**          | Short descriptive name                                         |
| **Type**           | Planning / Engineering / Content / Lab / Companion / Research-content / Curriculum / Publishing / Editing / Governance / Design / Project |
| **Purpose**        | What this card accomplishes and why it matters                  |
| **Inputs / Context** | What must exist or be understood before this work begins      |
| **Deliverables**   | Concrete artifacts produced                                    |
| **Acceptance Criteria** | How to know the card is done                              |
| **Dependencies**   | Other cards that must be completed first                       |
| **Suggested Owner**| Role best suited to do this work                               |
| **Notes / Risks**  | Warnings, open questions, or design tensions                   |

---

# Epic 0 — Course Vision and Scope

---

## Card 0.1 — Define course thesis and audience

| Field | Detail |
|---|---|
| **Type** | Planning |
| **Purpose** | Establish a clear, one-page statement of what the course is, who it serves, and how it differs from standard ML curricula. Every downstream content decision traces back to this document. |
| **Inputs / Context** | The course outline in `ml_ai_course_kanban_v2.md`; survey of comparable open ML curricula (fast.ai, Stanford CS229/CS231n, deeplearning.ai); understanding of category theory and Unity Theory integration goals. |
| **Deliverables** | `syllabus/course-overview.md` containing: course thesis paragraph, target learner persona(s), mathematical rigor target, scope/non-scope list, explicit statement of category theory and Unity Theory roles. |
| **Acceptance Criteria** | Clear learner persona with prerequisite expectations; explicit mathematical rigor target (proof density, derivation style); clear statement of how category theory and Unity Theory participate without dominating; scope and non-scope boundaries are crisp enough to resolve future content disputes. |
| **Dependencies** | None — this is the root planning card. |
| **Suggested Owner** | Course architect / lead author. |
| **Notes / Risks** | Risk of scope creep if the thesis is too broad. The thesis should be specific enough that someone could read it and decide whether this course is or is not for them. |

---

## Card 0.2 — Define pedagogical principles

| Field | Detail |
|---|---|
| **Type** | Planning |
| **Purpose** | Establish the instructional style, assessment philosophy, and repository-native learning design so that all modules share a consistent educational approach. |
| **Inputs / Context** | Card 0.1 (course thesis and audience); understanding of repo-native pedagogy; familiarity with proof-based vs. coding-based vs. intuition-based instruction tradeoffs. |
| **Deliverables** | `syllabus/learning-objectives.md` with per-module learning objectives; `syllabus/assessment-strategy.md` defining exercise tiers, grading philosophy, and assessment formats. |
| **Acceptance Criteria** | Every module follows a documented common internal structure (README, notes, derivations, notebooks, src, exercises, solutions, projects, references, unity); proof/coding/intuition balance is explicit and quantified (e.g., each module has at least N derivations, M coding exercises); assessment rubric exists for at least three exercise types. |
| **Dependencies** | Card 0.1. |
| **Suggested Owner** | Course architect / curriculum designer. |
| **Notes / Risks** | Overly rigid structure may stifle modules with different natural shapes (e.g., the RL module needs environments, not just notebooks). Allow for documented exceptions. |

---

## Card 0.3 — Define canonical-vs-speculative boundary

| Field | Detail |
|---|---|
| **Type** | Governance |
| **Purpose** | Explicitly and consistently separate standard, accepted ML material from original interpretive material (Unity Theory, novel categorical readings) throughout the entire course. This is essential for academic credibility. |
| **Inputs / Context** | Card 0.1 (course thesis); understanding of which content is standard in the literature vs. original to this course. |
| **Deliverables** | `shared/style-guides/content-boundary.md` defining: labeling conventions for standard vs. interpretive content; visual/typographic cues (e.g., boxed sections, distinct headers); rules for citation when crossing the boundary; reviewer checklist for boundary compliance. |
| **Acceptance Criteria** | Standard material is never presented as speculative; Unity Theory and novel categorical content is always labeled as interpretive or exploratory; a reviewer can audit any module page and immediately identify which category the content falls into. |
| **Dependencies** | Card 0.1. |
| **Suggested Owner** | Course architect with input from subject-matter reviewers. |
| **Notes / Risks** | This is one of the most important governance cards. If the boundary is unclear, the course loses credibility as an ML education resource. Err on the side of over-labeling speculative content. |

---

# Epic 1 — Repository and Tooling

---

## Card 1.1 — Scaffold repository structure

| Field | Detail |
|---|---|
| **Type** | Engineering |
| **Purpose** | Create the complete top-level repository layout with all directories, placeholder READMEs, and naming conventions so that all content authors work within a consistent structure from day one. |
| **Inputs / Context** | Repository architecture from Section 6 of the course plan; Card 0.2 (module internal structure). |
| **Deliverables** | Full directory tree as specified: `modules/00-math-toolkit/` through `modules/17-unity-theory-perspectives/`, each with subdirectories (`notes/`, `derivations/`, `notebooks/`, `src/`, `exercises/`, `solutions/`, `projects/`, `references/`, `unity/`); `shared/`, `tooling/`, `projects/`, `kanban/` top-level dirs; placeholder `README.md` in every directory; `shared/style-guides/naming-conventions.md`. |
| **Acceptance Criteria** | All 18 module directories exist with correct numbering; every directory has a placeholder README; naming convention document is written and linked from root README; `tree` command output matches the planned architecture. |
| **Dependencies** | Card 0.1, Card 0.2. |
| **Suggested Owner** | Repository engineer / DevOps contributor. |
| **Notes / Risks** | Naming must be locked early — renaming modules later causes link breakage. Consider whether `00-math-toolkit` should be split into sub-modules at this stage or deferred. |

---

## Card 1.2 — Define markdown and notebook standards

| Field | Detail |
|---|---|
| **Type** | Engineering |
| **Purpose** | Standardize all written content so that notes, derivations, notebooks, and exercises are visually and structurally consistent across the entire course. |
| **Inputs / Context** | Card 0.2 (pedagogical principles); Card 1.1 (repo structure); survey of existing math-heavy markdown and notebook conventions (e.g., LaTeX in markdown, nbformat metadata). |
| **Deliverables** | `shared/style-guides/markdown-style.md` covering: front matter template, heading hierarchy, theorem/definition/proof block formatting, inline and display math conventions, citation format, cross-reference conventions; `shared/style-guides/notebook-style.md` covering: cell ordering rules, narrative cell density, code cell naming, output expectations, kernel metadata. |
| **Acceptance Criteria** | Consistent front matter template exists and is demonstrated in at least one example file; math formatting conventions cover inline, display, aligned, and numbered equations; theorem blocks have a defined visual style; notebook naming pattern is documented (e.g., `NB-{module}-{topic}-{sequence}.ipynb`). |
| **Dependencies** | Card 0.2, Card 1.1. |
| **Suggested Owner** | Technical writer / style lead. |
| **Notes / Risks** | Over-specifying style can slow contributors. Keep rules minimal but enforced. Consider a linter or template generator. |

---

## Card 1.3 — Set up Python environment and reproducibility

| Field | Detail |
|---|---|
| **Type** | Engineering |
| **Purpose** | Ensure any contributor or learner can install dependencies and run all notebooks and code from a clean environment without manual patching. |
| **Inputs / Context** | Card 1.1 (repo structure); anticipated library dependencies (NumPy, SciPy, matplotlib, PyTorch, scikit-learn, Jupyter, etc.). |
| **Deliverables** | `pyproject.toml` or `requirements.txt` with pinned versions; environment setup script or Makefile target; `tooling/environment/README.md` with bootstrap instructions; optional Docker or devcontainer configuration. |
| **Acceptance Criteria** | Fresh `pip install` or `conda create` from the config file succeeds; core notebooks from at least three modules run without errors; environment setup documented for macOS, Linux, and Windows (or WSL). |
| **Dependencies** | Card 1.1. |
| **Suggested Owner** | DevOps / engineering contributor. |
| **Notes / Risks** | GPU-dependent notebooks (deep learning modules) need a separate or extended environment. Document GPU setup as optional. Pin versions aggressively to avoid silent breakage over time. |

---

## Card 1.4 — Add CI for notebooks and code checks

| Field | Detail |
|---|---|
| **Type** | Engineering |
| **Purpose** | Prevent repo drift, broken notebooks, and formatting inconsistencies through automated continuous integration. |
| **Inputs / Context** | Card 1.2 (style standards); Card 1.3 (environment); understanding of GitHub Actions or similar CI. |
| **Deliverables** | Linting config (e.g., `ruff` or `flake8` for Python, `markdownlint` for markdown); notebook smoke-test script that executes notebooks and checks for errors; CI pipeline definition (`.github/workflows/ci.yml` or equivalent); `tooling/ci/README.md` documenting what CI checks. |
| **Acceptance Criteria** | CI runs on every PR; Python files pass linting; notebooks execute without uncaught exceptions (smoke test); markdown files pass format checks; CI status badge is in the root README. |
| **Dependencies** | Card 1.2, Card 1.3. |
| **Suggested Owner** | DevOps / engineering contributor. |
| **Notes / Risks** | Notebook execution in CI can be slow and resource-intensive. Consider a lightweight smoke test (cell execution with timeout) rather than full execution. GPU notebooks may need to be excluded or tested separately. |

---

# Epic 2 — Mathematical Foundations

---

## Card 2.1 — Write linear algebra primer

| Field | Detail |
|---|---|
| **Type** | Content |
| **Purpose** | Build a mathematically serious, ML-focused linear algebra foundation that serves as the reference for all later modules dealing with vector spaces, matrix decompositions, and geometric intuition. |
| **Inputs / Context** | Card 0.2 (pedagogical principles); Card 1.2 (style standards); standard references (Strang, Axler); understanding of which linear algebra topics appear most in ML (eigendecomposition, SVD, PCA, matrix calculus connections). |
| **Deliverables** | `modules/00-math-toolkit/notes/linear-algebra.md` covering vectors, matrices, inner products, norms, orthogonality, eigenvalues, eigenvectors, spectral decomposition, SVD, positive definiteness; `modules/00-math-toolkit/derivations/svd-derivation.md`; `modules/00-math-toolkit/exercises/linear-algebra-exercises.md` with theory and computation problems; `modules/00-math-toolkit/solutions/linear-algebra-solutions.md`. |
| **Acceptance Criteria** | Includes full derivations (not just statements) for eigendecomposition and SVD; uses both geometric and algebraic viewpoints; ML-relevant examples appear throughout (e.g., PCA as spectral decomposition, covariance matrices, data whitening); exercises span proof-style, computational, and ML-application tiers. |
| **Dependencies** | Card 1.1, Card 1.2. |
| **Suggested Owner** | Mathematics content author. |
| **Notes / Risks** | Temptation to be encyclopedic — scope should be "what an ML practitioner needs to derive and understand algorithms," not a full linear algebra textbook. |

---

## Card 2.2 — Write multivariable calculus primer

| Field | Detail |
|---|---|
| **Type** | Content |
| **Purpose** | Cover the calculus machinery needed for optimization and backpropagation: gradients, Jacobians, Hessians, Taylor expansions, and the multivariate chain rule. |
| **Inputs / Context** | Card 2.1 (linear algebra, since matrix calculus builds on it); Card 1.2 (style standards); standard references (Rudin for rigor, Boyd for optimization applications). |
| **Deliverables** | `modules/00-math-toolkit/notes/multivariable-calculus.md` covering partial derivatives, gradients, directional derivatives, Jacobians, Hessians, Taylor approximations, chain rule in matrix form, constrained optimization basics (Lagrange multipliers preview); `modules/00-math-toolkit/derivations/chain-rule-matrices.md`; `modules/00-math-toolkit/exercises/calculus-exercises.md`. |
| **Acceptance Criteria** | Chain rule is derived in enough generality to support backpropagation derivation in Module 06; Hessian treatment supports second-order optimization discussion in Module 01; Lagrange multiplier introduction sets up constrained optimization; exercises include at least two backprop-prerequisite problems. |
| **Dependencies** | Card 2.1. |
| **Suggested Owner** | Mathematics content author. |
| **Notes / Risks** | Students often have calculus background but not matrix calculus. Bridge carefully from scalar to vector to matrix derivatives. |

---

## Card 2.3 — Write probability and statistics primer

| Field | Detail |
|---|---|
| **Type** | Content |
| **Purpose** | Establish the statistical foundation for learning theory, probabilistic models, Bayesian inference, and information-theoretic losses. |
| **Inputs / Context** | Card 2.1 (linear algebra for covariance, multivariate distributions); Card 1.2 (style standards); standard references (Murphy, Bishop). |
| **Deliverables** | `modules/00-math-toolkit/notes/probability-statistics.md` covering probability spaces, random variables, PMFs/PDFs, expectation, variance, covariance, joint and conditional distributions, Bayes' theorem, common distributions (Gaussian, Bernoulli, Categorical, Poisson), maximum likelihood estimation, MAP estimation, conjugate priors introduction; `modules/00-math-toolkit/exercises/probability-exercises.md`. |
| **Acceptance Criteria** | MLE derivation for Gaussian and Bernoulli is worked out fully; Bayes' theorem is motivated with ML examples (e.g., spam classification); covariance and correlation are connected to PCA and feature analysis; exercises include both proof-style and applied problems. |
| **Dependencies** | Card 2.1, Card 2.2. |
| **Suggested Owner** | Mathematics / statistics content author. |
| **Notes / Risks** | Measure-theoretic probability is out of scope — keep it at the level needed for ML, with pointers to deeper references. |

---

## Card 2.4a — Write category theory primer for ML

| Field | Detail |
|---|---|
| **Type** | Content |
| **Purpose** | Introduce category theory inside the mathematical foundations as a usable structural language for ML, not merely a formal exercise. This is the "primer mode" — a first pass that seeds ideas formalized later in Module 16. |
| **Inputs / Context** | Card 0.3 (canonical-vs-speculative boundary); Card 2.1 (linear algebra provides concrete categories like Vect); references (Fong & Spivak, Spivak for sciences). |
| **Deliverables** | `modules/00-math-toolkit/notes/category-theory-primer.md` covering objects, morphisms, identity, composition, commutative diagrams, products, coproducts, universal properties (intuitive), functors, natural transformations, monoidal intuition; `modules/00-math-toolkit/notebooks/CT-01-categories-and-diagrams.ipynb` with at least five ML constructions as compositional diagrams; `modules/00-math-toolkit/exercises/category-theory-exercises.md`. |
| **Acceptance Criteria** | Uses standard mathematical examples (Set, Vect, Grp) before ML examples; shows at least five ML constructions as compositional diagrams (e.g., feature pipeline, training loop, encoder-decoder, data augmentation chain, transfer learning); avoids unnecessary abstraction while remaining mathematically correct; a reader with linear algebra and basic set theory can follow it. |
| **Dependencies** | Card 2.1, Card 0.3. |
| **Suggested Owner** | Category theory content author with ML background. |
| **Notes / Risks** | Highest risk card in Epic 2. If too abstract, learners disengage; if too informal, it won't support Module 16. Aim for "working language" — precise enough to be useful, concrete enough to be motivating. |

---

## Card 2.5 — Map canonical ML concepts into categorical language

| Field | Detail |
|---|---|
| **Type** | Research-content |
| **Purpose** | Build an explicit translation layer from standard ML concepts into category-theoretic language, distinguishing precise structural correspondences from loose analogies. |
| **Inputs / Context** | Card 2.4a (category theory primer); familiarity with datasets, hypothesis classes, feature maps, loss functions, optimizers, evaluation loops as standard ML constructions. |
| **Deliverables** | `modules/00-math-toolkit/notes/ml-categorical-mapping.md` containing a structured mapping table and extended discussion; glossary of correspondences and their limitations; concrete examples from regression, classification, and neural network training. |
| **Acceptance Criteria** | Every mapping explicitly distinguishes analogy from exact categorical equivalence; includes concrete worked examples from at least three ML paradigms (regression, classification, neural nets); limitations and caveats are stated, not hidden; reviewed against Card 0.3 boundary guidelines. |
| **Dependencies** | Card 2.4a, Card 0.3. |
| **Suggested Owner** | Category theory content author. |
| **Notes / Risks** | Many "categorical" readings of ML are informal analogies. This card must be disciplined about what is and isn't a formal functor, natural transformation, etc. Over-claiming weakens the entire companion layer. |

---

## Card 2.6 — Define Unity Theory interface in the primer

| Field | Detail |
|---|---|
| **Type** | Research-content |
| **Purpose** | Introduce a disciplined bridge from category theory to Unity Theory without collapsing standard mathematics into speculative language. This sets the tone for all later Unity Theory companion content. |
| **Inputs / Context** | Card 2.4a (category theory primer); Card 0.3 (boundary guidelines); Unity Theory concepts: identity, relation, embodiment, coherence, multiplicity, transformation. |
| **Deliverables** | `modules/00-math-toolkit/unity/unity-theory-primer-interface.md` covering identity and relation in state representation, multiplicity as decomposition into coordinates/bases/observables, learning as stabilization of intelligible structure under transformation; at least three worked examples tying basis choice, representation, and invariance to Unity Theory concepts. |
| **Acceptance Criteria** | Canonical math is cleanly separated from interpretive synthesis (per Card 0.3); provides at least three examples that genuinely clarify later ML material rather than merely relabeling it; a skeptical reader finds the interpretive notes thought-provoking rather than dismissible. |
| **Dependencies** | Card 2.4a, Card 0.3. |
| **Suggested Owner** | Unity Theory author with mathematical training. |
| **Notes / Risks** | If this card is poorly executed, it undermines the credibility of all later Unity Theory content. Quality gate: have a skeptical ML reader review before merging. |

---

## Card 2.4b — Write information theory primer

| Field | Detail |
|---|---|
| **Type** | Content |
| **Purpose** | Introduce entropy, KL divergence, cross-entropy, and mutual information — the information-theoretic tools that underpin loss functions, representation learning, and generative modeling. |
| **Inputs / Context** | Card 2.3 (probability, since information theory is built on distributions); Card 1.2 (style standards); references (MacKay, Cover & Thomas). |
| **Deliverables** | `modules/00-math-toolkit/notes/information-theory.md` covering Shannon entropy, differential entropy, KL divergence, cross-entropy, mutual information, data processing inequality intuition; `modules/00-math-toolkit/notebooks/IT-01-entropy-and-divergence.ipynb` with visualizations; `modules/00-math-toolkit/exercises/information-theory-exercises.md`. |
| **Acceptance Criteria** | Cross-entropy is connected explicitly to classification loss; KL divergence is connected to VAE objectives and regularization; mutual information is connected to representation learning; notebooks include visualizations of entropy and divergence for common distributions. |
| **Dependencies** | Card 2.3. |
| **Suggested Owner** | Mathematics content author. |
| **Notes / Risks** | Keep scope practical — this is not an information theory course. Focus on the tools that appear repeatedly in ML losses and objectives. |

---

# Epic 3 — Optimization Module

---

## Card 3.1 — Write convexity and optimization notes

| Field | Detail |
|---|---|
| **Type** | Content |
| **Purpose** | Cover convex sets, convex functions, stationary points, first and second order conditions, Lagrange multipliers, and duality intuition — the theoretical backbone of all optimization-based ML. |
| **Inputs / Context** | Card 2.1 (linear algebra for quadratic forms, PSD matrices); Card 2.2 (calculus for gradients, Hessians); references (Boyd & Vandenberghe). |
| **Deliverables** | `modules/01-optimization/notes/convexity-and-optimization.md`; `modules/01-optimization/derivations/kkt-conditions.md`; `modules/01-optimization/exercises/optimization-theory-exercises.md`. |
| **Acceptance Criteria** | Includes geometric interpretation of convexity with figures; first and second order optimality conditions are derived; Lagrange multiplier method is derived and connected to constrained ML problems (e.g., SVM margin maximization preview); at least one non-convex example motivates later discussion of loss landscapes. |
| **Dependencies** | Card 2.1, Card 2.2. |
| **Suggested Owner** | Mathematics / optimization content author. |
| **Notes / Risks** | Duality theory can absorb enormous time. Keep it at intuitive level here; deeper treatment can live in Card 5.1 (kernel methods) where it's directly applied. |

---

## Card 3.2 — Implement optimization lab notebooks

| Field | Detail |
|---|---|
| **Type** | Lab |
| **Purpose** | Build interactive notebooks where students can run, visualize, and compare optimization algorithms (GD, SGD, momentum, Nesterov, Newton, Adam) on controlled objective functions. |
| **Inputs / Context** | Card 3.1 (optimization theory); Card 1.3 (Python environment); Card 1.2 (notebook standards). |
| **Deliverables** | `modules/01-optimization/notebooks/OPT-01-gradient-descent.ipynb`; `modules/01-optimization/notebooks/OPT-02-stochastic-and-momentum.ipynb`; `modules/01-optimization/notebooks/OPT-03-second-order-methods.ipynb`; `modules/01-optimization/notebooks/OPT-04-adam-and-modern-optimizers.ipynb`; shared plotting utilities in `modules/01-optimization/src/`. |
| **Acceptance Criteria** | Students can visualize optimizer trajectories on 2D surfaces; convergence rate comparisons are empirically observable; each notebook includes at least one "what happens when..." failure-mode experiment (e.g., saddle points, ill-conditioning); notebooks run cleanly in the standard environment. |
| **Dependencies** | Card 3.1, Card 1.3. |
| **Suggested Owner** | Lab / notebook author. |
| **Notes / Risks** | Visualization quality matters here — optimization intuition is largely geometric. Invest in good contour plots and trajectory animations. |

---

## Card 3.3 — Add category and Unity Theory companion note for optimization

| Field | Detail |
|---|---|
| **Type** | Companion |
| **Purpose** | Provide a structural and philosophical companion to the optimization module: optimization as transformation over structured spaces (categorical), and informational action under constraints (Unity Theory). |
| **Inputs / Context** | Card 3.1 (optimization math); Card 2.4a (category theory primer); Card 2.6 (Unity Theory interface); Card 0.3 (boundary guidelines). |
| **Deliverables** | `modules/01-optimization/unity/optimization-companion.md` covering: optimization as endomorphism on parameter spaces; gradient flow as a structured process; convergence as stabilization; informational action interpretation; connections to later modules. |
| **Acceptance Criteria** | Stays disciplined and non-handwavy — every interpretive claim is grounded in the actual math from Card 3.1; boundary between standard and interpretive content is clearly marked; a reader who skips this note loses nothing essential from the optimization module. |
| **Dependencies** | Card 3.1, Card 2.4a, Card 2.6, Card 0.3. |
| **Suggested Owner** | Category theory / Unity Theory author. |
| **Notes / Risks** | Optimization is the first major technical module with a companion note. This sets the template for all later companions — invest in getting the tone and depth right. |

---

# Epic 4 — Statistical Learning and Linear Models

---

## Card 4.1 — Write statistical learning theory module

| Field | Detail |
|---|---|
| **Type** | Content |
| **Purpose** | Explain the theoretical framework for supervised learning: empirical risk minimization, true risk, generalization, overfitting, the bias-variance tradeoff, VC-style capacity intuition, and regularization. |
| **Inputs / Context** | Card 2.3 (probability); Card 3.1 (optimization, since ERM is an optimization problem); references (Hastie et al., Shalev-Shwartz & Ben-David). |
| **Deliverables** | `modules/02-statistical-learning/notes/statistical-learning-foundations.md`; `modules/02-statistical-learning/derivations/bias-variance-decomposition.md`; `modules/02-statistical-learning/notebooks/SL-01-bias-variance-demo.ipynb`; `modules/02-statistical-learning/exercises/statistical-learning-exercises.md`. |
| **Acceptance Criteria** | Bias-variance decomposition is derived completely for squared loss; VC dimension or Rademacher complexity intuition is presented (not full proof, but conceptual understanding); regularization is motivated from both optimization and Bayesian perspectives; cross-validation methodology is explained and demonstrated; overfitting is shown empirically in a notebook. |
| **Dependencies** | Card 2.3, Card 3.1. |
| **Suggested Owner** | ML theory content author. |
| **Notes / Risks** | VC theory can become a rabbit hole. Focus on intuition and practical implications rather than full measure-theoretic proofs. |

---

## Card 4.2 — Write linear regression module

| Field | Detail |
|---|---|
| **Type** | Content |
| **Purpose** | Cover least squares, normal equations, regularized regression (ridge, lasso), and the geometric/probabilistic viewpoints on linear regression. |
| **Inputs / Context** | Card 2.1 (linear algebra for normal equations, projections); Card 2.3 (probability for MLE interpretation); Card 4.1 (statistical learning framework). |
| **Deliverables** | `modules/03-linear-models/notes/linear-regression.md`; `modules/03-linear-models/derivations/normal-equations.md`; `modules/03-linear-models/derivations/ridge-lasso.md`; `modules/03-linear-models/notebooks/LM-01-linear-regression.ipynb`; `modules/03-linear-models/exercises/linear-regression-exercises.md`. |
| **Acceptance Criteria** | Normal equations derived from both optimization (minimize squared loss) and probabilistic (MLE under Gaussian noise) viewpoints; ridge regression connected to MAP estimation; lasso sparsity property explained geometrically; notebook demonstrates underfitting, good fit, and overfitting on synthetic data. |
| **Dependencies** | Card 2.1, Card 2.3, Card 4.1. |
| **Suggested Owner** | ML content author. |
| **Notes / Risks** | Linear regression is deceptively deep. Balance between "first ML algorithm students see" accessibility and full mathematical treatment. |

---

## Card 4.3 — Write logistic regression module

| Field | Detail |
|---|---|
| **Type** | Content |
| **Purpose** | Cover binary and multiclass classification via logistic regression: sigmoid/softmax, log-likelihood, cross-entropy loss, gradient derivation, and decision boundaries. |
| **Inputs / Context** | Card 4.2 (linear regression as foundation); Card 2.4b (information theory for cross-entropy connection); Card 2.2 (calculus for gradient derivation). |
| **Deliverables** | `modules/03-linear-models/notes/logistic-regression.md`; `modules/03-linear-models/derivations/logistic-gradient.md`; `modules/03-linear-models/notebooks/LM-02-logistic-regression.ipynb`; `modules/03-linear-models/exercises/logistic-regression-exercises.md`. |
| **Acceptance Criteria** | Sigmoid and softmax derived from exponential family / maximum entropy motivation; cross-entropy loss connected to KL divergence and information theory; gradient of log-likelihood derived step-by-step; notebook shows decision boundaries on 2D data; exercises include both derivation and implementation tasks. |
| **Dependencies** | Card 4.2, Card 2.4b. |
| **Suggested Owner** | ML content author. |
| **Notes / Risks** | GLM generalization can be mentioned but should not dominate — keep the focus on logistic regression as the canonical classification model. |

---

## Card 4.4 — Write model evaluation toolkit

| Field | Detail |
|---|---|
| **Type** | Content / Lab |
| **Purpose** | Standardize evaluation methodology and build reusable utilities for metrics, train/validation/test splits, calibration, ROC/PR curves, and confusion matrices that all later modules use. |
| **Inputs / Context** | Card 4.1 (statistical learning for generalization concepts); Card 4.2, Card 4.3 (first models to evaluate); Card 1.3 (environment). |
| **Deliverables** | `modules/02-statistical-learning/notes/model-evaluation.md`; `modules/02-statistical-learning/notebooks/SL-02-evaluation-toolkit.ipynb`; `shared/src/evaluation.py` with reusable metric functions; documentation for accuracy, precision, recall, F1, AUC-ROC, AUC-PR, calibration plots, confusion matrices. |
| **Acceptance Criteria** | Evaluation utilities are importable and tested; notebook demonstrates evaluation on at least two model types (regression and classification); ROC and PR curves are explained with worked examples; calibration is covered; utilities are used by at least one later module (forward dependency). |
| **Dependencies** | Card 4.1, Card 4.2. |
| **Suggested Owner** | Lab / engineering author. |
| **Notes / Risks** | This is infrastructure — invest in clean API design since many modules will import these utilities. |

---

# Epic 5 — Kernel Methods and Probabilistic Modeling

---

## Card 5.1 — Write kernel methods module

| Field | Detail |
|---|---|
| **Type** | Content |
| **Purpose** | Build the theory of feature maps, the kernel trick, SVMs, and RKHS intuition — the bridge from linear models to nonlinear learning and the mathematical gateway to understanding implicit feature spaces. |
| **Inputs / Context** | Card 2.1 (linear algebra, inner products); Card 3.1 (optimization, duality); Card 4.1 (statistical learning); references (Bishop Ch. 6-7, Scholkopf & Smola). |
| **Deliverables** | `modules/04-kernel-methods/notes/kernel-methods.md`; `modules/04-kernel-methods/derivations/kernel-trick.md`; `modules/04-kernel-methods/derivations/svm-dual.md`; `modules/04-kernel-methods/notebooks/KM-01-kernels-and-svms.ipynb`; `modules/04-kernel-methods/exercises/kernel-exercises.md`. |
| **Acceptance Criteria** | Kernel trick is derived from feature map inner products; SVM dual formulation is derived via Lagrangian; RKHS is introduced at an intuitive level (what it means for a kernel to define a space); notebook demonstrates kernel SVM on non-linearly-separable data with visualization; at least three kernel types compared empirically. |
| **Dependencies** | Card 2.1, Card 3.1, Card 4.1. |
| **Suggested Owner** | ML theory content author. |
| **Notes / Risks** | RKHS can be extremely deep. Keep it at the "what it buys you" level, with references for students who want the functional analysis. |

---

## Card 5.2 — Write probabilistic modeling module

| Field | Detail |
|---|---|
| **Type** | Content |
| **Purpose** | Cover Bayesian inference, latent variable models, the EM algorithm, and graphical models — the probabilistic toolbox that underlies VAEs, topic models, and principled uncertainty quantification. |
| **Inputs / Context** | Card 2.3 (probability); Card 2.4b (information theory for KL divergence in EM/variational inference); references (Bishop Ch. 8-9, Murphy). |
| **Deliverables** | `modules/05-probabilistic-modeling/notes/bayesian-inference.md`; `modules/05-probabilistic-modeling/notes/em-algorithm.md`; `modules/05-probabilistic-modeling/notes/graphical-models.md`; `modules/05-probabilistic-modeling/derivations/em-derivation.md`; `modules/05-probabilistic-modeling/notebooks/PM-01-naive-bayes.ipynb`; `modules/05-probabilistic-modeling/notebooks/PM-02-em-gmm.ipynb`; `modules/05-probabilistic-modeling/exercises/probabilistic-exercises.md`. |
| **Acceptance Criteria** | EM algorithm is derived from the ELBO perspective; GMM fitting is demonstrated with EM in notebook; Naive Bayes is implemented and evaluated; graphical models (directed and undirected) are introduced with at least two examples; exact vs. approximate inference distinction is clear. |
| **Dependencies** | Card 2.3, Card 2.4b. |
| **Suggested Owner** | Probabilistic ML content author. |
| **Notes / Risks** | Graphical models alone could fill a course. Keep scope to what supports later modules (VAEs, latent variable thinking). |

---

## Card 5.3 — Build comparison project: discriminative vs generative models

| Field | Detail |
|---|---|
| **Type** | Project |
| **Purpose** | Give students a structured project to compare modeling assumptions, training procedures, and performance between discriminative (logistic regression, SVM) and generative (Naive Bayes, GMM) approaches. |
| **Inputs / Context** | Card 4.3 (logistic regression); Card 5.1 (kernel methods / SVM); Card 5.2 (probabilistic models); Card 4.4 (evaluation toolkit). |
| **Deliverables** | `projects/beginner/discriminative-vs-generative/README.md` with project spec; `projects/beginner/discriminative-vs-generative/template.ipynb` guided notebook; `projects/beginner/discriminative-vs-generative/rubric.md` grading rubric. |
| **Acceptance Criteria** | Project is completable with knowledge from Modules 02–05 only; requires students to train, evaluate, and compare at least two discriminative and two generative models on the same dataset; rubric evaluates both implementation correctness and written analysis; reusable as an assessment artifact. |
| **Dependencies** | Card 4.3, Card 5.1, Card 5.2, Card 4.4. |
| **Suggested Owner** | Curriculum designer / project author. |
| **Notes / Risks** | Dataset choice matters — pick something where the generative/discriminative tradeoff is visible (e.g., text classification, medical diagnosis with missing features). |

---

# Epic 6 — Neural Networks and Deep Learning

---

## Card 6.1 — Write neural networks from first principles

| Field | Detail |
|---|---|
| **Type** | Content |
| **Purpose** | Derive multilayer perceptrons and backpropagation from scratch, building the theoretical foundation for all deep learning modules. |
| **Inputs / Context** | Card 2.2 (calculus, chain rule); Card 3.1 (optimization); Card 4.1 (statistical learning framework); references (Goodfellow et al. Ch. 6). |
| **Deliverables** | `modules/06-neural-networks/notes/neural-networks-first-principles.md` covering perceptron, MLP architecture, universal approximation intuition, activation functions and their properties, loss functions; `modules/06-neural-networks/derivations/backpropagation.md` with step-by-step derivation; `modules/06-neural-networks/notes/initialization-and-normalization.md`; `modules/06-neural-networks/exercises/neural-network-exercises.md`. |
| **Acceptance Criteria** | Backprop is derived step-by-step using the chain rule on a concrete 2-hidden-layer network before generalizing; universal approximation theorem is stated with geometric intuition (not full proof); at least four activation functions are compared with gradient properties; loss landscape concept is introduced with visualization references. |
| **Dependencies** | Card 2.2, Card 3.1, Card 4.1. |
| **Suggested Owner** | Deep learning content author. |
| **Notes / Risks** | Backprop derivation is the single most important derivation in the course. Get it reviewed carefully. Computational graph viewpoint should also be introduced. |

---

## Card 6.2 — Implement neural nets from scratch

| Field | Detail |
|---|---|
| **Type** | Lab |
| **Purpose** | Build NumPy-first implementations of forward pass, backward pass, and training loop so students see the mechanics before framework abstraction hides them. |
| **Inputs / Context** | Card 6.1 (neural network theory); Card 1.3 (environment); Card 1.2 (notebook standards). |
| **Deliverables** | `modules/06-neural-networks/notebooks/NN-01-forward-pass.ipynb`; `modules/06-neural-networks/notebooks/NN-02-backprop-from-scratch.ipynb`; `modules/06-neural-networks/notebooks/NN-03-training-loop.ipynb`; `modules/06-neural-networks/src/nn_from_scratch.py` minimal library. |
| **Acceptance Criteria** | Forward pass computes activations layer by layer with explicit matrix operations; backward pass computes gradients matching the derivation in Card 6.1; training loop trains a 2-layer network on MNIST or synthetic data to reasonable accuracy; all matrix shapes are annotated in comments; students can modify architecture (add layers, change activations) and observe effects. |
| **Dependencies** | Card 6.1, Card 1.3. |
| **Suggested Owner** | Lab / notebook author. |
| **Notes / Risks** | Numerical stability matters even in scratch implementations. Include gradient checking as a validation step. |

---

## Card 6.3 — Write deep learning systems module

| Field | Detail |
|---|---|
| **Type** | Content |
| **Purpose** | Cover the practical engineering of training deep networks: normalization, regularization, initialization strategies, residual connections, and scaling considerations. |
| **Inputs / Context** | Card 6.1 (neural network theory); Card 6.2 (scratch implementation gives intuition); Card 3.2 (optimization labs); references (Goodfellow et al. Ch. 7-8). |
| **Deliverables** | `modules/07-deep-learning-systems/notes/training-deep-networks.md` covering batch norm, layer norm, dropout, weight decay, residual connections, learning rate schedules, gradient clipping; `modules/07-deep-learning-systems/notebooks/DL-01-normalization-comparison.ipynb`; `modules/07-deep-learning-systems/notebooks/DL-02-regularization-effects.ipynb`; `modules/07-deep-learning-systems/notebooks/DL-03-residual-connections.ipynb`; `modules/07-deep-learning-systems/exercises/deep-learning-exercises.md`. |
| **Acceptance Criteria** | Batch norm and layer norm are derived mathematically and compared empirically; dropout is explained from both regularization and ensemble perspectives; residual connections are motivated by gradient flow analysis; at least one notebook demonstrates a training pathology (vanishing gradients, internal covariate shift) and its fix; PyTorch is the framework for all labs. |
| **Dependencies** | Card 6.1, Card 6.2, Card 3.2. |
| **Suggested Owner** | Deep learning content author. |
| **Notes / Risks** | Hardware and scaling considerations should be mentioned but kept brief — this is a theory/practice course, not a systems engineering course. |

---

## Card 6.4 — Build training diagnostics toolkit

| Field | Detail |
|---|---|
| **Type** | Engineering / Lab |
| **Purpose** | Create reusable utilities for monitoring and diagnosing neural network training: loss curves, gradient statistics, activation distributions, and confusion matrices. |
| **Inputs / Context** | Card 6.2 (scratch implementation); Card 6.3 (training pathologies to diagnose); Card 4.4 (evaluation toolkit). |
| **Deliverables** | `shared/src/training_diagnostics.py` with functions for loss curve plotting, gradient norm tracking, activation histogram plotting, learning rate visualization; `modules/07-deep-learning-systems/notebooks/DL-04-diagnostics-demo.ipynb`; documentation in `shared/src/README.md`. |
| **Acceptance Criteria** | Utilities are importable and tested; demo notebook shows diagnostics on a real training run; gradient norm tracking catches vanishing/exploding gradients; activation histograms show saturation; used by at least Modules 07, 08, and 10. |
| **Dependencies** | Card 6.2, Card 6.3, Card 4.4. |
| **Suggested Owner** | Engineering / tooling author. |
| **Notes / Risks** | Consider TensorBoard integration as optional enhancement. Keep the core toolkit framework-agnostic where possible. |

---

# Epic 7 — Vision, Sequences, Transformers, Generative AI

---

## Card 7.1 — Write CNN and vision module

| Field | Detail |
|---|---|
| **Type** | Content |
| **Purpose** | Explain convolutions as equivariant structure, pooling and spatial hierarchy, classical CNN architectures, transfer learning, and vision task formulations. |
| **Inputs / Context** | Card 6.1 (neural networks); Card 6.3 (deep learning systems); Card 2.1 (linear algebra for convolution as matrix operation); references (Goodfellow et al. Ch. 9). |
| **Deliverables** | `modules/08-cnn-vision/notes/convolutional-networks.md` covering 1D and 2D convolution, stride, padding, pooling, spatial hierarchy, parameter sharing, equivariance; `modules/08-cnn-vision/notes/cnn-architectures.md` covering LeNet, AlexNet, VGG, ResNet, brief modern mentions; `modules/08-cnn-vision/notebooks/CNN-01-convolution-from-scratch.ipynb`; `modules/08-cnn-vision/notebooks/CNN-02-image-classification.ipynb`; `modules/08-cnn-vision/notebooks/CNN-03-transfer-learning.ipynb`; `modules/08-cnn-vision/exercises/cnn-exercises.md`. |
| **Acceptance Criteria** | Convolution is explained both algebraically (matrix multiplication view) and structurally (equivariance, weight sharing); at least three architectures are compared with parameter counts and design rationale; transfer learning is demonstrated on a real task; notebooks include visualization of learned filters and feature maps. |
| **Dependencies** | Card 6.1, Card 6.3. |
| **Suggested Owner** | Vision / deep learning content author. |
| **Notes / Risks** | Architecture survey can become a history lesson. Focus on design principles (depth, residuals, efficiency) rather than exhaustive enumeration. |

---

## Card 7.2 — Write sequence modeling module

| Field | Detail |
|---|---|
| **Type** | Content |
| **Purpose** | Cover RNNs, LSTMs, GRUs, and the fundamental challenges of sequence modeling that motivate the transition to attention mechanisms. |
| **Inputs / Context** | Card 6.1 (neural networks); Card 6.3 (deep learning systems); Card 2.2 (chain rule for backprop through time). |
| **Deliverables** | `modules/09-sequence-models/notes/sequence-modeling.md` covering Markov assumptions, RNN architecture and unrolling, BPTT derivation, vanishing/exploding gradient problem, LSTM gates and memory cell, GRU simplification, teacher forcing, beam search; `modules/09-sequence-models/notebooks/SEQ-01-rnn-from-scratch.ipynb`; `modules/09-sequence-models/notebooks/SEQ-02-lstm-language-model.ipynb`; `modules/09-sequence-models/exercises/sequence-exercises.md`. |
| **Acceptance Criteria** | BPTT is derived as a special case of backprop; vanishing gradient problem is demonstrated empirically and explained mathematically; LSTM gate mechanism is derived with clear motivation for each gate; at least one sequence-to-sequence example is included; attention precursor intuition is planted. |
| **Dependencies** | Card 6.1, Card 6.3. |
| **Suggested Owner** | NLP / sequence content author. |
| **Notes / Risks** | RNNs are somewhat historical at this point but essential for understanding why transformers work. Frame as "understanding the problem that transformers solve." |

---

## Card 7.3 — Write transformer foundations module

| Field | Detail |
|---|---|
| **Type** | Content |
| **Purpose** | Derive the self-attention mechanism, positional encoding, encoder-decoder architecture, pretraining objectives, and the foundations of modern LLMs. |
| **Inputs / Context** | Card 7.2 (sequence models as motivation); Card 2.1 (linear algebra for attention as matrix operations); Card 2.4b (information theory for pretraining objectives); references (Vaswani et al., modern transformer surveys). |
| **Deliverables** | `modules/10-transformers-llms/notes/transformer-foundations.md` covering scaled dot-product attention derivation, multi-head attention, positional encoding (sinusoidal and learned), encoder-decoder structure, masked attention; `modules/10-transformers-llms/notes/pretraining-and-scaling.md` covering MLM, CLM, scaling laws intuition, fine-tuning, instruction tuning, RLHF overview; `modules/10-transformers-llms/derivations/self-attention.md`; `modules/10-transformers-llms/notebooks/TF-01-attention-from-scratch.ipynb`; `modules/10-transformers-llms/notebooks/TF-02-transformer-training.ipynb`; `modules/10-transformers-llms/exercises/transformer-exercises.md`. |
| **Acceptance Criteria** | Self-attention Q/K/V derivation is explicit with matrix dimensions annotated; positional encoding is motivated (why attention is permutation-invariant without it); scaling factor √d_k is derived; at least one notebook implements attention from scratch; pretraining objectives (MLM, CLM) are explained with concrete examples; scaling laws are presented as empirical findings with references. |
| **Dependencies** | Card 7.2, Card 2.1, Card 2.4b. |
| **Suggested Owner** | NLP / transformer content author. |
| **Notes / Risks** | This is likely the most high-demand module. Balance mathematical depth with practical relevance. RLHF is a rapidly evolving area — present the framework, note that specifics change. |

---

## Card 7.4 — Write generative modeling module

| Field | Detail |
|---|---|
| **Type** | Content |
| **Purpose** | Compare the four major paradigms of generative modeling: autoregressive models, VAEs, GANs, and diffusion models, with emphasis on their mathematical foundations and tradeoffs. |
| **Inputs / Context** | Card 5.2 (probabilistic modeling, EM, latent variables); Card 2.4b (information theory for ELBO, KL divergence); Card 6.3 (deep learning systems); references (Goodfellow et al. Ch. 20, Kingma & Welling, Ho et al.). |
| **Deliverables** | `modules/11-generative-models/notes/generative-models-overview.md`; `modules/11-generative-models/notes/vaes.md`; `modules/11-generative-models/notes/gans.md`; `modules/11-generative-models/notes/diffusion-models.md`; `modules/11-generative-models/derivations/elbo-derivation.md`; `modules/11-generative-models/derivations/gan-minimax.md`; `modules/11-generative-models/notebooks/GEN-01-vae-mnist.ipynb`; `modules/11-generative-models/notebooks/GEN-02-gan-training.ipynb`; `modules/11-generative-models/notebooks/GEN-03-diffusion-basics.ipynb`; `modules/11-generative-models/exercises/generative-exercises.md`. |
| **Acceptance Criteria** | ELBO is derived from scratch for VAEs; GAN minimax objective is derived with Nash equilibrium intuition; diffusion forward/reverse process is explained mathematically; likelihood, sample quality, and latent structure tradeoffs are compared across all four paradigms; at least one notebook per paradigm with working training loop. |
| **Dependencies** | Card 5.2, Card 2.4b, Card 6.3. |
| **Suggested Owner** | Generative modeling content author. |
| **Notes / Risks** | Diffusion models are mathematically rich — balance depth with accessibility. Score-based and DDPM viewpoints can both be mentioned but pick one as primary exposition. |

---

# Epic 8 — Reinforcement Learning, Graph Learning, Causality

---

## Card 8.1 — Write RL module

| Field | Detail |
|---|---|
| **Type** | Content |
| **Purpose** | Cover MDPs, Bellman equations, value-based methods, policy gradients, and actor-critic methods — the mathematical and algorithmic foundation of reinforcement learning. |
| **Inputs / Context** | Card 2.3 (probability); Card 3.1 (optimization); Card 6.1 (neural networks for deep RL); references (Sutton & Barto). |
| **Deliverables** | `modules/12-reinforcement-learning/notes/rl-foundations.md` covering MDPs, Bellman equations, dynamic programming, temporal difference, Q-learning, SARSA; `modules/12-reinforcement-learning/notes/policy-gradients.md` covering REINFORCE, actor-critic, A2C/PPO overview; `modules/12-reinforcement-learning/derivations/bellman-derivation.md`; `modules/12-reinforcement-learning/derivations/policy-gradient-theorem.md`; `modules/12-reinforcement-learning/notebooks/RL-01-gridworld.ipynb`; `modules/12-reinforcement-learning/notebooks/RL-02-cartpole-pg.ipynb`; `modules/12-reinforcement-learning/exercises/rl-exercises.md`. |
| **Acceptance Criteria** | Bellman equations derived for both V and Q functions; policy gradient theorem derived step-by-step; exploration vs. exploitation tradeoff is formalized; model-based vs. model-free distinction is clear; notebooks include at least one tabular and one deep RL environment; small environments (gridworld, CartPole) are used to keep compute tractable. |
| **Dependencies** | Card 2.3, Card 3.1, Card 6.1. |
| **Suggested Owner** | RL content author. |
| **Notes / Risks** | RL environments require additional dependencies (gymnasium). Ensure Card 1.3 environment handles this. Deep RL notebooks may need GPU but should have CPU-feasible alternatives. |

---

## Card 8.2 — Write graph learning module

| Field | Detail |
|---|---|
| **Type** | Content |
| **Purpose** | Introduce graph-structured learning: message passing, GNNs, spectral methods, and graph transformers, connecting to relational reasoning and structured data. |
| **Inputs / Context** | Card 2.1 (linear algebra for graph Laplacian, spectral methods); Card 6.1 (neural networks); Card 7.3 (transformers, since graph transformers build on attention); references (Bronstein et al., Hamilton). |
| **Deliverables** | `modules/13-graph-learning/notes/graph-learning.md` covering graph representations, adjacency and Laplacian matrices, message passing framework, GCN, GraphSAGE, GAT; `modules/13-graph-learning/notes/spectral-graph-methods.md`; `modules/13-graph-learning/notebooks/GL-01-message-passing.ipynb`; `modules/13-graph-learning/notebooks/GL-02-node-classification.ipynb`; `modules/13-graph-learning/exercises/graph-exercises.md`. |
| **Acceptance Criteria** | Message passing is formalized as a general framework before specific architectures; spectral methods connect graph Laplacian to convolution on graphs; GCN is derived as a spectral approximation; at least one notebook demonstrates node classification on a standard graph dataset; connection to relational structure and categorical composition is noted (forward reference to Module 16). |
| **Dependencies** | Card 2.1, Card 6.1, Card 7.3. |
| **Suggested Owner** | Graph ML content author. |
| **Notes / Risks** | Graph learning libraries (PyG, DGL) add environment complexity. Consider minimal implementations before framework-dependent notebooks. |

---

## Card 8.3 — Write causality and reasoning module

| Field | Detail |
|---|---|
| **Type** | Content |
| **Purpose** | Introduce structural causal models, interventions, counterfactuals, and the boundary between statistical learning and causal reasoning — and connect to reasoning architectures. |
| **Inputs / Context** | Card 2.3 (probability); Card 5.2 (graphical models as precursor); references (Pearl, Peters et al.). |
| **Deliverables** | `modules/14-causality-reasoning/notes/causal-inference.md` covering SCMs, do-calculus intuition, interventions vs. observations, counterfactuals, confounding; `modules/14-causality-reasoning/notes/reasoning-architectures.md` covering symbolic-neural hybrids, chain-of-thought, and reasoning system overview; `modules/14-causality-reasoning/exercises/causality-exercises.md`. |
| **Acceptance Criteria** | Correlation vs. causation is formalized, not just stated; SCMs are introduced with concrete examples (e.g., medical treatment, A/B testing); interventions are distinguished from conditioning mathematically; causal discovery limitations are discussed; reasoning architectures section is appropriately tentative given rapid evolution. |
| **Dependencies** | Card 2.3, Card 5.2. |
| **Suggested Owner** | Causal inference content author. |
| **Notes / Risks** | Causality is a deep field — scope tightly to what supports ML understanding. Reasoning architectures section will date quickly; frame as a snapshot with pointers. |

---

# Epic 9 — Category Theory Companion Layer

---

## Card 9.1 — Define category theory syllabus for ML

| Field | Detail |
|---|---|
| **Type** | Planning |
| **Purpose** | Decide exactly how much category theory is pedagogically justified in Module 16 — the "consolidation mode" that formalizes ideas seeded in the primer. |
| **Inputs / Context** | Card 2.4a (primer content — what was introduced); full course outline (what ML content exists to illuminate); Card 0.3 (boundary guidelines). |
| **Deliverables** | `modules/16-category-theory-for-ml/notes/scope-document.md` defining: which categorical concepts are included and why; which are excluded and why; mapping from each concept to the ML modules it illuminates; estimated page count and difficulty level. |
| **Acceptance Criteria** | Avoids abstraction overload — every concept has a concrete ML application; supports specific ML modules with forward/backward references; scope is achievable within a single module (not a category theory textbook); reviewed by both a category theorist and an ML practitioner. |
| **Dependencies** | Card 2.4a, Card 0.3, all content cards in Epics 2–8 (ideally). |
| **Suggested Owner** | Category theory content author with curriculum design input. |
| **Notes / Risks** | Temptation to include too much. The guiding question should be: "Does this categorical concept make an ML idea clearer or more composable?" If not, exclude it. |

---

## Card 9.2 — Write category theory consolidation module

| Field | Detail |
|---|---|
| **Type** | Content |
| **Purpose** | Write the formal category theory module that consolidates primer material and extends it to support structural reasoning about ML architectures, learning procedures, and compositional systems. |
| **Inputs / Context** | Card 9.1 (scoping); Card 2.4a (primer to build on); references (Fong & Spivak, Spivak, relevant ACT4ML papers). |
| **Deliverables** | `modules/16-category-theory-for-ml/notes/categories-functors-nts.md`; `modules/16-category-theory-for-ml/notes/products-limits-colimits.md`; `modules/16-category-theory-for-ml/notes/monoidal-categories.md`; `modules/16-category-theory-for-ml/notes/operads-intuition.md`; `modules/16-category-theory-for-ml/notes/diagrammatic-reasoning.md`; `modules/16-category-theory-for-ml/exercises/category-theory-exercises.md`. |
| **Acceptance Criteria** | Each concept is first defined formally, then connected to at least one concrete ML construction; case studies cover supervised learning, representation learning, message passing, attention, and agent pipelines; exercises test structural reasoning, not just definition recall; accessible to a student who completed the primer and all ML modules. |
| **Dependencies** | Card 9.1, Card 2.4a, Epics 4–8 content cards. |
| **Suggested Owner** | Category theory content author. |
| **Notes / Risks** | This module must be the payoff for the primer investment. If students don't feel it illuminates their ML knowledge, the two-stage design fails. |

---

## Card 9.3 — Map category theory concepts onto ML pipeline structure

| Field | Detail |
|---|---|
| **Type** | Companion |
| **Purpose** | Create a cross-module companion document showing how compositionality appears concretely in feature maps, architectures, training pipelines, and datasets throughout the course. |
| **Inputs / Context** | Card 9.2 (formal module); all ML content modules (Epics 3–8). |
| **Deliverables** | `modules/16-category-theory-for-ml/notes/ml-pipeline-categorical-view.md` containing: at least eight worked examples mapping real ML pipelines to categorical diagrams; comparison table of where the categorical language adds clarity vs. where it's merely alternative notation. |
| **Acceptance Criteria** | Uses concrete examples from actual course modules, not abstract toy examples; explicitly states where the categorical view adds insight and where it doesn't; diagrams are publication-quality and reusable; a skeptical ML practitioner finds at least half the examples genuinely useful. |
| **Dependencies** | Card 9.2, all Epic 3–8 content cards. |
| **Suggested Owner** | Category theory content author. |
| **Notes / Risks** | Honesty is essential — if a categorical view doesn't illuminate a particular ML concept, say so. Forced mappings undermine credibility. |

---

## Card 9.4 — Build diagram library for categorical views

| Field | Detail |
|---|---|
| **Type** | Design / Content |
| **Purpose** | Create a reusable library of commutative diagrams, string diagrams, and structural figures used across the primer and consolidation module, with a consistent notation guide. |
| **Inputs / Context** | Card 9.2 (module content to illustrate); Card 1.2 (style standards); diagramming tools (TikZ, quiver, tikz-cd). |
| **Deliverables** | `shared/figures/category-theory/` containing: reusable diagram source files; `shared/style-guides/diagram-notation.md` with notation conventions; rendered PNGs/SVGs for each diagram; at least 15 diagrams covering basic categories, functors, natural transformations, products, ML pipeline compositions. |
| **Acceptance Criteria** | Consistent notation across all diagrams; every diagram has both source and rendered versions; notation guide is complete enough for a new contributor to add diagrams in the same style; diagrams are referenced by file path in the relevant module notes. |
| **Dependencies** | Card 9.2, Card 1.2. |
| **Suggested Owner** | Design / technical illustration contributor. |
| **Notes / Risks** | Diagram tooling choice matters for maintainability. TikZ-cd is powerful but requires LaTeX. Consider a simpler alternative (Mermaid, quiver app) if contributors aren't LaTeX-native. |

---

# Epic 10 — Unity Theory Companion Layer

---

## Card 10.1 — Define acceptable scope for Unity Theory integration

| Field | Detail |
|---|---|
| **Type** | Planning |
| **Purpose** | Decide where Unity Theory adds genuine illumination rather than confusion, setting boundaries for all companion content in Module 17 and scattered `unity/` directories. |
| **Inputs / Context** | Card 0.3 (canonical-vs-speculative boundary); Card 2.6 (primer interface); full course outline. |
| **Deliverables** | `modules/17-unity-theory-perspectives/notes/scope-memo.md` defining: which ML topics benefit from Unity Theory interpretation; which should be left alone; criteria for inclusion (does it clarify, provoke productive thought, or connect ideas?); criteria for exclusion (does it merely relabel, confuse, or overreach?). |
| **Acceptance Criteria** | Standard and original claims are clearly separated; at least three topics are explicitly excluded with rationale; inclusion criteria are specific enough to resolve future content disputes; reviewed by someone skeptical of the framework. |
| **Dependencies** | Card 0.3, Card 2.6. |
| **Suggested Owner** | Unity Theory author with editorial support. |
| **Notes / Risks** | This is the governance card for the entire Unity Theory layer. If scope is too loose, the companion content undermines the course's ML credibility. |

---

## Card 10.2 — Write Unity Theory glossary for ML readers

| Field | Detail |
|---|---|
| **Type** | Content |
| **Purpose** | Define the core Unity Theory terms (identity, relation, embodiment, coherence, multiplicity, informational action) in a technically disciplined way that ML readers can engage with. |
| **Inputs / Context** | Card 10.1 (scope); Card 2.6 (primer interface where terms were first introduced); Unity Theory source material. |
| **Deliverables** | `modules/17-unity-theory-perspectives/notes/glossary.md` with: precise definitions for each core term; at least one ML-grounded example per term; explicit statements of what each term does NOT mean; cross-references to where each term appears in companion notes. |
| **Acceptance Criteria** | Terms are precise enough to be used consistently across all companion notes; an ML reader who has never encountered Unity Theory can understand each definition; no term is defined circularly or only by reference to other Unity Theory terms; glossary is usable as a standalone reference. |
| **Dependencies** | Card 10.1, Card 2.6. |
| **Suggested Owner** | Unity Theory author. |
| **Notes / Risks** | Glossary quality determines whether the companion layer is taken seriously. Each definition should survive scrutiny from a technical reader. |

---

## Card 10.3 — Write optimization and learning through Unity Theory

| Field | Detail |
|---|---|
| **Type** | Companion |
| **Purpose** | Interpret training, generalization, regularization, and latent structure through the Unity Theory framework, grounded in the actual math from the optimization and learning modules. |
| **Inputs / Context** | Card 10.1 (scope); Card 10.2 (glossary); Card 3.1 (optimization math); Card 4.1 (statistical learning); Card 6.1 (neural networks); Card 0.3 (boundary guidelines). |
| **Deliverables** | `modules/17-unity-theory-perspectives/notes/optimization-and-learning.md` covering: training as directed transformation, generalization as structural preservation, regularization as coherence constraint, loss landscapes as multiplicity resolution; cross-references to specific equations and results from Modules 01–06. |
| **Acceptance Criteria** | Every interpretive claim references a specific mathematical result from the course; canonical and interpretive content clearly separated; a reader who disagrees with Unity Theory still finds the essay thought-provoking; does not claim Unity Theory "explains" standard results — frames as interpretive companion. |
| **Dependencies** | Card 10.1, Card 10.2, Card 3.1, Card 4.1, Card 6.1. |
| **Suggested Owner** | Unity Theory author. |
| **Notes / Risks** | Most important companion essay. Sets the quality bar for Card 10.4 and any future companion notes. |

---

## Card 10.4 — Write agency and sequence through Unity Theory

| Field | Detail |
|---|---|
| **Type** | Companion |
| **Purpose** | Connect sequence modeling, memory, decision-making, and policy learning to Unity Theory concepts of identity persistence across transformation and coherent action under uncertainty. |
| **Inputs / Context** | Card 10.1 (scope); Card 10.2 (glossary); Card 7.2 (sequence models); Card 8.1 (RL); Card 0.3 (boundary guidelines). |
| **Deliverables** | `modules/17-unity-theory-perspectives/notes/agency-and-sequence.md` covering: memory and sequence as identity across time, policy as coherent action, exploration as multiplicity, reward as evaluative coherence; grounded in specific results from Modules 09 and 12. |
| **Acceptance Criteria** | Grounded in sequence modeling and RL module content with specific cross-references; agency discussion connects to MDP formalism, not just metaphor; identity-across-time interpretation is connected to hidden states and value functions; clearly labeled as interpretive throughout. |
| **Dependencies** | Card 10.1, Card 10.2, Card 7.2, Card 8.1. |
| **Suggested Owner** | Unity Theory author. |
| **Notes / Risks** | Agency is the most philosophically loaded topic. Extra care needed to avoid overreach. Frame as "an interpretive lens" not "the correct reading." |

---

# Epic 11 — Projects and Assessments

---

## Card 11.1 — Design beginner project track

| Field | Detail |
|---|---|
| **Type** | Curriculum |
| **Purpose** | Create small, self-contained projects for students completing Part I and Part II (linear models, optimization, classification) that reinforce theory through applied work. |
| **Inputs / Context** | Card 4.2 (linear regression); Card 4.3 (logistic regression); Card 4.4 (evaluation toolkit); Card 0.2 (assessment philosophy). |
| **Deliverables** | `projects/beginner/` containing 3–4 project directories, each with `README.md` (project spec), `template.ipynb` (guided notebook), `rubric.md` (grading criteria); example projects: housing price prediction, binary classification with evaluation, regularization comparison study. |
| **Acceptance Criteria** | Each project is completable with knowledge from Modules 00–05 only; projects range from guided (template notebook) to semi-open (spec with rubric); rubrics evaluate correctness, analysis quality, and written interpretation; estimated completion time documented (4–8 hours each). |
| **Dependencies** | Cards 4.2, 4.3, 4.4, Card 0.2. |
| **Suggested Owner** | Curriculum designer. |
| **Notes / Risks** | Beginner projects set the tone for the course. They should feel achievable but not trivial. Dataset selection is critical — avoid datasets with preprocessing challenges that obscure the learning objectives. |

---

## Card 11.2 — Design intermediate project track

| Field | Detail |
|---|---|
| **Type** | Curriculum |
| **Purpose** | Create projects integrating deep learning, vision, NLP, or generative modeling that require both implementation skill and analytical writing. |
| **Inputs / Context** | Cards 7.1–7.4 (deep learning content); Card 6.4 (diagnostics toolkit); Card 0.2 (assessment philosophy). |
| **Deliverables** | `projects/intermediate/` containing 3–4 project directories; example projects: image classification with CNN comparison, text generation with transformers, generative model comparison (VAE vs GAN), sequence prediction with diagnostics analysis. |
| **Acceptance Criteria** | Each project integrates theory and implementation; requires students to make and justify design decisions; rubrics evaluate both code quality and written analysis; projects are feasible on a single GPU (or CPU with reduced scale); estimated completion time documented (10–20 hours each). |
| **Dependencies** | Cards 7.1–7.4, Card 6.4. |
| **Suggested Owner** | Curriculum designer with DL expertise. |
| **Notes / Risks** | Compute requirements must be realistic. Include "reduced scale" options for students without GPU access. |

---

## Card 11.3 — Design advanced research track

| Field | Detail |
|---|---|
| **Type** | Curriculum |
| **Purpose** | Create open-ended projects suitable for research-oriented students, involving categorical structure analysis, interpretability studies, or Unity Theory synthesis essays. |
| **Inputs / Context** | Cards 9.1–9.3 (category theory module); Cards 10.1–10.4 (Unity Theory module); all ML content modules; Card 0.2 (assessment philosophy). |
| **Deliverables** | `projects/advanced/` and `projects/research/` containing 3–5 project prompts; example projects: categorical analysis of a novel architecture, interpretability study using structural lens, Unity Theory essay on a chosen ML topic, novel algorithm design using compositional principles, cross-domain transfer analysis. |
| **Acceptance Criteria** | Projects are genuinely open-ended with room for original contribution; prompts are structured enough to guide (problem statement, suggested readings, evaluation criteria) but not prescriptive; suitable for publication-style final submissions; rubric distinguishes technical rigor from speculative quality. |
| **Dependencies** | Cards 9.1–9.3, Cards 10.1–10.4, all content modules. |
| **Suggested Owner** | Research-track curriculum designer. |
| **Notes / Risks** | These projects may produce original work. Clarify IP and authorship expectations. Consider requiring advisor-style check-ins for research projects. |

---

## Card 11.4 — Write exercise bank and solutions policy

| Field | Detail |
|---|---|
| **Type** | Curriculum |
| **Purpose** | Standardize exercise difficulty tiers, exercise formats, and the solution release policy across all modules. |
| **Inputs / Context** | Card 0.2 (assessment philosophy); all exercise files created across Epics 2–8; Card 1.2 (style standards). |
| **Deliverables** | `shared/style-guides/exercise-taxonomy.md` defining difficulty tiers (e.g., foundational / intermediate / challenge), exercise types (proof, derivation, implementation, analysis, open-ended), tagging conventions; `shared/style-guides/solutions-policy.md` defining when solutions are released, instructor-only vs. public solutions, honor code expectations. |
| **Acceptance Criteria** | Every module's exercises can be classified under the taxonomy; taxonomy covers at least three difficulty tiers and four exercise types; solutions policy is clear about what is publicly available vs. instructor-gated; at least one exercise per module has been retroactively tagged as a consistency check. |
| **Dependencies** | Card 0.2, all exercise cards in Epics 2–8. |
| **Suggested Owner** | Curriculum designer. |
| **Notes / Risks** | Solutions policy affects the course's value as an educational resource. Too open = cheating risk; too closed = self-learners can't use it. Consider a two-tier approach. |

---

# Epic 12 — Publishing, Polish, and Distribution

---

## Card 12.1 — Build master syllabus and pacing guide

| Field | Detail |
|---|---|
| **Type** | Publishing |
| **Purpose** | Turn the module set into coherent study paths for three usage modes: single semester, two-semester sequence, and self-study. |
| **Inputs / Context** | Full module set (Epics 2–10); Card 0.1 (audience); Card 0.2 (pedagogical principles). |
| **Deliverables** | `syllabus/pacing-guide.md` with: single-semester path (core ML focus, companion layers optional), two-semester path (full depth), self-study path (suggested weekly schedule with time estimates); `syllabus/reading-list.md` updated and organized by module. |
| **Acceptance Criteria** | Each path is feasible and complete (no orphan modules); time estimates are realistic (validated against module content volume); prerequisite chains are respected in all paths; companion layers are clearly optional in shorter paths; reading list is organized by module with priority tiers. |
| **Dependencies** | All content modules (Epics 2–10). |
| **Suggested Owner** | Course architect. |
| **Notes / Risks** | Self-study path is hardest to get right — no instructor to adjust pacing. Consider embedded self-assessment checkpoints. |

---

## Card 12.2 — Create figure and notation consistency pass

| Field | Detail |
|---|---|
| **Type** | Editing |
| **Purpose** | Harmonize mathematical notation, symbol usage, figure style, and cross-references across all modules to eliminate inconsistencies. |
| **Inputs / Context** | All module content; Card 1.2 (style standards); Card 9.4 (diagram library). |
| **Deliverables** | `shared/style-guides/notation-registry.md` listing every mathematical symbol used, its definition, and the module where it's first introduced; figure consistency audit report; updated figures where inconsistencies are found. |
| **Acceptance Criteria** | No symbol is used with conflicting meanings across modules; notation registry is complete and searchable; all figures follow the same visual style (colors, fonts, line weights); cross-references between modules are bidirectional and correct. |
| **Dependencies** | All content modules, Card 9.4. |
| **Suggested Owner** | Technical editor / style reviewer. |
| **Notes / Risks** | This is tedious but essential. Common collisions: θ for both parameters and angles, x for both features and inputs at different levels. Schedule enough time. |

---

## Card 12.3 — Create learner onboarding guide

| Field | Detail |
|---|---|
| **Type** | Publishing |
| **Purpose** | Help new students install the environment, navigate the repository, understand the module structure, and begin studying without instructor intervention. |
| **Inputs / Context** | Card 1.1 (repo structure); Card 1.3 (environment setup); Card 12.1 (pacing guide); Card 0.1 (audience profile). |
| **Deliverables** | Updated root `README.md` with onboarding flow; `syllabus/onboarding.md` covering: prerequisite self-assessment, environment setup walkthrough, repo navigation guide, suggested first steps for each study path, FAQ. |
| **Acceptance Criteria** | A learner matching the target persona can go from git clone to running their first notebook in under 30 minutes; prerequisite self-assessment helps learners identify gaps before starting; FAQ addresses at least five common questions; tested by at least one person unfamiliar with the repo. |
| **Dependencies** | Card 1.1, Card 1.3, Card 12.1. |
| **Suggested Owner** | Technical writer / UX contributor. |
| **Notes / Risks** | First impression matters enormously for self-study learners. Test the onboarding flow on a fresh machine. |

---

## Card 12.4 — Prepare public release roadmap

| Field | Detail |
|---|---|
| **Type** | Publishing |
| **Purpose** | Define alpha, beta, and full release milestones with scope, dependencies, quality gates, and a timeline for public availability. |
| **Inputs / Context** | Milestone plan from Section 11 of the course plan; all content and engineering cards; Card 0.1 (vision). |
| **Deliverables** | `kanban/release-roadmap.md` containing: alpha release scope (Milestone A — foundation release), beta release scope (Milestones A–C), full release scope (all milestones); quality gates for each stage (CI passing, content review, onboarding tested, notation audit complete); dependency graph between milestones; estimated timeline or velocity-based projection. |
| **Acceptance Criteria** | Each release stage has clear entry and exit criteria; dependencies between milestones are mapped and realistic; quality gates are specific and testable (not "content is good enough"); roadmap is actionable enough to drive sprint planning. |
| **Dependencies** | All planning cards (Epic 0), engineering cards (Epic 1), understanding of full content scope. |
| **Suggested Owner** | Project manager / course architect. |
| **Notes / Risks** | Release roadmap should be a living document updated as content is built. Avoid over-commitment on timeline — content quality should not be sacrificed for schedule. |
