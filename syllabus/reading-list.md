# Reading List

## Purpose
This reading list is organized by module and uses three priority tiers:
- `Core`: default required reading for that module;
- `Recommended`: strong supporting text for depth, alternate exposition, or worked examples;
- `Companion`: structural or interpretive material that is explicitly optional unless the pacing guide says otherwise.

The list preserves the canonical-first posture of the course. Category theory and Unity Theory readings appear only where they clarify structure or serve the dedicated companion modules.

## Module 00: Mathematical Toolkit for ML
- `Core`: Gilbert Strang, *Introduction to Linear Algebra*
- `Core`: Joseph K. Blitzstein and Jessica Hwang, *Introduction to Probability*
- `Core`: David J. C. MacKay, *Information Theory, Inference, and Learning Algorithms*
- `Recommended`: Deisenroth, Faisal, and Ong, *Mathematics for Machine Learning*
- `Recommended`: Boyd and Vandenberghe, *Introduction to Applied Linear Algebra*
- `Companion`: Fong and Spivak, *An Invitation to Applied Category Theory* chapters on compositional thinking

## Module 01: Optimization
- `Core`: Stephen Boyd and Lieven Vandenberghe, *Convex Optimization*
- `Recommended`: Nocedal and Wright, *Numerical Optimization*
- `Recommended`: Bottou, Curtis, and Nocedal, "Optimization Methods for Large-Scale Machine Learning"
- `Companion`: repo note `modules/01-optimization/unity/optimization-companion.md`

## Module 02: Statistical Learning Foundations
- `Core`: Hastie, Tibshirani, and Friedman, *The Elements of Statistical Learning*
- `Core`: James, Witten, Hastie, and Tibshirani, *An Introduction to Statistical Learning*
- `Recommended`: Bishop, *Pattern Recognition and Machine Learning*
- `Recommended`: Shalev-Shwartz and Ben-David, *Understanding Machine Learning*

## Module 03: Linear Models
- `Core`: Hastie, Tibshirani, and Friedman, *The Elements of Statistical Learning* chapters on linear methods
- `Core`: Murphy, *Probabilistic Machine Learning: An Introduction* sections on regression and classification
- `Recommended`: Bishop, *Pattern Recognition and Machine Learning* chapters on linear and logistic regression
- `Recommended`: Montgomery, Peck, and Vining, *Introduction to Linear Regression Analysis*

## Module 04: Kernel Methods and Margin-Based Learning
- `Core`: Hastie, Tibshirani, and Friedman, *The Elements of Statistical Learning* chapters on kernels and SVMs
- `Recommended`: Scholkopf and Smola, *Learning with Kernels*
- `Recommended`: Cristianini and Shawe-Taylor, *An Introduction to Support Vector Machines*

## Module 05: Probabilistic Modeling
- `Core`: Bishop, *Pattern Recognition and Machine Learning*
- `Core`: Murphy, *Probabilistic Machine Learning: An Introduction*
- `Recommended`: Koller and Friedman, *Probabilistic Graphical Models*
- `Recommended`: Barber, *Bayesian Reasoning and Machine Learning*

## Module 06: Neural Networks from First Principles
- `Core`: Goodfellow, Bengio, and Courville, *Deep Learning* chapters on feedforward nets and backpropagation
- `Recommended`: Nielsen, *Neural Networks and Deep Learning*
- `Recommended`: repo derivation `modules/06-neural-networks/derivations/backpropagation.md`

## Module 07: Deep Learning Systems
- `Core`: Goodfellow, Bengio, and Courville, *Deep Learning* chapters on regularization and optimization
- `Recommended`: Zhang et al., "Why Deep Learning Works: A View From the Loss Landscape"
- `Recommended`: Kaplan et al., "Scaling Laws for Neural Language Models"
- `Recommended`: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"

## Module 08: Convolutional Neural Networks and Vision
- `Core`: Goodfellow, Bengio, and Courville, *Deep Learning* chapter on convolutional networks
- `Recommended`: cs231n convolutional-network notes
- `Recommended`: He et al., "Deep Residual Learning for Image Recognition"

## Module 09: Sequence Models
- `Core`: Goodfellow, Bengio, and Courville, *Deep Learning* chapter on sequence modeling
- `Recommended`: Jurafsky and Martin, *Speech and Language Processing* sections on RNNs, LSTMs, and sequence modeling
- `Recommended`: repo note `modules/09-sequence-models/notes/sequence-modeling.md`

## Module 10: Transformers and LLM Foundations
- `Core`: Vaswani et al., "Attention Is All You Need"
- `Core`: Jurafsky and Martin, *Speech and Language Processing* transformer and language-model sections
- `Recommended`: Stanford CS25 or equivalent lecture notes on transformers and scaling
- `Recommended`: repo derivation `modules/10-transformers-llms/derivations/self-attention.md`
- `Companion`: selected alignment and RLHF overview readings as context, not as prerequisite theory

## Module 11: Generative Models
- `Core`: Goodfellow, Bengio, and Courville, *Deep Learning* sections on latent-variable and generative models
- `Core`: Kingma and Welling, "Auto-Encoding Variational Bayes"
- `Core`: Ho, Jain, and Abbeel, "Denoising Diffusion Probabilistic Models"
- `Recommended`: Goodfellow et al., "Generative Adversarial Nets"
- `Recommended`: Murphy, *Probabilistic Machine Learning: Advanced Topics* sections on generative modeling

## Module 12: Reinforcement Learning
- `Core`: Sutton and Barto, *Reinforcement Learning: An Introduction*
- `Recommended`: Szepesvari, *Algorithms for Reinforcement Learning*
- `Recommended`: repo derivations `modules/12-reinforcement-learning/derivations/bellman-derivation.md` and `modules/12-reinforcement-learning/derivations/policy-gradient-theorem.md`

## Module 13: Graph Learning
- `Core`: Hamilton, *Graph Representation Learning*
- `Recommended`: Kipf and Welling, "Semi-Supervised Classification with Graph Convolutional Networks"
- `Recommended`: Bronstein et al., "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges"

## Module 14: Causality and Reasoning
- `Core`: Pearl, Glymour, and Jewell, *Causal Inference in Statistics: A Primer*
- `Core`: Pearl, *Causality* selected chapters
- `Recommended`: Peters, Janzing, and Schoelkopf, *Elements of Causal Inference*
- `Recommended`: Russell and Norvig, *Artificial Intelligence: A Modern Approach* sections on reasoning and symbolic methods

## Module 15: Ethics, Safety, and Evaluation
- `Core`: Barocas, Hardt, and Narayanan, *Fairness and Machine Learning*
- `Core`: Mitchell, *Artificial Intelligence: A Guide for Thinking Humans* selected evaluation and deployment chapters
- `Recommended`: NIST AI Risk Management Framework
- `Recommended`: model evaluation and robustness papers assigned with the module exercises

## Module 16: Category Theory for Machine Learning
- `Core`: Spivak, *Category Theory for the Sciences*
- `Core`: Fong and Spivak, *An Invitation to Applied Category Theory*
- `Recommended`: Lawvere and Schanuel, *Conceptual Mathematics*
- `Recommended`: Awodey, *Category Theory*
- `Recommended`: repo notes in `modules/16-category-theory-for-ml/notes/`

## Module 17: Unity Theory Perspectives on AI and Learning
- `Core`: repo notes in `modules/17-unity-theory-perspectives/notes/`
- `Recommended`: the linked companion essays cross-referenced from Modules `00`, `01`, `09`, and `16`
- `Companion`: any additional Unity Theory material remains interpretive, exploratory, and non-canonical by default

## Cross-module reference spine
These texts recur across multiple modules and are worth owning or keeping close:
- Strang for linear algebra refresh and notation discipline
- Boyd and Vandenberghe for optimization
- Hastie, Tibshirani, and Friedman for classical ML
- Bishop and Murphy for probabilistic framing
- Goodfellow, Bengio, and Courville for deep learning
- Sutton and Barto for RL
- Spivak and Fong-Spivak for category-theory companion work

## Usage notes
- In shorter formats, assign only `Core` items and keep `Companion` optional.
- In the two-semester sequence, use `Recommended` items for student presentations, reading responses, or project literature reviews.
- Module-local `references/` directories should eventually mirror this syllabus-level spine with more exact chapter and paper assignments.
