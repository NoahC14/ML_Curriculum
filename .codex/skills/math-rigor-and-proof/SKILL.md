---
name: math-rigor-and-proof
description: Use when writing or revising mathematically serious content: primers, derivations, definitions, theorem statements, proof sketches, optimization formulations, probabilistic arguments, or formal notes for the ML/AI course. Do not use for project management cards or repo scaffolding.
---

# Purpose
Ensure that mathematical material is clear, rigorous, teachable, and explicitly connected to machine learning practice.

# Typical outputs
This skill is the default fit for mathematically serious artifacts such as:
- module `notes/`
- `derivations/`
- theory-heavy `README.md` sections
- `exercises/` and `solutions/`
- formal companion notes that still need standard mathematical discipline

# Required structure
For mathematical content, prefer this order when appropriate:
1. motivation
2. assumptions and notation
3. definitions
4. proposition/theorem/result
5. derivation or proof sketch
6. worked example
7. ML interpretation
8. limitations or caveats
9. exercises

# Rigor rules
- Define every symbol before use.
- State domains and codomains when they matter.
- Include dimensions for matrices and vectors when useful.
- Show intermediate steps for nontrivial derivations.
- Distinguish population from sample quantities.
- Distinguish exact statements from heuristics.
- Avoid hand-waving language.

# Pedagogical rules
- Pair formal statements with intuition.
- Include at least one concrete example.
- Surface common misconceptions.
- When possible, connect the mathematics to an implementable algorithm.
- Make sure the artifact satisfies the prerequisite burden implied by the course sequence.

# Category theory rules
When category theory is present:
- define the objects and morphisms explicitly;
- state what composition means in the example;
- give a concrete ML instantiation; and
- do not assume abstract comfort without a bridge example.

# Unity Theory rules
When Unity Theory is referenced:
- label the section as interpretive, formal, or speculative;
- do not claim formal equivalence without an actual mapping; and
- separate philosophical vocabulary from mathematical statements.

# Final check
Ask:
- Could a strong student reproduce the argument?
- Are the assumptions stated?
- Is the relation to ML explicit?
- Is any speculative content clearly marked?
- Does the artifact fit cleanly into the planned module structure?
