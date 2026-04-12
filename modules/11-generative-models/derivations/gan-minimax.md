---
title: "GAN Minimax Objective and Equilibrium"
module: "11-generative-models"
lesson: "gan-minimax"
doc_type: "derivation"
topic: "gan-minimax"
status: "draft"
prerequisites:
  - "00-math-toolkit/probability"
  - "01-optimization/convexity-and-optimization"
  - "11-generative-models/notes/gans"
updated: "2026-04-12"
owner: "curriculum-team"
tags:
  - "generative-models"
  - "gan"
  - "minimax"
  - "nash-equilibrium"
  - "js-divergence"
---

## Purpose

This derivation explains the original GAN objective, computes the optimal discriminator for a fixed generator, and connects the resulting game to a Nash-equilibrium interpretation.

## 1. Setup

Let:

- $p_{\mathrm{data}}$ be the data distribution;
- $\varepsilon \sim p(\varepsilon)$ be a simple noise variable;
- $x = G_\theta(\varepsilon)$ induce the generator distribution $p_\theta$;
- $D_\psi(x) \in (0,1)$ be the discriminator.

The original GAN value function is

$$
V(\psi,\theta)
=
\mathbb{E}_{x\sim p_{\mathrm{data}}}[\log D_\psi(x)]
+
\mathbb{E}_{x\sim p_\theta}[\log(1-D_\psi(x))].
$$

The optimization problem is

$$
\min_\theta \max_\psi V(\psi,\theta).
$$

## 2. Discriminator optimization for fixed generator

Fix $\theta$, so $p_\theta$ is fixed.
Write the objective as an integral:

$$
V(\psi,\theta)
=
\int p_{\mathrm{data}}(x)\log D_\psi(x)\,dx
+
\int p_\theta(x)\log(1-D_\psi(x))\,dx.
$$

For each point $x$, define

$$
a = p_{\mathrm{data}}(x),
\qquad
b = p_\theta(x),
\qquad
d = D_\psi(x).
$$

Then the integrand becomes

$$
f(d)=a\log d + b\log(1-d),
\qquad 0<d<1.
$$

To maximize $f(d)$, differentiate:

$$
f'(d)=\frac{a}{d}-\frac{b}{1-d}.
$$

Set the derivative to zero:

$$
\frac{a}{d}=\frac{b}{1-d}.
$$

Cross-multiplying gives

$$
a(1-d)=bd.
$$

Expand and collect terms:

$$
a = (a+b)d.
$$

Therefore the maximizer is

$$
d^*=\frac{a}{a+b}
=
\frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x)+p_\theta(x)}.
$$

Since

$$
f''(d)=-\frac{a}{d^2}-\frac{b}{(1-d)^2}<0,
$$

this critical point is indeed a maximum.

Thus the optimal discriminator is

$$
D^*(x)=\frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x)+p_\theta(x)}.
$$

## 3. Substitute the optimal discriminator

Plug $D^*$ back into the value function:

$$
V(D^*,\theta)
=
\int p_{\mathrm{data}}(x)
\log
\frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x)+p_\theta(x)}
\,dx
+
\int p_\theta(x)
\log
\frac{p_\theta(x)}{p_{\mathrm{data}}(x)+p_\theta(x)}
\,dx.
$$

Define the mixture distribution

$$
m(x)=\frac{1}{2}\bigl(p_{\mathrm{data}}(x)+p_\theta(x)\bigr).
$$

Then

$$
\frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x)+p_\theta(x)}
=
\frac{1}{2}\frac{p_{\mathrm{data}}(x)}{m(x)},
$$

and similarly

$$
\frac{p_\theta(x)}{p_{\mathrm{data}}(x)+p_\theta(x)}
=
\frac{1}{2}\frac{p_\theta(x)}{m(x)}.
$$

So

$$
V(D^*,\theta)
=
\int p_{\mathrm{data}}(x)\log \frac{1}{2}\frac{p_{\mathrm{data}}(x)}{m(x)}\,dx
+
\int p_\theta(x)\log \frac{1}{2}\frac{p_\theta(x)}{m(x)}\,dx.
$$

Separate the $\log \frac{1}{2}$ terms:

$$
V(D^*,\theta)
=
-\log 4
+
\int p_{\mathrm{data}}(x)\log \frac{p_{\mathrm{data}}(x)}{m(x)}\,dx
+
\int p_\theta(x)\log \frac{p_\theta(x)}{m(x)}\,dx.
$$

Recognize the two KL divergences:

$$
D_{\mathrm{KL}}(p_{\mathrm{data}}\|m)
=
\int p_{\mathrm{data}}(x)\log \frac{p_{\mathrm{data}}(x)}{m(x)}\,dx,
$$

$$
D_{\mathrm{KL}}(p_\theta\|m)
=
\int p_\theta(x)\log \frac{p_\theta(x)}{m(x)}\,dx.
$$

The Jensen-Shannon divergence is

$$
D_{\mathrm{JS}}(p_{\mathrm{data}}\|p_\theta)
=
\frac{1}{2}D_{\mathrm{KL}}(p_{\mathrm{data}}\|m)
+
\frac{1}{2}D_{\mathrm{KL}}(p_\theta\|m).
$$

Therefore

$$
V(D^*,\theta)
=
-\log 4 + 2D_{\mathrm{JS}}(p_{\mathrm{data}}\|p_\theta).
$$

## 4. Generator optimum

Because $D_{\mathrm{JS}}(\cdot\|\cdot)\geq 0$, we have

$$
V(D^*,\theta)\geq -\log 4.
$$

Equality holds exactly when

$$
p_\theta = p_{\mathrm{data}}.
$$

At that point,

$$
D^*(x)=\frac{1}{2}
$$

for all $x$ on the shared support.

So the ideal generator minimizes the divergence between the generated and data distributions, while the ideal discriminator becomes maximally uncertain.

## 5. Nash-equilibrium intuition

The GAN game is a two-player game with payoff $V$:

- the discriminator wants to maximize $V$;
- the generator wants to minimize $V$.

An equilibrium is a pair $(\theta^*,\psi^*)$ such that neither player can improve its own objective by changing strategy unilaterally.

In the idealized GAN analysis:

- if $p_{\theta^*}=p_{\mathrm{data}}$, the best discriminator is $D_{\psi^*}(x)=1/2$;
- if the discriminator is already optimal and equal to $1/2$ everywhere, the generator cannot reduce the value below $-\log 4$.

This is the basic Nash-equilibrium picture.

## 6. Why training is hard in practice

The theoretical result assumes:

- the discriminator can reach its pointwise optimum;
- the generator can represent the target distribution;
- optimization reaches the equilibrium.

In practice, neither network is optimized exactly at each step.
The learning dynamics are coupled, nonconvex, and can oscillate.
That is why the equilibrium intuition is useful conceptually but insufficient as an optimization guarantee.

## 7. Non-saturating alternative

The original minimax generator objective may produce weak gradients early in training.
So practitioners often replace it with

$$
\min_\theta
-\mathbb{E}_{\varepsilon\sim p(\varepsilon)}
\left[
\log D_\psi(G_\theta(\varepsilon))
\right].
$$

This does not change the goal of increasing discriminator confusion, but it often improves the gradient signal.

## 8. Summary

The GAN minimax objective leads to an optimal discriminator that estimates

$$
\frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x)+p_\theta(x)}.
$$

After substituting this optimum, the generator effectively minimizes a Jensen-Shannon divergence.
The target equilibrium is $p_\theta=p_{\mathrm{data}}$ with discriminator output $1/2$ everywhere.
This explains both the elegance of GAN theory and the fragility of GAN optimization.
