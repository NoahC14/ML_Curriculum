---
title: "Matrix-Form Derivation of the Multivariate Chain Rule"
module: "00-math-toolkit"
lesson: "chain-rule-matrices"
doc_type: "derivation"
topic: "matrix-calculus"
status: "draft"
prerequisites:
  - "00-math-toolkit/linear-algebra"
  - "00-math-toolkit/multivariable-calculus"
updated: "2026-04-09"
owner: "curriculum-team"
tags:
  - "chain-rule"
  - "jacobian"
  - "matrix-calculus"
  - "backpropagation"
---

## Goal

Derive the multivariate chain rule in matrix form and extract the scalar-loss special case used in backpropagation.

## Setup and Notation

Let

$$
\mathbf{g} : \mathbb{R}^d \to \mathbb{R}^m,
\qquad
\mathbf{f} : \mathbb{R}^m \to \mathbb{R}^p,
$$

and define

$$
\mathbf{h} = \mathbf{f} \circ \mathbf{g} : \mathbb{R}^d \to \mathbb{R}^p.
$$

Write

$$
\mathbf{g}(\mathbf{x}) =
\begin{bmatrix}
g_1(\mathbf{x}) \\
\vdots \\
g_m(\mathbf{x})
\end{bmatrix},
\qquad
\mathbf{f}(\mathbf{y}) =
\begin{bmatrix}
f_1(\mathbf{y}) \\
\vdots \\
f_p(\mathbf{y})
\end{bmatrix}.
$$

Assume $\mathbf{g}$ is differentiable at $\mathbf{x}$ and $\mathbf{f}$ is differentiable at $\mathbf{g}(\mathbf{x})$.

Their Jacobians are

$$
\mathbf{J}_{\mathbf{g}}(\mathbf{x}) \in \mathbb{R}^{m \times d},
\qquad
\mathbf{J}_{\mathbf{f}}(\mathbf{g}(\mathbf{x})) \in \mathbb{R}^{p \times m}.
$$

## Coordinate Derivation

Fix an output coordinate $i \in \{1, \ldots, p\}$. Since

$$
h_i(\mathbf{x}) = f_i(g_1(\mathbf{x}), \ldots, g_m(\mathbf{x})),
$$

the scalar multivariable chain rule gives, for each input coordinate $j \in \{1, \ldots, d\}$,

$$
\frac{\partial h_i}{\partial x_j}(\mathbf{x})
=
\sum_{k=1}^m
\frac{\partial f_i}{\partial y_k}(\mathbf{g}(\mathbf{x}))
\frac{\partial g_k}{\partial x_j}(\mathbf{x}).
$$

This identity already has matrix-product structure:

- the factor $\frac{\partial f_i}{\partial y_k}$ comes from row $i$, column $k$ of $\mathbf{J}_{\mathbf{f}}(\mathbf{g}(\mathbf{x}))$;
- the factor $\frac{\partial g_k}{\partial x_j}$ comes from row $k$, column $j$ of $\mathbf{J}_{\mathbf{g}}(\mathbf{x})$.

Therefore

$$
\left[\mathbf{J}_{\mathbf{h}}(\mathbf{x})\right]_{ij}
=
\sum_{k=1}^m
\left[\mathbf{J}_{\mathbf{f}}(\mathbf{g}(\mathbf{x}))\right]_{ik}
\left[\mathbf{J}_{\mathbf{g}}(\mathbf{x})\right]_{kj}.
$$

But this is exactly the $(i,j)$ entry of the matrix product

$$
\mathbf{J}_{\mathbf{f}}(\mathbf{g}(\mathbf{x})) \mathbf{J}_{\mathbf{g}}(\mathbf{x}).
$$

Hence

$$
\boxed{
\mathbf{J}_{\mathbf{f} \circ \mathbf{g}}(\mathbf{x})
=
\mathbf{J}_{\mathbf{f}}(\mathbf{g}(\mathbf{x})) \mathbf{J}_{\mathbf{g}}(\mathbf{x})
}.
$$

## Linearization Derivation

The same result can be seen more conceptually from local linear approximations.

Because $\mathbf{g}$ is differentiable at $\mathbf{x}$,

$$
\mathbf{g}(\mathbf{x} + \mathbf{h})
=
\mathbf{g}(\mathbf{x})
+ \mathbf{J}_{\mathbf{g}}(\mathbf{x})\mathbf{h}
+ \mathbf{r}_g(\mathbf{h}),
$$

with

$$
\frac{\|\mathbf{r}_g(\mathbf{h})\|_2}{\|\mathbf{h}\|_2} \to 0.
$$

Because $\mathbf{f}$ is differentiable at $\mathbf{g}(\mathbf{x})$, for any perturbation $\mathbf{u} \in \mathbb{R}^m$,

$$
\mathbf{f}(\mathbf{g}(\mathbf{x}) + \mathbf{u})
=
\mathbf{f}(\mathbf{g}(\mathbf{x}))
+ \mathbf{J}_{\mathbf{f}}(\mathbf{g}(\mathbf{x}))\mathbf{u}
+ \mathbf{r}_f(\mathbf{u}),
$$

with

$$
\frac{\|\mathbf{r}_f(\mathbf{u})\|_2}{\|\mathbf{u}\|_2} \to 0.
$$

Now substitute

$$
\mathbf{u}
=
\mathbf{J}_{\mathbf{g}}(\mathbf{x})\mathbf{h}
+ \mathbf{r}_g(\mathbf{h}).
$$

Then

$$
\mathbf{f}(\mathbf{g}(\mathbf{x} + \mathbf{h}))
=
\mathbf{f}(\mathbf{g}(\mathbf{x}))
+ \mathbf{J}_{\mathbf{f}}(\mathbf{g}(\mathbf{x}))
\left(\mathbf{J}_{\mathbf{g}}(\mathbf{x})\mathbf{h} + \mathbf{r}_g(\mathbf{h})\right)
+ \mathbf{r}_f(\mathbf{u}).
$$

Rearranging,

$$
\mathbf{f}(\mathbf{g}(\mathbf{x} + \mathbf{h}))
=
\mathbf{f}(\mathbf{g}(\mathbf{x}))
+ \mathbf{J}_{\mathbf{f}}(\mathbf{g}(\mathbf{x}))\mathbf{J}_{\mathbf{g}}(\mathbf{x})\mathbf{h}
+ \tilde{\mathbf{r}}(\mathbf{h}),
$$

where

$$
\tilde{\mathbf{r}}(\mathbf{h})
=
\mathbf{J}_{\mathbf{f}}(\mathbf{g}(\mathbf{x}))\mathbf{r}_g(\mathbf{h})
+ \mathbf{r}_f(\mathbf{u}).
$$

The remainder term is still little-$o(\|\mathbf{h}\|_2)$, so the derivative of the composition is the product of the derivative maps:

$$
D(\mathbf{f} \circ \mathbf{g})(\mathbf{x})
=
D\mathbf{f}(\mathbf{g}(\mathbf{x})) \circ D\mathbf{g}(\mathbf{x}).
$$

In matrix coordinates, this is the Jacobian product formula above.

## Scalar-Loss Special Case

Now let $f : \mathbb{R}^m \to \mathbb{R}$ be scalar-valued and $\mathbf{g} : \mathbb{R}^d \to \mathbb{R}^m$. Define

$$
\ell(\mathbf{x}) = f(\mathbf{g}(\mathbf{x})).
$$

Since $f$ is scalar-valued, its Jacobian is a row vector:

$$
\mathbf{J}_f(\mathbf{g}(\mathbf{x})) =
\begin{bmatrix}
\frac{\partial f}{\partial y_1} & \cdots & \frac{\partial f}{\partial y_m}
\end{bmatrix}.
$$

Using the chain rule,

$$
\mathbf{J}_\ell(\mathbf{x})
=
\mathbf{J}_f(\mathbf{g}(\mathbf{x})) \mathbf{J}_{\mathbf{g}}(\mathbf{x}).
$$

Under the column-gradient convention,

$$
\nabla f(\mathbf{g}(\mathbf{x}))
=
\mathbf{J}_f(\mathbf{g}(\mathbf{x}))^\top,
\qquad
\nabla \ell(\mathbf{x})
=
\mathbf{J}_\ell(\mathbf{x})^\top.
$$

Taking transposes gives

$$
\boxed{
\nabla \ell(\mathbf{x})
=
\mathbf{J}_{\mathbf{g}}(\mathbf{x})^\top \nabla f(\mathbf{g}(\mathbf{x}))
}.
$$

This is the exact form used when sensitivities are propagated from outputs back to inputs.

## Backpropagation Template

Consider a two-stage computation

$$
\mathbf{x}
\xrightarrow{\ \mathbf{g}\ }
\mathbf{z}
\xrightarrow{\ f\ }
\ell.
$$

Let

$$
\boldsymbol{\delta}_z = \nabla_{\mathbf{z}} f(\mathbf{z}) \in \mathbb{R}^m.
$$

Then the gradient with respect to the input is

$$
\nabla_{\mathbf{x}} \ell
=
\mathbf{J}_{\mathbf{g}}(\mathbf{x})^\top \boldsymbol{\delta}_z.
$$

If $\mathbf{g}$ itself is a composition, the same pattern repeats. For

$$
\mathbf{x}
\mapsto
\mathbf{z}^{(1)}
\mapsto
\mathbf{z}^{(2)}
\mapsto \cdots \mapsto
\ell,
$$

each backward step multiplies by the transpose of the local Jacobian. This is the algebraic core of reverse-mode differentiation.

## Worked Example: Affine Layer Plus Nonlinearity

Let

$$
\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b},
\qquad
\mathbf{a} = \phi(\mathbf{z}),
\qquad
\ell = f(\mathbf{a}),
$$

where $\phi$ acts coordinatewise. Let

$$
\boldsymbol{\delta}_a = \nabla_{\mathbf{a}} f(\mathbf{a}).
$$

Because $\phi$ is coordinatewise, its Jacobian is diagonal:

$$
\mathbf{J}_{\phi}(\mathbf{z})
=
\mathrm{diag}\left(\phi'(z_1), \ldots, \phi'(z_m)\right).
$$

Therefore

$$
\boldsymbol{\delta}_z
=
\nabla_{\mathbf{z}} \ell
=
\mathbf{J}_{\phi}(\mathbf{z})^\top \boldsymbol{\delta}_a
=
\mathbf{J}_{\phi}(\mathbf{z}) \boldsymbol{\delta}_a.
$$

Since $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$, the Jacobian with respect to $\mathbf{x}$ is $\mathbf{W}$. Hence

$$
\nabla_{\mathbf{x}} \ell
=
\mathbf{W}^\top \boldsymbol{\delta}_z.
$$

For the bias,

$$
\nabla_{\mathbf{b}} \ell = \boldsymbol{\delta}_z.
$$

For the weight matrix,

$$
\frac{\partial \ell}{\partial W_{ij}}
=
\frac{\partial \ell}{\partial z_i}\frac{\partial z_i}{\partial W_{ij}}
=
(\boldsymbol{\delta}_z)_i x_j,
$$

so

$$
\nabla_{\mathbf{W}} \ell
=
\boldsymbol{\delta}_z \mathbf{x}^\top.
$$

This is the layerwise derivative pattern reused throughout neural networks.

## Why Shape Checking Matters

The chain rule is often remembered symbolically but misapplied dimensionally. A useful discipline is:

1. state the domain and codomain of each map;
2. write the Jacobian shape explicitly;
3. verify that each matrix product is dimensionally valid.

For the vector-valued chain rule,

$$
(p \times m)(m \times d) = p \times d,
$$

which matches the expected shape of $\mathbf{J}_{\mathbf{f} \circ \mathbf{g}}(\mathbf{x})$.

For the scalar-loss case,

$$
(d \times m)(m \times 1) = d \times 1,
$$

which matches the expected shape of a gradient in $\mathbb{R}^d$.

## Result

The multivariate chain rule is the statement that derivatives compose as linear maps:

$$
\mathbf{J}_{\mathbf{f} \circ \mathbf{g}}(\mathbf{x})
=
\mathbf{J}_{\mathbf{f}}(\mathbf{g}(\mathbf{x})) \mathbf{J}_{\mathbf{g}}(\mathbf{x}).
$$

For scalar losses, this becomes

$$
\nabla (f \circ \mathbf{g})(\mathbf{x})
=
\mathbf{J}_{\mathbf{g}}(\mathbf{x})^\top \nabla f(\mathbf{g}(\mathbf{x})),
$$

which is precisely the algebraic rule that supports backpropagation.

## References

- Boyd, S., and Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
- Magnus, J. R., and Neudecker, H. (1999). *Matrix Differential Calculus with Applications in Statistics and Econometrics*. Wiley.
- Petersen, K. B., and Pedersen, M. S. (2012). *The Matrix Cookbook*.
