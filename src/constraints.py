"""
src/constraints.py
────────────────────
Constraint builders for the constrained-NLP formulation. Each function
returns (g_expr, lb, ub) -- a CasADi expression plus its numeric
bounds, in the format `casadi.nlpsol` expects for `g`, `lbg`, `ubg`.

These three constraints are the actual subject of the project: the
neural network is just the parametrized function they are imposed on.
"""

import numpy as np
import casadi as ca


def lipschitz_constraint(W1, W2, L_max):
    """
    Upper-bounds the network's Lipschitz constant.

    For a 1-hidden-layer tanh network, ||f(x)-f(x')|| <= ||W2||_2 * ||W1||_2 * ||x-x'||
    (tanh is 1-Lipschitz, so it does not add to the bound). We use the
    Frobenius norm as a tractable upper bound on the spectral norm:
    ||W||_2 <= ||W||_F.

    Constraint:  ||W1||_F^2 * ||W2||_F^2  <=  L_max^2

    This is bilinear in (||W1||_F^2, ||W2||_F^2) -- a product of two
    convex terms, hence a nonconvex constraint.
    """
    g = ca.sumsqr(W1) * ca.sumsqr(W2)
    return g, -np.inf, float(L_max ** 2)


def norm_ball_constraint(w, B_max):
    """
    Hard L2 norm-ball on all weights (the constrained analogue of L2
    regularization, enforced exactly rather than penalized).

    Constraint: ||w||_2^2 <= B_max^2
    """
    g = ca.sumsqr(w)
    return g, -np.inf, float(B_max ** 2)


def symmetry_breaking_constraints(b1):
    """
    Hidden units of a 1-layer network can be freely permuted (relabeled)
    without changing the function they represent -- every permutation
    is an equally good, equivalent local minimum. Forcing the hidden
    biases into ascending order removes this combinatorial symmetry
    without restricting the function class the network can represent.

        b1[0] <= b1[1] <= ... <= b1[H-1]

    Returns (g, lb, ub) for H-1 linear constraints, or (None, None, None)
    if there are fewer than 2 hidden units to order.
    """
    H = b1.shape[0]
    if H < 2:
        return None, None, None
    g = ca.vertcat(*[b1[i] - b1[i + 1] for i in range(H - 1)])
    lb = -np.inf * np.ones(H - 1)
    ub = np.zeros(H - 1)
    return g, lb, ub
