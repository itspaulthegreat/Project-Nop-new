import os
import sys
import numpy as np
import casadi as ca

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model import param_shapes, n_params, unflatten_symbolic, random_init
from src.constraints import (lipschitz_constraint, norm_ball_constraint,
                              symmetry_breaking_constraints)


def _split(w_val, shapes):
    n = n_params(shapes)
    w = ca.MX.sym('w', n)
    W1, b1, W2, b2 = unflatten_symbolic(w, shapes)
    fn = ca.Function('fn', [w], [W1, b1, W2, b2])
    return fn(w_val), (w, W1, b1, W2, b2)


def test_lipschitz_constraint_bounds():
    shapes = param_shapes(d_in=1, H=4, d_out=1)
    w_val = random_init(shapes, scale=1.0, seed=0)
    (W1v, b1v, W2v, b2v), (w, W1, b1, W2, b2) = _split(w_val, shapes)

    g, lb, ub = lipschitz_constraint(W1, W2, L_max=5.0)
    g_fn = ca.Function('g', [w], [g])
    g_val = float(g_fn(w_val))

    expected = float(np.sum(np.asarray(W1v) ** 2) * np.sum(np.asarray(W2v) ** 2))
    assert abs(g_val - expected) < 1e-9
    assert ub == 25.0
    assert lb == -np.inf


def test_norm_ball_constraint():
    shapes = param_shapes(d_in=1, H=4, d_out=1)
    w_val = random_init(shapes, scale=1.0, seed=1)
    n = n_params(shapes)
    w = ca.MX.sym('w', n)
    g, lb, ub = norm_ball_constraint(w, B_max=3.0)
    g_fn = ca.Function('g', [w], [g])
    g_val = float(g_fn(w_val))
    assert abs(g_val - float(np.sum(w_val ** 2))) < 1e-9
    assert ub == 9.0


def test_symmetry_breaking_shape():
    shapes = param_shapes(d_in=1, H=5, d_out=1)
    w_val = random_init(shapes, scale=1.0, seed=2)
    (W1v, b1v, W2v, b2v), (w, W1, b1, W2, b2) = _split(w_val, shapes)

    g, lb, ub = symmetry_breaking_constraints(b1)
    assert g.shape[0] == 4  # H - 1
    assert np.all(ub == 0.0)

    g_fn = ca.Function('g', [w], [g])
    g_val = np.asarray(g_fn(w_val)).flatten()
    expected = np.array([b1v[i] - b1v[i + 1] for i in range(4)]).flatten()
    np.testing.assert_allclose(g_val, expected, atol=1e-9)
