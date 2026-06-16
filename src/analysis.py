"""
src/analysis.py
─────────────────
Post-hoc metrics computed from a solved weight vector: the achieved
Lipschitz estimate and the worst constraint violation. Used to sanity
check that IPOPT/SQP solutions actually respect the constraints they
were given (violation should be ~1e-6 or smaller).
"""

import numpy as np
from src.model import unflatten_numpy


def lipschitz_estimate(w, shapes):
    """||W1||_F * ||W2||_F -- the upper bound the Lipschitz constraint enforces."""
    W1, b1, W2, b2 = unflatten_numpy(w, shapes)
    return float(np.linalg.norm(W1, 'fro') * np.linalg.norm(W2, 'fro'))


def max_constraint_violation(g, lbg, ubg):
    if len(g) == 0:
        return 0.0
    lo = np.maximum(np.asarray(lbg) - g, 0.0)
    hi = np.maximum(g - np.asarray(ubg), 0.0)
    return float(np.max(np.concatenate([lo, hi])))
