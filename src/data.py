"""
src/data.py
────────────
Synthetic teacher-student regression dataset.

A fixed random "teacher" network generates the ground-truth function;
the student network (the one whose weights we optimize) must recover
it from noisy samples. This gives a known, controllable regression
problem so the optimization behavior can be studied in isolation from
data-quality issues.
"""

import numpy as np


def _teacher_forward(X, W1, b1, W2, b2):
    Z1 = W1 @ X + b1
    A1 = np.tanh(Z1)
    return W2 @ A1 + b2


def make_teacher(d_in=1, H_teacher=6, d_out=1, seed=0):
    """Fixed random 1-hidden-layer tanh network used as ground truth."""
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1.0, size=(H_teacher, d_in))
    b1 = rng.normal(0, 1.0, size=(H_teacher, 1))
    W2 = rng.normal(0, 1.0, size=(d_out, H_teacher)) / np.sqrt(H_teacher)
    b2 = rng.normal(0, 0.2, size=(d_out, 1))
    return W1, b1, W2, b2


def generate_dataset(n_train=60, n_test=40, noise_std=0.05, seed=0,
                      d_in=1, d_out=1, H_teacher=6, x_range=(-3.0, 3.0)):
    """
    Returns (X_train, y_train, X_test, y_test) as numpy arrays of shape
    (d_in, N) / (d_out, N) -- the layout CasADi matrix ops expect.
    """
    rng = np.random.default_rng(seed)
    W1, b1, W2, b2 = make_teacher(d_in, H_teacher, d_out, seed=seed)

    def sample(n):
        X = rng.uniform(x_range[0], x_range[1], size=(d_in, n))
        y_clean = _teacher_forward(X, W1, b1, W2, b2)
        y = y_clean + rng.normal(0, noise_std, size=y_clean.shape)
        return X, y

    X_train, y_train = sample(n_train)
    X_test, y_test = sample(n_test)
    return X_train, y_train, X_test, y_test
