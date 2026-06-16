"""
=============================================================
  EXPERIMENT CONFIGURATION FILE
  Neural Network Weight Optimization as a Constrained NLP
=============================================================

This is the ONLY file you need to touch to run experiments.

HOW TO USE:
  - Each entry in EXPERIMENTS is one experiment run.
  - Set enabled=True/False to turn experiments on/off.
  - Change any parameter and re-run main.py.
  - Results are saved automatically to results/ and figures/.

PARAMETER REFERENCE:
  method      : 'ipopt' (interior point, constrained)
                'sqp'   (sequential quadratic programming, constrained)
                'adam'  (unconstrained gradient descent baseline)
  H           : number of hidden units in the student network
  L_max       : Lipschitz bound  ||W1||_F * ||W2||_F <= L_max  (only used if use_lipschitz)
  B_max       : weight norm-ball radius  ||w||_2 <= B_max      (only used if use_norm_ball)
  use_lipschitz / use_norm_ball / use_symmetry_break : turn each constraint on/off
  noise_std   : std of Gaussian noise added to the synthetic labels
  seed        : random seed (data generation + initial weight guess)
"""

import numpy as np

# ── Data generation (synthetic teacher-student regression) ───────────────────
DATA = dict(
    d_in=1, d_out=1,
    H_teacher=6,            # hidden units of the fixed random "ground truth" network
    n_train=60,
    n_test=40,
    x_range=(-3.0, 3.0),
)

# ── Default network / constraint settings (overridden per-experiment) ────────
DEFAULTS = dict(
    H=8,
    L_max=4.0,
    B_max=6.0,
    use_lipschitz=False,
    use_norm_ball=False,
    use_symmetry_break=False,
    noise_std=0.05,
    seed=0,
    init_scale=0.5,
)

# ── Solver settings ────────────────────────────────────────────────────────
IPOPT_OPTS = dict(
    ipopt=dict(max_iter=2000, tol=1e-8, print_level=0),
    print_time=False,
)

SQP_OPTS = dict(
    qpsol='qrqp',
    max_iter=3000,
    hessian_approximation='limited-memory',  # BFGS -- keeps QP subproblems convex;
                                              # the exact Hessian is indefinite here
                                              # because the Lipschitz constraint is bilinear
    tol_pr=1e-7,
    tol_du=1e-5,
    print_time=False,
    print_iteration=False,
    print_header=False,
    print_status=False,
    qpsol_options=dict(print_iter=False, print_header=False, error_on_fail=False),
)

ADAM_OPTS = dict(
    lr=0.02, n_iter=3000, beta1=0.9, beta2=0.999,
)


def _make(name, label, group, **overrides):
    cfg = dict(DATA)
    cfg.update(DEFAULTS)
    cfg.update(
        name=name, label=label, group=group, enabled=True,
        ipopt_opts=IPOPT_OPTS, sqp_opts=SQP_OPTS, adam_opts=ADAM_OPTS,
    )
    cfg.update(overrides)
    return cfg


# =============================================================================
#  EXPERIMENTS — edit this list freely
# =============================================================================
EXPERIMENTS = []

# ── GROUP 1: method comparison ────────────────────────────────────────────
# Same network, same data, same constraints (where applicable) -- only the
# optimization algorithm changes. Answers: how do IPOPT / SQP / Adam compare
# in solution quality and computational cost on the identical problem?
_method_common = dict(
    H=8, use_lipschitz=True, use_norm_ball=True, use_symmetry_break=True,
    L_max=4.0, B_max=6.0, noise_std=0.05,
)
EXPERIMENTS += [
    _make('exp_method_ipopt', 'Method comparison — IPOPT (constrained)',
          'method_comparison', method='ipopt', **_method_common),
    _make('exp_method_sqp', 'Method comparison — SQP (constrained)',
          'method_comparison', method='sqp', **_method_common),
    _make('exp_method_adam', 'Method comparison — Adam (unconstrained)',
          'method_comparison', method='adam', H=8, noise_std=0.05),
]

# ── GROUP 2: Lipschitz bound sweep ────────────────────────────────────────
# Only the Lipschitz constraint is active (isolated from the other two) so
# its effect on the train/test trade-off is visible on its own. Answers:
# how does tightening L_max trade off training fit against generalization?
for L in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
    tag = str(L).replace('.', 'p')
    EXPERIMENTS.append(_make(
        f'exp_lipschitz_L{tag}', f'Lipschitz sweep — L_max={L}',
        'lipschitz_sweep', method='ipopt', H=8, L_max=L,
        use_lipschitz=True, use_norm_ball=False, use_symmetry_break=False,
        noise_std=0.1,
    ))

# ── GROUP 3: NLP size scaling ─────────────────────────────────────────────
# Same constraints, growing network -> growing NLP. Answers: how does
# IPOPT's solve time and iteration count scale with problem dimension?
for H in [4, 8, 16, 32, 64]:
    EXPERIMENTS.append(_make(
        f'exp_size_H{H}', f'Size scaling — H={H} hidden units',
        'size_scaling', method='ipopt', H=H,
        use_lipschitz=True, use_norm_ball=True, use_symmetry_break=True,
        L_max=4.0, B_max=6.0, noise_std=0.05,
    ))

# ── GROUP 4: noise robustness ─────────────────────────────────────────────
# Constrained (IPOPT, hard Lipschitz + norm-ball) vs unconstrained (Adam),
# repeated at increasing label noise. Answers: do the hard constraints
# improve generalization under noisy data compared to unconstrained descent?
for noise in [0.0, 0.1, 0.3]:
    tag = str(noise).replace('.', 'p')
    EXPERIMENTS.append(_make(
        f'exp_noise_{tag}_ipopt', f'Noise robustness — sigma={noise} (IPOPT, constrained)',
        'noise_robustness', method='ipopt', H=8, noise_std=noise,
        use_lipschitz=True, use_norm_ball=True, use_symmetry_break=True,
        L_max=4.0, B_max=6.0,
    ))
    EXPERIMENTS.append(_make(
        f'exp_noise_{tag}_adam', f'Noise robustness — sigma={noise} (Adam, unconstrained)',
        'noise_robustness', method='adam', H=8, noise_std=noise,
    ))
