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

# =============================================================================
#  NEW GROUPS — deeper optimization-side studies (added on top of the
#  original four groups above; nothing above this line was changed)
# =============================================================================

# ── GROUP 5: multi-start / local minima study ─────────────────────────────
# The Lipschitz-constrained NLP is nonconvex (the constraint is bilinear,
# and the tanh network is already nonconvex on its own). IPOPT can converge
# to different local minima depending on the initial guess. All 20 runs use
# the SAME training data (fixed data_seed) and the SAME L_max -- only the
# random initialization seed changes -- so any spread in the result is
# genuinely about the shape of the optimization landscape, not the data.
# init_scale is intentionally larger than the project default (4.0 vs 0.5):
# small random jitters near the origin all fall in the same basin, so a
# wider initial spread is needed to actually land in different basins.
# Answers: does this NLP have a real nonconvex landscape with multiple,
# distinct local optima (a concern Adam users never have to think about)?
_MULTISTART_DATA_SEED = 999
for i in range(20):
    EXPERIMENTS.append(_make(
        f'exp_multistart_seed{i}', f'Multi-start — init seed {i}',
        'multistart', method='ipopt', H=8, L_max=4.0,
        use_lipschitz=True, use_norm_ball=False, use_symmetry_break=False,
        noise_std=0.05, seed=i, data_seed=_MULTISTART_DATA_SEED, init_scale=4.0,
    ))

# ── GROUP 6: KKT / dual-variable analysis ─────────────────────────────────
# Same Lipschitz sweep as 'lipschitz_sweep' (same H, same noise_std, same L
# range) but additionally records IPOPT's Lagrange multiplier (dual
# variable / "shadow price") for the Lipschitz constraint at each L_max.
# Answers: at which point does the constraint actually start binding
# (lambda > 0, costly) versus sit slack (lambda ~ 0, free)?
for L in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
    tag = str(L).replace('.', 'p')
    EXPERIMENTS.append(_make(
        f'exp_kkt_L{tag}', f'KKT analysis — L_max={L}',
        'kkt_analysis', method='ipopt', H=8, L_max=L,
        use_lipschitz=True, use_norm_ball=False, use_symmetry_break=False,
        noise_std=0.1,
    ))

# ── GROUP 7: penalty method vs. hard constraint ───────────────────────────
# The "everyone already does this" way to handle a constraint in ML is a
# penalty term in the loss, minimize f(w) + rho*violation(w), with plain
# Adam ('penalty_adam' method, see src/penalty_adam.py). Compared against
# IPOPT, which enforces the identical Lipschitz constraint exactly (zero
# violation by construction). Answers: how much does an unconstrained
# penalty method actually violate the constraint, and how does that trade
# off against fit, as rho varies from very loose to very strict?
#
# L_max=1.0 here (NOT the project default 4.0): the network's natural,
# unpenalized optimum for this data sits at Lipschitz~1.6, i.e. *below*
# L_max=4.0 -- so at the default bound the constraint never binds and the
# violation is 0 for every rho, which would make for a flat, uninteresting
# plot. L_max=1.0 sits below that natural optimum, so the constraint
# actually has something to fight against.
#
# rho range is empirically chosen rather than a naive log-decade sweep
# (0.1 .. 1000): Adam's per-parameter adaptive step normalization (it
# normalizes by the RMS of past gradients) means the penalty's *direction*
# matters far more than its magnitude, so enforcement saturates extremely
# fast -- the transition from "ignores the constraint" to "fully enforces
# it" happens between rho=1e-5 and rho=1e-4, not between 0.1 and 1000.
# rho=0.1..1000 all land in the already-saturated regime and look
# identical to each other; this range is the one that actually shows the
# trade-off the experiment is meant to demonstrate.
_PENALTY_COMMON = dict(H=8, L_max=1.0, noise_std=0.05,
                        use_lipschitz=True, use_norm_ball=False, use_symmetry_break=False)
EXPERIMENTS.append(_make(
    'exp_penalty_ipopt_ref', 'Penalty comparison — IPOPT (hard constraint, reference)',
    'penalty_vs_hard', method='ipopt', **_PENALTY_COMMON,
))
for rho in [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2, 1.0]:
    tag = str(rho).replace('.', 'p').replace('-', 'm')
    EXPERIMENTS.append(_make(
        f'exp_penalty_rho{tag}', f'Penalty comparison — Adam, rho={rho:g}',
        'penalty_vs_hard', method='penalty_adam', rho=rho, **_PENALTY_COMMON,
    ))
