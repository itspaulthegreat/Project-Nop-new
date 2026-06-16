"""
src/plotting.py
──────────────────
All figures. Each function takes result dict(s) and a save path, and
writes a PNG. No figure is ever shown interactively (Agg backend) so
this runs fine from the command line / over SSH.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.model import param_shapes, forward_numpy


# ─────────────────────────────────────────────────────────────────────────
#  Single-experiment fit plot (data scatter + learned curve)
# ─────────────────────────────────────────────────────────────────────────

def plot_fit(res, X_train, y_train, X_test, y_test, x_range, path):
    shapes = param_shapes(1, res['H'], 1)
    w = np.array(res['w'])

    xs = np.linspace(x_range[0], x_range[1], 300).reshape(1, -1)
    ys = forward_numpy(w, xs, shapes)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X_train.flatten(), y_train.flatten(), s=18, c='tab:blue',
               alpha=0.7, label='train data')
    ax.scatter(X_test.flatten(), y_test.flatten(), s=18, c='tab:orange',
               alpha=0.7, marker='^', label='test data')
    ax.plot(xs.flatten(), ys.flatten(), c='black', lw=2, label='learned f(x; w)')
    ax.set_title(f"{res['label']}\ntrain MSE={res['train_mse']:.4f}  "
                 f"test MSE={res['test_mse']:.4f}")
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
#  Method comparison: IPOPT vs SQP vs Adam
# ─────────────────────────────────────────────────────────────────────────

def plot_method_comparison(results, path):
    results = sorted(results, key=lambda r: r['method'])
    names = [r['method'].upper() for r in results]
    train_mse = [r['train_mse'] for r in results]
    test_mse = [r['test_mse'] for r in results]
    solve_time = [r['solve_time'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    x = np.arange(len(names))
    w = 0.35

    axes[0].bar(x - w / 2, train_mse, w, label='train MSE')
    axes[0].bar(x + w / 2, test_mse, w, label='test MSE')
    axes[0].set_xticks(x); axes[0].set_xticklabels(names)
    axes[0].set_ylabel('MSE'); axes[0].set_title('Fit quality')
    axes[0].legend(fontsize=8)

    axes[1].bar(x, solve_time, color='tab:green')
    axes[1].set_xticks(x); axes[1].set_xticklabels(names)
    axes[1].set_ylabel('solve time [s]'); axes[1].set_title('Computational cost')
    axes[1].set_yscale('log')

    fig.suptitle('IPOPT vs SQP vs Adam — same network, same data')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
#  Lipschitz bound sweep: train/test MSE trade-off
# ─────────────────────────────────────────────────────────────────────────

def plot_lipschitz_sweep(results, path):
    results = sorted(results, key=lambda r: r['L_max'])
    L = [r['L_max'] for r in results]
    train_mse = [r['train_mse'] for r in results]
    test_mse = [r['test_mse'] for r in results]
    lip_achieved = [r['lipschitz_estimate'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].plot(L, train_mse, 'o-', label='train MSE')
    axes[0].plot(L, test_mse, 's-', label='test MSE')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Lipschitz bound $L_{max}$')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Fit vs. generalization trade-off')
    axes[0].legend(fontsize=8)

    axes[1].plot(L, lip_achieved, 'd-', color='tab:purple', label='achieved $\\|W_1\\|_F\\|W_2\\|_F$')
    axes[1].plot(L, L, '--', color='gray', label='bound $L_{max}$')
    axes[1].set_xscale('log'); axes[1].set_yscale('log')
    axes[1].set_xlabel('Lipschitz bound $L_{max}$')
    axes[1].set_ylabel('achieved Lipschitz estimate')
    axes[1].set_title('Constraint activity')
    axes[1].legend(fontsize=8)

    fig.suptitle('Effect of tightening the Lipschitz constraint (IPOPT)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
#  NLP size scaling: solve time / iterations vs number of decision vars
# ─────────────────────────────────────────────────────────────────────────

def plot_size_scaling(results, path):
    results = sorted(results, key=lambda r: r['n_vars'])
    n_vars = [r['n_vars'] for r in results]
    solve_time = [r['solve_time'] for r in results]
    n_iter = [r['n_iter'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].plot(n_vars, solve_time, 'o-', color='tab:red')
    axes[0].set_xlabel('number of decision variables (NLP size)')
    axes[0].set_ylabel('solve time [s]')
    axes[0].set_title('Solve time vs. problem size')
    axes[0].set_xscale('log'); axes[0].set_yscale('log')

    axes[1].plot(n_vars, n_iter, 's-', color='tab:blue')
    axes[1].set_xlabel('number of decision variables (NLP size)')
    axes[1].set_ylabel('IPOPT iterations')
    axes[1].set_title('Iterations vs. problem size')
    axes[1].set_xscale('log')

    fig.suptitle('NLP scaling with network size (hidden units H)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
#  Noise robustness: constrained (IPOPT) vs unconstrained (Adam)
# ─────────────────────────────────────────────────────────────────────────

def plot_noise_robustness(results, path):
    noise_levels = sorted(set(r['noise_std'] for r in results))
    methods = ['ipopt', 'adam']
    colors = {'ipopt': 'tab:green', 'adam': 'tab:gray'}

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(noise_levels))
    w = 0.35

    for i, method in enumerate(methods):
        vals = []
        for nl in noise_levels:
            match = [r for r in results if r['method'] == method and r['noise_std'] == nl]
            vals.append(match[0]['test_mse'] if match else np.nan)
        offset = (i - 0.5) * w
        ax.bar(x + offset, vals, w, label=method.upper(), color=colors[method])

    ax.set_xticks(x)
    ax.set_xticklabels([f'σ={nl}' for nl in noise_levels])
    ax.set_ylabel('test MSE')
    ax.set_title('Generalization under noisy data:\nconstrained (IPOPT) vs unconstrained (Adam)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
#  Adam convergence curve
# ─────────────────────────────────────────────────────────────────────────

def plot_adam_convergence(res, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res['history'], lw=1.5)
    ax.set_yscale('log')
    ax.set_xlabel('iteration')
    ax.set_ylabel('training MSE (log scale)')
    ax.set_title(f"Adam convergence — {res['name']}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
