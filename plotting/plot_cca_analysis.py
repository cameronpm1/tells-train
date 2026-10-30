"""Figures for evals/marl/cca_analysis.py (see evals/marl/CCA_ANALYSIS.md): every method on the same
axes, scored against the truth. The figures used are task_ridgeline.png and centroid_map.png.

All values are raw R^2 = 1 - MSE / var(truth) of each belief's own estimate against the true
state - exactly what the policy receives, no fitting. Violins show the episode-bootstrap
distribution; the white dot is the full-data estimate. R^2 axes are cut at -1.

fixed_axis.png            R^2 along each policy task direction (fixed by the true model),
                          on the physical components (centroid, spread, adversary, ball) and
                          the mean over the task directions
fixed_axis_spectrum.png   top: the FO policy task eigenvalues (how much its actions depend on each
                          direction); bottom: each method's R^2 along those same directions
pred_vs_true.png          truth vs the raw belief along the policy task directions
centroid_map.png          local R^2 of the belief's teammate centroid over the true centroid
                          position (x, y relative to the agent)
task_ridgeline.png        one row per policy task eigenvector: distribution over episodes of
                          each belief's per-episode R^2 along it
task_error_ridgeline.png  same rows: distribution of each belief's raw error (estimate - truth)
actions.png               action agreement (action taken from the agent's own observation ==
                          the same policy's action on the true observation) and success rate

Usage:
    python plotting/plot_cca_analysis.py --in-dir results/football/cca_final
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

METHODS = ['fully_obs', 'allocentric', 'pf']
LABELS = {'fully_obs': 'Fully observed', 'allocentric': 'Allocentric', 'pf': 'PF'}
COLORS = {'fully_obs': '#1baf7a', 'allocentric': '#2a78d6', 'pf': '#eb6834'}
TEXT, TEXT_2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BLUE_RAMP = LinearSegmentedColormap.from_list(
    'blue', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
KIND = 'raw'
RIDGE_METHODS = ['allocentric', 'pf']  # ridgelines compare the two beliefs only
FLOOR = -1.0  # R^2 axes are cut here (raw R^2 is unbounded below)
WIDTH = 0.26
# ridgeline layout (data units): row spacing, curve height, and how far the back curve (PF) sits
# above the front one (allocentric) within a row - vertical only, so x values stay exact
RIDGE_STEP, RIDGE_H, RIDGE_DEPTH = 0.62, 0.5, 0.0
RIDGE_DEPTH_OF = {'allocentric': 0, 'pf': 1}  # 0 = front


def ridge_pair(ax, grid, heights, exact, base, k):
    """One ridgeline row: each method's curve (heights scaled to 0..1) on its own baseline (the back curve
    offset upwards by RIDGE_DEPTH), with see-through fills and every outline above every fill, so a curve
    behind another stays visible."""
    order = sorted(RIDGE_METHODS, key=lambda m: -RIDGE_DEPTH_OF[m])  # back first
    x0, x1 = grid[0], grid[-1]
    z = 10 * k  # lower rows on top
    for i, m in enumerate(order):
        b = base + RIDGE_DEPTH * RIDGE_DEPTH_OF[m]
        ax.plot([x0, x1], [b, b], color='#c9c8c3', linewidth=0.7, zorder=z)
        if m in exact:  # identical value everywhere: vertical line
            ax.plot([exact[m]] * 2, [b, b + 0.8 * RIDGE_H], color=COLORS[m], linewidth=2.5, zorder=z + 5 + i)
            continue
        h = heights[m]
        y = b + RIDGE_H * h
        ax.fill_between(grid, b, y, color=COLORS[m], alpha=0.22, linewidth=0, zorder=z + 1 + i)
        # outline only where there is density, so empty stretches do not read as data on the baseline
        ax.plot(grid, np.where(h / h.max() > 0.0005, y, np.nan), color=COLORS[m], linewidth=1.2, zorder=z + 5 + i)


def ridge_axes(ax, n):
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-0.1, (n - 1) * RIDGE_STEP + RIDGE_DEPTH + RIDGE_H + 0.3)


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#c9c8c3')
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(0, color='#c9c8c3', linewidth=1)


def violins(ax, samples, estimate, labels, colors=None):
    """samples: dict method -> (n_boot, n_axes); estimate: dict method -> (n_axes,)."""
    x = np.arange(len(labels))
    keys = list(samples)
    for j, m in enumerate(keys):
        xs = x + (j - (len(keys) - 1) / 2) * WIDTH
        data = np.asarray(samples[m], dtype=float)
        data = data.reshape(len(data), -1)
        for k, xi in enumerate(xs):
            col = data[:, k]
            if np.ptp(col) < 1e-9:  # constant (e.g. perfect observer): draw a flat tick
                ax.plot([xi - WIDTH * 0.4, xi + WIDTH * 0.4], [col[0]] * 2, color=(colors or COLORS)[m], lw=3)
                continue
            parts = ax.violinplot(col, positions=[xi], widths=WIDTH * 0.9, showextrema=False)
            for b in parts['bodies']:
                b.set_facecolor((colors or COLORS)[m])
                b.set_edgecolor((colors or COLORS)[m])
                b.set_alpha(0.85)
        est = np.asarray(estimate[m], dtype=float).ravel()
        ax.scatter(xs, np.maximum(est, FLOOR), s=14, color=SURFACE, edgecolors=TEXT, linewidths=0.8, zorder=5)
        for xi, v in zip(xs, est):
            if v < FLOOR:  # below the axis cut: label the value at the bottom edge
                ax.text(xi, FLOOR + 0.03, f'{v:.1f}', ha='center', va='bottom', fontsize=6.5, color=TEXT, rotation=90)
    ax.set_xticks(x, labels, fontsize=8)


def set_ylim(ax, *arrays):
    lo = min(np.nanmin(np.asarray(a, dtype=float)) for a in arrays)
    ax.set_ylim(max(FLOOR, min(0, lo - 0.05)), 1.05)


def legend(ax):
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in METHODS]
    handles.append(plt.Line2D([], [], marker='o', ls='', ms=5, color=SURFACE, markeredgecolor=TEXT, mew=0.8))
    ax.legend(handles, [LABELS[m] for m in METHODS] + ['full-data estimate'], loc='lower left', frameon=True,
              facecolor=SURFACE, edgecolor=GRID, fontsize=8, labelcolor=TEXT)


def group_names(meta):
    return {'adversary': meta['adversary'], 'difference': 'spread' if meta['n_teammates'] > 2 else 'difference'}


def plot_main(S, B, path):
    e, meta = S['estimate'], S['meta']
    r = e['r_ref']
    names = group_names(meta)

    def comp(k):
        return describe(S['task_composition'], k, names, sep='\n')

    task_labels = [f'task dir {k + 1}\nσ² {e["task_eig"][k]:.2f}\n{comp(k)}' for k in range(r)]
    groups = list(e['fully_obs'][KIND]['physical'])

    fig, axes = plt.subplots(1, 3, figsize=(min(19, max(14, 1.6 * r + 1.6 * len(groups) + 3)), 6.8),
                             facecolor=SURFACE, gridspec_kw=dict(width_ratios=[max(r, 2), len(groups), 1.3]))
    # fixed margins (constrained layout can collapse with long tick labels and off-axis value labels)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.85, bottom=0.24, wspace=0.14)
    for ax in axes:
        style(ax)

    task_s = {m: B[f'{m}/{KIND}/task'] for m in METHODS}
    task_e = {m: e[m][KIND]['task'] for m in METHODS}
    violins(axes[0], task_s, task_e, task_labels)
    axes[0].set_title('policy task directions (same axes for every method)', fontsize=10, color=TEXT, loc='left')
    axes[0].set_ylabel('R² against the true state', color=TEXT_2, fontsize=9)

    phys_s = {m: np.column_stack([B[f'{m}/{KIND}/physical/{g}'] for g in groups]) for m in METHODS}
    phys_e = {m: [e[m][KIND]['physical'][g] for g in groups] for m in METHODS}
    violins(axes[1], phys_s, phys_e, [names.get(g, g) for g in groups])
    axes[1].set_title('physical components', fontsize=10, color=TEXT, loc='left')

    cov_s = {m: B[f'{m}/{KIND}/coverage'][:, None] for m in METHODS}
    cov_e = {m: [e[m][KIND]['coverage']] for m in METHODS}
    violins(axes[2], cov_s, cov_e, ['mean over\ntask directions'])
    for j, m in enumerate(METHODS):
        v = e[m][KIND]['coverage']
        axes[2].text((j - 1) * WIDTH, v + 0.05, f'{v:.2f}', ha='center', fontsize=8, color=TEXT)
    axes[2].set_title('task coverage', fontsize=10, color=TEXT, loc='left')

    lows = [v for d in (task_s, phys_s, cov_s) for v in d.values()]
    for ax in axes:
        set_ylim(ax, *lows)
    axes[2].set_ylim(axes[2].get_ylim()[0], 1.12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in METHODS]
    handles.append(plt.Line2D([], [], marker='o', ls='', ms=5, color=SURFACE, markeredgecolor=TEXT, mew=0.8))
    fig.legend(handles, [LABELS[m] for m in METHODS] + ['full-data estimate'], loc='upper right', ncol=4,
               bbox_to_anchor=(0.99, 0.95), frameon=False, fontsize=8.5, labelcolor=TEXT)
    env = meta['env'].replace('_', '-').capitalize()
    fig.suptitle(f'{env}: belief vs true state on fixed axes, {meta["title"]}, all agents',
                 x=0.01, y=0.98, ha='left', fontsize=11, weight='bold', color=TEXT)
    fig.text(0.01, 0.015, "R² = 1 - MSE / var(truth) of each belief's own estimate, exactly what the policy receives (no "
             'fitting). Task directions come from the fully observed policy actions on the true state and are identical for every '
             f'method.\nViolins: distribution over 100 episode-bootstrap resamples; dot: full-data estimate. '
             f'Axis cut at {FLOOR:.0f}; values below it are printed at the bottom edge.', fontsize=7.5, color=TEXT_2)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_spectrum(S, B, path):
    e, meta = S['estimate'], S['meta']
    eig = np.array(e['task_eig_all'])
    d, r = len(eig), e['r_ref']
    comp_all = S['task_composition_all']
    names = group_names(meta)
    labels = []
    for k in range(d):
        if 'share' in comp_all:
            top_share, top_group = max((comp_all['share'][g][k], g) for g in comp_all['share'])
            labels.append(f'dir {k + 1}\n{100 * top_share:.0f}% {names.get(top_group, top_group)}')
        else:
            labels.append(f'dir {k + 1}')
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(max(10, 1.25 * d + 3), 7.2), facecolor=SURFACE, sharex=True,
                                   gridspec_kw=dict(height_ratios=[1, 2.2]), layout='constrained')
    for ax in (ax0, ax1):
        style(ax)
        ax.axvspan(-0.5, r - 0.5, color='#eef3fb', zorder=0)
    violins(ax0, {'policy': B['task_eig_all']}, {'policy': eig}, labels, colors={'policy': '#8f8e89'})
    ax0.axhline(0.1, color=TEXT_2, linestyle='--', linewidth=1)
    for xi, v in enumerate(eig):
        ax0.text(xi + 0.12, v, f'{v:.2f}', ha='left', va='center', fontsize=7.5, color=TEXT)
    ax0.set_ylim(0, 1.1)
    ax0.set_ylabel('FO policy\ntask eigenvalue σ²', color=TEXT_2, fontsize=9)
    ax0.set_title('How much the FO policy actions depend on each direction (true state)', fontsize=10, color=TEXT,
                  loc='left')
    samples = {m: B[f'{m}/{KIND}/task_all'] for m in METHODS}
    violins(ax1, samples, {m: e[m][KIND]['task_all'] for m in METHODS}, labels)
    set_ylim(ax1, *samples.values())
    ax1.set_ylabel('R² against the true state', color=TEXT_2, fontsize=9)
    ax1.set_title('How well each belief reproduces the true state along the same directions', fontsize=10,
                  color=TEXT, loc='left')
    legend(ax1)
    env = meta['env'].replace('_', '-').capitalize()
    fig.suptitle(f'{env}: policy task eigenvalues and belief recovery on the same axes, {meta["title"]}',
                 x=0.01, ha='left', fontsize=11, weight='bold', color=TEXT)
    fig.supxlabel('Directions = eigenvectors of the policy task CCA (FO policy actions vs true state), identical for every method; '
                  'shaded = above the rank cutoff σ² = 0.1 (dashed).\nR² of each belief\'s own estimate (no fitting). '
                  f'Violins: distribution over 100 episode-bootstrap resamples; dot: full-data estimate. Axis cut at {FLOOR:.0f}.',
                  fontsize=7.5, color=TEXT_2, x=0.01, ha='left')
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def describe(comp, k, names, sep=', '):
    """Label for task direction k. comp['share']: '60% prey, 38% centroid' (geometric share of each physical
    subspace); comp['explained'] (latent state, fire): 'R² from fire 0.62, teammates 0.46'."""
    if 'share' in comp:
        parts = sorted(((comp['share'][g][k], g) for g in comp['share']), reverse=True)
        return sep.join(f'{100 * v:.0f}% {names.get(g, g)}' for v, g in parts if v >= 0.05)
    parts = sorted(((comp['explained'][g][k], g) for g in comp['explained']), reverse=True)
    return 'R² from ' + sep.join(f'{g} {v:.2f}' for v, g in parts)


def composition(P):
    """Row-label composition saved in projections.npz ('share/<group>' or 'explained/<group>')."""
    out = {}
    for key in P.files:
        kind, _, g = key.partition('/')
        if kind in ('share', 'explained'):
            out.setdefault(kind, {})[g] = P[key]
    return out


def plot_pred_true(S, in_dir):
    """Hexbin of the truth vs the raw belief along each task direction, one row per method."""
    P = np.load(os.path.join(in_dir, 'projections.npz'))
    meta = S['meta']
    names = group_names(meta)
    comp = composition(P)
    n = P['functionals'].shape[1]
    fig, axes = plt.subplots(3, n, figsize=(4.3 * n + 0.5, 12), facecolor=SURFACE, squeeze=False)
    for k in range(n):
        allv = np.concatenate([np.concatenate([P[f'{m}/actual'][:, k], P[f'{m}/belief'][:, k]]) for m in METHODS])
        lim = np.percentile(allv, [0.5, 99.5])
        for i, m in enumerate(METHODS):
            ax = axes[i, k]
            style(ax)
            ax.yaxis.grid(False)
            a, b = P[f'{m}/actual'][:, k], P[f'{m}/belief'][:, k]
            ax.hexbin(a, b, gridsize=60, extent=(*lim, *lim), cmap=BLUE_RAMP, bins='log', mincnt=1, linewidths=0)
            ax.plot(lim, lim, color=TEXT, linestyle='--', linewidth=1)
            ax.text(0.03, 0.97, f'R² {1 - np.mean((b - a) ** 2) / np.var(a):.2f}\nbias {np.mean(b - a):+.2f}\n'
                    f'RMSE {np.sqrt(np.mean((b - a) ** 2)):.2f}', transform=ax.transAxes, va='top', fontsize=8,
                    color=TEXT)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect('equal')
            if i == 0:
                ax.set_title(f'task dir {k + 1}  (σ² {P["task_eig"][k]:.2f})\n{describe(comp, k, names)}',
                             fontsize=9, color=TEXT)
            if i == 2:
                ax.set_xlabel('true value (env units)', color=TEXT_2, fontsize=8)
            if k == 0:
                ax.set_ylabel(f'{LABELS[m]}\nbelief estimate (env units)', color=TEXT_2, fontsize=8)
    env = meta['env'].replace('_', '-').capitalize()
    fig.suptitle(f'{env}: belief vs truth along the policy task directions (fixed axes), {meta["title"]}',
                 x=0.02, ha='left', fontsize=11, weight='bold', color=TEXT)
    fig.text(0.02, 0.005, "Axes: projection of the true state (x) and of the belief's own estimate (y) onto each "
             'policy task direction, identical for every method. R² = 1 - MSE / var(true), no fitting. '
             'Dashed: perfect.', fontsize=7.5, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    fig.savefig(os.path.join(in_dir, 'pred_vs_true.png'), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_actions(S, in_dir):
    path = os.path.join(in_dir, 'bootstrap_actions.npz')
    if 'actions' not in S or not S['actions'] or not os.path.exists(path):
        return
    B = np.load(path)
    methods = [m for m in METHODS if m in S['actions']]
    panels = [('action_agreement', 'action agreement with the\ntrue-observation policy',
               'same action as the policy on the true observation'),
              ('success_rate', 'success rate', 'fraction of episodes that succeed')]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), facecolor=SURFACE, layout='constrained')
    for ax, (key, title, ylabel) in zip(axes, panels):
        style(ax)
        samples = {m: B[f'actions/{m}/{key}'][:, None] for m in methods}
        est = {m: [S['actions'][m][key]] for m in methods}
        violins(ax, samples, est, [''])
        for j, m in enumerate(methods):
            v = S['actions'][m][key]
            ax.text((j - (len(methods) - 1) / 2) * WIDTH, v + 0.04, f'{v:.0%}', ha='center', fontsize=9, color=TEXT)
        ax.set_xticks([])
        ax.set_ylim(0, 1.1)
        ax.set_title(title, fontsize=10, color=TEXT, loc='left')
        ax.set_ylabel(ylabel, color=TEXT_2, fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in methods]
    axes[1].legend(handles, [LABELS[m] for m in methods], loc='upper right', frameon=False, fontsize=8,
                   labelcolor=TEXT)
    env = S['meta']['env'].replace('_', '-').capitalize()
    fig.suptitle(f'{env}: does the belief make the policy act as it would with the true state?',
                 x=0.01, ha='left', fontsize=11, weight='bold', color=TEXT)
    fig.supxlabel('Action agreement: over all agent-steps of full episodes, how often the action taken from the '
                  "agent's own observation\n(belief / particle filter / noisy view) equals the action the same "
                  'policy takes from the true observation. Violins: 100 episode-bootstrap resamples.',
                  fontsize=7.5, color=TEXT_2, x=0.01, ha='left')
    fig.savefig(os.path.join(in_dir, 'actions.png'), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_centroid_map(S, in_dir, gridsize=70, min_count=20, methods=('allocentric', 'pf')):
    """Local R^2 of the belief's own centroid estimate over the true centroid position, on a hexagonal grid.

    Local R^2 in a hexagon = 1 - mean squared centroid error in the hexagon / total centroid variance,
    so hexagons are comparable and their count-weighted mean is the overall centroid R^2.
    """
    path = os.path.join(in_dir, 'centroid.npz')
    if not os.path.exists(path):
        return
    C = np.load(path)
    allv = np.concatenate([C[f'{m}/true'] for m in methods])
    axis_labels = S['meta'].get('centroid_axes')
    nx = gridsize
    ny = int(np.ceil(nx / np.sqrt(3)))
    if axis_labels:
        # scalar centroid (market): the axes are different quantities, so each gets its own range; the
        # centroid range stops at its 92nd percentile, past which only sparse start-of-episode samples lie
        (x0, y0), (x1, y1) = np.percentile(allv, 1, axis=0), [np.percentile(allv[:, 0], 92), np.percentile(allv[:, 1], 99)]
        lo, hi = np.array([x0, y0]), np.array([x1, y1])
    else:
        # square range covering the 1st-99th percentile of both coordinates, so x and y share one scale
        lo = min(np.percentile(allv, 1, axis=0))
        hi = max(np.percentile(allv, 99, axis=0))
        # matplotlib hexagons are regular only when the y bin height is sqrt(3) x the x bin width:
        # with nx hexagons across a width W, use ny rows over a height ny * sqrt(3) * W / nx (>= W)
        x0, x1, y0, y1 = lo, hi, lo, lo + ny * np.sqrt(3) * (hi - lo) / nx
    cmap = BLUE_RAMP.with_extremes(bad=(0, 0, 0, 0))  # hexagons with too few samples stay blank
    fig, axes = plt.subplots(1, len(methods), figsize=(13, 6), facecolor=SURFACE, layout='constrained')
    ims, overall = [], {}
    for ax, m in zip(axes, methods):
        ax.set_facecolor(SURFACE)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        for side in ['left', 'bottom']:
            ax.spines[side].set_color('#c9c8c3')
        ax.tick_params(colors=TEXT_2, labelsize=8)
        t, c = C[f'{m}/true'], C[f'{m}/raw']
        err = np.sum((c - t) ** 2, axis=1)
        tv = float(C[f'{m}/total_var'])
        overall[m] = 1 - err.mean() / tv
        keep = np.all((t >= lo) & (t <= hi), axis=1)  # inside the square plotted range
        # same extent and gridsize in every panel, so the hexagons line up across methods
        ims.append(ax.hexbin(t[keep, 0], t[keep, 1], C=err[keep], gridsize=(nx, ny), extent=(x0, x1, y0, y1),
                             reduce_C_function=lambda v: 1 - np.mean(v) / tv if len(v) >= min_count else np.nan,
                             cmap=cmap, linewidths=0.15, edgecolors='face'))
        name = 'PF (egocentric)' if m == 'pf' else LABELS[m]
        ax.set_title(f'{name}  (overall R² {overall[m]:.2f})', fontsize=10, color=TEXT)
        units = S['meta'].get('units', 'env units')
        if axis_labels:
            ax.set_xlabel(f'{axis_labels[0]} ({units})', color=TEXT_2, fontsize=8)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
        else:
            ax.set_xlabel(f'true centroid x, relative to the agent ({units})', color=TEXT_2, fontsize=8)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect('equal')
    # colour scale fitted to the data (shared by all panels), so spatial differences are visible
    vals = np.concatenate([np.asarray(im.get_array(), dtype=float) for im in ims])
    vmin = max(FLOOR, np.floor(np.nanpercentile(vals, 2) * 10) / 10)
    for im in ims:
        im.set_clim(vmin, 1)
    im = ims[-1]
    ylabel = f'{axis_labels[1]}' if axis_labels else 'true centroid y, relative to the agent'
    axes[0].set_ylabel(f'{ylabel} ({S["meta"].get("units", "env units")})', color=TEXT_2, fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label(f"local R² of the belief's centroid (values below {vmin:.1f} shown at {vmin:.1f})", color=TEXT_2, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=TEXT_2)
    env = S['meta']['env'].replace('_', '-').capitalize()
    fig.suptitle(f'{env}: allocentric vs egocentric (PF) recovery of the teammate centroid, by centroid position',
                 x=0.01, ha='left', fontsize=11, weight='bold', color=TEXT)
    fig.supxlabel(f"Local R² = 1 - mean squared error of the belief's own centroid estimate in the hexagon / total centroid "
                  f'variance (hexagons comparable; their count-weighted mean is the overall R²).\nHexagons with fewer '
                  f'than {min_count} samples left blank. No fitting: the estimate the policy receives.',
                  fontsize=7.5, color=TEXT_2, x=0.01, ha='left')
    fig.savefig(os.path.join(in_dir, 'centroid_map.png'), dpi=400, facecolor=SURFACE)
    plt.close(fig)


def plot_ridgeline(S, in_dir, overlap=1.05):
    """Ridgeline: one row per policy task eigenvector, one curve per method showing the
    distribution of the raw belief error (estimate - truth) along that eigenvector."""
    from scipy.stats import gaussian_kde
    P = np.load(os.path.join(in_dir, 'projections.npz'))
    meta = S['meta']
    names = group_names(meta)
    comp = composition(P)
    eig = P['task_eig']
    n = P['functionals'].shape[1]
    errs = {m: P[f'{m}/belief'] - P[f'{m}/actual'] for m in RIDGE_METHODS}
    allv = np.concatenate([e.ravel() for e in errs.values()])
    lo, hi = np.percentile(allv, [1, 99])
    half = max(abs(lo), abs(hi))
    grid = np.linspace(-half, half, 400)
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(11, 0.68 * n + 1.9), facecolor=SURFACE, layout='constrained')
    ax.set_facecolor(SURFACE)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#c9c8c3')
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.axvline(0, color=TEXT_2, linewidth=1, linestyle='--', zorder=10 * n + 5)
    labels = []
    for k in range(n):
        base = (n - 1 - k) * RIDGE_STEP
        dens, exact = {}, {}
        for m in RIDGE_METHODS:
            e = errs[m][:, k]
            sample = e[rng.choice(len(e), size=min(20000, len(e)), replace=False)]
            if np.ptp(sample) < 1e-9:  # exact estimate: all errors identical, no density to draw
                exact[m] = float(sample[0])
            else:
                dens[m] = gaussian_kde(sample)(grid)
        top = max((d.max() for d in dens.values()), default=1.0)
        ridge_pair(ax, grid, {m: d / top for m, d in dens.items()}, exact, base, k)
        short = {'fully_obs': 'FO', 'allocentric': 'Allo', 'pf': 'PF'}
        rmse = '  '.join(f'{short[m]} {np.sqrt(np.mean(errs[m][:, k] ** 2)):.2f}' for m in RIDGE_METHODS)
        labels.append((base, f'task dir {k + 1}  (σ² {eig[k]:.2f})\n{describe(comp, k, names)}\nRMSE  {rmse}'))
    ax.set_yticks([b for b, _ in labels], [t for _, t in labels], fontsize=8)
    ridge_axes(ax, n)
    ax.set_xlim(-half, half)
    ax.set_xlabel("raw belief error along the task eigenvector: estimate - truth (env units)", color=TEXT_2, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m], alpha=0.6) for m in RIDGE_METHODS]
    ax.legend(handles, [LABELS[m] for m in RIDGE_METHODS], loc='upper right', ncol=3, frameon=False, fontsize=8.5,
              labelcolor=TEXT)
    env = meta['env'].replace('_', '-').capitalize()
    fig.suptitle(f'{env}: belief error along each policy task eigenvector, {meta["title"]}',
                 x=0.01, ha='left', fontsize=11, weight='bold', color=TEXT)
    fig.supxlabel("Each row is one eigenvector of the policy task CCA (FO policy actions vs true state; strongest first), identical for every method. "
                  "Curves: distribution of each belief's own estimate minus the truth, projected on that eigenvector\n"
                  "(no fitting). Narrow and centred on 0 (dashed) = accurate; wide = noisy; shifted = biased. "
                  "Curves scaled to the tallest in each row; x-range = 1st-99th percentile of all errors.\n"
                  "An exact estimate (zero error, e.g. a noise-free observer) is drawn as a vertical line.",
                  fontsize=7.5, color=TEXT_2, x=0.01, ha='left')
    fig.savefig(os.path.join(in_dir, 'task_error_ridgeline.png'), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_r2_ridgeline(S, in_dir, overlap=1.05):
    """Ridgeline: one row per policy task eigenvector, one curve per method showing the distribution
    over episodes of the per-episode R^2 (raw estimate) along that eigenvector."""
    from scipy.stats import gaussian_kde
    path = os.path.join(in_dir, 'episode_r2.npz')
    if not os.path.exists(path):
        return
    E = np.load(path)
    P = np.load(os.path.join(in_dir, 'projections.npz'))
    meta = S['meta']
    names = group_names(meta)
    comp = composition(P)
    eig = P['task_eig']
    n = E[RIDGE_METHODS[0]].shape[1]
    allv = np.concatenate([E[m].ravel() for m in RIDGE_METHODS])
    lo = max(FLOOR, np.floor(np.percentile(allv, 1) * 10) / 10)
    grid = np.linspace(lo, 1.0, 400)

    fig, ax = plt.subplots(figsize=(11, 0.68 * n + 1.9), facecolor=SURFACE, layout='constrained')
    ax.set_facecolor(SURFACE)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#c9c8c3')
    ax.tick_params(colors=TEXT_2, labelsize=8)
    labels = []
    for k in range(n):
        base = (n - 1 - k) * RIDGE_STEP
        dens, exact = {}, {}
        for m in RIDGE_METHODS:
            v = np.clip(E[m][:, k], lo, 1.0)
            if np.ptp(v) < 1e-9:  # identical in every episode (or all below the axis cut)
                exact[m] = float(v[0])
                continue
            try:
                dens[m] = gaussian_kde(v)(grid)
            except np.linalg.LinAlgError:  # nearly constant: fall back to a smoothed histogram
                h, _ = np.histogram(v, bins=len(grid), range=(lo, 1.0))
                dens[m] = np.convolve(h, np.ones(9) / 9, mode='same').astype(float)
        exact = {m: min(max(v, lo), 1.0) for m, v in exact.items()}
        ridge_pair(ax, grid, {m: d / d.max() for m, d in dens.items()}, exact, base, k)  # each to its own peak
        short = {'fully_obs': 'FO', 'allocentric': 'Allo', 'pf': 'PF'}
        med = '  '.join(f'{short[m]} {np.median(E[m][:, k]):.2f}' for m in RIDGE_METHODS)
        labels.append((base, f'task dir {k + 1}  (σ² {eig[k]:.2f})\n{describe(comp, k, names)}\nmedian R²  {med}'))
    ax.set_yticks([b for b, _ in labels], [t for _, t in labels], fontsize=8)
    ridge_axes(ax, n)
    ax.set_xlim(lo, 1.0)
    ax.set_xlabel('per-episode R² of the belief along the task eigenvector', color=TEXT_2, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m], alpha=0.6) for m in RIDGE_METHODS]
    ax.legend(handles, [LABELS[m] for m in RIDGE_METHODS], loc='upper left', ncol=3, frameon=False, fontsize=8.5,
              labelcolor=TEXT)
    env = meta['env'].replace('_', '-').capitalize()
    fig.suptitle(f'{env}: per-episode R² along each policy task eigenvector, {meta["title"]}',
                 x=0.01, ha='left', fontsize=11, weight='bold', color=TEXT)
    fig.supxlabel("Each row is one eigenvector of the policy task CCA (FO policy actions vs true state; strongest first), identical for every method. "
                  "Curves: distribution over episodes of R² = 1 - mean squared error in\nthe episode / variance of the "
                  "truth over all episodes (raw estimate, no fitting). Right = better; wide = inconsistent across "
                  f"episodes. Each curve scaled to its own peak. Values below {lo:.1f} are piled up at {lo:.1f};\nan identical R² in every episode "
                  "is drawn as a vertical line.",
                  fontsize=7.5, color=TEXT_2, x=0.01, ha='left')
    fig.savefig(os.path.join(in_dir, 'task_ridgeline.png'), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in-dir', required=True)
    args = p.parse_args()
    S = json.load(open(os.path.join(args.in_dir, 'summary.json')))
    B = np.load(os.path.join(args.in_dir, 'bootstrap.npz'))
    plot_main(S, B, os.path.join(args.in_dir, 'fixed_axis.png'))
    plot_spectrum(S, B, os.path.join(args.in_dir, 'fixed_axis_spectrum.png'))
    plot_pred_true(S, args.in_dir)
    plot_actions(S, args.in_dir)
    plot_centroid_map(S, args.in_dir)
    plot_ridgeline(S, args.in_dir)
    plot_r2_ridgeline(S, args.in_dir)
    print(f'saved fixed_axis.png, fixed_axis_spectrum.png, pred_vs_true.png, actions.png, centroid_map.png, task_ridgeline.png, task_error_ridgeline.png in {args.in_dir}')


if __name__ == '__main__':
    main()
