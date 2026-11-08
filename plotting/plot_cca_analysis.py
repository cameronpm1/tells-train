"""Paper figures for evals/marl/cca_analysis.py, scored against the truth
as raw R^2 = 1 - MSE / var(truth) of each belief's own estimate - exactly what the policy receives, no fitting.

centroid_map.png    local R^2 of the belief's teammate centroid over the true centroid position
                    (x, y relative to the agent)
task_ridgeline.png  one row per policy task eigenvector: distribution over episodes of each belief's
                    per-episode R^2 along it
noise_sweep.png     (--sweep) weighted task R^2 against success rate across belief-noise levels

Usage:
    python plotting/plot_cca_analysis.py --in-dir results/football/cca
    python plotting/plot_cca_analysis.py --sweep results/football/noise_sweep/sweep.json
    python plotting/plot_cca_analysis.py --sweep-combined   # -> results/noise_sweep_combined.png
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.util import RESULTS_ROOT  # noqa: E402

LABELS = {'fully_obs': 'Fully observed', 'allocentric': 'Allocentric', 'pf': 'PF'}
COLORS = {'fully_obs': '#1baf7a', 'allocentric': '#2a78d6', 'pf': '#eb6834'}
TEXT, TEXT_2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BLUE_RAMP = LinearSegmentedColormap.from_list(
    'blue', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
RIDGE_METHODS = ['allocentric', 'pf']  # ridgelines compare the two beliefs only
FLOOR = -1.0  # R^2 axes are cut here (raw R^2 is unbounded below)
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
        labels.append((base, f'task dir {k + 1}  (σ² {eig[k]:.2f})\nmedian R²  {med}'))
    ax.set_yticks([b for b, _ in labels], [t for _, t in labels], fontsize=8)
    ridge_axes(ax, n)
    ax.set_xlim(lo, 1.0)
    ax.set_xlabel('per-episode R² of the belief along the task eigenvector', color=TEXT_2, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m], alpha=0.6) for m in RIDGE_METHODS]
    ax.legend(handles, [LABELS[m] for m in RIDGE_METHODS], loc='upper left', ncol=3, frameon=False, fontsize=8.5,
              labelcolor=TEXT)
    short = {'allocentric': 'Aloc.', 'pf': 'Egoc.'}
    # eigenvalue-weighted mean of the raw task R^2 (sum_k sigma_k^2 R_k^2 / sum_k sigma_k^2), as in the noise sweep
    w = np.asarray(S['estimate']['task_eig'], dtype=float)
    wr2 = ',  '.join(f"{short[m]} {float((w * np.asarray(S['estimate'][m]['raw']['task'])).sum() / w.sum()):.3f}"
                     for m in RIDGE_METHODS)
    steps = S.get('settings', {}).get('max_steps', 200)
    window = f'   (first {steps} steps of each episode)' if steps < 200 else ''  # only when episodes were cut short
    fig.suptitle(f'$R^2_w$:  {wr2}{window}', x=0.01, ha='left', fontsize=11, weight='bold', color=TEXT)
    fig.savefig(os.path.join(in_dir, 'task_ridgeline.png'), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def load_sweep(path):
    """Sweep runs as {label: dict}: from a sweep.json (cca_analysis.py --sweep), or from a noise_sweep_results/
    folder of one .npz per run (results/compact_noise_sweeps.py)."""
    if os.path.isdir(path):
        R = {}
        for f in sorted(os.listdir(path)):
            if f.endswith('.npz'):
                d = np.load(os.path.join(path, f))
                label = 'egocentric' if str(d['method']) == 'egocentric' else f"allocentric:{float(d['noise_level']):g}"
                R[label] = dict(weighted_coverage=float(d['weighted_r2']), success_rate=float(d['success_rate']))
        return R
    return json.load(open(path))['runs']


def plot_sweep(path):
    """Belief-noise sweep: weighted task R^2 (x) against success rate (y), in seaborn. Allocentric runs (labels
    'allocentric:<sigma>') are blue and joined in order of sigma; the egocentric (PF) run is an orange dot.
    path: sweep.json or a noise_sweep_results/ folder; the figure is written next to it as noise_sweep.png."""
    import pandas as pd
    import seaborn as sns
    R = load_sweep(path)
    rows = [dict(sigma=float(k.split(':')[1]) if k.startswith('allocentric') else np.nan,
                 method='allocentric' if k.startswith('allocentric') else 'egocentric',
                 weighted_r2=v['weighted_coverage'], success=v['success_rate']) for k, v in R.items()]
    df = pd.DataFrame(rows).sort_values('sigma')
    allo, ego = df[df.method == 'allocentric'], df[df.method == 'egocentric']
    palette = {'allocentric': sns.color_palette()[0], 'egocentric': sns.color_palette()[1]}  # seaborn blue; orange
    with sns.axes_style('whitegrid'):
        fig, ax = plt.subplots(figsize=(8, 4.2), layout='constrained')
        sns.lineplot(data=allo, x='weighted_r2', y='success', sort=False, color=palette['allocentric'],
                     linewidth=2, ax=ax)
        sns.scatterplot(data=pd.concat([allo, ego]), x='weighted_r2', y='success', hue='method', palette=palette,
                        s=70, edgecolor='white', linewidth=0.8, zorder=3, ax=ax)
    ax.set_xlabel('weighted task-direction R²')
    ax.set_ylabel('success rate')
    ax.set_ylim(-0.03, 1.03)
    ax.legend(title=None, frameon=False, loc='upper left')
    out = os.path.join(os.path.dirname(path.rstrip('/')), 'noise_sweep.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f'saved {out}')


SWEEP_ENVS = {  # env label: (noise_sweep_results folder, color, marker)
    'Predator-prey': (os.path.join(RESULTS_ROOT, 'predator_prey/noise_sweep/noise_sweep_results'), '#2a78d6', 'o'),
    'Drone caravan': (os.path.join(RESULTS_ROOT, 'drones/noise_sweep/noise_sweep_results'), '#eb6834', 's'),
    'Fire': (os.path.join(RESULTS_ROOT, 'fire/noise_sweep/noise_sweep_results'), '#1baf7a', '^'),
    'Market': (os.path.join(RESULTS_ROOT, 'market_game/noise_sweep/noise_sweep_results'), '#8e4ec6', 'D'),
    'Football': (os.path.join(RESULTS_ROOT, 'football/noise_sweep/noise_sweep_results'), '#d6336c', 'v'),
}


def logistic(x, lo, hi, k, x0):
    return lo + (hi - lo) / (1 + np.exp(-k * (x - x0)))


def fit_logistic(x, y):
    """4-parameter logistic fit of success rate on R^2, asymptotes kept inside [0, 1]."""
    from scipy.optimize import curve_fit
    p0 = [max(y.min(), 0), min(y.max(), 1), 10.0, x[np.argmin(np.abs(y - (y.min() + y.max()) / 2))]]
    p, _ = curve_fit(logistic, x, y, p0=p0, bounds=([0, 0, 0, -2], [1, 1, 200, 2]), maxfev=20000)
    return p


def plot_sweep_combined(out=os.path.join(RESULTS_ROOT, 'noise_sweep_combined.png'), envs=SWEEP_ENVS):
    """Belief-noise sweeps of several envs on one axis: per env, a logistic fit of success rate against weighted
    task R^2 over the allocentric noise levels, with those points on top (one color + marker per env)."""
    import seaborn as sns
    with sns.axes_style('whitegrid'):
        fig, ax = plt.subplots(figsize=(8, 4.6), layout='constrained')
    for name, (path, color, marker) in envs.items():
        if not os.path.exists(path):
            print(f'skipping {name}: {path} not found')
            continue
        R = load_sweep(path)
        allo = np.array([(v['weighted_coverage'], v['success_rate']) for k, v in R.items() if k.startswith('allocentric')])
        allo = allo[np.argsort(allo[:, 0])]
        x, y = allo[:, 0], allo[:, 1]
        p = fit_logistic(x, y)
        xs = np.linspace(x.min(), x.max(), 300)
        ax.plot(xs, logistic(xs, *p), color=color, linewidth=2.2, alpha=0.9, zorder=2)
        ax.scatter(x, y, color=color, marker=marker, s=48 * (1.5 if marker in '^v' else 1), edgecolor='white', linewidth=0.8, zorder=3)
    ax.set_xlabel('weighted task-direction R²')
    ax.set_ylabel('success rate')
    ax.set_ylim(-0.03, 1.03)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f'saved {out}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep-combined', default=None, nargs='?', const=os.path.join(RESULTS_ROOT, 'noise_sweep_combined.png'),
                   help='draw the combined noise sweep (predator-prey, drones, fire, market, football) to this path')
    p.add_argument('--in-dir', default=None)
    p.add_argument('--sweep', default=None, help='sweep.json from cca_analysis.py --sweep, or a noise_sweep_results/ '
                                                 'folder of per-level .npz files: draw noise_sweep.png only')
    args = p.parse_args()
    if args.sweep_combined:
        plot_sweep_combined(args.sweep_combined)
        return
    if args.sweep:
        plot_sweep(args.sweep)
        return
    S = json.load(open(os.path.join(args.in_dir, 'summary.json')))
    plot_centroid_map(S, args.in_dir)
    plot_r2_ridgeline(S, args.in_dir)
    print(f'saved centroid_map.png, task_ridgeline.png in {args.in_dir}')


if __name__ == '__main__':
    main()
