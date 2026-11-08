"""Figures and table for the 60-company market game with the allocentric belief.

market_schematic.png    one frame of the controller rollout: every company on a 1d stockpile axis (beeswarm), and
                        parallel below it the stockpile distribution with its median, 10th and 90th percentiles
decomposed_reward.png   per-step decomposed reward (MarketGameEnv.decompose_reward) of the allocentric run, in the
                        format of predator_prey_evaluate.py, with the controller reference
centroid_map.png        local R^2 of the belief's centroid (teammates' median stockpile) over the true median and
                        the true standard deviation of teammates' stockpiles
results_table_market_large.tex   success rate and episode length, fully observable vs allocentric

Usage:
    python plotting/market_game_large_figures.py
    python plotting/market_game_large_figures.py --results-root results/market_game_large --out-dir <dir>
"""
import argparse
import glob
import os
import sys
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluate'))

from envs.marl.market_game_env import MarketGameEnv
from plot_cca_analysis import BLUE_RAMP, FLOOR, TEXT, TEXT_2

_rc = plt.rcParams.copy()
from predator_prey_evaluate import _draw_average_with_std, _set_endpoint_yticks  # its import sets bold text globally
plt.rcParams.update(_rc)
from market_game_evaluate import load_env_stub
from util.util import find_run_config, RESULTS_ROOT as _RESULTS
BOLD = {'font.weight': 'bold', 'axes.titleweight': 'bold', 'axes.labelweight': 'bold'}

RESULTS_ROOT = os.path.join(_RESULTS, 'market_game_large')  # results_fully_obs/, results_allocentric/, config.yaml
OUT_DIR = os.path.join(RESULTS_ROOT, 'figures')
TABLE_PATH = os.path.join(OUT_DIR, 'results_table_market_large.tex')

# mean decomposed task reward (team reward without the imitation term) of the centralized controller over 200
# 60-company episodes (run config), with the terminal step of episodes that end early left out as in load_episode
CONTROLLER_MEAN_DECOMPOSED_REWARD = -0.004454

Q_MEDIAN = 2  # position of the median in the (min, q10, median, q90, max) teammate summary


def load_episode(path, results_dir):
    '''
    per step, every agent: true and believed teammate quantiles (material units), plus the agent-averaged
    decomposed task reward (the team reward without its imitation term); success and length of the episode
    '''
    d = np.load(path, allow_pickle=True)
    obs_map = d['obs_map'].item()
    stub = load_env_stub(results_dir, obs_map)
    with open(find_run_config(results_dir)) as f:
        bc_scale = yaml.safe_load(f)['env']['reward_kwargs']['bc_scale']
    steps = sorted((f for f in d.files if f.isdigit()), key=int)
    true, pred, team_std, reward = [], [], [], []
    for k in steps:
        _, rewards, _, _, info = d[k]
        c = info['__common__']
        if 'sampled_predictions' in c:
            true.append([c['obs_no_noise'][a][obs_map['team']] for a in stub.agents])
            pred.append([c['sampled_predictions'][a] for a in stub.agents])
            stocks = np.array([c['obs_no_noise'][a][obs_map['self_stock']][0] for a in stub.agents])
            team_std.append([np.std(np.delete(stocks, i)) for i in range(len(stocks))])
        m = c[stub.agents[0]]
        if not (m['dead'] or m['short']):
            joint = float(np.mean([rewards[a] for a in stub.agents])) - bc_scale * m['controller_action_reward']
            reward.append(np.mean(list(MarketGameEnv.decompose_reward(stub, joint, c['obs_no_noise']).values())))
        else:
            reward.append(np.nan)
    last = d[steps[-1]][4]['__common__'][stub.agents[0]]
    return dict(true=np.array(true) * stub.scale, pred=np.array(pred) * stub.scale,
                team_std=np.array(team_std) * stub.scale, reward=np.array(reward),
                success=bool(last['success']), length=len(steps))


def load_all(results_dir, workers):
    files = sorted(glob.glob(os.path.join(results_dir, '*.npz')), key=lambda f: int(os.path.basename(f)[:-4]))
    with Pool(workers) as p:
        return p.map(partial(load_episode, results_dir=results_dir), files)


def controller_rollout(cfg, seed):
    '''
    centralized controller rollout from the run's config: every company's stockpile per step (material units)
    '''
    from copy import deepcopy
    from envs.marl.make_env import make_marl_env
    from controllers.market_game_large_control import compute_market_actions
    env = make_marl_env(cfg, seed=seed, wrap=None)
    obs, _ = env.reset(seed=seed)
    stocks = []
    while True:
        obs, _, te, tr, _ = env.step(compute_market_actions(deepcopy(obs), env.obs_map, env.controller_cfg))
        stocks.append(env.stocks.copy())
        if te['__all__'] or tr['__all__']:
            return np.array(stocks)


def beeswarm(x, radius):
    '''
    vertical offsets that keep circles of the given radius from overlapping, alternating above and below the line
    '''
    y = np.zeros_like(x)
    placed = []
    for i in np.argsort(x):
        for k in range(len(x)):
            cand = (k + 1) // 2 * (1 if k % 2 else -1) * 2 * radius * 0.9
            if all((x[i] - x[j]) ** 2 + (cand - y[j]) ** 2 >= (2 * radius) ** 2 for j in placed):
                y[i] = cand
                break
        placed.append(i)
    return y


def plot_schematic(cfg, out_dir, seed=0, start=20, radius=0.55):
    '''
    one frame of the controller rollout: every company placed by its stockpile on a 1d axis, and, parallel below it,
    the distribution of stockpiles with its median, 10th and 90th percentiles; no labels, for the paper figure
    '''
    from scipy.stats import gaussian_kde
    stocks = controller_rollout(cfg, seed)
    q = np.quantile(stocks, [0.1, 0.5, 0.9], axis=1)
    t = start + int(np.argmax(q[2, start:] - q[0, start:]))
    s, (q10, med, q90) = stocks[t], q[:, t]

    blue, edge, tint = '#3f6fb5', '#2b4f86', '#9db8e0'
    kde = gaussian_kde(s)
    tail = 3 * np.sqrt(kde.covariance[0, 0])  # the density is ~1% of its peak this far past the outermost company
    lo, hi = s.min() - tail, s.max() + tail
    y_dots = beeswarm(s, radius)
    gap = 1.5 - y_dots.min() + radius
    grid = np.linspace(lo, hi, 400)
    density = kde(grid)
    density *= 1.1 * (y_dots.max() - y_dots.min() + 2 * radius) / density.max()
    base = -gap - density.max()

    fig, ax = plt.subplots(figsize=(10, 10 * (gap + density.max() + y_dots.max() + 2 * radius + 1.6) / (hi - lo)),
                           facecolor='white')
    fig.subplots_adjust(0, 0, 1, 1)
    for y in [0, base]:
        ax.plot([lo, hi], [y, y], color='#c9c8c3', linewidth=1.2, zorder=0)
    for v in [q10, q90]:
        ax.plot([v, v], [base, y_dots.max() + radius], color='#c9c8c3', linewidth=1, linestyle=(0, (2, 3)), zorder=0)
    ax.fill_between(grid, base, base + density, color=tint, alpha=0.25, linewidth=0, zorder=1)
    ax.plot(grid, base + density, color=tint, linewidth=6, alpha=0.8, solid_capstyle='round', zorder=2)
    top = np.interp([q10, med, q90], grid, base + density)
    for v, h in zip([q10, q90], top[[0, 2]]):
        ax.plot([v, v], [base, h], color=TEXT, linewidth=4, linestyle=(0, (1, 1.5)), dash_capstyle='round', zorder=3)
    ax.plot([med, med], [base, top[1]], color=TEXT, linewidth=2.5, solid_capstyle='round', zorder=3)
    ax.scatter(med, base, s=380, color=TEXT, edgecolors='white', linewidths=3, zorder=6, clip_on=False)
    for x, y in zip(s, y_dots):
        ax.add_patch(plt.Circle((x, y), radius, facecolor=blue, edgecolor=edge, linewidth=0.8, zorder=5))

    ax.set_xlim(lo, hi)
    ax.set_ylim(base - 0.8, y_dots.max() + radius + 0.8)
    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.savefig(os.path.join(out_dir, 'market_schematic.png'), dpi=300, facecolor='white', bbox_inches='tight',
                pad_inches=0.1)
    plt.close(fig)
    print(f'schematic: controller seed {seed}, step {t + 1}, q10 {q10:.1f}, median {med:.1f}, q90 {q90:.1f}')


def plot_decomposed_reward(eps, out_dir):
    '''
    format of predator_prey_evaluate.py (seaborn darkgrid, bold), without a legend and with only the end y-ticks,
    as in the swarm figure
    '''
    import seaborn as sns
    values_by_step = {}
    for e in eps:
        for step, v in enumerate(e['reward'], start=1):
            if not np.isnan(v):
                values_by_step.setdefault(step, []).append(v)
    values_by_step = {s: np.array(v) for s, v in values_by_step.items()}
    with plt.rc_context(BOLD):
        sns.set_theme(style='darkgrid')
        fig, ax = plt.subplots(figsize=(10, 4))
        _draw_average_with_std(ax, {'Allocentric': values_by_step}, ylabel='Decomposed Reward',
                               reference_line=CONTROLLER_MEAN_DECOMPOSED_REWARD, min_samples=2, show_legend=False)
        _set_endpoint_yticks(ax)
        ax.set_xlabel('step')
        ax.set_title('Average Decomposed Reward over Episode Steps', fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'decomposed_reward.png'), dpi=300)
        plt.close(fig)
        sns.reset_orig()


def plot_centroid_map(eps, out_dir, gridsize=45, min_count=20):
    '''
    hexagons placed by the true median and standard deviation of teammates' stockpiles, coloured by the local R^2 of
    the believed median: 1 - mean squared error in the hexagon / total variance of the true median
    '''
    true = np.concatenate([e['true'].reshape(-1, 5) for e in eps])
    pred = np.concatenate([e['pred'].reshape(-1, 5) for e in eps])
    xy = np.stack([true[:, Q_MEDIAN], np.concatenate([e['team_std'].reshape(-1) for e in eps])], 1)
    err = (pred[:, Q_MEDIAN] - true[:, Q_MEDIAN]) ** 2
    tv = np.var(true[:, Q_MEDIAN])

    lo, hi = np.percentile(xy, 0.05, 0), np.percentile(xy, 99.95, 0)
    pad = 0.15 * (hi - lo)
    fig, ax = plt.subplots(figsize=(6.8, 6), facecolor='white', layout='constrained')
    im = ax.hexbin(xy[:, 0], xy[:, 1], C=err, gridsize=(gridsize, int(np.ceil(gridsize / np.sqrt(3)))),
                   extent=(lo[0], hi[0], lo[1], hi[1]),
                   reduce_C_function=lambda v: 1 - np.mean(v) / tv if len(v) >= min_count else np.nan,
                   cmap=BLUE_RAMP.with_extremes(bad=(0, 0, 0, 0)), linewidths=0.15, edgecolors='face')
    ax.set_facecolor('white')
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#c9c8c3')
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(max(0, lo[1] - pad[1]), hi[1] + pad[1])
    ax.set_box_aspect(1)
    ax.set_xlabel("teammates' median stockpile", color=TEXT_2, fontsize=9)
    ax.set_ylabel("teammates' stockpile standard deviation", color=TEXT_2, fontsize=9)
    ax.set_title(f'centroid  (overall R² {1 - err.mean() / tv:.2f})', loc='left', fontsize=10.5, color=TEXT)

    vals = np.asarray(im.get_array(), dtype=float)
    im.set_clim(max(FLOOR, np.floor(np.nanpercentile(vals, 2) * 10) / 10), 1)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=TEXT_2)
    cb.set_label('local R²', color=TEXT_2, fontsize=9)
    fig.savefig(os.path.join(out_dir, 'centroid_map.png'), dpi=300, facecolor='white')
    plt.close(fig)


def write_table(runs, path):
    '''
    success rate and episode length, mean +/- std over evaluation episodes, in the format of results_table_swarm.tex
    '''
    cells = {}
    for name, eps in runs.items():
        success = 100 * np.array([e['success'] for e in eps], dtype=float)
        length = np.array([e['length'] for e in eps], dtype=float)
        cells[name] = (f'${success.mean():.2f}\\% \\pm {success.std():.2f}$', f'${length.mean():.2f} \\pm {length.std():.2f}$')
    n = len(next(iter(runs.values())))
    table = (
        '\\begin{table}[h]\n'
        f'\\caption{{Success rate on the 60-company market game for the fully observable and allocentric methods, mean '
        f'$\\pm$ standard deviation over {n} evaluation episodes. Both use the same policy checkpoint; the allocentric '
        'agents observe no teammate information and infer the quantiles of their teammates\' stockpiles with the belief '
        'model.}\\label{tab:market_large_success}\n'
        '\\small\n'
        '\\begin{tabular}{@{}lcc@{}}\n'
        '\\toprule\n'
        ' & \\multicolumn{2}{c}{Market Game ($N = 60$)} \\\\\n'
        '\\cmidrule(lr){2-3}\n'
        'Metric & Fully Obs. & Alloc. \\\\\n'
        '\\midrule\n'
        f'Success Rate & {cells["fully_obs"][0]} & {cells["allocentric"][0]} \\\\\n'
        f'Ep.\\ Length & {cells["fully_obs"][1]} & {cells["allocentric"][1]} \\\\\n'
        '\\botrule\n'
        '\\end{tabular}\n'
        '\\end{table}\n'
    )
    with open(path, 'w') as f:
        f.write(table)
    print(table)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-root', default=RESULTS_ROOT,
                   help='folder holding results_fully_obs/ and results_allocentric/ (and the run config.yaml)')
    p.add_argument('--out-dir', default=OUT_DIR)
    p.add_argument('--table', default=TABLE_PATH)
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(find_run_config(os.path.join(args.results_root, 'results_allocentric'))) as f:
        plot_schematic(yaml.safe_load(f), args.out_dir)
    runs = {name: load_all(os.path.join(args.results_root, sub), args.workers)
            for name, sub in [('fully_obs', 'results_fully_obs'), ('allocentric', 'results_allocentric')]}
    for name, eps in runs.items():
        print(f'{name}: {len(eps)} episodes, success {np.mean([e["success"] for e in eps]):.3f}')
    write_table(runs, args.table)
    plot_decomposed_reward(runs['allocentric'], args.out_dir)
    plot_centroid_map(runs['allocentric'], args.out_dir)
    print(f'saved market_schematic.png, decomposed_reward.png, centroid_map.png in {args.out_dir}')


if __name__ == '__main__':
    main()
