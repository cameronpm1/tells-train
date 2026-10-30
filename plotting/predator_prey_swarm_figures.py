"""Figures for the 40-agent predator-prey swarm with the allocentric belief.

swarm_schematic.png   one frame of the swarm (scripted controller) with the team's 1.5-sigma covariance ellipse,
                      major axis and orientation arc, centroid, prey trail and goal zone; no labels
belief_error.png      per-step error of the belief's swarm summary: centroid, log size, orientation
decomposed_reward.png per-step decomposed reward (PredatorPreyEnv.decompose_reward), in the format of
                      predator_prey_evaluate.py, with the controller reference
belief_maps.png       local R^2 of the belief's (a) centroid over the true centroid position and (b) orientation
                      over the true elongation vector

Usage:
    python plotting/predator_prey_swarm_figures.py
    python plotting/predator_prey_swarm_figures.py --results <checkpoint>/results_allocentric --out-dir <dir>
"""
import argparse
import glob
import os
import sys
from functools import partial
from multiprocessing import Pool
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Arc, Ellipse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.marl.predator_prey_swarm_env import PredatorPreySwarmEnv
from plot_cca_analysis import BLUE_RAMP, COLORS, FLOOR, GRID, SURFACE, TEXT, TEXT_2, style

_rc = plt.rcParams.copy()
from predator_prey_evaluate import _draw_average_with_std, _set_endpoint_yticks  # its import sets bold text globally
plt.rcParams.update(_rc)
BOLD = {'font.weight': 'bold', 'axes.titleweight': 'bold', 'axes.labelweight': 'bold'}

RESULTS_DIR = '/home/cameron/tells/logs/marl/predator_prey_swarm_40a_fully_obs/checkpoint250/results_allocentric'
OUT_DIR = '/home/cameron/tells/results/predator_prey_swarm/figures'

# mean decomposed reward of the scripted swarm controller over 200 40-agent episodes, terminal
# success / oob steps left out as in decomposed_reward_by_step
CONTROLLER_MEAN_DECOMPOSED_REWARD = 0.0084

FOCAL = COLORS['pf']  # the agent's own position in the centroid map
MIN_ELONGATION = 0.2  # orientation is undefined for a round swarm: scored only where |e| >= this


def orientation(e):
    """Major-axis angle phi in (-90, 90] degrees from the elongation vector e = |e| (cos 2phi, sin 2phi)."""
    return np.degrees(0.5 * np.arctan2(e[..., 1], e[..., 0]))


def angle_error(a, b):
    """|a - b| for axis angles in degrees (period 180), in [0, 90]."""
    return np.abs((a - b + 90) % 180 - 90)


def run_config(results_dir):
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(results_dir))), 'config.yaml')) as f:
        return yaml.safe_load(f)


def load_episode(path, controller_cfg):
    """Per step, every agent: true and believed swarm summary, plus the agent-averaged decomposed reward."""
    d = np.load(path, allow_pickle=True)
    obs_map = d['obs_map'].item()
    steps = sorted((f for f in d.files if f.isdigit()), key=int)
    agents = sorted(d[steps[0]][4]['__common__']['obs_no_noise'], key=lambda a: int(a[5:]))
    stub = SimpleNamespace(agents=agents, controller_cfg=controller_cfg, obs_map=obs_map)
    true, pred, reward = [], [], []
    for k in steps:
        _, rewards, _, _, info = d[k]
        c = info['__common__']
        true.append([c['obs_no_noise'][a][obs_map['team']] for a in agents])
        pred.append([c['sampled_predictions'][a] for a in agents])
        m = c[agents[0]]
        if not (m['success'] or m['oob']):
            joint = float(np.mean([rewards[a] for a in agents]))
            reward.append(np.mean(list(PredatorPreySwarmEnv.decompose_reward(stub, joint, c['obs_no_noise']).values())))
        else:
            reward.append(np.nan)
    last = d[steps[-1]][4]['__common__'][agents[0]]
    return dict(true=np.array(true), pred=np.array(pred), reward=np.array(reward), success=bool(last['success']))


def load_all(results_dir, workers):
    files = sorted(glob.glob(os.path.join(results_dir, '*.npz')), key=lambda f: int(os.path.basename(f)[:-4]))
    with Pool(workers) as p:
        return p.map(partial(load_episode, controller_cfg=run_config(results_dir)['env']['controller_kwargs']), files)


def by_step(series, min_episodes=20):
    """Per-step mean and std over episodes (each episode contributes its agent-averaged value), up to the
    last step reached by at least min_episodes episodes."""
    n = max(len(s) for s in series)
    pad = np.full((len(series), n), np.nan)
    for i, s in enumerate(series):
        pad[i, :len(s)] = s
    keep = np.sum(~np.isnan(pad), 0) >= min_episodes
    steps = np.arange(1, n + 1)[keep]
    return steps, np.nanmean(pad[:, keep], 0), np.nanstd(pad[:, keep], 0)


def band(ax, series, color, label):
    steps, mean, std = by_step(series)
    ax.plot(steps, mean, color=color, linewidth=1.6, label=label)
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)


# ---------------------------------------------------------------- schematic

def controller_rollout(cfg, seed):
    """Scripted swarm controller rollout from the run's config: prey and predator positions per step, and the goal."""
    from envs.marl.make_env import make_marl_env
    from controllers.predator_prey_swarm_control import compute_slot_actions
    env = make_marl_env(cfg, seed=seed, wrap=None)
    obs, _ = env.reset(seed=seed)
    prey, predators = [], []
    for _ in range(cfg['env']['max_episode_length']):
        target = obs['target']
        prey.append(target[2:4])
        predators.append(target[2:4] + target[env.obs_map['target_obs']].reshape(-1, 2))
        goal = target[2:4] + target[4:6]
        obs, _, te, tr, _ = env.step(compute_slot_actions(obs, env.obs_map))
        if te['__all__'] or tr['__all__']:
            break
    return np.array(prey), np.array(predators), goal


def pick_frame(prey, predators, goal):
    """A mid-push frame: the swarm in an arc behind the prey (all on the side away from the goal, at a
    similar distance) while the prey is still well short of the goal."""
    best = None
    for t in range(20, len(prey)):
        to_goal = goal - prey[t]
        rel = predators[t] - prey[t]
        dist = np.linalg.norm(rel, axis=1)
        if np.linalg.norm(to_goal) < 2.0 or not 1.5 < dist.mean() < 3.2 or dist.min() < 0.8:
            continue
        score = np.mean(rel @ to_goal < 0) - dist.std() / dist.mean()
        if best is None or score > best[0]:
            best = (score, t)
    return best[1]


def plot_schematic(results_dir, out_dir, seed=3, agent_radius=0.1, trail=30, trail_every=3):
    """One frame of the simulation (scripted controller) with the team's 1.5-sigma covariance ellipse, its
    major axis and orientation arc, and the centroid. No labels, for the paper figure."""
    cfg = run_config(results_dir)
    prey_path, predators, goal = controller_rollout(cfg, seed)
    t = pick_frame(prey_path, predators, goal)
    team, prey = predators[t], prey_path[t]
    goal_radius = cfg['env']['reward_kwargs']['hold_goal_radius']

    mu = team.mean(0)
    cov = (team - mu).T @ (team - mu) / len(team)
    lam, vec = np.linalg.eigh(cov)
    phi = (np.degrees(np.arctan2(vec[1, 1], vec[0, 1])) + 90) % 180 - 90
    half_axis = 2 * np.sqrt(lam[1]) * np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))])
    path = prey_path[max(0, t - trail):t + 1]

    box = 1.5 * np.sqrt(np.diag(cov)) + 0.15
    pts = np.vstack([team, path, goal + goal_radius, goal - goal_radius, mu + box, mu - box])
    lo, hi = pts.min(0) - 0.45, pts.max(0) + 0.45
    w, h = hi - lo
    fig, ax = plt.subplots(figsize=(8, 8 * h / w), facecolor='white')
    fig.subplots_adjust(0, 0, 1, 1)

    blue, tint, prey_c, goal_c = '#3f6fb5', '#9db8e0', '#e4572e', '#59a14f'
    ax.add_patch(plt.Circle(goal, goal_radius, color=goal_c, alpha=0.15, linewidth=0, zorder=0))
    ax.add_patch(plt.Circle(goal, agent_radius * 1.2, facecolor=goal_c, edgecolor='#3b7a34', linewidth=1, zorder=5))
    past = path[:-1][::-1][trail_every - 1::trail_every][::-1]  # every trail_every-th step before now, oldest first
    for k, q in enumerate(past):
        alpha = 0.1 + 0.5 * (k + 1) / len(past)
        ax.add_patch(plt.Circle(q, 0.7 * 1.2 * agent_radius, facecolor=prey_c, edgecolor='#a83a1c', linewidth=0.6,
                                alpha=alpha, zorder=4))
    ax.add_patch(Ellipse(mu, 3 * np.sqrt(lam[1]), 3 * np.sqrt(lam[0]), angle=phi, facecolor=tint, alpha=0.08,
                         linewidth=0, zorder=1))
    ax.add_patch(Ellipse(mu, 3 * np.sqrt(lam[1]), 3 * np.sqrt(lam[0]), angle=phi, fill=False, edgecolor=tint,
                         alpha=0.55, linewidth=12, zorder=2))
    # long axis: as long as it can be along phi while both ends stay 0.2 inside the frame
    d = half_axis / np.linalg.norm(half_axis)
    inner_lo, inner_hi = lo + 0.2, hi - 0.2
    reach = min(np.min(np.where(d != 0, np.maximum((inner_hi - mu) / d, (inner_lo - mu) / d), np.inf)),
                np.min(np.where(d != 0, np.maximum((inner_hi - mu) / -d, (inner_lo - mu) / -d), np.inf)))
    ax.plot(*np.array([mu - reach * d, mu + reach * d]).T, color=TEXT, linewidth=1.4, solid_capstyle='round', zorder=5.5)
    r = 0.28 * np.linalg.norm(half_axis)
    ax.plot([mu[0], mu[0] + 3 * r], [mu[1], mu[1]], color=TEXT, linewidth=4, linestyle=(0, (1, 1.5)),
            dash_capstyle='round', zorder=3)
    ax.add_patch(Arc(mu, 2 * r, 2 * r, theta1=min(0, phi), theta2=max(0, phi), color=TEXT, linewidth=2.5, zorder=4))
    for p in team:
        ax.add_patch(plt.Circle(p, agent_radius, facecolor=blue, edgecolor='#2b4f86', linewidth=0.8, zorder=5))
    ax.add_patch(plt.Circle(prey, agent_radius * 1.2, facecolor=prey_c, edgecolor='#a83a1c', linewidth=1, zorder=5))
    ax.scatter(*mu, s=380, color=TEXT, edgecolors='white', linewidths=3, zorder=6)

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.savefig(os.path.join(out_dir, 'swarm_schematic.png'), dpi=300, facecolor='white')
    plt.close(fig)
    print(f'schematic: controller seed {seed}, step {t}, φ {phi:.0f}°')


# ---------------------------------------------------------------- belief error and reward

def belief_errors(e):
    t, p = e['true'], e['pred']
    phi_err = angle_error(orientation(p[..., 3:]), orientation(t[..., 3:]))
    scored = np.linalg.norm(t[..., 3:], axis=-1) >= MIN_ELONGATION
    n = scored.sum(1)
    return dict(centroid=np.linalg.norm(p[..., :2] - t[..., :2], axis=-1).mean(1),
                size=np.abs(p[..., 2] - t[..., 2]).mean(1),
                phi=np.where(n > 0, np.sum(phi_err * scored, 1) / np.maximum(n, 1), np.nan))


def plot_belief_error(eps, out_dir):
    errs = [belief_errors(e) for e in eps]
    true = np.concatenate([e['true'].reshape(-1, 5) for e in eps])
    baseline = dict(centroid=np.linalg.norm(true[:, :2] - true[:, :2].mean(0), axis=1).mean(),
                    size=np.abs(true[:, 2] - true[:, 2].mean()).mean(), phi=45.0)
    panels = [('centroid', 'centroid error\n‖μ̂ − μ‖ (env units)', 'predict the mean centroid'),
              ('size', 'log-size error\n|ŝ − s|', 'predict the mean size'),
              ('phi', 'orientation error\n|φ̂ − φ| (degrees)', 'random orientation')]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.4), facecolor=SURFACE, sharex=True, layout='constrained')
    for ax, (key, ylabel, ref) in zip(axes, panels):
        style(ax)
        band(ax, [x[key] for x in errs], COLORS['allocentric'], 'allocentric belief (mean ± 1 std over episodes)')
        ax.axhline(baseline[key], color=TEXT_2, linestyle='--', linewidth=1, label=f'no information: {ref}')
        ax.set_ylim(0, None)
        ax.set_ylabel(ylabel, color=TEXT_2, fontsize=9)
        ax.legend(loc='upper right', fontsize=8, frameon=False, labelcolor=TEXT)
    axes[-1].set_xlabel('episode step', color=TEXT_2, fontsize=9)
    fig.suptitle(f'Belief error over the episode, 40-agent swarm ({len(eps)} episodes, all agents)',
                 x=0.01, ha='left', fontsize=11, weight='bold', color=TEXT)
    fig.supxlabel(f'Each episode contributes its agent-averaged error per step; steps reached by fewer than 20 episodes '
                  f'are not shown.\nOrientation scored only where the true elongation |e| ≥ {MIN_ELONGATION} '
                  '(a rounder swarm has no defined orientation).',
                  fontsize=7.5, color=TEXT_2, x=0.01, ha='left')
    fig.savefig(os.path.join(out_dir, 'belief_error.png'), dpi=300, facecolor=SURFACE)
    plt.close(fig)


def plot_decomposed_reward(eps, out_dir):
    """Format of predator_prey_evaluate.py (seaborn darkgrid, bold), without a legend and with only the end
    y-ticks, like the decomposed reward panel of its combined figure."""
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


# ---------------------------------------------------------------- maps

def hex_r2(ax, xy, sq_err, total_var, extent, gridsize, min_count=20):
    """Local R^2 = 1 - mean squared error in the hexagon / total variance, on regular hexagons."""
    nx, ny = gridsize, int(np.ceil(gridsize / np.sqrt(3)))
    x0, x1, y0, _ = extent
    return ax.hexbin(xy[:, 0], xy[:, 1], C=sq_err, gridsize=(nx, ny),
                     extent=(x0, x1, y0, y0 + ny * np.sqrt(3) * (x1 - x0) / nx),
                     reduce_C_function=lambda v: 1 - np.mean(v) / total_var if len(v) >= min_count else np.nan,
                     cmap=BLUE_RAMP.with_extremes(bad=(0, 0, 0, 0)), linewidths=0.15, edgecolors='face')


def map_axes(ax, lim):
    ax.set_facecolor('white')
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#c9c8c3')
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect('equal')


def plot_maps(eps, out_dir, gridsize=45):
    true = np.concatenate([e['true'].reshape(-1, 5) for e in eps])
    pred = np.concatenate([e['pred'].reshape(-1, 5) for e in eps])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor='white', layout='constrained')

    # (a) centroid: hexagons over the full data range, sparse edges left blank, axes padded around them
    t, c = true[:, :2], pred[:, :2]
    err = np.sum((c - t) ** 2, 1)
    tv = np.sum(np.var(t, 0))
    lo, hi = np.min(np.percentile(t, 0.05, 0)), np.max(np.percentile(t, 99.95, 0))
    pad = 0.15 * (hi - lo)
    ims = [hex_r2(axes[0], t, err, tv, (lo, hi, lo, hi), gridsize)]
    map_axes(axes[0], (lo - pad, hi + pad))
    axes[0].set_xlabel('centroid x relative to agent', color=TEXT_2, fontsize=9)
    axes[0].set_ylabel('centroid y relative to agent', color=TEXT_2, fontsize=9)
    axes[0].scatter(0, 0, s=60, color=FOCAL, edgecolors=TEXT, linewidths=0.8, zorder=5)
    axes[0].set_title(f'a   centroid μ  (overall R² {1 - err.mean() / tv:.2f})', loc='left', fontsize=10.5, color=TEXT)

    # (b) orientation: R^2 of the doubled-angle unit vector (cos 2φ, sin 2φ), every agent-step
    e, e_hat = true[:, 3:], pred[:, 3:]
    u = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
    u_hat = e_hat / np.maximum(np.linalg.norm(e_hat, axis=1, keepdims=True), 1e-9)
    err_u = np.sum((u_hat - u) ** 2, 1)
    tv_u = np.sum(np.var(u, 0))
    ims.append(hex_r2(axes[1], e, err_u, tv_u, (-1, 1, -1, 1), gridsize))
    map_axes(axes[1], (-1.1, 1.1))
    axes[1].set_xticks([-1, 0, 1])
    axes[1].set_yticks([-1, 0, 1])
    axes[1].set_xlabel('elongation e₁ = |e| cos 2φ', color=TEXT_2, fontsize=9)
    axes[1].set_ylabel('elongation e₂ = |e| sin 2φ', color=TEXT_2, fontsize=9)
    axes[1].set_title(f'b   orientation φ  (overall R² {1 - err_u.mean() / tv_u:.2f})', loc='left', fontsize=10.5,
                      color=TEXT)

    vals = np.concatenate([np.asarray(im.get_array(), dtype=float) for im in ims])
    vmin = max(FLOOR, np.floor(np.nanpercentile(vals, 2) * 10) / 10)
    for im in ims:
        im.set_clim(vmin, 1)
    cb = fig.colorbar(ims[-1], ax=axes, fraction=0.02, pad=0.01)
    cb.ax.tick_params(labelsize=7, colors=TEXT_2)
    cb.set_label('local R²', color=TEXT_2, fontsize=9)
    fig.savefig(os.path.join(out_dir, 'belief_maps.png'), dpi=300, facecolor='white')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default=RESULTS_DIR)
    p.add_argument('--out-dir', default=OUT_DIR)
    p.add_argument('--workers', type=int, default=32)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plot_schematic(args.results, args.out_dir)
    eps = load_all(args.results, args.workers)
    print(f'loaded {len(eps)} episodes, success {np.mean([e["success"] for e in eps]):.2f}')
    plot_belief_error(eps, args.out_dir)
    plot_decomposed_reward(eps, args.out_dir)
    plot_maps(eps, args.out_dir)
    print(f'saved swarm_schematic.png, belief_error.png, decomposed_reward.png, belief_maps.png in {args.out_dir}')


if __name__ == '__main__':
    main()
