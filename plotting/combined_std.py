import argparse
import importlib
import os
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluate'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


'''
combined belief error / belief std / decomposed reward plot for each environment, the same figure each
<env>_evaluate.plot_combined_metrics draws but with the cross-observer belief spread as a standard deviation
instead of a variance: per estimated target, the root-mean-square distance of the observers' estimates from their
average estimate (the square root of the trace of their covariance, i.e. of the variance panel's per-target value),
in the env's distance units, then averaged over targets

belief error, decomposed reward, the teammate matching and the drawing all come from the env's own evaluate module,
so only the middle panel differs from its combined_metrics figure
'''

from util.util import RESULTS_ROOT

ENVS = {
    'predator_prey': dict(module='predator_prey_evaluate', results_root=f'{RESULTS_ROOT}/predator_prey'),
    'drones': dict(module='drones_evaluate', results_root=f'{RESULTS_ROOT}/drones'),
    'fire': dict(module='fire_evaluate', results_root=f'{RESULTS_ROOT}/fire'),
    'football': dict(module='football_evaluate', results_root=f'{RESULTS_ROOT}/football'),
    # market: market_game_evaluate.episode_metrics already gives belief error and the cross-observer std, and its
    # combined_metrics figure is already the three-panel std version, so only the two-panel plot is drawn
    'market_game': dict(module='market_game_evaluate', results_root=f'{RESULTS_ROOT}/market_game',
                        two_panel_only=True),
}


def spread(estimates) -> float:
    '''
    root-mean-square distance of a set of estimates of one target from their average estimate
    '''
    return float(np.sqrt(np.var(np.stack(estimates), axis=0).sum()))


def estimate_std_by_step(env: str, results_dir: str, episode: int, permutation_invariant: bool = True) -> dict:
    '''
    per step, the cross-observer spread of each target's estimates averaged over targets, with the teammates matched
    to the observers' belief slots exactly as the env's agent_estimate_variance_by_step does
    '''
    ev = importlib.import_module(ENVS[env]['module'])
    if env in ('predator_prey', 'drones'):
        return {
            step: float(np.mean([spread(estimates) for estimates in by_target.values()]))
            for step, by_target in ev.matched_belief_estimates_by_step(results_dir, episode).items()
        }

    if env == 'fire':
        beliefs = ev.load_episode_beliefs(results_dir=results_dir, episode=episode, permutation_invariant=permutation_invariant)
    else:
        beliefs = ev.load_episode_beliefs(results_dir=results_dir, episode=episode)
        ball_positions = ev.load_episode_positions(results_dir=results_dir, episode=episode)[3]
    names = sorted(beliefs)
    n_steps = next(iter(beliefs[names[0]].values())).shape[0]

    std_by_step = {}
    for t in range(n_steps):
        spreads = [
            spread([beliefs[observer][target][t] for observer in names if observer != target])
            for target in names
        ]
        if env == 'football':
            spreads.append(spread([ball_positions[observer][t] for observer in names]))
        std_by_step[t + 1] = float(np.mean(spreads))

    return std_by_step


def episode_values(episode: int, env: str, results_dir: str, label: str, controller_cfg) -> tuple:
    '''
    belief error, belief std and decomposed reward for one episode, each step -> value, computed with the same calls
    (and arguments) as the env's evaluate script
    '''
    ev = importlib.import_module(ENVS[env]['module'])
    if env == 'market_game':
        return ev.episode_metrics(episode, results_dir)
    permutation_invariant = label != 'PF'
    if env == 'fire':
        belief_error = ev.belief_error_by_step(results_dir, episode, permutation_invariant)
    else:
        belief_error = ev.belief_error_by_step(results_dir, episode)
    if env == 'football':
        decomposed_reward = ev.decomposed_reward_by_step(results_dir, episode)
    else:
        decomposed_reward = ev.decomposed_reward_by_step(results_dir, episode, controller_cfg)
    belief_std = estimate_std_by_step(env, results_dir, episode, permutation_invariant)

    return belief_error, belief_std, decomposed_reward


def pool_by_step(per_episode: list, max_step: int = None) -> dict:
    combined = {}
    for values_by_step in per_episode:
        for step, value in values_by_step.items():
            if max_step is not None and step > max_step:
                continue
            combined.setdefault(step, []).append(value)
    return {step: np.array(values) for step, values in combined.items()}


def plot_combined_std(env: str, belief_error: dict, belief_std: dict, decomposed_reward: dict, save_path: str):
    '''
    the env's plot_combined_metrics layout with 'Belief Std' as the middle panel: same panel settings (fire's 125-step
    horizon and zero clamp on the spread panel, football's minimum sample count on the reward panel)
    '''
    ev = importlib.import_module(ENVS[env]['module'])
    sns.set_theme(style='darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    extra = {'max_step': 125} if env == 'fire' else {}
    reward_min_samples = ev.MIN_SAMPLES if env == 'football' else 2
    ev._draw_average_with_std(axes[0], belief_error, ylabel='Belief Error', show_legend=False, **extra)
    ev._draw_average_with_std(axes[1], belief_std, ylabel='Belief Std', show_legend=False, **extra,
                              **({'clamp_min': 0} if env == 'fire' else {}))
    ev._draw_average_with_std(
        axes[2], decomposed_reward, ylabel='Decomposed Reward',
        reference_line=ev.CONTROLLER_MEAN_DECOMPOSED_REWARD, reference_label='controller mean',
        min_samples=reward_min_samples, show_legend=False, **extra,
    )

    ev._set_endpoint_yticks(axes[0], y_min=0)
    ev._set_endpoint_yticks(axes[1], y_min=0)
    ev._set_endpoint_yticks(axes[2])
    axes[-1].set_xlabel('step')

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f'saved plot to {save_path}', flush=True)


def plot_combined_std_belief(env: str, belief_error: dict, belief_std: dict, save_path: str):
    '''
    the belief error and belief std panels of plot_combined_std on their own, sharing a step axis
    '''
    ev = importlib.import_module(ENVS[env]['module'])
    sns.set_theme(style='darkgrid')
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True)

    extra = {'max_step': 125} if env == 'fire' else {}
    ev._draw_average_with_std(axes[0], belief_error, ylabel='Belief Error', show_legend=False, **extra)
    ev._draw_average_with_std(axes[1], belief_std, ylabel='Belief Std', show_legend=False, **extra,
                              **({'clamp_min': 0} if env == 'fire' else {}))

    ev._set_endpoint_yticks(axes[0], y_min=0)
    ev._set_endpoint_yticks(axes[1], y_min=0)
    axes[-1].set_xlabel('step')

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f'saved plot to {save_path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', nargs='+', default=list(ENVS), choices=list(ENVS))
    parser.add_argument('--out-root', type=str, default=RESULTS_ROOT,
                        help='plots are saved as <out-root>/<env>/combined_std.png (belief error, belief std and '
                             'decomposed reward) and <out-root>/<env>/combined_std_belief.png (belief error and '
                             'belief std only); the market env (results/market_game) gets only the latter')
    parser.add_argument('--results-root', type=str, default=None,
                        help='folder holding results_allocentric/ and results_pf/ (one env only; '
                             'default: the env\'s results_root in ENVS)')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='save the plots here instead of <out-root>/<env> (one env only)')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    if (args.results_root or args.out_dir) and len(args.envs) != 1:
        parser.error('--results-root / --out-dir need exactly one env in --envs')

    for env in args.envs:
        ev = importlib.import_module(ENVS[env]['module'])
        results_root = args.results_root or ENVS[env]['results_root']
        out_dir = args.out_dir or os.path.join(args.out_root, env)
        result_dirs = {'Allocentric': os.path.join(results_root, 'results_allocentric'),
                       'PF': os.path.join(results_root, 'results_pf')}
        max_step = ev.MAX_STEP if env == 'football' else None

        belief_error, belief_std, decomposed_reward = {}, {}, {}
        for label, results_dir in result_dirs.items():
            controller_cfg = None if env in ('football', 'market_game') else ev.load_controller_cfg(results_dir)
            episodes = ev.list_episodes(results_dir)
            with Pool(args.workers) as pool:
                per_episode = pool.map(partial(episode_values, env=env, results_dir=results_dir, label=label,
                                               controller_cfg=controller_cfg), episodes)
            belief_error[label] = pool_by_step([v[0] for v in per_episode], max_step)
            belief_std[label] = pool_by_step([v[1] for v in per_episode], max_step)
            decomposed_reward[label] = pool_by_step([v[2] for v in per_episode], max_step)
            print(f'{env} {label}: {len(episodes)} episodes', flush=True)

        os.makedirs(out_dir, exist_ok=True)
        if not ENVS[env].get('two_panel_only'):
            plot_combined_std(env, belief_error, belief_std, decomposed_reward, os.path.join(out_dir, 'combined_std.png'))
        plot_combined_std_belief(env, belief_error, belief_std, os.path.join(out_dir, 'combined_std_belief.png'))
