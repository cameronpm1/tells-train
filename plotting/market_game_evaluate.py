import argparse
import os
import sys
from functools import partial
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.marl.market_game_env import MarketGameEnv
from predator_prey_evaluate import list_episodes, _draw_average_with_std, _set_endpoint_yticks, plot_average_with_std


RESULTS_DIR = '/home/cameron/tells/logs/marl/market_game_7a_fully_obs/checkpoint829/results_allocentric'
RESULTS_PF_DIR = '/home/cameron/tells/logs/marl/market_game_7a_fully_obs/checkpoint829/results_pf'
SAVE_DIR = '/home/cameron/tells/results/market_game'

# mean decomposed reward over 500 controller-driven episodes of the 7 company game (logs/marl/market_game_7a_fully_obs
# config), with the terminal step of episodes that end early (a company dies or the team falls under rho) filtered out
# the same way decomposed_reward_by_step does below
CONTROLLER_MEAN_DECOMPOSED_REWARD = 0.140489


def load_env_stub(results_dir: str, obs_map: dict) -> SimpleNamespace:
    '''
    walk up from a checkpoint's results dir (<run_dir>/<checkpoint>/results) to the run's config.yaml and build the
    attributes MarketGameEnv.decompose_reward reads, so it can be called unbound on logged observations
    '''
    checkpoint_dir = os.path.dirname(os.path.normpath(results_dir))
    with open(os.path.join(os.path.dirname(checkpoint_dir), 'config.yaml'), 'r') as stream:
        cfg = yaml.safe_load(stream)

    env_cfg = cfg['env']
    agents = env_cfg['learned_agent_list']
    stock_cut = env_cfg['supplier_kwargs']['stock_cut_frac'] * len(agents) * env_cfg['market_kwargs']['stock_cap']

    return SimpleNamespace(
        agents=agents,
        n_agents=len(agents),
        obs_map=obs_map,
        scale=env_cfg['market_kwargs']['obs_scale'],
        supplier_cfg={'stock_cut': stock_cut},
    )


def episode_metrics(episode: int, results_dir: str) -> tuple:
    '''
    load a single episode's eval results and compute, for each step:
        belief error        each observer's teammate stockpile estimates matched to the true stockpiles by sorting
                            (the optimal one-to-one assignment for scalars, and the same matching the belief model's
                            permutation invariant loss uses), mean absolute error over teammates and observers
        estimate std        how much the observers' estimates of the same company disagree: each observer's sorted
                            estimates are assigned to its teammates in order of their true stockpile, then the std
                            over the observers estimating each company, averaged over companies
        decomposed reward   MarketGameEnv.decompose_reward on the true observations, averaged over agents; the
                            terminal step of an episode that ends early is skipped, since its early end penalty is a
                            flat team-wide term the sales share can't attribute to individual companies

    all in material units (observations are scaled by obs_scale)

    returns
    -------
    tuple of dicts: step (int) -> value, for belief error, estimate std and decomposed reward
    '''
    data = np.load(os.path.join(results_dir, f'{episode}.npz'), allow_pickle=True)
    obs_map = data['obs_map'].item()
    env_stub = load_env_stub(results_dir, obs_map)
    agents, team_slice = env_stub.agents, obs_map['team']
    steps = sorted(int(f) for f in data.files if f != 'obs_map')

    belief_error, estimate_std, decomposed_reward = {}, {}, {}
    for step in steps:
        _obs, rewards, _terminations, _truncations, infos = data[str(step)]
        common = infos['__common__']
        obs_no_noise = common.get('obs_no_noise')
        if not obs_no_noise:
            continue

        predictions = common['sampled_predictions']
        errors, estimates_by_company = [], {agent: [] for agent in agents}
        for observer in agents:
            others = [agent for agent in agents if agent != observer]
            true = np.asarray(obs_no_noise[observer][team_slice], dtype=float) * env_stub.scale
            estimate = np.sort(np.asarray(predictions[observer], dtype=float).reshape(-1)) * env_stub.scale
            errors.append(np.abs(estimate - np.sort(true)).mean())
            for rank, idx in enumerate(np.argsort(true)):
                estimates_by_company[others[idx]].append(estimate[rank])

        belief_error[step] = float(np.mean(errors))
        estimate_std[step] = float(np.mean([np.std(v) for v in estimates_by_company.values()]))

        metrics = common.get(agents[0], {})
        if not (metrics.get('dead') or metrics.get('short')):
            joint_reward = float(np.mean([rewards[agent] for agent in agents]))
            decomposed = MarketGameEnv.decompose_reward(env_stub, joint_reward, obs_no_noise)
            decomposed_reward[step] = float(np.mean(list(decomposed.values())))

    return belief_error, estimate_std, decomposed_reward


def pool_by_step(per_episode: list) -> dict:
    '''
    pool per-episode step -> value dicts by step

    returns
    -------
    dict: step (int) -> array of per-episode values at that step
    '''
    combined = {}
    for values_by_step in per_episode:
        for step, value in values_by_step.items():
            combined.setdefault(step, []).append(value)

    return {step: np.array(values) for step, values in combined.items()}


def plot_combined_metrics(
    belief_error: dict,
    decomposed_reward: dict,
    estimate_std: dict,
    save_path: str,
    reference_line: float = None,
    reference_label: str = None,
):
    '''
    stack the belief error, cross-observer estimate std and decomposed reward panels into a single figure sharing a
    step axis, as predator_prey_evaluate.plot_combined_metrics does (with the estimate spread as a std, in the same
    units as the belief error)
    '''
    sns.set_theme(style='darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    _draw_average_with_std(axes[0], belief_error, ylabel='Belief Error', show_legend=False)
    _draw_average_with_std(axes[1], estimate_std, ylabel='Belief Std', show_legend=False)
    _draw_average_with_std(
        axes[2], decomposed_reward, ylabel='Decomposed Reward',
        reference_line=reference_line, reference_label=reference_label, min_samples=2, show_legend=False,
    )

    _set_endpoint_yticks(axes[0], y_min=0)
    _set_endpoint_yticks(axes[1], y_min=0)
    _set_endpoint_yticks(axes[2])

    axes[-1].set_xlabel('step')

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    print(f'saved plot to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default=RESULTS_DIR,
                        help='directory containing the allocentric per-episode <idx>.npz result files')
    parser.add_argument('--results-pf', type=str, default=RESULTS_PF_DIR,
                        help='directory containing the particle-filter per-episode <idx>.npz result files')
    parser.add_argument('--save-dir', type=str, default=SAVE_DIR,
                        help='where to save the belief error / estimate std / decomposed reward / combined plots')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    result_dirs = {'Allocentric': args.results, 'PF': args.results_pf}

    belief_error, estimate_std, decomposed_reward, episode_counts = {}, {}, {}, {}
    for label, results_dir in result_dirs.items():
        episodes = list_episodes(results_dir)
        episode_counts[label] = len(episodes)
        with Pool(args.workers) as pool:
            per_episode = pool.map(partial(episode_metrics, results_dir=results_dir), episodes)

        belief_error[label] = pool_by_step([m[0] for m in per_episode])
        estimate_std[label] = pool_by_step([m[1] for m in per_episode])
        decomposed_reward[label] = pool_by_step([m[2] for m in per_episode])

    run_counts = ', '.join(f'{label}: {n}' for label, n in episode_counts.items())

    plot_average_with_std(
        belief_error,
        save_path=os.path.join(args.save_dir, 'belief_error.png'),
        ylabel=f'belief error, material units ({run_counts})',
        title='Average Belief Error over Episode Steps',
    )

    plot_average_with_std(
        decomposed_reward,
        save_path=os.path.join(args.save_dir, 'decomposed_reward.png'),
        ylabel=f'decomposed reward ({run_counts})',
        title='Average Decomposed Reward over Episode Steps',
        reference_line=CONTROLLER_MEAN_DECOMPOSED_REWARD,
        reference_label='controller mean',
        min_samples=2,
    )

    plot_average_with_std(
        estimate_std,
        save_path=os.path.join(args.save_dir, 'estimate_std.png'),
        ylabel=f'cross-observer estimate std, material units ({run_counts})',
        title='Average Cross-Observer Estimate Std over Episode Steps',
    )

    plot_combined_metrics(
        belief_error,
        decomposed_reward,
        estimate_std,
        save_path=os.path.join(args.save_dir, 'combined_metrics.png'),
        reference_line=CONTROLLER_MEAN_DECOMPOSED_REWARD,
        reference_label='controller mean',
    )
