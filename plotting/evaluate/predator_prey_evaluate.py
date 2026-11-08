import argparse
import glob
import os
import sys
from types import SimpleNamespace

import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from envs.marl.predator_prey_env import PredatorPreyEnv
from loaders import load_predator_prey_positions as load_episode_positions, load_predator_prey_beliefs as load_episode_beliefs
from util.util import find_run_config, RESULTS_ROOT

# bold text for every figure (set at import, so modules that import this one get it too)
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'


RESULTS_DIR = os.path.join(RESULTS_ROOT, 'predator_prey', 'results_allocentric')
RESULTS_PF_DIR = os.path.join(RESULTS_ROOT, 'predator_prey', 'results_pf')
BELIEF_ERROR_SAVE_PATH = os.path.join(RESULTS_ROOT, 'predator_prey', 'belief_error.png')
DECOMPOSED_REWARD_SAVE_PATH = os.path.join(RESULTS_ROOT, 'predator_prey', 'decomposed_reward.png')
ESTIMATE_VARIANCE_SAVE_PATH = os.path.join(RESULTS_ROOT, 'predator_prey', 'estimate_variance.png')
COMBINED_SAVE_PATH = os.path.join(RESULTS_ROOT, 'predator_prey', 'combined_metrics.png')

# mean decomposed reward over 500 controller-driven episodes
# (evals/marl/collect_decomposed_rewards.py), with terminal success/oob
# steps filtered out the same way decomposed_reward_by_step does below
CONTROLLER_MEAN_DECOMPOSED_REWARD = 0.448082


def list_episodes(results_dir: str):
    '''
    return the sorted episode indices for the <idx>.npz result files
    found directly inside results_dir
    '''
    paths = glob.glob(os.path.join(results_dir, '*.npz'))
    return sorted(int(os.path.splitext(os.path.basename(p))[0]) for p in paths)


def load_controller_cfg(results_dir: str) -> dict:
    '''
    the run's config.yaml for a results dir (util.util.find_run_config: next to it in
    results/<env>, or two levels up in the logs layout) and pull out env.controller_kwargs - needed to
    recompute the adversary force each predator applies to the prey, the
    same way PredatorPreyEnv.decompose_reward does
    '''
    with open(find_run_config(results_dir), 'r') as stream:
        cfg = yaml.safe_load(stream)

    return cfg['env']['controller_kwargs']


def compute_decomposed_reward(obs_no_noise: dict, rewards: dict, obs_map: dict, controller_cfg: dict) -> dict:
    '''
    recompute the per-agent decomposed reward for a single step by calling
    the real PredatorPreyEnv.decompose_reward (envs/marl/predator_prey_env.py)
    directly, rather than reimplementing its logic here - decompose_reward
    only touches self.agents / self.controller_cfg / self.obs_map, so a
    lightweight stub standing in for the env is enough to call it unbound

    inputs
    ------
    obs_no_noise:dict
        agent -> ground-truth observation for that step (infos['__common__']['obs_no_noise'])
    rewards:dict
        agent -> reward for that step
    obs_map:dict
        slices dict loaded from the eval npz ('obs_map')
    controller_cfg:dict
        env.controller_kwargs from the run's config.yaml

    returns
    -------
    dict: agent -> decomposed reward
    '''
    env_stub = SimpleNamespace(
        agents=sorted(obs_no_noise.keys()),
        controller_cfg=controller_cfg,
        obs_map=obs_map,
    )
    joint_reward = float(np.mean(list(rewards.values())))

    return PredatorPreyEnv.decompose_reward(env_stub, joint_reward, obs_no_noise)


def belief_error_by_step(results_dir: str, episode: int) -> dict:
    '''
    load a single episode's eval results and, for each step, compute each
    predator's individual belief error the same way loaders.load_predator_prey_beliefs
    does: reconstruct absolute ground-truth predator positions
    (load_episode_positions) and each observer's believed absolute position
    of its teammates (load_episode_beliefs), then take the position error
    between them for every observer/teammate pair and average across pairs

    this differs from infos['__common__']['belief_error'] (a
    permutation-invariant error the env computes internally) by matching
    teammates to their known identity instead

    returns
    -------
    dict: step (int) -> belief error averaged over agent pairs (float)
    '''
    predator_positions, _target_positions, _true_target_positions, _goal_position = load_episode_positions(
        results_dir=results_dir, episode=episode,
    )
    beliefs = load_episode_beliefs(results_dir=results_dir, episode=episode)
    predator_names = sorted(predator_positions)
    n_steps = predator_positions[predator_names[0]].shape[0]

    errors_by_step = {}
    for t in range(n_steps):
        pair_errors = [
            np.linalg.norm(predator_positions[target][t] - beliefs[observer][target][t])
            for observer in predator_names
            for target in predator_names
            if target != observer
        ]
        errors_by_step[t + 1] = float(np.mean(pair_errors))

    return errors_by_step


def decomposed_reward_by_step(results_dir: str, episode: int, controller_cfg: dict) -> dict:
    '''
    load a single episode's eval results and, for each step, recompute the
    per-agent decomposed reward and average it across agents

    steps where the episode terminated with a success bonus or an
    out-of-bounds penalty are skipped: decompose_reward apportions the
    joint reward by each predator's adversary-force contribution, which
    only makes sense for the smooth per-step reward - a terminal
    success/oob bonus is a flat team-wide add-on that force_share can't
    meaningfully attribute to individual agents

    returns
    -------
    dict: step (int) -> decomposed reward averaged over agents (float)
    '''
    npz_path = os.path.join(results_dir, f'{episode}.npz')
    data = np.load(npz_path, allow_pickle=True)

    obs_map = data['obs_map'].item()
    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)

    reward_by_step = {}
    for step in steps:
        _obs, rewards, _terminations, _truncations, infos = data[step]
        common_infos = infos['__common__']
        obs_no_noise = common_infos.get('obs_no_noise')
        if not obs_no_noise:
            continue

        terminal_bonus_or_penalty = any(
            common_infos.get(agent, {}).get('success') or common_infos.get(agent, {}).get('oob')
            for agent in obs_no_noise
        )
        if terminal_bonus_or_penalty:
            continue

        decomposed_reward = compute_decomposed_reward(obs_no_noise, rewards, obs_map, controller_cfg)
        reward_by_step[int(step)] = float(np.mean(list(decomposed_reward.values())))

    return reward_by_step


def matched_belief_estimates_by_step(results_dir: str, episode: int) -> dict:
    '''
    load a single episode's eval results and, for each step, recover every
    observer's belief about its 2 teammates as absolute positions, matching
    the model's 2 predicted slots to the 2 actual teammates by whichever
    assignment (direct vs swapped) has the lower total distance

    this matching is required because the belief model's teammate slots are
    permutation-invariant by construction (learn/belief/models.py
    PredPreyPermutationInvariantMSE.permutation_invariant_loss takes the min
    of the direct/swapped loss every step, so training never penalizes slot
    order) - the same direct/swapped comparison PredatorPreyEnv.team_error
    uses to score these predictions is used here to recover identity instead,
    unlike load_episode_beliefs (loaders.load_predator_prey_beliefs), which zips
    slots to teammates in a fixed order and silently mislabels them whenever
    the model outputs the swapped order

    returns
    -------
    dict: step (int) -> {agent: list of estimated absolute (x, y) positions,
        one from each observer that isn't that agent}
    '''
    npz_path = os.path.join(results_dir, f'{episode}.npz')
    data = np.load(npz_path, allow_pickle=True)

    obs_map = data['obs_map'].item()
    self_pos_slice = obs_map['self_pos']
    team_slice = obs_map['team']

    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)

    first_common = data[steps[0]][4]['__common__']
    agents = sorted(first_common['obs_no_noise'].keys())
    dim = (team_slice.stop - team_slice.start) // (len(agents) - 1)

    estimates_by_step = {}
    for step in steps:
        _obs, _rewards, _terminations, _truncations, infos = data[step]
        common = infos['__common__']
        obs_no_noise = common.get('obs_no_noise')
        if not obs_no_noise:
            continue

        sampled_predictions = common['sampled_predictions']
        estimates_by_target = {agent: [] for agent in agents}

        for observer in agents:
            others = sorted(agent for agent in agents if agent != observer)
            observer_pos = np.asarray(obs_no_noise[observer][self_pos_slice])
            slots = np.asarray(sampled_predictions[observer]).reshape(-1, dim)
            teammate_positions = np.stack(
                [np.asarray(obs_no_noise[other][self_pos_slice]) for other in others]
            )

            direct = (np.linalg.norm(slots[0] - teammate_positions[0])
                      + np.linalg.norm(slots[1] - teammate_positions[1]))
            swapped = (np.linalg.norm(slots[0] - teammate_positions[1])
                       + np.linalg.norm(slots[1] - teammate_positions[0]))
            slot_order = (0, 1) if direct <= swapped else (1, 0)

            for other, slot_idx in zip(others, slot_order):
                estimates_by_target[other].append(observer_pos + slots[slot_idx])

        estimates_by_step[int(step)] = estimates_by_target

    return estimates_by_step


def agent_estimate_variance_by_step(results_dir: str, episode: int) -> dict:
    '''
    load a single episode's eval results and, for each step, compute how
    much the different observers' estimates of a given agent's position
    disagree with each other, then average that disagreement across agents

    an agent's estimates (matched_belief_estimates_by_step) disagreement is
    the trace of their 2x2 covariance matrix - the sum of the x and y
    componentwise variances, i.e. the mean squared distance of each
    estimate from their shared centroid

    returns
    -------
    dict: step (int) -> estimate variance averaged over agents (float)
    '''
    estimates_by_step = matched_belief_estimates_by_step(results_dir, episode)

    variance_by_step = {}
    for step, estimates_by_target in estimates_by_step.items():
        agent_variances = [
            np.var(np.stack(estimates), axis=0).sum()
            for estimates in estimates_by_target.values()
        ]
        variance_by_step[step] = float(np.mean(agent_variances))

    return variance_by_step


def average_over_episodes(per_episode_fn, results_dir: str, episodes) -> dict:
    '''
    run per_episode_fn(results_dir, episode) over every episode and pool
    the results by step

    returns
    -------
    dict: step (int) -> array of per-episode values at that step
    '''
    combined = {}
    for episode in episodes:
        values_by_step = per_episode_fn(results_dir, episode)
        for step, value in values_by_step.items():
            combined.setdefault(step, []).append(value)

    return {step: np.array(values) for step, values in combined.items()}


def _draw_average_with_std(
    ax,
    series: dict,
    ylabel: str,
    reference_line: float = None,
    reference_label: str = None,
    min_samples: int = 2,
    show_legend: bool = True,
):
    '''
    draw the per-step mean (+/- 1 stdev, pooled across episodes) for one or
    more named series onto ax, each as its own curve with a shaded std-dev
    band - factored out of plot_average_with_std so plot_combined_metrics
    can reuse the same per-panel drawing logic across stacked subplots

    inputs
    ------
    series:dict
        label -> values_by_step (step -> array of per-episode values, as
        returned by average_over_episodes)
    ylabel:str
        y-axis label
    reference_line:float
        if given, draw a black dotted horizontal line at this value (e.g.
        CONTROLLER_MEAN_DECOMPOSED_REWARD)
    reference_label:str
        legend label for the reference line
    min_samples:int
        drop steps pooling fewer than this many per-episode values before
        plotting - late steps that only a handful of (long-running)
        episodes reach are otherwise single noisy datapoints with a
        meaningless (zero-width) stdev band
    show_legend:bool
        whether to draw the per-panel legend - combined figures with
        several stacked panels turn this off to avoid repeating the same
        legend on every subplot
    '''
    for label, values_by_step in series.items():
        steps = sorted(s for s in values_by_step if len(values_by_step[s]) >= min_samples)
        if not steps:
            raise ValueError(f'values_by_step for "{label}" is empty - regenerate the eval results')

        means = np.array([values_by_step[s].mean() for s in steps])
        stds = np.array([values_by_step[s].std() for s in steps])

        line = sns.lineplot(x=steps, y=means, ax=ax, label=f'{label} mean', legend=False)
        color = line.get_lines()[-1].get_color()
        ax.fill_between(steps, means - stds, means + stds, alpha=0.2, color=color, label=f'{label} +/- 1 stdev')

    if reference_line is not None:
        ax.axhline(reference_line, color='black', linestyle=':', label=reference_label)

    ax.set_ylabel(ylabel)
    if show_legend:
        ax.legend()


def _set_endpoint_yticks(ax, y_min: float = None):
    '''
    keep only the lowest and highest y-tick so stacked combined-metric
    panels stay visually light

    inputs
    ------
    y_min:float
        if given, floor the low tick at this value (e.g. 0 for a metric
        that can never go negative) instead of using the auto-computed
        tick, which can dip below the true data range due to matplotlib's
        default axis padding
    '''
    yticks = ax.get_yticks()
    low = yticks[0] if y_min is None else max(y_min, yticks[0])
    ax.set_yticks([low, yticks[-1]])


def plot_average_with_std(
    series: dict,
    save_path: str,
    ylabel: str,
    title: str,
    reference_line: float = None,
    reference_label: str = None,
    min_samples: int = 2,
):
    '''
    plot the per-step mean (+/- 1 stdev, pooled across episodes) for one or
    more named series, each as its own curve with a shaded std-dev band

    inputs
    ------
    series:dict
        label -> values_by_step (step -> array of per-episode values, as
        returned by average_over_episodes)
    save_path:str
        where to save the resulting figure
    ylabel:str
        y-axis label
    title:str
        plot title
    reference_line:float
        if given, draw a black dotted horizontal line at this value (e.g.
        CONTROLLER_MEAN_DECOMPOSED_REWARD)
    reference_label:str
        legend label for the reference line
    min_samples:int
        drop steps pooling fewer than this many per-episode values before
        plotting - late steps that only a handful of (long-running)
        episodes reach are otherwise single noisy datapoints with a
        meaningless (zero-width) stdev band
    '''
    sns.set_theme(style='darkgrid')
    fig, ax = plt.subplots(figsize=(10, 4))

    _draw_average_with_std(ax, series, ylabel, reference_line, reference_label, min_samples)

    ax.set_xlabel('step')
    ax.set_title(title, fontweight='bold')

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    print(f'saved plot to {save_path}')


def plot_combined_metrics(
    belief_error: dict,
    decomposed_reward: dict,
    estimate_variance: dict,
    save_path: str,
    reference_line: float = None,
    reference_label: str = None,
):
    '''
    stack the belief error, decomposed reward, and cross-observer estimate
    variance plots into a single figure - one panel per metric sharing a
    step axis, so trends across metrics can be lined up and compared
    directly instead of flipping between 3 separate images

    inputs
    ------
    belief_error, decomposed_reward, estimate_variance:dict
        label -> values_by_step, as produced by average_over_episodes for
        belief_error_by_step / decomposed_reward_by_step /
        agent_estimate_variance_by_step respectively
    save_path:str
        where to save the resulting figure
    reference_line, reference_label:
        forwarded to the decomposed reward panel (e.g.
        CONTROLLER_MEAN_DECOMPOSED_REWARD)
    '''
    sns.set_theme(style='darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    _draw_average_with_std(axes[0], belief_error, ylabel='Belief Error', show_legend=False)
    _draw_average_with_std(axes[1], estimate_variance, ylabel='Belief Variance', show_legend=False)
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
    parser.add_argument('--belief-error-save-path', type=str, default=BELIEF_ERROR_SAVE_PATH,
                         help='where to save the belief error plot')
    parser.add_argument('--decomposed-reward-save-path', type=str, default=DECOMPOSED_REWARD_SAVE_PATH,
                         help='where to save the decomposed reward plot')
    parser.add_argument('--estimate-variance-save-path', type=str, default=ESTIMATE_VARIANCE_SAVE_PATH,
                         help='where to save the cross-observer estimate variance plot')
    parser.add_argument('--combined-save-path', type=str, default=COMBINED_SAVE_PATH,
                         help='where to save the combined (stacked) belief error / decomposed reward / '
                              'estimate variance plot')
    args = parser.parse_args()

    result_dirs = {'Allocentric': args.results, 'PF': args.results_pf}

    belief_error = {}
    decomposed_reward = {}
    estimate_variance = {}
    episode_counts = {}

    for label, results_dir in result_dirs.items():
        episodes = list_episodes(results_dir)
        episode_counts[label] = len(episodes)
        controller_cfg = load_controller_cfg(results_dir)

        belief_error[label] = average_over_episodes(belief_error_by_step, results_dir, episodes)
        decomposed_reward[label] = average_over_episodes(
            lambda rdir, ep: decomposed_reward_by_step(rdir, ep, controller_cfg),
            results_dir,
            episodes,
        )
        estimate_variance[label] = average_over_episodes(agent_estimate_variance_by_step, results_dir, episodes)

    run_counts = ', '.join(f'{label}: {n}' for label, n in episode_counts.items())

    plot_average_with_std(
        belief_error,
        save_path=args.belief_error_save_path,
        ylabel=f'belief error ({run_counts})',
        title='Average Belief Error over Episode Steps',
    )

    plot_average_with_std(
        decomposed_reward,
        save_path=args.decomposed_reward_save_path,
        ylabel=f'decomposed reward ({run_counts})',
        title='Average Decomposed Reward over Episode Steps',
        reference_line=CONTROLLER_MEAN_DECOMPOSED_REWARD,
        reference_label=f'controller mean',
        min_samples=2,
    )

    plot_average_with_std(
        estimate_variance,
        save_path=args.estimate_variance_save_path,
        ylabel=f'cross-observer estimate variance ({run_counts})',
        title='Average Cross-Observer Estimate Variance over Episode Steps',
    )

    plot_combined_metrics(
        belief_error,
        decomposed_reward,
        estimate_variance,
        save_path=args.combined_save_path,
        reference_line=CONTROLLER_MEAN_DECOMPOSED_REWARD,
        reference_label='controller mean',
    )



