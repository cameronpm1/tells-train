import argparse
import glob
import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from envs.marl.football_env import CirclePass5v1Env
from util.util import RESULTS_ROOT
from loaders import load_football_positions as load_episode_positions, load_football_beliefs as load_episode_beliefs

# bold text for every figure (set at import, so modules that import this one get it too)
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'


RESULTS_DIR = os.path.join(RESULTS_ROOT, 'football', 'results_allocentric')
RESULTS_PF_DIR = os.path.join(RESULTS_ROOT, 'football', 'results_pf')
BELIEF_ERROR_SAVE_PATH = os.path.join(RESULTS_ROOT, 'football', 'belief_error.png')
DECOMPOSED_REWARD_SAVE_PATH = os.path.join(RESULTS_ROOT, 'football', 'decomposed_reward.png')
ESTIMATE_VARIANCE_SAVE_PATH = os.path.join(RESULTS_ROOT, 'football', 'estimate_variance.png')
COMBINED_SAVE_PATH = os.path.join(RESULTS_ROOT, 'football', 'combined_metrics.png')

# mean decomposed reward over 500 controller-driven episodes
# (evals/marl/collect_decomposed_rewards.py), with terminal oob/steal
# steps filtered out the same way decomposed_reward_by_step does below
CONTROLLER_MEAN_DECOMPOSED_REWARD = 1.161578

# steps past this are dropped from every plot
MAX_STEP = 200

# steps reached by fewer episodes than this are dropped from every plot -
# late steps only a handful of long episodes reach are too noisy to read
MIN_SAMPLES = 5


def list_episodes(results_dir: str):
    '''
    return the sorted episode indices for the <idx>.npz result files
    found directly inside results_dir
    '''
    paths = glob.glob(os.path.join(results_dir, '*.npz'))
    return sorted(int(os.path.splitext(os.path.basename(p))[0]) for p in paths)


def compute_decomposed_reward(obs_no_noise: dict, last_obs_no_noise: dict, rewards: dict, obs_map: dict) -> dict:
    '''
    recompute the per-agent decomposed reward for a single step by calling
    the real CirclePass5v1Env.decompose_reward (envs/marl/football_env.py)
    directly, rather than reimplementing its logic here - decompose_reward
    only touches self.agents / self.obs_map (plus the _ball_owner_index
    helper, bound onto the stub below since decompose_reward calls it as
    self._ball_owner_index), so a lightweight stub standing in for the env
    is enough to call it unbound

    unlike predator-prey's decompose_reward, this one needs both the
    current and previous step's ground-truth obs to detect a completed
    pass (the ball-owner one-hot changing between the two)

    inputs
    ------
    obs_no_noise:dict
        agent -> ground-truth observation for this step (infos['__common__']['obs_no_noise'])
    last_obs_no_noise:dict
        agent -> ground-truth observation for the previous step
    rewards:dict
        agent -> reward for this step
    obs_map:dict
        slices dict loaded from the eval npz ('obs_map')

    returns
    -------
    dict: agent -> decomposed reward
    '''
    env_stub = SimpleNamespace(
        agents=sorted(obs_no_noise.keys()),
        obs_map=obs_map,
    )
    env_stub._ball_owner_index = types.MethodType(CirclePass5v1Env._ball_owner_index, env_stub)
    joint_reward = float(np.mean(list(rewards.values())))

    return CirclePass5v1Env.decompose_reward(env_stub, joint_reward, obs_no_noise, last_obs_no_noise)


def belief_error_by_step(results_dir: str, episode: int) -> dict:
    '''
    load a single episode's eval results and, for each step, compute each
    agent's individual belief error the same way loaders.load_football_beliefs
    does: reconstruct absolute ground-truth agent positions
    (load_episode_positions) and each observer's believed absolute position
    of its teammates (load_episode_beliefs), then take the position error
    between them for every observer/teammate pair and average across pairs

    returns
    -------
    dict: step (int) -> belief error averaged over agent pairs (float)
    '''
    agent_positions, _target_positions, _true_target_positions, _ball_positions, _true_ball_positions = (
        load_episode_positions(results_dir=results_dir, episode=episode)
    )
    beliefs = load_episode_beliefs(results_dir=results_dir, episode=episode)
    agent_names = sorted(agent_positions)
    n_steps = agent_positions[agent_names[0]].shape[0]

    errors_by_step = {}
    for t in range(n_steps):
        pair_errors = [
            np.linalg.norm(agent_positions[target][t] - beliefs[observer][target][t])
            for observer in agent_names
            for target in agent_names
            if target != observer
        ]
        errors_by_step[t + 1] = float(np.mean(pair_errors))

    return errors_by_step


def agent_estimate_variance_by_step(results_dir: str, episode: int) -> dict:
    '''
    load a single episode's eval results and, for each step, compute how
    much the different observers' estimates of a given agent's position
    disagree with each other (load_episode_beliefs), then average that
    disagreement across agents and the ball (treating the ball as an extra
    "agent" whose estimated position is each observer's own obs_map['ball_pos']
    reading, via load_episode_positions - unlike sampled_predictions, which
    only ever covers obs_map['team'] (envs/marl/pf_wrapper.py:179), so the
    ball's cross-observer estimate spread isn't otherwise visible)

    unlike a teammate agent, every observer (including the ball's own
    non-existent "self") independently estimates the ball, so its spread is
    taken across all agent_names rather than all-but-the-target

    unlike predator_prey_evaluate.py/drones_evaluate.py, this does NOT need
    a direct-vs-swapped matching step to recover which slot belongs to
    which teammate: football's belief model is not trained with a
    permutation-invariant loss (learn/belief/models.py's football_NN uses a
    plain nn.MSELoss, with no direct/swapped matching term like
    PredPreyPermutationInvariantMSE/DronesPermutationInvariantMSE) - both
    the allocentric path (CirclePass5v1Env._convert_obs, envs/marl/football_env.py)
    and the particle-filter path (FootballParticleFilter's clusters, seeded
    in PFWrapper.__init__ via 'for agent2 in self.agents: if agent !=
    agent2') build a given observer's team slice as self.agents with the
    observer's own index deleted, preserving self.agents' original relative
    order - so load_episode_beliefs' fixed-order zip (which walks
    infos['__common__']['obs_no_noise'].keys(), preserving that same order)
    already recovers each teammate's identity correctly

    a target's estimates' disagreement is the trace of their 2x2
    covariance matrix - the sum of the x and y componentwise variances,
    i.e. the mean squared distance of each estimate from their shared
    centroid

    returns
    -------
    dict: step (int) -> estimate variance averaged over agents and the ball (float)
    '''
    beliefs = load_episode_beliefs(results_dir=results_dir, episode=episode)
    agent_names = sorted(beliefs)
    n_steps = next(iter(beliefs[agent_names[0]].values())).shape[0]

    _agent_positions, _target_positions, _true_target_positions, ball_positions, _true_ball_positions = (
        load_episode_positions(results_dir=results_dir, episode=episode)
    )

    variance_by_step = {}
    for t in range(n_steps):
        target_variances = [
            np.var(
                np.stack([beliefs[observer][target][t] for observer in agent_names if observer != target]),
                axis=0,
            ).sum()
            for target in agent_names
        ]
        target_variances.append(
            np.var(np.stack([ball_positions[observer][t] for observer in agent_names]), axis=0).sum()
        )
        variance_by_step[t + 1] = float(np.mean(target_variances))

    return variance_by_step


def decomposed_reward_by_step(results_dir: str, episode: int) -> dict:
    '''
    load a single episode's eval results and, for each step, recompute the
    per-agent decomposed reward and average it across agents

    a step is skipped if it ends in an out-of-bounds penalty or the
    defender stealing the ball (CirclePass5v1Env._compute_team_reward
    returns oob_penalty/target_steal_penalty directly as a flat
    team-wide value - +/-750 in this run's config - bypassing the
    pass/openness logic that decompose_reward attributes to individual
    agents, same reasoning as predator_prey_evaluate.py's
    decomposed_reward_by_step for its success/oob terminal bonus) or if
    it's the episode's first step (decompose_reward needs the previous
    step's ground-truth obs to detect a completed pass, and there is none
    before the first step)

    returns
    -------
    dict: step (int) -> decomposed reward averaged over agents (float)
    '''
    npz_path = os.path.join(results_dir, f'{episode}.npz')
    data = np.load(npz_path, allow_pickle=True)

    obs_map = data['obs_map'].item()
    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)

    reward_by_step = {}
    last_obs_no_noise = None
    for step in steps:
        _obs, rewards, _terminations, _truncations, infos = data[step]
        common_infos = infos['__common__']
        obs_no_noise = common_infos.get('obs_no_noise')
        if not obs_no_noise:
            continue

        oob = any(common_infos.get(agent, {}).get('oob') for agent in obs_no_noise)
        target_stole_ball = bool(infos['target'][obs_map['target_ball_owned']][0])
        if not oob and not target_stole_ball and last_obs_no_noise is not None:
            decomposed_reward = compute_decomposed_reward(obs_no_noise, last_obs_no_noise, rewards, obs_map)
            reward_by_step[int(step)] = float(np.mean(list(decomposed_reward.values())))

        last_obs_no_noise = obs_no_noise

    return reward_by_step


def average_over_episodes(per_episode_fn, results_dir: str, episodes, max_step: int = None) -> dict:
    '''
    run per_episode_fn(results_dir, episode) over every episode and pool
    the results by step, dropping any step past max_step (if given)

    returns
    -------
    dict: step (int) -> array of per-episode values at that step
    '''
    combined = {}
    for episode in episodes:
        values_by_step = per_episode_fn(results_dir, episode)
        for step, value in values_by_step.items():
            if max_step is not None and step > max_step:
                continue
            combined.setdefault(step, []).append(value)

    return {step: np.array(values) for step, values in combined.items()}


def _draw_average_with_std(
    ax,
    series: dict,
    ylabel: str,
    reference_line: float = None,
    reference_label: str = None,
    min_samples: int = MIN_SAMPLES,
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
    min_samples: int = MIN_SAMPLES,
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
        reference_line=reference_line, reference_label=reference_label, min_samples=MIN_SAMPLES, show_legend=False,
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
    parser.add_argument('--max-step', type=int, default=MAX_STEP,
                         help='drop steps past this from every plot')
    args = parser.parse_args()

    result_dirs = {'Allocentric': args.results, 'PF': args.results_pf}

    belief_error = {}
    decomposed_reward = {}
    estimate_variance = {}
    episode_counts = {}

    for label, results_dir in result_dirs.items():
        episodes = list_episodes(results_dir)
        episode_counts[label] = len(episodes)

        belief_error[label] = average_over_episodes(belief_error_by_step, results_dir, episodes, args.max_step)
        decomposed_reward[label] = average_over_episodes(decomposed_reward_by_step, results_dir, episodes, args.max_step)
        estimate_variance[label] = average_over_episodes(
            agent_estimate_variance_by_step, results_dir, episodes, args.max_step
        )

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
        reference_label='controller mean',
        min_samples=MIN_SAMPLES,
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
