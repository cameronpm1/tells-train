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

from envs.marl.fire_env import DroneFireEnv
from loaders import load_fire_positions as load_episode_positions, load_fire_beliefs as load_episode_beliefs
from util.util import find_run_config, RESULTS_ROOT

# bold text for every figure (set at import, so modules that import this one get it too)
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'


RESULTS_DIR = os.path.join(RESULTS_ROOT, 'fire', 'results_allocentric')
RESULTS_PF_DIR = os.path.join(RESULTS_ROOT, 'fire', 'results_pf')
BELIEF_ERROR_SAVE_PATH = os.path.join(RESULTS_ROOT, 'fire', 'belief_error.png')
DECOMPOSED_REWARD_SAVE_PATH = os.path.join(RESULTS_ROOT, 'fire', 'decomposed_reward.png')
ESTIMATE_VARIANCE_SAVE_PATH = os.path.join(RESULTS_ROOT, 'fire', 'estimate_variance.png')
COMBINED_SAVE_PATH = os.path.join(RESULTS_ROOT, 'fire', 'combined_metrics.png')

# mean decomposed reward over 500 controller-driven episodes
# (evals/marl/collect_fire_decomposed_rewards.py), with terminal
# spread/extinguish/contained steps filtered out the same way
# decomposed_reward_by_step does below
CONTROLLER_MEAN_DECOMPOSED_REWARD = 0.021037


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
    recompute min_slot_dist the same way DroneFireEnv.decompose_reward does
    '''
    with open(find_run_config(results_dir), 'r') as stream:
        cfg = yaml.safe_load(stream)

    return cfg['env'].get('controller_kwargs', {})


def compute_decomposed_reward(decomposed_obs: dict, fire_grid: np.ndarray, rewards: dict, controller_cfg: dict) -> dict:
    '''
    recompute the per-agent decomposed reward for a single step by calling
    the real DroneFireEnv.decompose_reward (envs/marl/fire_env.py) directly,
    rather than reimplementing its logic here - decompose_reward only
    touches self.agents / self.controller_kwargs, so a lightweight stub
    standing in for the env is enough to call it unbound

    decompose_reward reads obs['target'] (the full ground-truth fire grid,
    needed to score each drone's own pixel against the nearest green/black
    cells) - that key is stripped from decomposed_obs (see DroneFireEnv.step,
    which stashes info['__common__']['decomposed_obs'] before info['target']
    is set), but is separately available as the top-level infos['target']
    the wrapper records, so it's passed in here and restored under the
    'target' key

    inputs
    ------
    decomposed_obs:dict
        agent -> ground-truth decomposed obs for that step
        (infos['__common__']['decomposed_obs'])
    fire_grid:np.ndarray
        the ground-truth (GRID_SIZE, GRID_SIZE) fire grid for that step
        (infos['target'])
    rewards:dict
        agent -> reward for that step
    controller_cfg:dict
        env.controller_kwargs from the run's config.yaml

    returns
    -------
    dict: agent -> decomposed reward
    '''
    agents = sorted(decomposed_obs.keys())
    env_stub = SimpleNamespace(agents=agents, controller_kwargs=controller_cfg)
    obs = dict(decomposed_obs)
    obs['target'] = fire_grid
    joint_reward = float(np.mean(list(rewards.values())))

    return DroneFireEnv.decompose_reward(env_stub, joint_reward, obs)


def belief_error_by_step(results_dir: str, episode: int, permutation_invariant: bool = True) -> dict:
    '''
    load a single episode's eval results and, for each step, compute each
    drone's individual belief error the same way loaders.load_fire_beliefs does:
    reconstruct absolute ground-truth drone positions (load_episode_positions)
    and each observer's believed absolute position of its teammates
    (load_episode_beliefs), then take the position error between them for
    every observer/teammate pair and average across pairs

    unlike predator_prey_evaluate.py/drones_evaluate.py, this does NOT need
    a separate direct-vs-swapped matching step here: load_episode_beliefs
    already resolves each predicted slot to a specific teammate name itself
    (loaders._fire_best_teammate_permutation) when permutation_invariant
    is True - the right setting for the allocentric NN belief model, which
    (like drones/predator_prey, unlike football) doesn't commit its output
    to a fixed teammate-slot order

    permutation_invariant must be False for particle-filter results:
    FireParticleFilter tracks each teammate in its own identity-labeled
    cluster (envs/marl/particle_filters.py), so a PF sampled_predictions
    slot already corresponds to a specific, known teammate - searching for
    the best-fit permutation there would silently paper over genuine PF
    tracking/identity confusion instead of reporting it as belief error
    (see load_episode_beliefs)

    inputs
    ------
    results_dir, episode:
        forwarded to load_episode_positions / load_episode_beliefs
    permutation_invariant:bool
        True for allocentric results, False for particle-filter results
        (see above)

    returns
    -------
    dict: step (int) -> belief error averaged over agent pairs (float)
    '''
    drone_positions, _fire_grids = load_episode_positions(results_dir=results_dir, episode=episode)
    beliefs = load_episode_beliefs(results_dir=results_dir, episode=episode, permutation_invariant=permutation_invariant)
    drone_names = sorted(drone_positions)
    n_steps = drone_positions[drone_names[0]].shape[0]

    errors_by_step = {}
    for t in range(n_steps):
        pair_errors = [
            np.linalg.norm(drone_positions[target][t] - beliefs[observer][target][t])
            for observer in drone_names
            for target in drone_names
            if target != observer
        ]
        errors_by_step[t + 1] = float(np.mean(pair_errors))

    return errors_by_step


def agent_estimate_variance_by_step(results_dir: str, episode: int, permutation_invariant: bool = True) -> dict:
    '''
    load a single episode's eval results and, for each step, compute how
    much the different observers' estimates of a given drone's position
    disagree with each other (load_episode_beliefs, already teammate-
    identity resolved when permutation_invariant is True), then average
    that disagreement across drones

    permutation_invariant must be False for particle-filter results, same
    reasoning as belief_error_by_step: PF sampled_predictions slots already
    correspond to a specific, known teammate, so they shouldn't be run
    through the best-fit permutation search

    a drone's estimates' disagreement is the trace of their 2x2 covariance
    matrix - the sum of the row/col componentwise variances, i.e. the mean
    squared distance of each estimate from their shared centroid

    returns
    -------
    dict: step (int) -> estimate variance averaged over drones (float)
    '''
    beliefs = load_episode_beliefs(results_dir=results_dir, episode=episode, permutation_invariant=permutation_invariant)
    drone_names = sorted(beliefs)
    n_steps = next(iter(beliefs[drone_names[0]].values())).shape[0]

    variance_by_step = {}
    for t in range(n_steps):
        drone_variances = [
            np.var(
                np.stack([beliefs[observer][target][t] for observer in drone_names if observer != target]),
                axis=0,
            ).sum()
            for target in drone_names
        ]
        variance_by_step[t + 1] = float(np.mean(drone_variances))

    return variance_by_step


def decomposed_reward_by_step(results_dir: str, episode: int, controller_cfg: dict) -> dict:
    '''
    load a single episode's eval results and, for each step, recompute the
    per-agent decomposed reward and average it across agents

    a step is skipped if it ends the episode: DroneFireEnv._compute_reward
    adds a flat team-wide lose_penalty/extinguish_reward bonus whenever the
    episode terminates (fire fully spread, fully extinguished, or contained
    - see terminations['__all__'], which DroneFireEnv.step sets True for all
    3 cases), which score_share can't meaningfully attribute to individual
    agents, same reasoning as predator_prey_evaluate.py's
    decomposed_reward_by_step for its success/oob terminal bonus

    returns
    -------
    dict: step (int) -> decomposed reward averaged over agents (float)
    '''
    npz_path = os.path.join(results_dir, f'{episode}.npz')
    data = np.load(npz_path, allow_pickle=True)

    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)

    reward_by_step = {}
    for step in steps:
        _obs, rewards, terminations, _truncations, infos = data[step]
        common_infos = infos['__common__']
        decomposed_obs = common_infos.get('decomposed_obs')
        if not decomposed_obs:
            continue

        if terminations.get('__all__', False):
            continue

        decomposed_reward = compute_decomposed_reward(decomposed_obs, infos['target'], rewards, controller_cfg)
        reward_by_step[int(step)] = float(np.mean(list(decomposed_reward.values())))

    return reward_by_step


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
    max_step: int = 125,
    clamp_min: float = None,
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
    max_step:int
        drop steps beyond this value so plots cut off at a fixed horizon
        instead of trailing off into the long tail of a few surviving
        episodes
    clamp_min:float
        if given, floor the lower edge of the shaded stdev band at this
        value (e.g. 0 for a metric like variance that can never go
        negative) instead of letting mean - stdev dip below it
    '''
    for label, values_by_step in series.items():
        steps = sorted(s for s in values_by_step if len(values_by_step[s]) >= min_samples and s <= max_step)
        if not steps:
            raise ValueError(f'values_by_step for "{label}" is empty - regenerate the eval results')

        means = np.array([values_by_step[s].mean() for s in steps])
        stds = np.array([values_by_step[s].std() for s in steps])

        line = sns.lineplot(x=steps, y=means, ax=ax, label=f'{label} mean', legend=False)
        color = line.get_lines()[-1].get_color()
        lower = means - stds if clamp_min is None else np.maximum(means - stds, clamp_min)
        ax.fill_between(steps, lower, means + stds, alpha=0.2, color=color, label=f'{label} +/- 1 stdev')

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
    max_step: int =125,
    clamp_min: float = None,
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
    max_step:int
        drop steps beyond this value so the plot cuts off at a fixed
        horizon instead of trailing off into the long tail of a few
        surviving episodes
    clamp_min:float
        if given, floor the lower edge of the shaded stdev band at this
        value (e.g. 0 for a metric like variance that can never go
        negative) instead of letting mean - stdev dip below it
    '''
    sns.set_theme(style='darkgrid')
    fig, ax = plt.subplots(figsize=(10, 4))

    _draw_average_with_std(
        ax, series, ylabel, reference_line, reference_label, min_samples, max_step=max_step, clamp_min=clamp_min,
    )

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
    max_step: int = 125,
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
    max_step:int
        drop steps beyond this value so all 3 panels cut off at a fixed
        horizon instead of trailing off into the long tail of a few
        surviving episodes
    '''
    sns.set_theme(style='darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    _draw_average_with_std(axes[0], belief_error, ylabel='Belief Error', show_legend=False, max_step=max_step)
    _draw_average_with_std(
        axes[1], estimate_variance, ylabel='Belief Variance', show_legend=False, max_step=max_step, clamp_min=0,
    )
    _draw_average_with_std(
        axes[2], decomposed_reward, ylabel='Decomposed Reward',
        reference_line=reference_line, reference_label=reference_label, min_samples=2, show_legend=False,
        max_step=max_step,
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
        # PF sampled_predictions slots are already identity-known (see
        # load_episode_beliefs) - only the allocentric NN model's output
        # needs the best-fit permutation search
        permutation_invariant = (label != 'PF')

        belief_error[label] = average_over_episodes(
            lambda rdir, ep: belief_error_by_step(rdir, ep, permutation_invariant),
            results_dir,
            episodes,
        )
        decomposed_reward[label] = average_over_episodes(
            lambda rdir, ep: decomposed_reward_by_step(rdir, ep, controller_cfg),
            results_dir,
            episodes,
        )
        estimate_variance[label] = average_over_episodes(
            lambda rdir, ep: agent_estimate_variance_by_step(rdir, ep, permutation_invariant),
            results_dir,
            episodes,
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
        min_samples=2,
    )

    plot_average_with_std(
        estimate_variance,
        save_path=args.estimate_variance_save_path,
        ylabel=f'cross-observer estimate variance ({run_counts})',
        title='Average Cross-Observer Estimate Variance over Episode Steps',
        clamp_min=0,
    )

    plot_combined_metrics(
        belief_error,
        decomposed_reward,
        estimate_variance,
        save_path=args.combined_save_path,
        reference_line=CONTROLLER_MEAN_DECOMPOSED_REWARD,
        reference_label='controller mean',
    )
