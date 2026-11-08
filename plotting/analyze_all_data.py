import argparse
import glob
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluate'))

import drones_evaluate
import fire_evaluate
import football_evaluate
import predator_prey_evaluate


# success is defined purely from episode length, per env: predator_prey
# terminates early (agents caught the prey) before its 400-step cap, while
# football/drones only reach their 300/125-step caps by avoiding the early
# failure terminations (defender steal/oob for football, breach/oob for
# drones) - fire's episodes terminate early (fire fully extinguished or
# contained) before its 125-step cap (max_episode_length in confs/fire/*.yaml),
# so a successful fire episode is one that doesn't run out the clock
SUCCESS_RULES = {
    'predator_prey': ('episode ends before 400 steps', lambda length: length < 400),
    'football': ('episode reaches >= 300 steps', lambda length: length >= 300),
    'drones': ('episode reaches >= 125 steps', lambda length: length >= 125),
    'fire': ('episode ends before 125 steps', lambda length: length < 125),
}


def list_episodes(results_dir: str):
    '''
    return the sorted episode indices for the <idx>.npz result files found
    directly inside results_dir
    '''
    paths = glob.glob(os.path.join(results_dir, '*.npz'))
    return sorted(int(os.path.splitext(os.path.basename(p))[0]) for p in paths)


def episode_length_and_reward(results_dir: str, episode: int):
    '''
    load a single episode's eval results and return its length (number of
    steps) and total reward - summed across agents and steps, matching the
    episode_rewards.append(sum(total_reward.values())) convention used by
    evals/marl/eval.py

    returns
    -------
    (length:int, reward:float)
    '''
    npz_path = os.path.join(results_dir, f'{episode}.npz')
    data = np.load(npz_path, allow_pickle=True)
    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)

    total_reward = 0.0
    for step in steps:
        _obs, rewards, _terminations, _truncations, _infos = data[step]
        total_reward += float(np.sum(list(rewards.values())))

    return len(steps), total_reward


def mean_over_steps(values_by_step: dict):
    '''
    collapse a step -> value dict (as returned by e.g. belief_error_by_step)
    into a single per-episode scalar by averaging across steps - returns
    None if the episode contributed no values (e.g. every step was a
    terminal step skipped by decomposed_reward_by_step)
    '''
    if not values_by_step:
        return None
    return float(np.mean(list(values_by_step.values())))


def matched_belief_error_by_step(module, results_dir: str, episode: int) -> dict:
    '''
    for predator_prey/drones, whose belief model is trained with a
    permutation-invariant loss (PredPreyPermutationInvariantMSE /
    DronesPermutationInvariantMSE in learn/belief/models.py), the model's
    per-teammate slots aren't guaranteed to stay in a fixed order - so the
    plain belief_error_by_step in predator_prey_evaluate.py/drones_evaluate.py
    (which zips slots to teammates via loaders.load_<env>_beliefs's
    FIXED order) can silently mislabel a teammate whenever the model
    outputs its 2 slots swapped, inflating the reported error for the
    allocentric NN model specifically (PF's sampled_predictions slots are
    already identity-known, so it isn't affected)

    matched_belief_estimates_by_step (used elsewhere only for
    agent_estimate_variance_by_step) already resolves this the same way
    PredatorPreyEnv.team_error / CaravanAviary.team_error do: picking
    whichever of the direct/swapped slot assignments has the lower total
    distance to the true teammate positions - this reuses that matched
    identity to compute belief error instead of the raw fixed-order beliefs

    returns
    -------
    dict: step (int) -> belief error averaged over agent pairs (float)
    '''
    estimates_by_step = module.matched_belief_estimates_by_step(results_dir, episode)
    positions = module.load_episode_positions(results_dir=results_dir, episode=episode)[0]

    errors_by_step = {}
    for step, estimates_by_target in estimates_by_step.items():
        pair_errors = [
            np.linalg.norm(np.asarray(estimate) - positions[target][step - 1])
            for target, estimates in estimates_by_target.items()
            for estimate in estimates
        ]
        errors_by_step[step] = float(np.mean(pair_errors))

    return errors_by_step


def get_metric_fns(env_name: str, module, results_dir: str, permutation_invariant: bool):
    '''
    each *_evaluate.py module exposes belief_error_by_step /
    decomposed_reward_by_step / agent_estimate_variance_by_step with a
    slightly different signature (fire's take a permutation_invariant flag,
    predator_prey/drones/fire's decomposed_reward_by_step needs
    controller_cfg loaded from the run's config.yaml, football needs
    neither) - this wraps each up into a common episode -> step->value dict
    interface

    returns
    -------
    (belief_error_fn, decomposed_reward_fn, estimate_variance_fn), each
    episode:int -> dict
    '''
    if env_name == 'fire':
        controller_cfg = module.load_controller_cfg(results_dir)
        return (
            lambda ep: module.belief_error_by_step(results_dir, ep, permutation_invariant),
            lambda ep: module.decomposed_reward_by_step(results_dir, ep, controller_cfg),
            lambda ep: module.agent_estimate_variance_by_step(results_dir, ep, permutation_invariant),
        )

    if env_name == 'football':
        return (
            lambda ep: module.belief_error_by_step(results_dir, ep),
            lambda ep: module.decomposed_reward_by_step(results_dir, ep),
            lambda ep: module.agent_estimate_variance_by_step(results_dir, ep),
        )

    # predator_prey, drones - belief error uses the permutation-matched
    # estimates (see matched_belief_error_by_step), not the plain
    # fixed-order belief_error_by_step
    controller_cfg = module.load_controller_cfg(results_dir)
    return (
        lambda ep: matched_belief_error_by_step(module, results_dir, ep),
        lambda ep: module.decomposed_reward_by_step(results_dir, ep, controller_cfg),
        lambda ep: module.agent_estimate_variance_by_step(results_dir, ep),
    )


def mean_std(values):
    if not values:
        return None
    return float(np.mean(values)), float(np.std(values))


def wilson_interval(successes: int, n: int, z: float = 1.959964):
    '''
    Wilson score interval for a binomial proportion (95% for the default z): unlike the normal approximation it
    stays inside [0, 1] and keeps a sensible width when the success rate is near 0 or 1
    '''
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(center - half), float(center + half)


def analyze_result_dir(env_name: str, module, results_dir: str, permutation_invariant: bool) -> dict:
    '''
    compute every key metric (episode length, episode reward, success rate,
    belief error, belief variance, decomposed reward) over every episode
    found in results_dir, each first collapsed to a single per-episode
    scalar (mean over steps for the per-step metrics) and then reduced to a
    (mean, std) pair across episodes

    returns
    -------
    dict: metric name -> (mean, std) across episodes, or None if the metric
        isn't defined for this env (e.g. fire's success rate) or no episode
        produced a value for it; plus 'success_ci95', the (low, high) Wilson
        95% interval of the success rate
    '''
    episodes = list_episodes(results_dir)
    belief_error_fn, decomposed_reward_fn, estimate_variance_fn = get_metric_fns(
        env_name, module, results_dir, permutation_invariant,
    )
    _success_desc, success_fn = SUCCESS_RULES[env_name]

    lengths, rewards, successes = [], [], []
    belief_errors, variances, decomposed_rewards = [], [], []

    for episode in episodes:
        length, reward = episode_length_and_reward(results_dir, episode)
        lengths.append(length)
        rewards.append(reward)
        if success_fn is not None:
            successes.append(success_fn(length))

        belief_error = mean_over_steps(belief_error_fn(episode))
        if belief_error is not None:
            belief_errors.append(belief_error)

        variance = mean_over_steps(estimate_variance_fn(episode))
        if variance is not None:
            variances.append(variance)

        decomposed_reward = mean_over_steps(decomposed_reward_fn(episode))
        if decomposed_reward is not None:
            decomposed_rewards.append(decomposed_reward)

    return {
        'n_episodes': len(episodes),
        'episode_length': mean_std(lengths),
        'episode_reward': mean_std(rewards),
        'success_rate': mean_std(successes),
        'success_ci95': wilson_interval(int(np.sum(successes)), len(successes)) if successes else None,
        'belief_error': mean_std(belief_errors),
        'belief_variance': mean_std(variances),
        'decomposed_reward': mean_std(decomposed_rewards),
    }


METRIC_LABELS = [
    ('episode_length', 'avg episode length'),
    ('episode_reward', 'avg episode reward'),
    ('success_rate', 'success rate'),
    ('belief_error', 'avg belief error'),
    ('belief_variance', 'avg belief variance'),
    ('decomposed_reward', 'avg decomposed reward'),
]


def format_report(all_stats: dict) -> str:
    '''
    all_stats: env_name -> source_label -> results_dir -> stats dict (as
    returned by analyze_result_dir)
    '''
    lines = []
    for env_name, by_source in all_stats.items():
        lines.append(f'== {env_name} ==')
        for source_label, results_dir_stats in by_source.items():
            for results_dir, stats in results_dir_stats.items():
                lines.append(f'-- {source_label}: {results_dir} (n={stats["n_episodes"]}) --')
                for key, label in METRIC_LABELS:
                    value = stats[key]
                    if value is None:
                        lines.append(f'  {label}: n/a')
                        continue
                    mean, std = value
                    line = f'  {label}: {mean:.4f} +/- {std:.4f}'
                    if key == 'success_rate' and stats.get('success_ci95') is not None:
                        lo, hi = stats['success_ci95']
                        line += f'  (Wilson 95% CI [{lo:.4f}, {hi:.4f}])'
                    lines.append(line)
        lines.append('')

    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', type=str, default=None,
                         help='if given, also write the report to this text file')
    parser.add_argument('--envs', nargs='+', default=list(SUCCESS_RULES), choices=list(SUCCESS_RULES))
    parser.add_argument('--results-root', type=str, default=None,
                        help='folder holding results_allocentric/ and results_pf/ (one env only; '
                             'default: the RESULTS_DIR / RESULTS_PF_DIR of the env\'s evaluate module)')
    args = parser.parse_args()
    if args.results_root and len(args.envs) != 1:
        parser.error('--results-root needs exactly one env in --envs')

    # same RESULTS_DIR / RESULTS_PF_DIR constants (and, for fire, the
    # Allocentric-vs-PF permutation_invariant handling) each *_evaluate.py's
    # __main__ already uses
    env_configs = {
        'predator_prey': (predator_prey_evaluate, {
            'Allocentric': (predator_prey_evaluate.RESULTS_DIR, True),
            'PF': (predator_prey_evaluate.RESULTS_PF_DIR, True),
        }),
        'football': (football_evaluate, {
            'Allocentric': (football_evaluate.RESULTS_DIR, True),
            'PF': (football_evaluate.RESULTS_PF_DIR, True),
        }),
        'drones': (drones_evaluate, {
            'Allocentric': (drones_evaluate.RESULTS_DIR, True),
            'PF': (drones_evaluate.RESULTS_PF_DIR, True),
        }),
        'fire': (fire_evaluate, {
            'Allocentric': (fire_evaluate.RESULTS_DIR, True),
            'PF': (fire_evaluate.RESULTS_PF_DIR, False),
        }),
    }

    env_configs = {env: env_configs[env] for env in args.envs}
    if args.results_root:
        (env_name, (module, sources)), = env_configs.items()
        folders = {'Allocentric': 'results_allocentric', 'PF': 'results_pf'}
        env_configs[env_name] = (module, {label: (os.path.join(args.results_root, folders[label]), perm)
                                          for label, (_, perm) in sources.items()})

    all_stats = {}
    for env_name, (module, sources) in env_configs.items():
        all_stats[env_name] = {}
        for source_label, (results_dir, permutation_invariant) in sources.items():
            print(f'analyzing {env_name} / {source_label}: {results_dir}')
            stats = analyze_result_dir(env_name, module, results_dir, permutation_invariant)
            all_stats[env_name][source_label] = {results_dir: stats}

    report = format_report(all_stats)
    print()
    print(report)

    if args.save_path:
        with open(args.save_path, 'w') as f:
            f.write(report)
        print(f'saved report to {args.save_path}')
