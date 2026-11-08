"""Episode loaders for the plotting/*_evaluate.py scripts: read an eval run's <episode>.npz and return the
per-step positions and per-agent beliefs. One pair of functions per env, moved here from the old
overlay_plot_<env>.py files.
"""
import itertools
import os

import numpy as np


# ---------------- predator_prey ----------------

def load_predator_prey_positions(results_dir: str, episode: int = 1):
    '''
    load a single episode's eval results and reconstruct absolute (x, y)
    positions for every predator, the target (as seen by each predator),
    and the goal at each step

    positions are recovered from each predator's own observation, using
    obs_map['self_pos'] for the predator's own absolute position and the
    relative slices obs_map['target_pos'] / obs_map['target_goal'] (both
    stored relative to that predator) to recover target/goal position

    the target is also reconstructed separately per predator (rather than
    once, shared) because some pipelines (e.g. particle-filter belief data)
    give each agent a different observed/estimated target position - using
    one agent's reading for every panel would silently mislabel the others'

    true_target_positions comes from infos['target'] instead - the
    target's own observation, as returned directly by the env (see
    RLLibWrapper.step / PFWrapper.step: `infos['target'] = obs['target']`
    before the 'target' key is stripped from the returned obs dict) - this
    is the actual target position, not any predator's (possibly noisy or
    estimated) reading of it, so it's what the overall ground-truth panel
    should use

    inputs
    ------
    results_dir:str
        directory containing the per-episode <idx>.npz result files
    episode:int
        which episode file to load (loads '<episode>.npz')

    returns
    -------
    predator_positions:dict of agent -> (T, 2) array
    target_positions:dict of agent -> (T, 2) array, that agent's own
        observed target position
    true_target_positions:(T, 2) array, the target's own reported position
    goal_position:(2,) array
    '''
    npz_path = os.path.join(results_dir, f'{episode}.npz')
    data = np.load(npz_path, allow_pickle=True)

    obs_map = data['obs_map'].item()
    self_pos_slice = obs_map['self_pos']
    target_pos_slice = obs_map['target_pos']
    goal_slice = obs_map['target_goal']

    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)
    if not steps:
        raise ValueError(f'no steps found in {npz_path}')

    first_obs = data[steps[0]][0]
    predator_names = sorted(first_obs.keys())

    predator_positions = {name: [] for name in predator_names}
    target_positions = {name: [] for name in predator_names}
    true_target_positions = []
    goal_position = None

    for step in steps:
        obs, _rewards, _terminations, _truncations, infos = data[step]
        for name in predator_names:
            self_pos = np.asarray(obs[name][self_pos_slice])
            predator_positions[name].append(self_pos)
            target_positions[name].append(self_pos + np.asarray(obs[name][target_pos_slice]))

        true_target_positions.append(np.asarray(infos['target'][self_pos_slice]))

        if goal_position is None:
            ref_obs = obs[predator_names[0]]
            ref_pos = np.asarray(ref_obs[self_pos_slice])
            goal_position = ref_pos + np.asarray(ref_obs[goal_slice])

    predator_positions = {name: np.stack(pos) for name, pos in predator_positions.items()}
    target_positions = {name: np.stack(pos) for name, pos in target_positions.items()}
    true_target_positions = np.stack(true_target_positions)

    return predator_positions, target_positions, true_target_positions, goal_position


def load_predator_prey_beliefs(results_dir: str, episode: int = 1):
    '''
    load a single episode's eval results and, for each step, convert every
    agent's belief about its teammates (infos['__common__']['sampled_predictions'],
    produced by the belief model) into absolute positions

    sampled_predictions[observer] is that observer's estimate of the
    obs_map['team'] slice - the relative positions of its teammates, in the
    same order as infos['__common__']['obs_no_noise'].keys() with the
    observer itself removed
    - each estimate is anchored to the observer's own ground-truth position
    (obs_no_noise) to recover an absolute estimated position for each teammate

    inputs
    ------
    results_dir:str
        directory containing the per-episode <idx>.npz result files
    episode:int
        which episode file to load (loads '<episode>.npz')

    returns
    -------
    dict: observer -> {target_agent: (T, 2) array of estimated positions}
    '''
    npz_path = os.path.join(results_dir, f'{episode}.npz')
    data = np.load(npz_path, allow_pickle=True)

    obs_map = data['obs_map'].item()
    self_pos_slice = obs_map['self_pos']
    team_slice = obs_map['team']

    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)
    if not steps:
        raise ValueError(f'no steps found in {npz_path}')

    first_common = data[steps[0]][4]['__common__']
    agents = list(first_common['obs_no_noise'].keys())
    dim = (team_slice.stop - team_slice.start) // (len(agents) - 1)

    estimated = {observer: {target: [] for target in agents if target != observer} for observer in agents}

    for step in steps:
        _obs, _rewards, _terminations, _truncations, infos = data[step]
        common = infos['__common__']
        obs_no_noise = common['obs_no_noise']
        sampled_predictions = common['sampled_predictions']

        for observer in agents:
            others = [agent for agent in agents if agent != observer]
            observer_pos = np.asarray(obs_no_noise[observer][self_pos_slice])
            estimate = np.asarray(sampled_predictions[observer]).reshape(-1, dim)

            for idx, target in enumerate(others):
                estimated[observer][target].append(observer_pos + estimate[idx])

    return {
        observer: {target: np.stack(positions) for target, positions in targets.items()}
        for observer, targets in estimated.items()
    }


# ---------------- drones ----------------

def load_drones_positions(results_dir: str, episode: int = 1):
    '''
    load a single episode's eval results and reconstruct absolute (x, y, z)
    positions for every protector drone, the adversary ("target"), and the
    goal boxes at each step

    protector positions are recovered from each drone's own observation via
    obs_map['self_pos'] (never noised, see RLLibWrapper/PFWrapper.step,
    which only add noise to obs_map['target_pos'] and obs_map['team'])

    target_positions is that drone's own (possibly noisy - see
    RLLibWrapper.step/PFWrapper.step, which add noise to
    obs_map['target_pos']) reading of the adversary, recovered from
    self_pos + obs_map['target_pos']

    true_target_positions comes from infos['target'] instead - the
    adversary's own observation, as returned directly by the env (see
    RLLibWrapper.step / PFWrapper.step: `infos['target'] = obs['target']`
    before the 'target' key is stripped from the returned obs dict) - this
    is the actual adversary position, not any protector's (possibly noisy)
    reading of it

    box_positions is read off obs_map['target_goal'] (also never noised)
    via a single reference drone's observation - CaravanAviary.box_state
    defaults to zero velocity so these are normally static, but are
    reconstructed per step in case nonzero box velocities are ever used

    inputs
    ------
    results_dir:str
        directory containing the per-episode <idx>.npz result files
    episode:int
        which episode file to load (loads '<episode>.npz')

    returns
    -------
    drone_positions:dict of agent -> (T, 3) array
    target_positions:dict of agent -> (T, 3) array, that agent's own
        observed adversary position
    true_target_positions:(T, 3) array, the adversary's own reported position
    box_positions:(T, num_goal_boxes, 3) array
    '''
    npz_path = os.path.join(results_dir, f'{episode}.npz')
    data = np.load(npz_path, allow_pickle=True)

    obs_map = data['obs_map'].item()
    self_pos_slice = obs_map['self_pos']
    target_pos_slice = obs_map['target_pos']
    target_goal_slice = obs_map['target_goal']
    num_boxes = (target_goal_slice.stop - target_goal_slice.start) // 3

    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)
    if not steps:
        raise ValueError(f'no steps found in {npz_path}')

    first_obs = data[steps[0]][0]
    drone_names = sorted(first_obs.keys())

    drone_positions = {name: [] for name in drone_names}
    target_positions = {name: [] for name in drone_names}
    true_target_positions = []
    box_positions = []

    for step in steps:
        obs, _rewards, _terminations, _truncations, infos = data[step]

        for name in drone_names:
            self_pos = np.asarray(obs[name][self_pos_slice])
            drone_positions[name].append(self_pos)
            target_positions[name].append(self_pos + np.asarray(obs[name][target_pos_slice]))

        true_target_positions.append(np.asarray(infos['target'][self_pos_slice]))

        ref_obs = obs[drone_names[0]]
        ref_pos = np.asarray(ref_obs[self_pos_slice])
        box_positions.append(ref_pos + np.asarray(ref_obs[target_goal_slice]).reshape(num_boxes, 3))

    drone_positions = {name: np.stack(pos) for name, pos in drone_positions.items()}
    target_positions = {name: np.stack(pos) for name, pos in target_positions.items()}
    true_target_positions = np.stack(true_target_positions)
    box_positions = np.stack(box_positions)

    return drone_positions, target_positions, true_target_positions, box_positions


def load_drones_beliefs(results_dir: str, episode: int = 1):
    '''
    load a single episode's eval results and, for each step, convert every
    drone's belief about its teammates (infos['__common__']['sampled_predictions'],
    produced by the belief model) into absolute positions

    sampled_predictions[observer] is that observer's estimate of the
    obs_map['team'] slice - the relative positions of its teammates, in the
    same order as infos['__common__']['obs_no_noise'].keys() with the
    observer itself removed
    - each estimate is anchored to the observer's own ground-truth position
    (obs_no_noise) to recover an absolute estimated position for each teammate

    inputs
    ------
    results_dir:str
        directory containing the per-episode <idx>.npz result files
    episode:int
        which episode file to load (loads '<episode>.npz')

    returns
    -------
    dict: observer -> {target_drone: (T, 3) array of estimated positions}
    '''
    npz_path = os.path.join(results_dir, f'{episode}.npz')
    data = np.load(npz_path, allow_pickle=True)

    obs_map = data['obs_map'].item()
    self_pos_slice = obs_map['self_pos']
    team_slice = obs_map['team']

    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)
    if not steps:
        raise ValueError(f'no steps found in {npz_path}')

    first_common = data[steps[0]][4]['__common__']
    drone_names = list(first_common['obs_no_noise'].keys())
    dim = (team_slice.stop - team_slice.start) // (len(drone_names) - 1)

    estimated = {observer: {target: [] for target in drone_names if target != observer} for observer in drone_names}

    for step in steps:
        _obs, _rewards, _terminations, _truncations, infos = data[step]
        common = infos['__common__']
        obs_no_noise = common['obs_no_noise']
        sampled_predictions = common['sampled_predictions']

        for observer in drone_names:
            others = [drone for drone in drone_names if drone != observer]
            observer_pos = np.asarray(obs_no_noise[observer][self_pos_slice])
            estimate = np.asarray(sampled_predictions[observer]).reshape(-1, dim)

            for idx, target in enumerate(others):
                estimated[observer][target].append(observer_pos + estimate[idx])

    return {
        observer: {target: np.stack(positions) for target, positions in targets.items()}
        for observer, targets in estimated.items()
    }


# ---------------- football ----------------

def load_football_positions(results_dir: str, episode: int = 1):
    '''
    load a single episode's eval results and reconstruct absolute (x, y)
    positions for every passer agent, the defender ("target", as seen by
    each agent), and the ball at each step

    positions are recovered from each agent's own observation, using
    obs_map['self_pos'] for the agent's own absolute position (never
    noised) and the relative slices obs_map['target_pos'] / obs_map['ball_pos']
    (both stored relative to that agent) to recover defender/ball position

    the defender and ball are reconstructed separately per agent (rather
    than once, shared) because noise/particle-filter belief pipelines give
    each agent a different observed/estimated reading - using one agent's
    reading for every panel would silently mislabel the others'

    true_target_positions comes from infos['target'] instead - the
    defender's own observation, as returned directly by the env (see
    RLLibWrapper.step / PFWrapper.step: `infos['target'] = obs['target']`
    before the 'target' key is stripped from the returned obs dict) - this
    is the actual defender position, not any agent's (possibly noisy or
    estimated) reading of it

    true_ball_positions comes from infos['__common__']['obs_no_noise']
    instead - a snapshot of every agent's own obs taken before per-step
    noise (team/target_pos/ball_pos) is applied (see RLLibWrapper.step /
    PFWrapper.step), so self_pos + ball_pos read off obs_no_noise recovers
    the ball's true absolute position rather than any agent's noisy reading

    inputs
    ------
    results_dir:str
        directory containing the per-episode <idx>.npz result files
    episode:int
        which episode file to load (loads '<episode>.npz')

    returns
    -------
    agent_positions:dict of agent -> (T, 2) array
    target_positions:dict of agent -> (T, 2) array, that agent's own
        observed defender position
    true_target_positions:(T, 2) array, the defender's own reported position
    ball_positions:dict of agent -> (T, 2) array, that agent's own
        observed ball position
    true_ball_positions:(T, 2) array, the ball's true position
    '''
    npz_path = os.path.join(results_dir, f'{episode}.npz')
    data = np.load(npz_path, allow_pickle=True)

    obs_map = data['obs_map'].item()
    self_pos_slice = obs_map['self_pos']
    target_pos_slice = obs_map['target_pos']
    ball_pos_slice = obs_map['ball_pos']

    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)
    if not steps:
        raise ValueError(f'no steps found in {npz_path}')

    first_obs = data[steps[0]][0]
    agent_names = sorted(first_obs.keys())

    agent_positions = {name: [] for name in agent_names}
    target_positions = {name: [] for name in agent_names}
    ball_positions = {name: [] for name in agent_names}
    true_target_positions = []
    true_ball_positions = []

    for step in steps:
        obs, _rewards, _terminations, _truncations, infos = data[step]
        obs_no_noise = infos['__common__']['obs_no_noise']

        for name in agent_names:
            self_pos = np.asarray(obs[name][self_pos_slice])
            agent_positions[name].append(self_pos)
            target_positions[name].append(self_pos + np.asarray(obs[name][target_pos_slice]))
            ball_positions[name].append(self_pos + np.asarray(obs[name][ball_pos_slice]))

        true_target_positions.append(np.asarray(infos['target'][self_pos_slice]))

        ref_obs = np.asarray(obs_no_noise[agent_names[0]])
        true_ball_positions.append(ref_obs[self_pos_slice] + ref_obs[ball_pos_slice])

    agent_positions = {name: np.stack(pos) for name, pos in agent_positions.items()}
    target_positions = {name: np.stack(pos) for name, pos in target_positions.items()}
    ball_positions = {name: np.stack(pos) for name, pos in ball_positions.items()}
    true_target_positions = np.stack(true_target_positions)
    true_ball_positions = np.stack(true_ball_positions)

    return agent_positions, target_positions, true_target_positions, ball_positions, true_ball_positions


def load_football_beliefs(results_dir: str, episode: int = 1):
    '''
    load a single episode's eval results and, for each step, convert every
    agent's belief about its teammates (infos['__common__']['sampled_predictions'],
    produced by the particle-filter/belief pipeline) into absolute positions

    sampled_predictions[observer] is that observer's estimate of the
    obs_map['team'] slice - the relative positions of its teammates, in the
    same order as infos['__common__']['obs_no_noise'].keys() with the
    observer itself removed
    - each estimate is anchored to the observer's own ground-truth position
    (obs_no_noise) to recover an absolute estimated position for each teammate

    inputs
    ------
    results_dir:str
        directory containing the per-episode <idx>.npz result files
    episode:int
        which episode file to load (loads '<episode>.npz')

    returns
    -------
    dict: observer -> {target_agent: (T, 2) array of estimated positions}
    '''
    npz_path = os.path.join(results_dir, f'{episode}.npz')
    data = np.load(npz_path, allow_pickle=True)

    obs_map = data['obs_map'].item()
    self_pos_slice = obs_map['self_pos']
    team_slice = obs_map['team']

    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)
    if not steps:
        raise ValueError(f'no steps found in {npz_path}')

    first_common = data[steps[0]][4]['__common__']
    agents = list(first_common['obs_no_noise'].keys())
    dim = (team_slice.stop - team_slice.start) // (len(agents) - 1)

    estimated = {observer: {target: [] for target in agents if target != observer} for observer in agents}

    for step in steps:
        _obs, _rewards, _terminations, _truncations, infos = data[step]
        common = infos['__common__']
        obs_no_noise = common['obs_no_noise']
        sampled_predictions = common['sampled_predictions']

        for observer in agents:
            others = [agent for agent in agents if agent != observer]
            observer_pos = np.asarray(obs_no_noise[observer][self_pos_slice])
            estimate = np.asarray(sampled_predictions[observer]).reshape(-1, dim)

            for idx, target in enumerate(others):
                estimated[observer][target].append(observer_pos + estimate[idx])

    return {
        observer: {target: np.stack(positions) for target, positions in targets.items()}
        for observer, targets in estimated.items()
    }


# ---------------- fire ----------------

def _fire_best_teammate_permutation(estimate, true_relative):
    '''
    unlike the belief models in the drones/football/predator_prey envs,
    DroneFireEnv's belief model doesn't commit its output to a fixed
    teammate-slot order - DroneFireEnv.team_error (envs/marl/fire_env.py)
    resolves this itself by checking every permutation of the true
    teammate order against the model's (fixed-slot) prediction and keeping
    whichever permutation minimizes total distance; this does the same
    thing for a single (step, observer) pair so a given predicted slot can
    be labeled with a specific teammate name for plotting/error purposes

    inputs
    ------
    estimate:(N, 2) array, the belief model's raw (fixed-slot) prediction
    true_relative:(N, 2) array, ground-truth relative teammate positions,
        in `others` order (see load_episode_beliefs)

    returns
    -------
    tuple of length N: perm such that estimate[i] best matches
    true_relative[perm[i]] - i.e. estimate slot i should be labeled
    others[perm[i]]
    '''
    best_perm, best_dist = None, np.inf

    for perm in itertools.permutations(range(len(true_relative))):
        dist = np.linalg.norm(estimate - true_relative[list(perm)], axis=1).sum()
        if dist < best_dist:
            best_dist = dist
            best_perm = perm

    return best_perm


def load_fire_positions(results_dir: str, episode: int = 1):
    '''
    load a single episode's eval results and reconstruct each drone's
    absolute (row, col) position and the ground-truth fire grid at every
    step

    drone positions come from infos['__common__']['decomposed_obs'][agent]
    ['self_pos'] (see DroneFireEnv.step, which stashes each agent's own
    decomposed obs - fire/team/self_pos - into info before the noise-free
    positions get folded into the packaged observation); fire_env never
    perturbs self_pos, so this is exact ground truth, not a noisy reading

    the fire grid comes from infos['target'], which DroneFireEnv._compute_obs
    sets to self.env.fire_state directly (the full (GRID_SIZE, GRID_SIZE)
    simulation grid) rather than any agent's cropped observation window

    inputs
    ------
    results_dir:str
        directory containing the per-episode <idx>.npz result files
    episode:int
        which episode file to load (loads '<episode>.npz')

    returns
    -------
    drone_positions:dict of agent -> (T, 2) array of (row, col) positions
    fire_grids:(T, GRID_SIZE, GRID_SIZE) array of the ground-truth fire grid
    '''
    npz_path = os.path.join(results_dir, f'{episode}.npz')
    data = np.load(npz_path, allow_pickle=True)

    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)
    if not steps:
        raise ValueError(f'no steps found in {npz_path}')

    first_common = data[steps[0]][4]['__common__']
    agent_names = sorted(first_common['decomposed_obs'].keys())

    drone_positions = {name: [] for name in agent_names}
    fire_grids = []

    for step in steps:
        _obs, _rewards, _terminations, _truncations, infos = data[step]
        decomposed_obs = infos['__common__']['decomposed_obs']

        for name in agent_names:
            drone_positions[name].append(np.asarray(decomposed_obs[name]['self_pos']))

        fire_grids.append(np.asarray(infos['target']))

    drone_positions = {name: np.stack(pos) for name, pos in drone_positions.items()}
    fire_grids = np.stack(fire_grids)

    return drone_positions, fire_grids


def load_fire_beliefs(results_dir: str, episode: int = 1, permutation_invariant: bool = True):
    '''
    load a single episode's eval results and, for each step, convert every
    drone's belief about its teammates (infos['__common__']['sampled_predictions'],
    produced by the belief model) into absolute positions

    sampled_predictions[observer] is that observer's (fixed-slot) estimate
    of decomposed_obs[observer]['team'] - see _fire_best_teammate_permutation
    for how a slot gets matched up with a specific teammate name, since
    the allocentric NN belief model doesn't guarantee a fixed slot ordering
    (it's trained with FirePermutationInvariantLoss, learn/belief/models.py,
    which takes the min over all 3 teammate orderings every step, so
    training never penalizes slot order)

    the particle-filter path is different: FireParticleFilter
    (envs/marl/particle_filters.py) keeps one identity-labeled cluster per
    named teammate (self.clusters[name]), and pf_wrapper.py concatenates
    sampled_predictions in that same fixed per-observer `others` order
    (start_dict's key order in PFWrapper.__init__, which walks self.agents
    the same way `others` below does) - so a PF sampled_predictions slot
    already corresponds to a specific, known teammate, and searching for
    the best-fit permutation would silently paper over genuine PF
    tracking/identity confusion instead of reporting it as belief error

    each estimate is anchored to the observer's own ground-truth position
    (decomposed_obs[observer]['self_pos'], never noised - see
    DroneFireEnv._compute_obs) to recover an absolute estimated position
    for each teammate

    inputs
    ------
    results_dir:str
        directory containing the per-episode <idx>.npz result files
    episode:int
        which episode file to load (loads '<episode>.npz')
    permutation_invariant:bool
        True (default) for the allocentric NN belief model: match each
        predicted slot to whichever teammate minimizes error, the same way
        DroneFireEnv.team_error scores it. False for the particle-filter
        path: skip the search and zip predicted slots to `others` in their
        already-known fixed order

    returns
    -------
    dict: observer -> {target_drone: (T, 2) array of estimated positions}
    '''
    npz_path = os.path.join(results_dir, f'{episode}.npz')
    data = np.load(npz_path, allow_pickle=True)

    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)
    if not steps:
        raise ValueError(f'no steps found in {npz_path}')

    first_common = data[steps[0]][4]['__common__']
    agent_names = sorted(first_common['decomposed_obs'].keys())

    first_predictions = first_common.get('sampled_predictions', {})
    if not all(agent in first_predictions for agent in agent_names):
        raise ValueError(
            f"{npz_path} has no per-agent 'sampled_predictions' (found keys "
            f'{list(first_predictions.keys())} at step {steps[0]}) - this looks like a '
            f"particle-filter eval run, which only records an aggregate scalar 'belief_error' "
            f'per step rather than per-agent position estimates; use '
            f'a results directory with populated sampled_predictions (e.g. a non-"_pf" results '
            f'directory) to plot per-agent belief panels/error curves'
        )

    estimated = {observer: {target: [] for target in agent_names if target != observer} for observer in agent_names}

    for step in steps:
        _obs, _rewards, _terminations, _truncations, infos = data[step]
        common = infos['__common__']
        decomposed_obs = common['decomposed_obs']
        sampled_predictions = common['sampled_predictions']

        for observer in agent_names:
            others = [agent for agent in agent_names if agent != observer]
            observer_pos = np.asarray(decomposed_obs[observer]['self_pos'], dtype=float)
            true_relative = np.asarray(decomposed_obs[observer]['team'], dtype=float)
            estimate = np.asarray(sampled_predictions[observer], dtype=float).reshape(len(others), -1)

            if permutation_invariant:
                perm = _fire_best_teammate_permutation(estimate, true_relative)
            else:
                perm = range(len(others))

            for slot_idx, true_idx in enumerate(perm):
                target = others[true_idx]
                estimated[observer][target].append(observer_pos + estimate[slot_idx])

    return {
        observer: {target: np.stack(positions) for target, positions in targets.items()}
        for observer, targets in estimated.items()
    }
