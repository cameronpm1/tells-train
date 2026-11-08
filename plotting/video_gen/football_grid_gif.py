import os
import sys
import argparse
from copy import deepcopy

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from envs.marl.make_env import make_marl_env
from learn.marl.train import make_ray_config, marl_policy_mapping_fn
from controllers.football_control import compute_rondo_actions
from plotting.video_gen.predator_prey_grid_gif import COLUMNS, CROP_COLUMNS, compose_grid
from plotting.video_gen.drones_grid_gif import reset_pf_bookkeeping
from util.util import load_config, save_rgb_gif, MODELS_ROOT


CHECKPOINT_DIR = os.path.join(MODELS_ROOT, 'marl', 'football_fully_obs/checkpoint3000')
BELIEF_CONFIG = 'confs/belief/football.yaml'
BELIEF_DIR = os.path.join(MODELS_ROOT, 'belief', '_football_fully_obs_NN_1/epoch1026')
# each column uses the config its reported results were evaluated with; the centralized
# controller uses the fully observable one
FULLY_OBS_CONFIG = 'confs/football/5a_game.yaml'
ALLOCENTRIC_CONFIG = 'confs/football/5a_game_allocentric.yaml'
EGOCENTRIC_CONFIG = 'confs/football/5a_game_pf.yaml'

RENDER_SIZE = 700   # square render resolution, downscaled to the tile size
CROP_MARGIN = 0.06  # pitch units kept around the furthest in-frame player / ball
# raw GRF observation fields CirclePass5v1Env.render_rgb draws from
RENDER_KEYS = ['left_team', 'right_team', 'left_team_direction', 'right_team_direction',
               'left_team_roles', 'right_team_roles', 'ball']


def build_envs():
    '''
    one env per column, each with the wrapper / observation model that column is evaluated with
    '''
    from evals.belief.eval import load_model

    fo_cfg = load_config(FULLY_OBS_CONFIG)
    allo_cfg = load_config(ALLOCENTRIC_CONFIG)
    ego_cfg = load_config(EGOCENTRIC_CONFIG)

    belief_kwargs = {
        'on': True,
        'model': load_model(config_dir=BELIEF_CONFIG, ckpt_dir=BELIEF_DIR),
        'min_obs': allo_cfg['min_obs'],
    }

    return {
        'Centralized': make_marl_env(deepcopy(fo_cfg), wrap='rllib', eval=True, belief_kwargs={'on': False}),
        'Fully Observable': make_marl_env(deepcopy(fo_cfg), wrap='rllib', eval=True, belief_kwargs={'on': False}),
        'Allocentric': make_marl_env(deepcopy(allo_cfg), wrap='rllib', eval=True, belief_kwargs=belief_kwargs),
        'Egocentric': make_marl_env(deepcopy(ego_cfg), wrap='pf', eval=True, belief_kwargs={}),
    }, fo_cfg


def snapshot(base):
    '''copy of the raw GRF observation fields needed to re-render this step later'''
    return {k: np.array(base.raw_obs[0][k], copy=True) for k in RENDER_KEYS}


def entity_positions(snap):
    '''x-y positions of every outfield player (goalkeepers are not drawn) and the ball'''
    left = snap['left_team'][snap['left_team_roles'] != 0]
    right = snap['right_team'][snap['right_team_roles'] != 0]
    return np.concatenate([left, right, snap['ball'][None, :2]]).astype(float)


def run_episode(name, env, algo, seed):
    '''
    roll out one episode from a reset with the global numpy RNG seeded to seed (the scenario
    builder draws the start positions from it)

    the GRF engine is not deterministic, so rollouts are recorded as raw-observation snapshots
    ('frames') and rendered later instead of being replayed. Returns dict with frames,
    positions, start_state, outcome and the mean belief error (None without a belief model)
    '''
    np.random.seed(seed)
    obs, infos = env.reset(seed=seed)
    reset_pf_bookkeeping(env)
    base = env.env  # CirclePass5v1Env

    frames = [snapshot(base)]
    errors = []
    done = False

    while not done:
        if name == 'Centralized':
            actions = compute_rondo_actions(deepcopy(base.obs), base.obs_map)
            actions = {a: actions[a] for a in obs.keys()}
        else:
            actions = {
                agent_id: algo.compute_single_action(
                    agent_obs,
                    policy_id=marl_policy_mapping_fn(agent_id, 0),
                    explore=False,
                )
                for agent_id, agent_obs in obs.items()
            }

        obs, rewards, terminations, truncations, infos = env.step(actions)
        done = terminations['__all__'] or truncations['__all__']
        if infos['__common__'].get('belief_error') is not None:
            errors.append(infos['__common__']['belief_error'])
        frames.append(snapshot(base))

    if base.oob:
        outcome = 'out of play'
    elif base._step >= base.max_episode_length:
        outcome = 'kept possession'
    else:
        outcome = 'stolen'

    positions = np.array([entity_positions(f) for f in frames])
    return {
        'frames': frames,
        'positions': positions,
        'start_state': positions[0],
        'outcome': outcome,
        'belief_error': float(np.mean(errors)) if errors else None,
    }


def search_seeds(env, algo, n_search, n_rows):
    '''
    run the allocentric column on seeds 0..n_search-1 and keep the n_rows seeds where the team
    kept possession with the lowest mean belief error. Returns (seeds, cached results by seed):
    the engine is not deterministic, so the selected runs themselves are reused in the grid.
    '''
    results = {}
    for seed in range(n_search):
        results[seed] = run_episode('Allocentric', env, algo, seed)
        r = results[seed]
        print(f'search seed {seed}: belief error {r["belief_error"]:.4f}, {r["outcome"]}, '
              f'{len(r["frames"]) - 1} steps')
    candidates = [s for s in results if results[s]['outcome'] == 'kept possession'] or list(results)
    seeds = sorted(candidates, key=lambda s: results[s]['belief_error'])[:n_rows]
    return seeds, {s: results[s] for s in seeds}


def interpolate_frames(frames, n):
    '''
    n - 1 linearly blended snapshots between each pair of consecutive steps (positions, directions, ball), for
    smooth slow motion; player roles are categorical and are kept from the earlier step
    '''
    if n <= 1 or len(frames) < 2:
        return frames
    out = []
    for a, b in zip(frames[:-1], frames[1:]):
        for k in range(n):
            w = k / n
            out.append({key: a[key] if key.endswith('roles') else (1 - w) * a[key] + w * b[key] for key in a})
    out.append(frames[-1])
    return out


def view_bounds(rows, scale=1.0):
    '''
    one square view shared by every tile, framing the 1st-99th percentile of every player and ball position in
    CROP_COLUMNS over the whole episode, in every row: a rare excursion (e.g. an episode ending out of play)
    briefly leaves the frame instead of zooming the whole grid out
    '''
    pts = np.concatenate([r[c]['positions'].reshape(-1, 2) for r in rows for c in CROP_COLUMNS])
    lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
    center = (lo + hi) / 2
    half = (np.max(hi - lo) / 2 + CROP_MARGIN) * scale  # scale < 1 zooms in
    print(f'view: center ({center[0]:.3f}, {center[1]:.3f}), half-width {half:.3f}')
    return (center[0] - half, center[0] + half, center[1] - half, center[1] + half)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='one grid row per seed; if omitted, seeds are chosen by --search_seeds')
    parser.add_argument('--search_seeds', type=int, default=60,
                        help='number of seeds to try when picking rows by lowest allocentric belief error')
    parser.add_argument('--rows', type=int, default=4, help='number of rows when searching for seeds')
    parser.add_argument('--save_path', type=str, default='football_grid.gif')
    parser.add_argument('--scale', type=float, default=0.5, help='tile size as a fraction of the 700px render')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--view-scale', type=float, default=1.0,
                        help='shrink (<1, zoom in) or grow the shared view around the players')
    parser.add_argument('--interp', type=int, default=1,
                        help='frames per simulation step (in-between frames are interpolated); playback speed is '
                             'fps / interp steps per second')
    parser.add_argument('--episodes-cache', type=str, default=None,
                        help='pickle of the simulated episodes: loaded if it exists (no simulation; the engine is not '
                             'deterministic, so this keeps the same games across re-renders), else written')
    args = parser.parse_args()

    if args.episodes_cache and os.path.exists(args.episodes_cache):
        import pickle
        with open(args.episodes_cache, 'rb') as f:
            rows = pickle.load(f)
        print(f'loaded {len(rows)} rows of episodes from {args.episodes_cache}')
        render_grid(rows, args)
        return

    import ray
    ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False)

    envs, fo_cfg = build_envs()

    # all decentralized columns run the same trained policy; only the observation pipeline differs
    algo = make_ray_config({**fo_cfg, 'alg': {**fo_cfg['alg'], 'nenvs': 0}}).build_algo()
    algo.restore(os.path.abspath(CHECKPOINT_DIR))

    if args.seeds is None:
        seeds, cached = search_seeds(envs['Allocentric'], algo, args.search_seeds, args.rows)
        print(f'selected seeds (lowest allocentric belief error): {seeds}')
    else:
        seeds, cached = args.seeds, {}

    rows = []
    for seed in seeds:
        row = {}
        for c in COLUMNS:
            if c == 'Allocentric' and seed in cached:
                row[c] = cached[seed]
            else:
                row[c] = run_episode(c, envs[c], algo, seed)
            err = row[c]['belief_error']
            print(f'seed {seed} | {c}: {len(row[c]["frames"]) - 1} steps, {row[c]["outcome"]}'
                  + (f', belief error={err:.4f}' if err is not None else ''))
        ref = row[COLUMNS[0]]['start_state']
        for c in COLUMNS[1:]:
            assert np.allclose(row[c]['start_state'], ref), f'{c} did not start from the same state as {COLUMNS[0]}'
        rows.append(row)

    if args.episodes_cache:
        import pickle
        with open(args.episodes_cache, 'wb') as f:
            pickle.dump([{c: {'frames': r[c]['frames'], 'positions': r[c]['positions']} for c in r} for r in rows], f)
    render_grid(rows, args)


def render_grid(rows, args):
    from envs.marl.football_env import CirclePass5v1Env
    bounds = view_bounds(rows, args.view_scale)
    side = int(round(RENDER_SIZE * args.scale))
    renderer = CirclePass5v1Env.__new__(CirclePass5v1Env)  # render_rgb only reads raw_obs: no engine needed
    rows = [{c: {**r[c], 'frames': interpolate_frames(r[c]['frames'], args.interp)} for c in r} for r in rows]

    def tile_fn(snap):
        renderer.raw_obs = [snap]
        frame = renderer.render_rgb(width=RENDER_SIZE, height=RENDER_SIZE, bounds=bounds)
        return cv2.resize(frame, (side, side), interpolation=cv2.INTER_AREA)

    grid = compose_grid(rows, side, tile_fn)
    print(f'saving {len(grid)} frames ({grid[0].shape[1]}x{grid[0].shape[0]}) to {args.save_path}')
    save_rgb_gif(grid, args.save_path, fps=args.fps)


if __name__ == '__main__':
    main()
