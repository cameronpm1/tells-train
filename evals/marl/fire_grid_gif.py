import os
import sys
import argparse
from copy import deepcopy

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from envs.marl.make_env import make_marl_env
from envs.marl.fire_env import GRID_SIZE, DroneFireSim, render_fire_frame
from learn.marl.train import make_ray_config, marl_policy_mapping_fn
from controllers.fire_control import extinguish_controller
from evals.marl.predator_prey_grid_gif import COLUMNS, CROP_COLUMNS, compose_grid
from evals.marl.drones_grid_gif import reset_pf_bookkeeping
from util.util import load_config, save_rgb_gif


CHECKPOINT_DIR = 'logs/marl/fire_fully_obs/checkpoint5000'
BELIEF_CONFIG = 'confs/belief/fire.yaml'
BELIEF_DIR = 'logs/belief/_fire_fully_obs_fire_NN/lightning_logs/checkpoints/{epoch}_{val_loss}'
# each column uses the config its reported results were evaluated with; the centralized
# controller uses the fully observable one
FULLY_OBS_CONFIG = 'confs/fire/4a_game.yaml'
ALLOCENTRIC_CONFIG = 'confs/fire/4a_game_allocentric.yaml'
EGOCENTRIC_CONFIG = 'confs/fire/4a_game_pf.yaml'

CROP_MARGIN = 2.0  # cells kept around the furthest in-frame drone / burning or burned cell


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
    '''copy of the fire grid and drone positions needed to re-render this step later'''
    return {
        'fire': base.env.fire_state.copy(),
        'drones': {k: v.copy() for k, v in base.env.drone_states.items()},
    }


def entity_positions(snap):
    '''
    (col, row) points that must stay in view: every drone, plus the corners of the bounding
    box of the burning / burned cells
    '''
    drones = [(c, r) for r, c in (snap['drones'][d] for d in sorted(snap['drones']))]
    rows, cols = np.nonzero(snap['fire'] != DroneFireSim.GREEN)
    fire = [(cols.min(), rows.min()), (cols.max(), rows.max())]
    return np.array(drones + fire, dtype=float)


def run_episode(name, env, algo, seed):
    '''
    roll out one episode from reset(seed=seed), recording snapshots ('frames') that are rendered
    after the shared view is known. Returns dict with frames, positions, start_state, outcome and
    the mean belief error (None without a belief model)
    '''
    np.random.seed(seed)
    obs, infos = env.reset(seed=seed)
    reset_pf_bookkeeping(env)
    base = env.env  # DroneFireEnv

    frames = [snapshot(base)]
    errors = []
    done = False

    while not done:
        if name == 'Centralized':
            actions = extinguish_controller(deepcopy(base.info['__common__']['decomposed_obs']),
                                            **base.controller_kwargs)
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

    fire = base.env.fire_state
    if np.all(fire != DroneFireSim.RED):
        outcome = 'extinguished'
    elif np.all(fire != DroneFireSim.GREEN):
        outcome = 'burned out'
    elif not base._fire_can_spread():
        outcome = 'contained'
    else:
        outcome = 'timeout'

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
    run the allocentric column on seeds 0..n_search-1 and keep the n_rows seeds where the fire
    was put out (extinguished or contained) with the lowest mean belief error
    '''
    results = {}
    for seed in range(n_search):
        results[seed] = run_episode('Allocentric', env, algo, seed)
        r = results[seed]
        print(f'search seed {seed}: belief error {r["belief_error"]:.4f}, {r["outcome"]}, '
              f'{len(r["frames"]) - 1} steps')
    candidates = [s for s in results if results[s]['outcome'] in ('extinguished', 'contained')] or list(results)
    seeds = sorted(candidates, key=lambda s: results[s]['belief_error'])[:n_rows]
    return seeds, {s: results[s] for s in seeds}


def view_bounds(rows):
    '''
    one square view (col_min, col_max, row_min, row_max) shared by every tile, just large enough
    that every drone and every burning / burned cell in CROP_COLUMNS stays in view for the whole
    episode, in every row; shifted (never shrunk) to stay on the grid
    '''
    pts = np.concatenate([r[c]['positions'].reshape(-1, 2) for r in rows for c in CROP_COLUMNS])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2
    half = min(np.max(hi - lo) / 2 + 0.5 + CROP_MARGIN, GRID_SIZE / 2)
    grid_lo, grid_hi = -0.5, GRID_SIZE - 0.5
    center = np.clip(center, grid_lo + half, grid_hi - half)
    print(f'zoom {GRID_SIZE / (2 * half):.2f}x: center (col {center[0]:.1f}, row {center[1]:.1f}), '
          f'half-width {half:.1f} cells')
    return (center[0] - half, center[0] + half, center[1] - half, center[1] + half)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='one grid row per seed; if omitted, seeds are chosen by --search_seeds')
    parser.add_argument('--search_seeds', type=int, default=60,
                        help='number of seeds to try when picking rows by lowest allocentric belief error')
    parser.add_argument('--rows', type=int, default=4, help='number of rows when searching for seeds')
    parser.add_argument('--save_path', type=str, default='fire_grid.gif')
    parser.add_argument('--tile_size', type=int, default=350, help='tile size in pixels')
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()

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

    bounds = view_bounds(rows)
    side = args.tile_size

    def tile_fn(snap):
        frame = render_fire_frame(snap['fire'], snap['drones'], bounds=bounds)
        return cv2.resize(frame, (side, side), interpolation=cv2.INTER_AREA)

    grid = compose_grid(rows, side, tile_fn)
    print(f'saving {len(grid)} frames ({grid[0].shape[1]}x{grid[0].shape[0]}) to {args.save_path}')
    save_rgb_gif(grid, args.save_path, fps=args.fps)


if __name__ == '__main__':
    main()
