import os
import sys
import argparse
from copy import deepcopy

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from envs.marl.make_env import make_marl_env
from envs.marl.pf_wrapper import PFWrapper
from learn.marl.train import make_ray_config, marl_policy_mapping_fn
from controllers.drone_control import drone_controller
from evals.marl.predator_prey_grid_gif import COLUMNS, CROP_COLUMNS, compose_grid
from util.util import load_config, save_rgb_gif


CHECKPOINT_DIR = 'logs/marl/drones_fully_obs/checkpoint1500'
BELIEF_CONFIG = 'confs/belief/drones.yaml'
BELIEF_DIR = 'logs/belief/_drones_fully_obs_drones_NN/lightning_logs/checkpoints/{epoch}_{val_loss}'
# each column uses the config its reported results were evaluated with (the env / adversary
# settings differ between them); the centralized controller uses the fully observable one
FULLY_OBS_CONFIG = 'confs/drones/3a_game.yaml'
ALLOCENTRIC_CONFIG = 'confs/drones/3a_game_allocentric.yaml'
EGOCENTRIC_CONFIG = 'confs/drones/3a_game_pf.yaml'

RENDER_SIZE = 700   # CaravanAviary.render_rgb default resolution, downscaled to the tile size
CROP_MARGIN = 1.0   # world units kept around the furthest in-frame drone / box edge


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
        # centralized controller acts on the true drone states, so the wrapper's noise is irrelevant
        'Centralized': make_marl_env(deepcopy(fo_cfg), wrap='rllib', eval=True, belief_kwargs={'on': False}),
        'Fully Observable': make_marl_env(deepcopy(fo_cfg), wrap='rllib', eval=True, belief_kwargs={'on': False}),
        'Allocentric': make_marl_env(deepcopy(allo_cfg), wrap='rllib', eval=True, belief_kwargs=belief_kwargs),
        'Egocentric': make_marl_env(deepcopy(ego_cfg), wrap='pf', eval=True, belief_kwargs={}),
    }, fo_cfg


def reset_pf_bookkeeping(env):
    '''
    clear the PF wrapper's observation-switching state, which PFWrapper.reset leaves over from
    the previous episode (so the same seed would not reproduce the same rollout)
    '''
    if isinstance(env, PFWrapper):
        env.consecutive_agent_count = {agent: 0 for agent in env.agents}
        env.min_confidence_agent = {}
        env.temp_noise = {}
        env._step = 0


def world_state(base):
    '''x-y positions of every drone (protectors then adversary) and every goal box'''
    drones = base._get_team_state()[:, 0:2]
    boxes = base.box_state[:base.num_goal_boxes, 0:2]
    return np.concatenate([drones, boxes]).astype(float)


def run_episode(name, env, algo, seed, bounds=None, side=None):
    '''
    roll out one episode from reset(seed=seed)

    frames are only rendered when bounds (a fixed xmin, xmax, ymin, ymax view) is given, and are
    downscaled to side x side tiles as they are made. Returns dict with frames, positions,
    start_state, breached, oob and the mean belief error (None without a belief model)
    '''
    # observation noise and the particle filter draw from the global numpy RNG
    np.random.seed(seed)
    obs, infos = env.reset(seed=seed)
    base = env.env  # CaravanAviary
    # BaseAviary.reset does not reset the per-drone PID velocity controllers, so their integral
    # terms would otherwise carry over from the previous episode and the same seed would not
    # reproduce the same rollout
    for ctrl in base.ctrl:
        ctrl.reset()
    reset_pf_bookkeeping(env)

    def render():
        frame = base.render_rgb(bounds=bounds)
        return cv2.resize(frame, (side, side), interpolation=cv2.INTER_AREA)

    positions = [world_state(base)]
    frames = [render()] if bounds is not None else []
    errors = []
    done = False
    breached, oob = False, False

    while not done:
        if name == 'Centralized':
            actions = drone_controller(deepcopy(base.obs), base.obs_map, base.controller_cfg)
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

        agent_info = infos['__common__'][base.agents[0]]
        breached = breached or bool(agent_info.get('breached', False))
        oob = oob or bool(agent_info.get('oob', False))
        if infos['__common__'].get('belief_error') is not None:
            errors.append(infos['__common__']['belief_error'])
        positions.append(world_state(base))
        if bounds is not None:
            frames.append(render())

    return {
        'frames': frames,
        'positions': np.array(positions),
        'start_state': positions[0],
        'breached': breached,
        'oob': oob,
        'belief_error': float(np.mean(errors)) if errors else None,
    }


def outcome(r):
    return 'breached' if r['breached'] else ('out of bounds' if r['oob'] else 'defended')


def search_seeds(env, algo, n_search, n_rows):
    '''
    run the allocentric column on seeds 0..n_search-1 and keep the n_rows seeds where the
    boxes were defended with the lowest mean belief error
    '''
    results = {}
    for seed in range(n_search):
        results[seed] = run_episode('Allocentric', env, algo, seed)
        r = results[seed]
        print(f'search seed {seed}: belief error {r["belief_error"]:.4f}, {outcome(r)}, '
              f'{len(r["positions"]) - 1} steps')
    candidates = [s for s in results if outcome(results[s]) == 'defended'] or list(results)
    return sorted(candidates, key=lambda s: results[s]['belief_error'])[:n_rows]


def view_bounds(rows, num_boxes, box_half_extents):
    '''
    one square view shared by every tile, just large enough that every drone in CROP_COLUMNS
    and every goal box stays in view for the whole episode, in every row
    '''
    hx, hy = box_half_extents[0], box_half_extents[1]
    pts = []
    for r in rows:
        for c in CROP_COLUMNS:
            p = r[c]['positions']
            pts.append(p.reshape(-1, 2))
        # box corners (boxes are the trailing entries of every position row)
        boxes = r[CROP_COLUMNS[0]]['positions'][0][-num_boxes:]
        pts.append(boxes + [-hx, -hy])
        pts.append(boxes + [hx, hy])
    pts = np.concatenate(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2
    half = np.max(hi - lo) / 2 + CROP_MARGIN
    print(f'view: center ({center[0]:.2f}, {center[1]:.2f}), half-width {half:.2f}')
    return (center[0] - half, center[0] + half, center[1] - half, center[1] + half)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='one grid row per seed; if omitted, seeds are chosen by --search_seeds')
    parser.add_argument('--search_seeds', type=int, default=60,
                        help='number of seeds to try when picking rows by lowest allocentric belief error')
    parser.add_argument('--rows', type=int, default=4, help='number of rows when searching for seeds')
    parser.add_argument('--save_path', type=str, default='drones_grid.gif')
    parser.add_argument('--scale', type=float, default=0.5, help='tile size as a fraction of the 700px render')
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()

    import ray
    ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False)

    envs, fo_cfg = build_envs()

    # all decentralized columns run the same trained policy; only the observation pipeline differs
    algo = make_ray_config({**fo_cfg, 'alg': {**fo_cfg['alg'], 'nenvs': 0}}).build_algo()
    algo.restore(os.path.abspath(CHECKPOINT_DIR))

    if args.seeds is None:
        seeds = search_seeds(envs['Allocentric'], algo, args.search_seeds, args.rows)
        print(f'selected seeds (lowest allocentric belief error): {seeds}')
    else:
        seeds = args.seeds

    # pass 1: positions only, to size the shared view
    rows = []
    for seed in seeds:
        row = {c: run_episode(c, envs[c], algo, seed) for c in COLUMNS}
        ref = row[COLUMNS[0]]['start_state']
        for c in COLUMNS[1:]:
            assert np.allclose(row[c]['start_state'], ref), f'{c} did not start from the same state as {COLUMNS[0]}'
        rows.append(row)
    base = envs['Centralized'].env
    bounds = view_bounds(rows, base.num_goal_boxes, base.goal_box_half_extents)

    # pass 2: replay the same episodes, rendering against the fixed view
    side = int(round(RENDER_SIZE * args.scale))
    for seed, row in zip(seeds, rows):
        for c in COLUMNS:
            r = run_episode(c, envs[c], algo, seed, bounds=bounds, side=side)
            assert np.allclose(r['positions'], row[c]['positions']), f'seed {seed} {c} replay diverged'
            row[c] = r
            err = r['belief_error']
            print(f'seed {seed} | {c}: {len(r["frames"]) - 1} steps, {outcome(r)}'
                  + (f', belief error={err:.4f}' if err is not None else ''))

    grid = compose_grid(rows, side)
    print(f'saving {len(grid)} frames ({grid[0].shape[1]}x{grid[0].shape[0]}) to {args.save_path}')
    save_rgb_gif(grid, args.save_path, fps=args.fps)


if __name__ == '__main__':
    main()
