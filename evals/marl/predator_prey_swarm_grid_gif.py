import os
import sys
import argparse
from copy import deepcopy

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from envs.marl.make_env import make_marl_env
from learn.marl.train import make_ray_config
from controllers.predator_prey_swarm_control import compute_slot_actions
from evals.marl.predator_prey_grid_gif import build_grid, run_episode, search_seeds
from util.util import load_config, save_rgb_gif


CHECKPOINT_DIR = 'logs/marl/predator_prey_swarm_40a_fully_obs/checkpoint250'
BELIEF_CONFIG = 'confs/belief/predator_prey_swarm.yaml'
BELIEF_DIR = ('logs/belief/_predator_prey_swarm_40a_fully_obs_predator_prey_swarm_NN_1/'
              'lightning_logs/checkpoints/{epoch}_{val_loss}')
FULLY_OBS_CONFIG = 'confs/predator_prey_swarm/40a_game.yaml'
ALLOCENTRIC_CONFIG = 'confs/predator_prey_swarm/40a_game_allocentric.yaml'

# no egocentric column: the PF wrapper is not implemented for the swarm env
COLUMNS = ['Centralized', 'Fully Observable', 'Allocentric']


def build_envs():
    '''
    one env per column, each with the wrapper / observation model that column is evaluated with
    '''
    from evals.belief.eval import load_model

    fo_cfg = load_config(FULLY_OBS_CONFIG)
    allo_cfg = load_config(ALLOCENTRIC_CONFIG)

    belief_kwargs = {
        'on': True,
        'model': load_model(config_dir=BELIEF_CONFIG, ckpt_dir=BELIEF_DIR),
        'min_obs': allo_cfg['min_obs'],
    }

    return {
        # centralized controller acts on the true world state, so the wrapper's noise is irrelevant
        'Centralized': make_marl_env(deepcopy(fo_cfg), wrap='rllib', eval=True, render_mode='rgb_array',
                                     belief_kwargs={'on': False}),
        'Fully Observable': make_marl_env(deepcopy(fo_cfg), wrap='rllib', eval=True, render_mode='rgb_array',
                                          belief_kwargs={'on': False}),
        'Allocentric': make_marl_env(deepcopy(allo_cfg), wrap='rllib', eval=True, render_mode='rgb_array',
                                     belief_kwargs=belief_kwargs),
    }, fo_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='one grid row per seed; if omitted, seeds are chosen by --search_seeds')
    parser.add_argument('--search_seeds', type=int, default=60,
                        help='number of seeds to try when picking rows by lowest allocentric belief error')
    parser.add_argument('--rows', type=int, default=4, help='number of rows when searching for seeds')
    parser.add_argument('--save_path', type=str, default='predator_prey_swarm_grid.gif')
    parser.add_argument('--scale', type=float, default=0.5, help='resize factor applied to each tile')
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()

    import ray
    ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False)

    envs, fo_cfg = build_envs()

    # both decentralized columns run the same trained policy; only the observation pipeline differs
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
                row[c] = run_episode(c, envs[c], algo, seed, controller=compute_slot_actions)
            err = row[c]['belief_error']
            print(f'seed {seed} | {c}: {len(row[c]["frames"]) - 1} steps, success={row[c]["success"]}, '
                  f'oob={row[c]["oob"]}' + (f', belief error={err:.4f}' if err is not None else ''))
        ref = row[COLUMNS[0]]['start_state']
        for c in COLUMNS[1:]:
            assert np.allclose(row[c]['start_state'], ref), f'{c} did not start from the same state as {COLUMNS[0]}'
        rows.append(row)

    grid = build_grid(rows, args.scale, columns=COLUMNS, crop_columns=COLUMNS)
    print(f'saving {len(grid)} frames ({grid[0].shape[1]}x{grid[0].shape[0]}) to {args.save_path}')
    save_rgb_gif(grid, args.save_path, fps=args.fps, loop=0)


if __name__ == '__main__':
    main()
