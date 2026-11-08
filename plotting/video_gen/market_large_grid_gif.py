import os
import sys
import argparse
from copy import deepcopy

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from envs.marl.make_env import make_marl_env
from learn.marl.train import make_ray_config
from controllers.market_game_large_control import compute_market_actions
from plotting.video_gen.predator_prey_grid_gif import GAP, OUTLINE, compose_grid
from plotting.video_gen.market_grid_gif import (
    TileRenderer, draw_line_legend, run_episode, save_gif_shared_palette, search_seeds,
)
from util.util import load_config, MODELS_ROOT


CHECKPOINT_DIR = os.path.join(MODELS_ROOT, 'marl', 'market_game_large_fully_obs/checkpoint250')
BELIEF_CONFIG = 'confs/belief/market_game_large.yaml'
# snapshot of the label-1 belief model taken when results_allocentric was evaluated
BELIEF_DIR = os.path.join(MODELS_ROOT, 'belief', '_market_game_large_fully_obs_NN_1/epoch3500')
# fully observable and allocentric were both evaluated with 60a_game.yaml (the belief wrapper
# turns off the team noise); the centralized controller uses it too
CONFIG = 'confs/market_game_large/60a_game.yaml'

# no egocentric column: the PF wrapper is not implemented for the large market env
COLUMNS = ['Centralized', 'Fully Observable', 'Allocentric']


def build_envs():
    '''
    one env per column, each with the wrapper / observation model that column is evaluated with
    '''
    from evals.belief.eval import load_model

    cfg = load_config(CONFIG)
    belief_kwargs = {
        'on': True,
        'model': load_model(config_dir=BELIEF_CONFIG, ckpt_dir=BELIEF_DIR),
        'min_obs': cfg['min_obs'],
    }

    return {
        'Centralized': make_marl_env(deepcopy(cfg), wrap='rllib', eval=True, belief_kwargs={'on': False}),
        'Fully Observable': make_marl_env(deepcopy(cfg), wrap='rllib', eval=True, belief_kwargs={'on': False}),
        'Allocentric': make_marl_env(deepcopy(cfg), wrap='rllib', eval=True, belief_kwargs=belief_kwargs),
    }, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='one grid row per seed; if omitted, seeds are chosen by --search_seeds')
    parser.add_argument('--search_seeds', type=int, default=60,
                        help='number of seeds to try when picking rows by lowest allocentric belief error')
    parser.add_argument('--rows', type=int, default=4, help='number of rows when searching for seeds')
    parser.add_argument('--save_path', type=str, default='market_large_grid.gif')
    parser.add_argument('--tile_size', type=int, default=350, help='tile size in pixels')
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()

    import ray
    ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False)

    envs, cfg = build_envs()

    # both decentralized columns run the same trained policy; only the observation pipeline differs
    algo = make_ray_config({**cfg, 'alg': {**cfg['alg'], 'nenvs': 0}}).build_algo()
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
                row[c] = run_episode(c, envs[c], algo, seed, controller=compute_market_actions)
            err = row[c]['belief_error']
            print(f'seed {seed} | {c}: {len(row[c]["frames"]) - 1} steps, {row[c]["outcome"]}'
                  + (f', belief error={err:.4f}' if err is not None else ''))
        ref = row[COLUMNS[0]]['start_state']
        for c in COLUMNS[1:]:
            assert np.allclose(row[c]['start_state'], ref), f'{c} did not start from the same state as {COLUMNS[0]}'
        rows.append(row)

    base = envs['Centralized'].env
    renderer = TileRenderer(base, args.tile_size, rows, columns=COLUMNS)
    band = draw_line_legend(len(COLUMNS) * (args.tile_size + 2 * OUTLINE + GAP))

    grid = compose_grid(rows, args.tile_size, renderer, extra_band=band, columns=COLUMNS)
    print(f'saving {len(grid)} frames ({grid[0].shape[1]}x{grid[0].shape[0]}) to {args.save_path}')
    save_gif_shared_palette(grid, args.save_path, fps=args.fps)


if __name__ == '__main__':
    main()
