import os
import sys
import argparse
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from envs.marl.make_env import make_marl_env
from learn.marl.train import make_ray_config, marl_policy_mapping_fn
from controllers.predator_prey_control import compute_slot_actions
from envs.marl.predator_prey_env import GRID_SIZE
from util.util import load_config, save_rgb_gif


CHECKPOINT_DIR = 'logs/marl/predator_prey_fully_obs/checkpoint1000'
BELIEF_CONFIG = 'confs/belief/predator_prey.yaml'
BELIEF_DIR = 'logs/belief/_predator_prey_fully_obs_predator_prey_NN/lightning_logs/checkpoints/{epoch}_{val_loss}'
FULLY_OBS_CONFIG = 'confs/predator_prey/3a_game.yaml'
ALLOCENTRIC_CONFIG = 'confs/predator_prey/3a_game_allocentric.yaml'
EGOCENTRIC_CONFIG = 'confs/predator_prey/3a_game_pf.yaml'

# grid columns, left to right, with the outline color (RGB) used for each
COLUMNS = ['Centralized', 'Fully Observable', 'Allocentric', 'Egocentric']
COLUMN_COLORS = {
    'Centralized': (77, 77, 77),
    'Fully Observable': (230, 159, 0),
    'Allocentric': (142, 68, 173),
    'Egocentric': (23, 190, 207),
}
# Nimbus Sans is a metric-compatible Helvetica clone
FONT_PATH = '/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf'
FONT_SIZE = 22
OUTLINE = 5         # colored outline thickness around each tile
GAP = 18            # spacing between tiles and around the grid edge
LEGEND_HEIGHT = 60
SEED_LABEL_WIDTH = 34  # left strip holding the vertical seed label
SEED_LABEL_MARGIN = 6  # distance from the image's left edge to the label
CAM_RANGE = GRID_SIZE + 0.15  # half-width of the world shown by ScenarioEnv.draw
CROP_MARGIN = 1.0  # world units kept around the furthest in-frame entity
# columns whose entities must stay inside the zoomed window; egocentric agents may leave it
CROP_COLUMNS = ['Centralized', 'Fully Observable', 'Allocentric']


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
        # centralized controller acts on the true world state, so the wrapper's noise is irrelevant
        'Centralized': make_marl_env(deepcopy(fo_cfg), wrap='rllib', eval=True, render_mode='rgb_array',
                                     belief_kwargs={'on': False}),
        'Fully Observable': make_marl_env(deepcopy(fo_cfg), wrap='rllib', eval=True, render_mode='rgb_array',
                                          belief_kwargs={'on': False}),
        'Allocentric': make_marl_env(deepcopy(allo_cfg), wrap='rllib', eval=True, render_mode='rgb_array',
                                     belief_kwargs=belief_kwargs),
        'Egocentric': make_marl_env(deepcopy(ego_cfg), wrap='pf', eval=True, render_mode='rgb_array',
                                    belief_kwargs={}),
    }, fo_cfg


def world_state(env):
    '''positions of every entity in the underlying mpe world, used to check columns start identically'''
    world = env.env.env.unwrapped.world
    return np.concatenate([e.state.p_pos for e in world.entities])


def run_episode(name, env, algo, seed, controller=compute_slot_actions, render=True):
    '''
    roll out one episode from reset(seed=seed)

    returns dict with frames (empty when render is False), start_state, success, oob and the
    mean belief error over the episode (None for columns without a belief model)
    '''
    # observation noise and the particle filter draw from the global numpy RNG
    np.random.seed(seed)
    obs, infos = env.reset(seed=seed)
    start_state = world_state(env)
    base = env.env  # PredatorPreyEnv
    frames = [env.render_rgb()] if render else []
    positions = [start_state.reshape(-1, 2)]
    errors = []
    done = False
    success, oob = False, False

    while not done:
        if name == 'Centralized':
            actions = controller(deepcopy(base.obs), base.obs_map)
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
        success = success or bool(agent_info.get('success', False))
        oob = oob or bool(agent_info.get('oob', False))
        if infos['__common__'].get('belief_error') is not None:
            errors.append(infos['__common__']['belief_error'])
        if render:
            frames.append(env.render_rgb())
        positions.append(world_state(env).reshape(-1, 2))

    return {
        'frames': frames,
        'positions': np.array(positions),
        'start_state': start_state,
        'success': success,
        'oob': oob,
        'belief_error': float(np.mean(errors)) if errors else None,
    }


def search_seeds(env, algo, n_search, n_rows):
    '''
    run the allocentric column on seeds 0..n_search-1 and keep the n_rows successful seeds
    with the lowest mean belief error. Returns (seeds, cache): rollouts are deterministic, so
    the search runs without rendering (keeping every frame would use tens of GB) and the
    selected seeds are simply re-run, leaving the cache empty.
    '''
    results = {}
    for seed in range(n_search):
        results[seed] = run_episode('Allocentric', env, algo, seed, render=False)
        r = results[seed]
        print(f'search seed {seed}: belief error {r["belief_error"]:.4f}, success={r["success"]}, '
              f'{len(r["positions"]) - 1} steps')
    candidates = [s for s in results if results[s]['success']] or list(results)
    seeds = sorted(candidates, key=lambda s: results[s]['belief_error'])[:n_rows]
    return seeds, {}


def text_size(draw, text, font):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top, top


def draw_legend(width, font, columns=COLUMNS):
    '''legend band: outline color per controller'''
    img = Image.new('RGB', (width, LEGEND_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    swatch = 18
    item_gap = 34
    cy = LEGEND_HEIGHT // 2

    widths = [swatch + 8 + text_size(draw, c, font)[0] for c in columns]
    total = sum(widths) + item_gap * (len(columns) - 1)
    x = (width - total) // 2

    for label, w in zip(columns, widths):
        draw.rectangle([x, cy - swatch // 2, x + swatch, cy + swatch // 2], fill=COLUMN_COLORS[label])
        # center on the cap height so labels with descenders share a baseline
        _, th, top = text_size(draw, 'H', font)
        draw.text((x + swatch + 8, cy - th // 2 - top), label, fill=(0, 0, 0), font=font)
        x += w + item_gap
    return np.asarray(img)


def draw_seed_label(height, row_number, font):
    '''vertical "Seed N" label, read bottom to top, for the left side of a row'''
    text = f'Seed {row_number}'
    tmp = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    tw, th, top = text_size(tmp, text, font)
    label = Image.new('RGB', (tw, th), (255, 255, 255))
    ImageDraw.Draw(label).text((0, -top), text, fill=(0, 0, 0), font=font)
    label = label.rotate(90, expand=True)

    img = Image.new('RGB', (SEED_LABEL_WIDTH, height), (255, 255, 255))
    img.paste(label, (SEED_LABEL_MARGIN, (height - label.height) // 2))
    return np.asarray(img)


def crop_window(rows, frame_shape, crop_columns=CROP_COLUMNS):
    '''
    one square pixel window shared by every tile, just large enough that every entity in
    crop_columns stays in view for the whole episode, in every row
    '''
    pts = np.concatenate([r[c]['positions'].reshape(-1, 2) for r in rows for c in crop_columns])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2
    half = min(np.max(hi - lo) / 2 + CROP_MARGIN, CAM_RANGE)

    # world -> pixel, matching ScenarioEnv.draw (y axis flipped)
    h, w = frame_shape[:2]
    size = int(round(2 * half / CAM_RANGE * (w / 2)))
    cx = center[0] / CAM_RANGE * (w / 2) + w / 2
    cy = -center[1] / CAM_RANGE * (h / 2) + h / 2
    # shift (never shrink) the window so it stays inside the rendered frame
    x0 = int(np.clip(round(cx - size / 2), 0, w - size))
    y0 = int(np.clip(round(cy - size / 2), 0, h - size))
    print(f'zoom {CAM_RANGE / half:.2f}x: window center ({center[0]:.2f}, {center[1]:.2f}), half-width {half:.2f}')
    return x0, y0, size


def outlined_tile(frame, color):
    frame = np.ascontiguousarray(frame[..., :3])
    return cv2.copyMakeBorder(frame, OUTLINE, OUTLINE, OUTLINE, OUTLINE, cv2.BORDER_CONSTANT, value=color)


def build_grid(rows, scale, columns=COLUMNS, crop_columns=CROP_COLUMNS):
    '''crop every frame to the shared zoom window, then lay out the grid'''
    frame_shape = rows[0][columns[0]]['frames'][0].shape
    x0, y0, size = crop_window(rows, frame_shape, crop_columns)
    side = int(round(frame_shape[1] * scale))

    def tile_fn(frame):
        return cv2.resize(frame[y0:y0 + size, x0:x0 + size], (side, side), interpolation=cv2.INTER_AREA)

    return compose_grid(rows, side, tile_fn, columns=columns)


def compose_grid(rows, side, tile_fn=None, extra_band=None, columns=COLUMNS):
    '''
    lay out the legend, seed labels and outlined tiles for every frame of the grid

    rows: list of {column name: episode result with 'frames'}. Shorter episodes hold their
    last frame until the longest episode in the grid ends.
    side: tile size in pixels (before the outline)
    tile_fn: maps a stored frame to a side x side tile; frames are used as-is when None
    extra_band: optional RGB image placed under the legend, centered over the tiles (e.g. a
        second legend row explaining what is drawn inside each tile)
    columns: grid columns, left to right (each needs an entry in COLUMN_COLORS)
    '''
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    tile_h, tile_w = side + 2 * OUTLINE, side + 2 * OUTLINE

    band_h = 0 if extra_band is None else extra_band.shape[0]
    top = LEGEND_HEIGHT + band_h
    width = SEED_LABEL_WIDTH + len(columns) * (tile_w + GAP)
    height = top + len(rows) * (tile_h + GAP)

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    # legend is centered over the tiles, not the seed label strip
    canvas[:LEGEND_HEIGHT, SEED_LABEL_WIDTH:] = draw_legend(width - SEED_LABEL_WIDTH, font, columns)
    if extra_band is not None:
        x0 = SEED_LABEL_WIDTH + (width - SEED_LABEL_WIDTH - extra_band.shape[1]) // 2
        canvas[LEGEND_HEIGHT:top, x0:x0 + extra_band.shape[1]] = extra_band
    for i in range(len(rows)):
        y = top + i * (tile_h + GAP)
        canvas[y:y + tile_h, :SEED_LABEL_WIDTH] = draw_seed_label(tile_h, i + 1, font)

    n_steps = max(len(r[c]['frames']) for r in rows for c in columns)
    grid = []
    for t in range(n_steps):
        frame = canvas.copy()
        for i, r in enumerate(rows):
            y = top + i * (tile_h + GAP)
            for j, c in enumerate(columns):
                frames = r[c]['frames']
                tile = frames[min(t, len(frames) - 1)]
                if tile_fn is not None:
                    tile = tile_fn(tile)
                x = SEED_LABEL_WIDTH + j * (tile_w + GAP)
                frame[y:y + tile_h, x:x + tile_w] = outlined_tile(tile, COLUMN_COLORS[c])
        grid.append(frame)
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='one grid row per seed; if omitted, seeds are chosen by --search_seeds')
    parser.add_argument('--search_seeds', type=int, default=20,
                        help='number of seeds to try when picking rows by lowest allocentric belief error')
    parser.add_argument('--rows', type=int, default=1, help='number of rows when searching for seeds')
    parser.add_argument('--save_path', type=str, default='predator_prey_grid.gif')
    parser.add_argument('--scale', type=float, default=0.5, help='resize factor applied to each tile')
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
            print(f'seed {seed} | {c}: {len(row[c]["frames"]) - 1} steps, success={row[c]["success"]}, '
                  f'oob={row[c]["oob"]}' + (f', belief error={err:.4f}' if err is not None else ''))
        ref = row[COLUMNS[0]]['start_state']
        for c in COLUMNS[1:]:
            assert np.allclose(row[c]['start_state'], ref), f'{c} did not start from the same state as {COLUMNS[0]}'
        rows.append(row)

    grid = build_grid(rows, args.scale)
    print(f'saving {len(grid)} frames ({grid[0].shape[1]}x{grid[0].shape[0]}) to {args.save_path}')
    save_rgb_gif(grid, args.save_path, fps=args.fps)


if __name__ == '__main__':
    main()
