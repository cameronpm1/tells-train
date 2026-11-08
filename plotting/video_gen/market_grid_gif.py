import os
import sys
import argparse
from copy import deepcopy

import cv2
import numpy as np
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from envs.marl.make_env import make_marl_env
from learn.marl.train import make_ray_config, marl_policy_mapping_fn
from controllers.market_game_control import compute_market_actions
from plotting.video_gen.predator_prey_grid_gif import COLUMNS, FONT_PATH, GAP, OUTLINE, compose_grid, text_size
from plotting.video_gen.drones_grid_gif import reset_pf_bookkeeping
from util.util import load_config, MODELS_ROOT


CHECKPOINT_DIR = os.path.join(MODELS_ROOT, 'marl', 'market_game_fully_obs/checkpoint829')
BELIEF_CONFIG = 'confs/belief/market_game.yaml'
# snapshot of the label-2 belief model taken before results_allocentric was evaluated
BELIEF_DIR = os.path.join(MODELS_ROOT, 'belief', '_market_game_fully_obs_NN_1/epoch989')
# fully observable and allocentric were both evaluated with 7a_game.yaml (the belief wrapper
# turns off the team noise); the centralized controller uses it too
FULLY_OBS_CONFIG = 'confs/market_game/7a_game.yaml'
ALLOCENTRIC_CONFIG = 'confs/market_game/7a_game.yaml'
EGOCENTRIC_CONFIG = 'confs/market_game/7a_game_pf.yaml'

FONT_REGULAR_PATH = '/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf'
TILE_DPI = 200        # tiles are drawn at 2x and downscaled
LEGEND2_FONT_SIZE = 18
LEGEND2_HEIGHT = 40

# line styles shared by the tiles and the second legend row: (label, RGB, style)
DEMAND = ('Demand', (34, 34, 34), 'solid')
SOLD = ('Team sold', (47, 158, 68), 'solid')
RELEASED = ('Material released', (214, 69, 65), 'solid')
RHO_D = (r'Failure line (ρ·demand)', (34, 34, 34), 'dotted')
REQUIRED = ('Required sales (ρ)', (214, 69, 65), 'dotted')
FAILURE = ('Episode failed', (214, 69, 65), 'cross')
LEGEND2_ITEMS = [DEMAND, SOLD, RELEASED, RHO_D, REQUIRED, FAILURE]


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


def run_episode(name, env, algo, seed, controller=compute_market_actions):
    '''
    roll out one episode from reset(seed=seed)

    the market history is recorded once and each frame is (history, t): the tile for frame t
    draws the first t steps. Returns dict with frames, start_state, outcome and the mean belief
    error (None without a belief model)
    '''
    np.random.seed(seed)
    obs, infos = env.reset(seed=seed)
    reset_pf_bookkeeping(env)
    base = env.env  # MarketGameEnv
    start_state = np.concatenate([base.rates, [base.demand], base.release])

    errors = []
    done = False
    while not done:
        if name == 'Centralized':
            actions = controller(deepcopy(base.obs), base.obs_map, base.controller_cfg)
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

    metrics = base.last_metrics
    if metrics['success']:
        outcome = 'success'
    elif metrics['dead']:
        outcome = 'stock ran out'
    else:
        outcome = 'short of demand'

    history = {
        'demand': np.array(base.history['demand']),
        'sold': np.array(base.history['sold']),
        'release': np.array(base.history['release']),
        'stocks': np.array(base.history['stocks']).reshape(-1, base.n_agents),
    }
    return {
        'frames': [(history, t) for t in range(len(history['demand']) + 1)],
        'start_state': start_state,
        'outcome': outcome,
        'belief_error': float(np.mean(errors)) if errors else None,
    }


def search_seeds(env, algo, n_search, n_rows):
    '''
    run the allocentric column on seeds 0..n_search-1 and keep the n_rows seeds that ran the full
    episode without failing, with the lowest mean belief error
    '''
    results = {}
    for seed in range(n_search):
        results[seed] = run_episode('Allocentric', env, algo, seed)
        r = results[seed]
        print(f'search seed {seed}: belief error {r["belief_error"]:.4f}, {r["outcome"]}, '
              f'{len(r["frames"]) - 1} steps')
    candidates = [s for s in results if results[s]['outcome'] == 'success'] or list(results)
    seeds = sorted(candidates, key=lambda s: results[s]['belief_error'])[:n_rows]
    return seeds, {s: results[s] for s in seeds}


def mpl_style(style):
    return {'solid': '-', 'dotted': ':', 'dashed': '--'}[style]


def rgb01(color):
    return tuple(v / 255 for v in color)


class TileRenderer:
    '''
    draws the tile for (history, t): team sales as a percentage of demand on top, against the
    required rho (an episode ends once sales drop below it), the full market panel below. Caches the last
    tile drawn for each history so frames held after an episode ends are not redrawn
    '''

    def __init__(self, base, side, rows, columns=COLUMNS):
        self.side = side
        self.max_t = base.max_episode_length
        self.rho = base.market_cfg['rho']
        self.cache = {}

        # y ranges shared by every tile, fit to the data in the grid; the top panel is zoomed
        # to just the sales and failure lines
        histories = [r[c]['frames'][0][0] for r in rows for c in columns]
        self.market_max = 1.05 * max(max(h[k].max(initial=0.0) for k in ('demand', 'sold', 'release')) for h in histories)
        # percentiles, so a few spikes don't flatten the gap to the required line
        pct = np.concatenate([100 * h['sold'] / h['demand'] for h in histories])
        lo, hi = np.percentile(pct, [1, 99])
        lo, hi = min(lo, 100 * self.rho), max(hi, 100 * self.rho)
        pad = 0.08 * (hi - lo)
        self.sales_range = (lo - pad, hi + pad)

        font_manager.fontManager.addfont(FONT_REGULAR_PATH)
        font_manager.fontManager.addfont(FONT_PATH)
        self.family = font_manager.FontProperties(fname=FONT_REGULAR_PATH).get_name()

    def __call__(self, frame):
        history, t = frame
        key = (id(history), t)
        if key not in self.cache:
            self.cache = {k: v for k, v in self.cache.items() if k[0] != id(history)}
            self.cache[key] = self.draw(history, t)
        return self.cache[key]

    def draw(self, history, t):
        size_in = self.side * 2 / TILE_DPI
        fig = Figure(figsize=(size_in, size_in), dpi=TILE_DPI)
        canvas = FigureCanvasAgg(fig)
        ax_sales, ax_market = fig.subplots(2, 1, sharex=True)
        steps = np.arange(1, t + 1)
        demand = history['demand'][:t]

        pct = 100 * history['sold'][:t] / demand
        ax_sales.plot(steps, pct, color=rgb01(SOLD[1]), lw=1.2)
        ax_sales.axhline(100 * self.rho, color=rgb01(REQUIRED[1]), lw=1.2, ls=mpl_style(REQUIRED[2]))
        if t > 0 and pct[-1] < 100 * self.rho:
            ax_sales.plot(t, max(pct[-1], self.sales_range[0]), marker='x', color=rgb01(FAILURE[1]),
                          markersize=7, mew=2.0, clip_on=False, zorder=5)
        ax_sales.set_ylim(*self.sales_range)
        ax_sales.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax_sales.set_ylabel('Sales / demand', fontweight='bold')

        ax_market.plot(steps, demand, color=rgb01(DEMAND[1]), lw=1.2)
        ax_market.plot(steps, history['sold'][:t], color=rgb01(SOLD[1]), lw=1.2)
        ax_market.plot(steps, history['release'][:t], color=rgb01(RELEASED[1]), lw=1.0, alpha=0.75)
        ax_market.plot(steps, self.rho * demand, color=rgb01(RHO_D[1]), lw=1.0, ls=mpl_style(RHO_D[2]))
        ax_market.set_ylim(0, self.market_max)
        ax_market.set_ylabel('Market', fontweight='bold')
        ax_market.set_xlim(0, self.max_t)
        ax_market.set_xlabel('Step', fontweight='bold')

        for ax in (ax_sales, ax_market):
            ax.tick_params(labelsize=6, length=2)
            ax.spines[['top', 'right']].set_visible(False)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_family(self.family)
            ax.yaxis.label.set_size(7)
            ax.xaxis.label.set_size(7)

        fig.tight_layout(pad=0.4, h_pad=0.6)
        canvas.draw()
        img = np.asarray(canvas.buffer_rgba())[..., :3]
        return cv2.resize(img, (self.side, self.side), interpolation=cv2.INTER_AREA)


def draw_line_sample(draw, x, cy, length, color, style, width=3):
    if style == 'solid':
        draw.line([(x, cy), (x + length, cy)], fill=color, width=width)
        return
    dash, gap = (3, 4) if style == 'dotted' else (8, 5)
    pos = x
    while pos < x + length:
        draw.line([(pos, cy), (min(pos + dash, x + length), cy)], fill=color, width=width)
        pos += dash + gap


def draw_line_legend(width):
    '''second legend row: what each line inside a tile means, shrinking the font to fit width'''
    img = Image.new('RGB', (width, LEGEND2_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    sample, pad, item_gap = 30, 7, 24
    cy = LEGEND2_HEIGHT // 2

    for font_size in range(LEGEND2_FONT_SIZE, 9, -1):
        font = ImageFont.truetype(FONT_PATH, font_size)
        widths = [sample + pad + text_size(draw, label, font)[0] for label, _, _ in LEGEND2_ITEMS]
        total = sum(widths) + item_gap * (len(widths) - 1)
        if total <= width - 2 * pad:
            break
    _, cap_h, cap_top = text_size(draw, 'H', font)
    x = (width - total) // 2
    for (label, color, style), w in zip(LEGEND2_ITEMS, widths):
        if style == 'cross':
            mx, r = x + sample // 2, 7
            draw.line([(mx - r, cy - r), (mx + r, cy + r)], fill=color, width=3)
            draw.line([(mx - r, cy + r), (mx + r, cy - r)], fill=color, width=3)
        else:
            draw_line_sample(draw, x, cy, sample, color, style)
        draw.text((x + sample + pad, cy - cap_h // 2 - cap_top), label, fill=(0, 0, 0), font=font)
        x += w + item_gap
    return np.asarray(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='one grid row per seed; if omitted, seeds are chosen by --search_seeds')
    parser.add_argument('--search_seeds', type=int, default=60,
                        help='number of seeds to try when picking rows by lowest allocentric belief error')
    parser.add_argument('--rows', type=int, default=4, help='number of rows when searching for seeds')
    parser.add_argument('--save_path', type=str, default='market_grid.gif')
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

    base = envs['Centralized'].env
    renderer = TileRenderer(base, args.tile_size, rows)
    band = draw_line_legend(len(COLUMNS) * (args.tile_size + 2 * OUTLINE + GAP))

    grid = compose_grid(rows, args.tile_size, renderer, extra_band=band)
    print(f'saving {len(grid)} frames ({grid[0].shape[1]}x{grid[0].shape[0]}) to {args.save_path}')
    save_gif_shared_palette(grid, args.save_path, fps=args.fps)


def save_gif_shared_palette(frames, path, fps=10):
    '''
    save with one palette for every frame and no dithering: the antialiased plot lines would
    otherwise get per-frame dithering noise that compresses badly
    '''
    palette = Image.fromarray(frames[-1]).quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    images = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps),
                   loop=0, optimize=False)


if __name__ == '__main__':
    main()
