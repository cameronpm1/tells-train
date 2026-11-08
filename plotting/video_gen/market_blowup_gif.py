import os
import sys
import argparse

import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import plotting.video_gen.market_grid_gif as grid
from plotting.video_gen.predator_prey_grid_gif import COLUMNS
from learn.marl.train import make_ray_config


# a grid tile is 3.5 x 3.5 in with the two panels stacked; side by side each panel keeps the tile's 3.5 in width
PANEL_IN = (3.5, 2.4)
MARGIN = 16  # white border around the plots, in pixels


class SideBySideRenderer(grid.TileRenderer):
    '''
    market_grid_gif's tile with the two panels side by side: the market panel on the left and sales as a percentage
    of demand on the right, with the same lines, colors, y ranges and failure marker. Drawn at the grid's physical
    panel size and a higher dpi, so fonts and line widths scale with the output width
    '''

    def __init__(self, base, width, rows, columns):
        super().__init__(base, width, rows, columns=columns)
        self.width = width
        self.dpi = 2 * width / (2 * PANEL_IN[0])
        # drawn at 2x and downscaled by half in both directions
        self.height = int(round(PANEL_IN[1] * self.dpi / 2))

    def draw(self, history, t):
        fig = Figure(figsize=(2 * PANEL_IN[0], PANEL_IN[1]), dpi=self.dpi)
        canvas = FigureCanvasAgg(fig)
        ax_market, ax_sales = fig.subplots(1, 2)
        steps = np.arange(1, t + 1)
        demand = history['demand'][:t]

        ax_market.plot(steps, demand, color=grid.rgb01(grid.DEMAND[1]), lw=1.2)
        ax_market.plot(steps, history['sold'][:t], color=grid.rgb01(grid.SOLD[1]), lw=1.2)
        ax_market.plot(steps, history['release'][:t], color=grid.rgb01(grid.RELEASED[1]), lw=1.0, alpha=0.75)
        ax_market.plot(steps, self.rho * demand, color=grid.rgb01(grid.RHO_D[1]), lw=1.0, ls=grid.mpl_style(grid.RHO_D[2]))
        ax_market.set_ylim(0, self.market_max)
        ax_market.set_ylabel('Market', fontweight='bold')

        pct = 100 * history['sold'][:t] / demand
        ax_sales.plot(steps, pct, color=grid.rgb01(grid.SOLD[1]), lw=1.2)
        ax_sales.axhline(100 * self.rho, color=grid.rgb01(grid.REQUIRED[1]), lw=1.2, ls=grid.mpl_style(grid.REQUIRED[2]))
        if t > 0 and pct[-1] < 100 * self.rho:
            ax_sales.plot(t, max(pct[-1], self.sales_range[0]), marker='x', color=grid.rgb01(grid.FAILURE[1]),
                          markersize=7, mew=2.0, clip_on=False, zorder=5)
        ax_sales.set_ylim(*self.sales_range)
        ax_sales.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax_sales.set_ylabel('Sales / demand', fontweight='bold')

        for ax in (ax_market, ax_sales):
            ax.set_xlim(0, self.max_t)
            ax.set_xlabel('Step', fontweight='bold')
            ax.tick_params(labelsize=6, length=2)
            ax.spines[['top', 'right']].set_visible(False)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_family(self.family)
            ax.yaxis.label.set_size(7)
            ax.xaxis.label.set_size(7)

        fig.tight_layout(pad=0.4, w_pad=1.6)
        canvas.draw()
        img = np.asarray(canvas.buffer_rgba())[..., :3]
        return cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)


def main():
    '''
    one run of the 7-company market grid (market_grid_gif.py), enlarged and laid out side by side: the grid's
    line legend on top, then the market panel (left) and sales / demand panel (right)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--column', type=str, default='Allocentric', choices=COLUMNS)
    parser.add_argument('--seed', type=int, default=14, help='env seed; 14 is row 1 of market_grid.gif')
    parser.add_argument('--save_path', type=str, default='market_allocentric_blowup.gif')
    parser.add_argument('--width', type=int, default=1600, help='width of the two plots together, in pixels')
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()

    import ray
    ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False)

    envs, fo_cfg = grid.build_envs()
    algo = make_ray_config({**fo_cfg, 'alg': {**fo_cfg['alg'], 'nenvs': 0}}).build_algo()
    algo.restore(os.path.abspath(grid.CHECKPOINT_DIR))

    result = grid.run_episode(args.column, envs[args.column], algo, args.seed)
    err = result['belief_error']
    print(f'seed {args.seed} | {args.column}: {len(result["frames"]) - 1} steps, {result["outcome"]}'
          + (f', belief error={err:.4f}' if err is not None else ''))

    renderer = SideBySideRenderer(envs[args.column].env, args.width, [{args.column: result}], columns=[args.column])
    grid.LEGEND2_FONT_SIZE = 24
    grid.LEGEND2_HEIGHT = 52
    band = grid.draw_line_legend(args.width)

    frames = []
    for frame in result['frames']:
        tile = renderer(frame)
        canvas = np.full((MARGIN + band.shape[0] + tile.shape[0] + MARGIN, args.width + 2 * MARGIN, 3), 255, dtype=np.uint8)
        canvas[MARGIN:MARGIN + band.shape[0], MARGIN:MARGIN + args.width] = band
        canvas[MARGIN + band.shape[0]:MARGIN + band.shape[0] + tile.shape[0], MARGIN:MARGIN + args.width] = tile
        frames.append(canvas)
    print(f'saving {len(frames)} frames ({frames[0].shape[1]}x{frames[0].shape[0]}) to {args.save_path}')
    grid.save_gif_shared_palette(frames, args.save_path, fps=args.fps)


if __name__ == '__main__':
    main()
