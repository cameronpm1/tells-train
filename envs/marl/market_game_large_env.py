import numpy as np
from copy import deepcopy
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from controllers.market_game_control import N_SELL
from controllers.market_game_large_control import compute_market_actions, action_to_fractions
from envs.marl.market_game_env import MarketGameEnv


class MarketGameLargeEnv(MarketGameEnv):

    '''
    market game for a large team, teammates observed through quantiles of their stockpiles and orders are multiples of the fair share
    '''

    def __init__(self, *args, quantiles: tuple = (0.0, 0.1, 0.5, 0.9, 1.0), **kwargs):
        super().__init__(*args, **kwargs)

        self.quantiles = np.asarray(quantiles, dtype=float)
        n_q = len(self.quantiles)
        self.obs_map.update({
            'team': slice(8, 8 + n_q),
            'team_stock': slice(8, 8 + n_q),
            'state_space': (8 + n_q,),
        })

    def action_to_fractions(self, action: int):
        return action_to_fractions(action, self.n_agents)

    def _get_obs(self):
        obs = super()._get_obs()

        for i, agent in enumerate(self.agents):
            others = np.delete(self.stocks, i)
            obs[agent] = np.concatenate((
                obs[agent][:8],
                np.quantile(others, self.quantiles) / self.scale,
            )).astype(np.float32)

        return obs

    def compute_team_metrics(self, market_metrics: dict):
        metrics = super().compute_team_metrics(market_metrics)
        metrics['order_sum'] = float(self.orders.sum())
        metrics['oversubscription'] = max(0.0, metrics['order_sum'] - 1.0)
        return metrics

    def compute_team_reward(self, metrics: dict):
        return super().compute_team_reward(metrics) - self.reward_cfg['oversubscription_scale'] * metrics['oversubscription']

    def behavior_cloning_reward(self, action_dict: dict) -> dict:
        controller_actions = compute_market_actions(deepcopy(self.obs), self.obs_map, self.controller_cfg)
        action_matches = []
        for agent in self.agents:
            if agent not in action_dict:
                continue
            learned_order, learned_sell = divmod(int(action_dict[agent]), N_SELL)
            controller_order, controller_sell = divmod(controller_actions[agent], N_SELL)
            action_matches.append(0.5 * float(learned_order == controller_order) + 0.5 * float(learned_sell == controller_sell))
        controller_action_match = float(np.mean(action_matches)) if action_matches else 0.0

        return {
            'controller_action_error': 1.0 - controller_action_match,
            'controller_action_match': controller_action_match,
            'controller_action_reward': controller_action_match,
        }

    def team_error(
        self,
        estimate: np.ndarray,
        agent: str
    ):
        return float(np.linalg.norm(estimate[self.obs_map['team']] - self.obs[agent][self.obs_map['team']]))

    def render_rgb(self):
        cfg = self.market_cfg
        fig = Figure(figsize=(9, 3.6), dpi=80)
        canvas = FigureCanvasAgg(fig)
        ax_market, ax_stock = fig.subplots(1, 2)

        steps = np.arange(1, len(self.history['demand']) + 1)
        demand = np.array(self.history['demand'])
        ax_market.plot(steps, demand, color='k', label='demand')
        ax_market.plot(steps, self.history['sold'], color=(0.35, 0.35, 0.85), label='team sold')
        ax_market.plot(steps, self.history['release'], color=(0.85, 0.35, 0.35), alpha=0.6, label='material released')
        ax_market.plot(steps, cfg['rho'] * demand, color='k', ls=':', label=r'$\rho$ D')
        ax_market.set_xlim(0, self.max_episode_length)
        ax_market.set_ylim(0, 1.3 * cfg['mean_demand'])
        ax_market.set_title(f'market, t = {self.ts}')
        ax_market.legend(loc='lower right', fontsize=7)

        stocks = np.array(self.history['stocks']).reshape(-1, self.n_agents)
        q = np.quantile(stocks, [0.0, 0.1, 0.5, 0.9, 1.0], axis=1)
        blue = (0.35, 0.35, 0.85)
        ax_stock.fill_between(steps, q[0], q[4], color=blue, alpha=0.15, lw=0, label='min - max')
        ax_stock.fill_between(steps, q[1], q[3], color=blue, alpha=0.3, lw=0, label='10% - 90%')
        ax_stock.plot(steps, q[2], color=blue, label='median')
        ax_stock.axhline(self.supplier_cfg['stock_cut'] / self.n_agents, color=(0.85, 0.35, 0.35), ls='--', label='supplier target')
        ax_stock.axhline(cfg['min_stock'], color='k', ls=':', label='min stock')
        ax_stock.set_xlim(0, self.max_episode_length)
        ax_stock.set_ylim(0, cfg['stock_cap'] * 1.05)
        ax_stock.set_title(f'stockpiles ({self.n_agents} companies)')
        ax_stock.legend(loc='upper right', fontsize=7)

        fig.tight_layout()
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[..., :3].copy()
