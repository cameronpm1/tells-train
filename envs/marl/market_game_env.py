import gymnasium
import numpy as np
from copy import deepcopy
from typing import Optional
from gymnasium import spaces
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from controllers.market_game_control import compute_market_actions, action_to_fractions, N_ACTIONS, N_SELL


class MarketGameEnv(gymnasium.Env):

    '''
    companies buy cornered material from a scripted supplier cartel and sell product, teammates' rates and stockpiles are only revealed through the supplier
    '''

    def __init__(
        self,
        agents: list[str],
        supplier_controller,
        market_kwargs: dict,
        supplier_kwargs: dict,
        reward_kwargs: Optional[dict] = None,
        controller_kwargs: Optional[dict] = None,
        max_episode_length: int = 200,
        seed: Optional[int] = None,
    ):

        self.agents = agents
        self.n_agents = len(agents)
        self.supplier_controller = supplier_controller
        self.market_cfg = deepcopy(market_kwargs)
        self.supplier_cfg = deepcopy(supplier_kwargs)
        self.reward_cfg = reward_kwargs
        self.max_episode_length = max_episode_length
        self.rng = np.random.default_rng(seed)

        self.scale = self.market_cfg['obs_scale']
        self.controller_cfg = {
            **deepcopy(controller_kwargs or {}),
            'rho': self.market_cfg['rho'],
            'stock_cap': self.market_cfg['stock_cap'],
            'obs_scale': self.scale,
        }

        self.n_distributors = int(self.supplier_cfg['n_distributors'])
        self.cartel = self.supplier_cfg['cartel']
        self.supplier_cfg['stock_cut'] = self.supplier_cfg['stock_cut_frac'] * self.n_agents * self.market_cfg['stock_cap']
        if self.cartel != 'stocked':
            self.supplier_cfg['own_stock_gain'] = 0.0
        if self.cartel == 'regional':
            self.presence = self._regional_presence(np.random.default_rng(self.supplier_cfg['region_seed']))
        else:
            self.presence = np.full((self.n_distributors, self.n_agents), 1.0 / self.n_agents)

        n_team = self.n_agents - 1
        self.obs_map = {
            'self': slice(0, 4),
            'self_pos': slice(0, 1),
            'self_stock': slice(0, 1),
            'self_rate': slice(1, 2),
            'self_sales': slice(2, 3),
            'self_received': slice(3, 4),
            'target': slice(4, 8),
            'target_pos': slice(5, 6),
            'demand': slice(4, 5),
            'release': slice(5, 6),
            'release_spread': slice(6, 7),
            'release_weighted': slice(7, 8),
            'team': slice(8, 8 + n_team),
            'team_stock': slice(8, 8 + n_team),
            'supplier_team_stock': slice(0, 1),
            'supplier_mix': slice(1, 1 + self.n_distributors),
            'supplier_demand': slice(1 + self.n_distributors, 2 + self.n_distributors),
            'supplier_own_stock': slice(2 + self.n_distributors, 2 + 2 * self.n_distributors),
            'state_space': (8 + n_team,),
        }

        self.obs = None
        self.ts = 0
        self.last_metrics = {}

    def _observation_space(self, agent):
        return spaces.Box(-np.inf, np.inf, self.obs_map['state_space'], np.float32)

    def _action_space(self, agent):
        return spaces.Discrete(N_ACTIONS)

    def action_to_fractions(self, action: int):
        return action_to_fractions(action)

    def _regional_presence(self, rng, iters: int = 100):
        presence = rng.dirichlet(np.ones(self.n_agents), self.n_distributors)
        for _ in range(iters):
            presence *= (self.n_distributors / self.n_agents) / presence.sum(axis=0)
            presence /= presence.sum(axis=1, keepdims=True)
        return presence

    def step(
        self,
        action_dict: dict
    ):
        self.ts += 1

        controller_metrics = self.behavior_cloning_reward(action_dict)
        fractions = np.array([self.action_to_fractions(action_dict[agent]) for agent in self.agents])
        self.orders, self.sells = fractions[:, 0], fractions[:, 1]

        metrics = self.compute_team_metrics(self.clear_market())
        metrics.update(controller_metrics)
        team_reward = self.compute_team_reward(metrics)
        rewards = {agent: team_reward for agent in self.agents}

        self.demand = self._next_demand()
        self._supplier_step()
        self.obs = self._get_obs()

        failed = metrics['dead'] or metrics['short']
        truncated = not failed and self.ts >= self.max_episode_length
        terminations = {agent: failed for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        terminations["__all__"] = failed
        truncations["__all__"] = truncated

        info = {}
        info['__common__'] = {
            agent: {
                'team_reward': team_reward,
                **metrics,
            }
            for agent in self.agents
        }

        return deepcopy(self.obs), rewards, terminations, truncations, info

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        cfg = self.market_cfg
        self.ts = 0
        self.last_metrics = {}
        self.history = {key: [] for key in ('demand', 'sold', 'release', 'team_stock', 'cut_active', 'stocks')}

        self.rates = self.rng.uniform(*cfg['rate_range'], self.n_agents)
        self.stocks = np.full(self.n_agents, cfg['stock_cap'] / 2.0)
        self.sales = np.zeros(self.n_agents)
        self.received = np.zeros(self.n_agents)
        self.distributor_stocks = np.full(self.n_distributors, self.supplier_cfg['own_stock_init'] / self.n_distributors)
        self.demand = max(0.0, cfg['mean_demand'] + self.rng.normal(0, cfg['demand_noise'] / np.sqrt(1 - cfg['demand_phi'] ** 2)))

        self._supplier_step()
        self.obs = self._get_obs()

        return deepcopy(self.obs), {'__common__':{a: {} for a in self.agents}}

    def get_obs(self):
        return self.obs

    def _next_demand(self):
        cfg = self.market_cfg
        return max(0.0, cfg['mean_demand'] + cfg['demand_phi'] * (self.demand - cfg['mean_demand']) + self.rng.normal(0, cfg['demand_noise']))

    def _get_supplier_obs(self):
        return np.concatenate((
            [self.stocks.sum()],
            self.presence @ (1.0 / self.rates),
            [self.demand],
            self.distributor_stocks,
        ))

    def _supplier_step(self):
        self.supplier_obs = self._get_supplier_obs()
        release = self.supplier_controller(self.supplier_obs, self.obs_map, self.supplier_cfg)
        mix = self.supplier_obs[self.obs_map['supplier_mix']]
        noise = mix * self.demand * self.rng.normal(0, self.supplier_cfg['noise'], self.n_distributors) / self.n_distributors
        release = np.maximum(release + noise, 0.0)

        if self.cartel == 'stocked':
            release = np.minimum(release, self.distributor_stocks)
            self.distributor_stocks += mix * self.market_cfg['mean_demand'] / self.n_distributors - release

        self.release = release

    def _get_obs(self):
        obs = {'target': self.supplier_obs.copy()}
        release_total = self.release.sum()

        for i, agent in enumerate(self.agents):
            others = [j for j in range(self.n_agents) if j != i]
            if self.cartel == 'regional':
                weighted = self.presence[:, i] @ self.release / self.presence[:, i].mean()
            else:
                weighted = 0.0

            obs[agent] = np.concatenate((
                [self.stocks[i] / self.scale, self.rates[i], self.sales[i] / self.scale, self.received[i] / self.scale],
                np.array([self.demand, release_total, np.std(self.release), weighted]) / self.scale,
                self.stocks[others] / self.scale,
            )).astype(np.float32)

        return obs

    def _fill(self, requests: np.ndarray, available: float) -> np.ndarray:
        if not self.market_cfg['random_fill']:
            return requests * min(1.0, available / max(requests.sum(), 1e-8))
        filled = np.zeros_like(requests)
        for i in self.rng.permutation(len(requests)):
            filled[i] = min(requests[i], available)
            available -= filled[i]
        return filled

    def clear_market(self):
        cfg = self.market_cfg
        team_stock_start = self.stocks.sum()
        release_total = self.release.sum()

        self.received = self._fill(self.orders * release_total, release_total)
        cap_hit = bool(np.any(self.stocks + self.received > cfg['stock_cap']))
        self.stocks = np.minimum(self.stocks + self.received, cfg['stock_cap'])

        product = self.rates * self.sells * self.stocks
        self.stocks = (1.0 - self.sells) * self.stocks
        self.sales = self._fill(product, self.demand)
        sold = float(self.sales.sum())

        return {
            'demand': self.demand,
            'release': release_total,
            'team_stock_start': team_stock_start,
            'cut_active': bool(team_stock_start > self.supplier_cfg['stock_cut']),
            'team_sold': sold,
            'team_product': float(product.sum()),
            'wasted': max(0.0, float(product.sum()) - self.demand),
            'demand_error': abs(float(product.sum()) - self.demand),
            'team_stock': float(self.stocks.sum()),
            'min_stock': float(self.stocks.min()),
            'cap_hit': cap_hit,
            'share': sold / max(self.demand, 1e-8),
            'dead': bool(np.any(self.stocks < cfg['min_stock'])),
            'short': bool(sold < cfg['rho'] * self.demand),
        }

    def compute_team_metrics(self, market_metrics: dict):
        history = self.history
        history['demand'].append(market_metrics['demand'])
        history['sold'].append(market_metrics['team_sold'])
        history['release'].append(market_metrics['release'])
        history['team_stock'].append(market_metrics['team_stock'])
        history['cut_active'].append(market_metrics['cut_active'])
        history['stocks'].append(self.stocks.copy())

        failed = market_metrics['dead'] or market_metrics['short']
        metrics = {
            **market_metrics,
            'cut_fraction': float(np.mean(history['cut_active'])),
            'success': not failed and self.ts >= self.max_episode_length,
        }
        self.last_metrics = metrics
        return metrics

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

    def compute_team_reward(self, metrics: dict):
        cfg = self.reward_cfg

        team_reward = (
            - cfg['reward_scale'] * metrics['demand_error']
            + cfg['bc_scale'] * metrics['controller_action_reward']
        )

        if metrics['dead'] or metrics['short']:
            team_reward -= cfg['reward_scale'] * cfg['early_end_penalty']

        return team_reward

    def decompose_reward(
        self,
        joint_reward,
        obs
    ):
        sales = np.array([obs[agent][self.obs_map['self_sales']][0] for agent in self.agents])
        stocks = np.array([obs[agent][self.obs_map['self_stock']][0] for agent in self.agents]) * self.scale
        safe_stock = self.supplier_cfg['stock_cut'] / self.n_agents

        sales_share = sales / max(sales.sum(), 1e-8)
        stock_factor = np.clip(stocks / safe_stock, 0.0, 1.0)

        return {agent: sales_share[i] * stock_factor[i] * joint_reward for i, agent in enumerate(self.agents)}

    def team_error(
        self,
        estimate: np.ndarray,
        agent: str
    ):
        return float(np.linalg.norm(np.sort(estimate[self.obs_map['team']]) - np.sort(self.obs[agent][self.obs_map['team']])))

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
        for i, agent in enumerate(self.agents):
            ax_stock.plot(steps, stocks[:, i], label=f'{agent} (k={self.rates[i]:.2f})')
        ax_stock.axhline(self.supplier_cfg['stock_cut'] / self.n_agents, color=(0.85, 0.35, 0.35), ls='--', label='supplier target')
        ax_stock.axhline(cfg['min_stock'], color='k', ls=':', label='min stock')
        ax_stock.set_xlim(0, self.max_episode_length)
        ax_stock.set_ylim(0, cfg['stock_cap'] * 1.05)
        ax_stock.set_title('stockpiles')
        ax_stock.legend(loc='upper right', fontsize=7)

        fig.tight_layout()
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[..., :3].copy()
