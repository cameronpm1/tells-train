import numpy as np

from controllers.market_game_control import market_fractions, team_sales_target, supplier_controller, SELL_STEP, N_SELL

ORDER_SHARE_STEP, N_ORDER = 0.25, 13
N_ACTIONS = N_ORDER * N_SELL


def action_to_fractions(action: int, n_agents: int) -> tuple[float, float]:
    '''
    order is a multiple of the fair share 1 / n_agents of the release, so the action space is the same for any team size
    '''
    order_idx, sell_idx = divmod(int(action), N_SELL)
    return order_idx * ORDER_SHARE_STEP / n_agents, sell_idx * SELL_STEP

def fractions_to_action(order: float, sell: float, n_agents: int) -> int:
    order_idx = int(np.clip(np.rint(order * n_agents / ORDER_SHARE_STEP), 0, N_ORDER - 1))
    sell_idx = int(np.clip(np.rint(sell / SELL_STEP), 0, N_SELL - 1))
    return order_idx * N_SELL + sell_idx

def large_market_fractions(demand, release, rates, stocks, controller_cfg):
    '''
    equal split of the team target, with each company's sell fraction capped at max_sell and the excess passed to companies with room
    '''
    orders, _ = market_fractions(demand, release, rates, stocks, controller_cfg)
    n = len(rates)
    target = team_sales_target(demand, stocks.sum(), np.mean(1.0 / rates), controller_cfg)
    stocks_after = np.minimum(stocks + orders * release, controller_cfg['stock_cap'])
    cap = controller_cfg['max_sell'] * rates * stocks_after
    product = np.full(n, target / n)

    for _ in range(n):
        over = product > cap
        free = ~over & (product < cap)
        if not over.any() or not free.any():
            break
        excess = np.sum(product[over] - cap[over])
        product[over] = cap[over]
        product[free] += excess / free.sum()

    product = np.minimum(product, cap)
    return orders, product / np.maximum(rates * stocks_after, 1e-8)

def compute_market_actions(
    obs: dict,
    obs_map: dict,
    controller_cfg: dict,
):
    '''
    centralized controller, takes the full marl obs dict and reads every company's own rate and stock
    '''
    scale = controller_cfg['obs_scale']
    agent_names = [name for name in obs.keys() if 'agent' in name]
    own = obs[agent_names[0]]
    rates = np.array([obs[name][obs_map['self_rate']][0] for name in agent_names])
    stocks = np.array([obs[name][obs_map['self_stock']][0] for name in agent_names]) * scale

    demand = own[obs_map['demand']][0] * scale
    release = own[obs_map['release']][0] * scale
    orders, sells = large_market_fractions(demand, release, rates, stocks, controller_cfg)

    return {
        name: fractions_to_action(order, sell, len(agent_names))
        for name, order, sell in zip(agent_names, orders, sells)
    }
