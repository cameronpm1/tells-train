import numpy as np

ORDER_STEP, N_ORDER = 0.05, 13
SELL_STEP, N_SELL = 0.02, 51
N_ACTIONS = N_ORDER * N_SELL


def action_to_fractions(action: int) -> tuple[float, float]:
    order_idx, sell_idx = divmod(int(action), N_SELL)
    return order_idx * ORDER_STEP, sell_idx * SELL_STEP

def fractions_to_action(order: float, sell: float) -> int:
    order_idx = int(np.clip(np.rint(order / ORDER_STEP), 0, N_ORDER - 1))
    sell_idx = int(np.clip(np.rint(sell / SELL_STEP), 0, N_SELL - 1))
    return order_idx * N_SELL + sell_idx

def team_sales_target(demand: float, team_stock: float, material_per_product: float, controller_cfg: dict) -> float:
    return float(np.clip(
        demand - controller_cfg['kappa'] * (controller_cfg['healthy_stock'] - team_stock) / material_per_product,
        (controller_cfg['rho'] + controller_cfg['margin']) * demand,
        demand,
    ))

def market_fractions(demand, release, rates, stocks, controller_cfg):
    '''
    order and sell fractions for every company from the full state, raw (unscaled) units
    '''
    n = len(rates)
    target = team_sales_target(demand, stocks.sum(), np.mean(1.0 / rates), controller_cfg)
    product = np.full(n, target / n)

    orders = (1.0 / rates) / np.sum(1.0 / rates)
    orders = orders + controller_cfg.get('balance_gain', 0.0) * (stocks.mean() - stocks) / max(release, 1e-8)
    stocks_after = np.minimum(stocks + orders * release, controller_cfg['stock_cap'])
    sells = product / np.maximum(rates * stocks_after, 1e-8)
    return orders, sells

def compute_market_actions(
    obs,
    obs_map,
    controller_cfg: dict,
    team_rates: np.ndarray = None,
):
    '''
    full marl obs dict gives every company's rate and stock (centralized controller), a single obs uses its team slice
    for stocks and team_rates for teammates' rates, assuming they match its own rate if none are given
    '''
    scale = controller_cfg['obs_scale']

    if isinstance(obs, dict):
        agent_names = [name for name in obs.keys() if 'agent' in name]
        own = obs[agent_names[0]]
        rates = np.array([obs[name][obs_map['self_rate']][0] for name in agent_names])
        stocks = np.array([obs[name][obs_map['self_stock']][0] for name in agent_names]) * scale
    else:
        own = obs
        if team_rates is None:
            team_rates = np.full(len(obs[obs_map['team_stock']]), obs[obs_map['self_rate']][0])
        rates = np.concatenate((obs[obs_map['self_rate']], team_rates))
        stocks = np.concatenate((obs[obs_map['self_stock']], obs[obs_map['team_stock']])) * scale

    demand = own[obs_map['demand']][0] * scale
    release = own[obs_map['release']][0] * scale
    orders, sells = market_fractions(demand, release, np.asarray(rates, dtype=float), stocks, controller_cfg)
    actions = [fractions_to_action(order, sell) for order, sell in zip(orders, sells)]

    if isinstance(obs, dict):
        return {name: action for name, action in zip(agent_names, actions)}
    return actions[0]

def supplier_controller(
    obs: np.ndarray,
    obs_map: dict,
    controller_cfg: dict,
) -> np.ndarray:
    '''
    assumes recieves obs['target'], returns the noiseless release of every distributor in the cartel
    '''
    material_per_product = obs[obs_map['supplier_mix']]
    n_distributors = len(material_per_product)
    base = material_per_product * obs[obs_map['supplier_demand']][0]
    excess = max(0.0, obs[obs_map['supplier_team_stock']][0] - controller_cfg['stock_cut'])
    cut = min(controller_cfg['eta'] * excess, controller_cfg['max_cut_frac'] * base.mean())

    release = (
        (base - cut) / n_distributors
        + controller_cfg['own_stock_gain'] * (obs[obs_map['supplier_own_stock']] - controller_cfg['own_stock_init'] / n_distributors)
    )

    return np.maximum(release, 0.0)
