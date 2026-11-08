import yaml
import numpy
from typing import Optional, Union

from envs.marl.pf_wrapper import PFWrapper
from evals.marl.observation_packaging import *
from envs.marl.drones_env import CaravanAviary
from envs.marl.rllib_wrapper import RLLibWrapper
from envs.marl.predator_prey_env import PredatorPreyEnv, parallel_env
from envs.marl.particle_filters import PredatorPreyParticleFilter, DronesParticleFilter, FootballParticleFilter, FireParticleFilter, MarketFilter

def make_marl_env(
        config: dict,
        seed: Optional[int] = None,
        wrap: Optional[str] = 'rllib',
        eval: bool = False,
        render_mode: Optional[Union[str,bool]] = None,
        belief_kwargs: Optional[dict] = None,
):
    '''
    initialize marl env

    input
    -----
    config_dir:dict
        config dictionary
    seed:int
        seed
    wrap:bool
        if True wraps env in rllib wrapper
    '''

    env_name = config['env']['scenario']

    if 'predator_prey_swarm' in env_name:
        return make_predator_prey_swarm_env(config,seed,wrap,eval,render_mode,belief_kwargs)

    elif 'predator_prey' in env_name:
        return make_predator_prey_env(config,seed,wrap,eval,render_mode,belief_kwargs)
    
    elif 'drones' in env_name:
        return make_drones_env(config,seed,wrap,eval,render_mode,belief_kwargs)

    elif 'footbal' in env_name:
        return make_football_env(config,seed,wrap,eval,render_mode,belief_kwargs)

    elif 'fire' in env_name:
        return make_fire_env(config,seed,wrap,eval,render_mode,belief_kwargs)

    elif 'market_game_large' in env_name:
        return make_market_game_large_env(config,seed,wrap,eval,render_mode,belief_kwargs)

    elif 'market' in env_name:
        return make_market_game_env(config,seed,wrap,eval,render_mode,belief_kwargs)

    else:
        raise NotImplementedError(f'Env {env_name} not implemented yet')

def make_predator_prey_env(
        config: dict,
        seed: Optional[int] = None,
        wrap: Optional[str] = 'rllib',
        eval: bool = False,
        render_mode: Optional[str] = None,
        belief_kwargs: Optional[dict] = None,
):
    '''
    initialize usv_gym_env

    input
    -----
    config_dir:dict
        config dictionary
    seed:int
        seed
    wrap:bool
        if True wraps env in rllib wrapper
    '''

    from controllers.predator_prey_control import adversary_controller, compute_slot_actions, action_to_vec

    scenario_kwargs = config['env']['scenario_kwargs']
    scenario_kwargs['agent_list'] = config['env']['agent_list']
    env = parallel_env(
        scenario_kwargs=scenario_kwargs,
        max_cycles=config['env']['max_episode_length'],
        render_mode = render_mode,
    )

    env = PredatorPreyEnv(
        mpeEnv=env,
        agents=config['env']['learned_agent_list'],
        adversary_controller=adversary_controller,
        reward_kwargs=config['env'].get('reward_kwargs'),
        controller_kwargs=config['env'].get('controller_kwargs'),
        seed = seed,
        local_observations = config['env']['local_observations'],
    )

    if wrap == 'rllib':
        env = RLLibWrapper(
            env=env,
            name='predator_prey',
            obs_packaging_func=predator_prey_obs_packaging,
            eval=eval,
            belief_kwargs=belief_kwargs,
            noise=config['env']['noise']
        )
    elif wrap == 'pf':
        env = PFWrapper(
            env,
            name='predator_prey',
            particle_filter = PredatorPreyParticleFilter,
            agent_control_function = lambda obs, obs_map: action_to_vec(compute_slot_actions(obs, obs_map)),
            target_control_function = lambda obs, obs_map: action_to_vec(adversary_controller(obs, obs_map, config['env']['controller_kwargs'])),
            dim=2,
            eval=eval,
            belief_kwargs=config['env']['belief_kwargs'],
            noise=config['env']['noise'],
        )
    elif wrap is None:
        pass
    else:
        print('Wrapper not supported for mpe2 env...')

    return env


def make_predator_prey_swarm_env(
        config: dict,
        seed: Optional[int] = None,
        wrap: Optional[str] = 'rllib',
        eval: bool = False,
        render_mode: Optional[str] = None,
        belief_kwargs: Optional[dict] = None,
):
    '''
    initialize predator_prey_swarm_env

    input
    -----
    config_dir:dict
        config dictionary
    seed:int
        seed
    wrap:bool
        if True wraps env in rllib wrapper
    '''

    from controllers.predator_prey_swarm_control import adversary_controller
    from envs.marl.predator_prey_swarm_env import PredatorPreySwarmEnv, parallel_env as swarm_parallel_env

    scenario_kwargs = config['env']['scenario_kwargs']
    scenario_kwargs['agent_list'] = config['env']['agent_list']
    env = swarm_parallel_env(
        scenario_kwargs=scenario_kwargs,
        max_cycles=config['env']['max_episode_length'],
        render_mode = render_mode,
    )

    env = PredatorPreySwarmEnv(
        mpeEnv=env,
        agents=config['env']['learned_agent_list'],
        adversary_controller=adversary_controller,
        reward_kwargs=config['env'].get('reward_kwargs'),
        controller_kwargs=config['env'].get('controller_kwargs'),
        seed = seed,
        local_observations = config['env']['local_observations'],
    )

    if wrap == 'rllib':
        env = RLLibWrapper(
            env=env,
            name='predator_prey_swarm',
            obs_packaging_func=predator_prey_swarm_obs_packaging,
            eval=eval,
            belief_kwargs=belief_kwargs,
            noise=config['env']['noise']
        )
    elif wrap is None:
        pass
    else:
        print('Wrapper not supported for predator prey swarm env...')

    return env


def make_drones_env(
        config: dict,
        seed: Optional[int] = None,
        wrap: Optional[str] = 'rllib',
        eval: bool = False,
        render_mode: Optional[str] = None,
        belief_kwargs: Optional[dict] = None,
):
    '''
    initialize drones_env using pybullet-drones

    input
    -----
    config_dir:dict
        config dictionary
    seed:int
        seed
    wrap:bool
        if True wraps env in rllib wrapper
    '''

    from envs.marl.drones_env import CaravanAviary
    from gym_pybullet_drones.utils.enums import ActionType
    from controllers.drone_control import drone_controller, action_to_vec, vec_to_action

    env = CaravanAviary(
        agent_list=config['env']['agent_list'],
        learned_agent_list=config['env']['learned_agent_list'],
        controller_kwargs=config['env'].get('controller_kwargs'),
        reward_kwargs=config['env'].get('reward_kwargs'),
        gui=False,
        act=ActionType.VEL,
        max_episode_length=config['env']['max_episode_length']*config['timestep'], # 10Hz step rate
        **config['env']['env_kwargs'],
    )

    if wrap == 'rllib':
        env = RLLibWrapper(
            env=env,
            name='drones',
            obs_packaging_func=drones_obs_packaging,
            eval=eval,
            dimension=3,
            belief_kwargs=belief_kwargs,
            noise=config['env']['noise']
        )
    elif wrap == 'pf':
        env = PFWrapper(
            env,
            name='drones',
            particle_filter = DronesParticleFilter,
            agent_control_function = lambda obs, obs_map: action_to_vec(drone_controller(obs, obs_map, config['env']['controller_kwargs'])),
            target_control_function = lambda obs, obs_map: action_to_vec(vec_to_action(env.env._scripted_target_action(obs=obs)[0:3])),
            dim=3,
            eval=eval,
            belief_kwargs=config['env']['belief_kwargs'],
            noise=config['env']['noise'],
        )
    elif wrap is None:
        pass
    else:
        print('Wrapper not supported for drone env...')

    return env


def make_football_env(
        config: dict,
        seed: Optional[int] = None,
        wrap: Optional[str] = 'rllib',
        eval: bool = False,
        render_mode: Optional[bool] = False,
        belief_kwargs: Optional[dict] = None,
):
    '''
    initialize drones_env using pybullet-drones

    input
    -----
    config_dir:dict
        config dictionary
    seed:int
        seed
    wrap:bool
        if True wraps env in rllib wrapper
    '''
    from envs.marl.football_env import CirclePass5v1Env
    from controllers.football_control import adversary_controller, compute_rondo_actions, action_to_vec

    env = CirclePass5v1Env(
        agent_list=config['env']['learned_agent_list'],
        full_agent_list=config['env']['agent_list'],
        adversary_controller=adversary_controller,
        max_episode_length=config['env']['max_episode_length'],
        controller_kwargs=config['env'].get('controller_kwargs'),
        reward_kwargs = config['env'].get('reward_kwargs'),
        render=False,
    )

    if wrap == 'rllib':
        env = RLLibWrapper(
            env=env,
            name='football',
            obs_packaging_func=football_obs_packaging,
            eval=eval,
            belief_kwargs=belief_kwargs,
            # applied in every run (until 2026-11-06 football never received it, so its fully observed runs and
            # belief data were noise-free); 5a_game_collect.yaml (belief data) sets team (and ball) noise 0.03; 5a_game.yaml and the allocentric config 0
            noise=config['env'].get('noise'),
        )
    elif wrap == 'pf':
        env = PFWrapper(
            env,
            name='football',
            particle_filter = FootballParticleFilter,
            agent_control_function = lambda obs, obs_map: action_to_vec(compute_rondo_actions(obs, obs_map)),
            target_control_function = lambda obs, obs_map: action_to_vec(adversary_controller(obs, obs_map, config['env']['controller_kwargs'])),
            dim=2,
            eval=eval,
            belief_kwargs=config['env']['belief_kwargs'],
            noise=config['env']['noise'],
        )
    elif wrap is None:
        pass
    else:
        print('Wrapper not supported for football env...')

    return env

def make_fire_env(
        config: dict,
        seed: Optional[int] = None,
        wrap: Optional[str] = 'rllib',
        eval: bool = False,
        render_mode: Optional[bool] = False,
        belief_kwargs: Optional[dict] = None,
):
    '''
    initialize drones_env using pybullet-drones

    input
    -----
    config_dir:dict
        config dictionary
    seed:int
        seed
    wrap:bool
        if True wraps env in rllib wrapper
    '''
    from controllers.fire_control import extinguish_controller
    from envs.marl.fire_env import DroneFireEnv, probabilistic_fire_controller

    env = DroneFireEnv(
        drone_names=config['env']['learned_agent_list'],
        fire_controller=probabilistic_fire_controller,
        controller_kwargs=config['env'].get('controller_kwargs'),
        fire_controller_kwargs=config['env'].get('fire_controller_kwargs'),
        reward_kwargs = config['env'].get('reward_kwargs'),
        max_episode_length=config['env']['max_episode_length'],
        **config['env']['env_kwargs'],
    )

    if wrap == 'rllib':
        env = RLLibWrapper(
            env=env,
            name='fire',
            obs_packaging_func=fire_obs_packaging,
            eval=eval,
            belief_kwargs=belief_kwargs,
            noise=config['env']['noise'],
        )
    elif wrap == 'pf':
        env = PFWrapper(
            env,
            name='fire',
            particle_filter = FireParticleFilter,
            agent_control_function = lambda obs: extinguish_controller(obs, **config['env']['controller_kwargs']),
            target_control_function = lambda fire, drone_positions=None: probabilistic_fire_controller(
                fire, drone_positions=drone_positions, **config['env'].get('fire_controller_kwargs', {})
            ),
            dim=2,
            eval=eval,
            belief_kwargs=config['env']['belief_kwargs'],
        )
    elif wrap is None:
        pass
    else:
        print('Wrapper not supported for fire env...')

    return env

def make_market_game_env(
        config: dict,
        seed: Optional[int] = None,
        wrap: Optional[str] = 'rllib',
        eval: bool = False,
        render_mode: Optional[str] = None,
        belief_kwargs: Optional[dict] = None,
):
    '''
    initialize market_game_env

    input
    -----
    config_dir:dict
        config dictionary
    seed:int
        seed
    wrap:bool
        if True wraps env in rllib wrapper
    '''
    from controllers.market_game_control import supplier_controller, compute_market_actions
    from envs.marl.market_game_env import MarketGameEnv

    env = MarketGameEnv(
        agents=config['env']['learned_agent_list'],
        supplier_controller=supplier_controller,
        market_kwargs=config['env']['market_kwargs'],
        supplier_kwargs=config['env']['supplier_kwargs'],
        reward_kwargs=config['env'].get('reward_kwargs'),
        controller_kwargs=config['env'].get('controller_kwargs'),
        max_episode_length=config['env']['max_episode_length'],
        seed=seed,
    )

    if wrap == 'rllib':
        env = RLLibWrapper(
            env=env,
            name='market_game',
            obs_packaging_func=market_game_obs_packaging,
            eval=eval,
            dimension=1,
            belief_kwargs=belief_kwargs,
            noise=config['env']['noise'],
        )
    elif wrap == 'pf':
        market_env = env
        env = PFWrapper(
            env,
            name='market_game',
            particle_filter = MarketFilter,
            agent_control_function = lambda obs, obs_map, team_rates=None: compute_market_actions(obs, obs_map, market_env.controller_cfg, team_rates),
            target_control_function = lambda obs, obs_map: supplier_controller(obs, obs_map, market_env.supplier_cfg),
            dim=1,
            eval=eval,
            belief_kwargs={
                **config['env']['belief_kwargs'],
                'agents': market_env.agents,
                'presence': market_env.presence,
                'stock_init': market_env.market_cfg['stock_cap'] / 2.0 / market_env.scale,
                'stock_cap': market_env.market_cfg['stock_cap'] / market_env.scale,
                'rate_range': market_env.market_cfg['rate_range'],
                'obs_scale': market_env.scale,
                'release_noise': market_env.supplier_cfg['noise'],
                'distributor_stock': market_env.supplier_cfg['own_stock_init'] / market_env.n_distributors,
            },
            noise=config['env']['noise'],
        )
    elif wrap is None:
        pass
    else:
        print('Wrapper not supported for market game env...')

    return env

def make_market_game_large_env(
        config: dict,
        seed: Optional[int] = None,
        wrap: Optional[str] = 'rllib',
        eval: bool = False,
        render_mode: Optional[str] = None,
        belief_kwargs: Optional[dict] = None,
):
    '''
    initialize market_game_large_env

    input
    -----
    config_dir:dict
        config dictionary
    seed:int
        seed
    wrap:bool
        if True wraps env in rllib wrapper
    '''
    from controllers.market_game_large_control import supplier_controller
    from envs.marl.market_game_large_env import MarketGameLargeEnv

    env = MarketGameLargeEnv(
        agents=config['env']['learned_agent_list'],
        supplier_controller=supplier_controller,
        market_kwargs=config['env']['market_kwargs'],
        supplier_kwargs=config['env']['supplier_kwargs'],
        reward_kwargs=config['env'].get('reward_kwargs'),
        controller_kwargs=config['env'].get('controller_kwargs'),
        max_episode_length=config['env']['max_episode_length'],
        seed=seed,
        quantiles=config['env']['quantiles'],
    )

    if wrap == 'rllib':
        env = RLLibWrapper(
            env=env,
            name='market_game_large',
            obs_packaging_func=market_game_obs_packaging,
            eval=eval,
            dimension=1,
            belief_kwargs=belief_kwargs,
            noise=config['env']['noise'],
        )
    elif wrap is None:
        pass
    else:
        print('Wrapper not supported for large market game env...')

    return env
