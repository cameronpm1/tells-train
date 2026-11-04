import time
import torch
import gymnasium
import numpy as np
from copy import deepcopy
from gymnasium import spaces
from collections import OrderedDict
from typing import Any, Dict, Type, Optional, Union
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class RLLibWrapper(MultiAgentEnv):
    '''
    wrapper for multiagent envs
    env should handle dict format, wrapper is
    for rllib MultiAgentEnv inheritance
    '''

    def __init__(
            self,
            env,
            name,
            obs_packaging_func,
            eval: bool = False,
            belief_kwargs: Optional[dict] = None,
            dimension: int = 2,
            noise: Optional[float] = None,
    ): 
        super().__init__()

        self.env = env
        self.eval = eval
        self.name = name
        self.dim = dimension
        self.n_agents = env.n_agents
        self.belief_kwargs = belief_kwargs
        self.agents = deepcopy(env.agents)
        self.obs_packaging_func = obs_packaging_func
        self.last_raw_reward = None
        self.noise = noise
        self.temp_noise = {}

        self.single_observation_spaces = {
            agent: self.env._observation_space(agent)
            for agent in self.agents
        }

        self.single_action_spaces = {
            agent: self.env._action_space(agent)
            for agent in self.agents
        }

        # disable deprecated paths
        self.observation_spaces = None
        self.action_spaces = None

        self.observation_space = None
        self.action_space = None

        if belief_kwargs is not None and belief_kwargs['on']:
            self.belief = True
            self.belief_model = belief_kwargs['model']
            self.belief_n = belief_kwargs['min_obs']
            self.obs_history = []
        else:
            self.belief = False

    def get_observation_space(self, agent):
        return self.single_observation_spaces[agent]

    def get_action_space(self, agent):
        return self.single_action_spaces[agent]
    
    def step(self, action_dict):

        terminated_all = False
        truncated_all = False

        obs,rew,terminated,truncated,infos = self.env.step(action_dict)
        rew = dict(rew)

        if self.eval:
            infos['target'] = obs['target']

        obs.pop("target", None)
        rew.pop('target', None)
        terminated.pop('target', None)
        truncated.pop('target', None)

        terminated["__all__"] = all(terminated.values())
        truncated["__all__"] = all(truncated.values())
        #print(obs.keys(),rew.keys(),terminated.keys(),truncated.keys(),_.keys())
        
        infos['__common__']['raw_reward'] = sum(rew.values())
        infos['__common__']['obs_no_noise'] = deepcopy(obs)

        if self.noise is not None:
            if 'fire' in self.name:
                for agent in self.agents:
                    noise_old = self.temp_noise.get(agent, np.zeros((self.n_agents-1)*self.dim))
                    self.temp_noise[str(agent)+'_old'] = noise_old
                    self.temp_noise[agent] = infos['__common__']['decomposed_obs'][agent]['noise']
            else:
                team_size = self.env.obs_map['team'].stop - self.env.obs_map['team'].start
                for agent in self.agents:
                    target_noise = np.random.normal(0, self.noise['target'], size=self.dim,)
                    obs[agent][self.env.obs_map['target_pos']] = obs[agent][self.env.obs_map['target_pos']] + target_noise
                    if not self.belief:
                        team_old = self.temp_noise.get(agent, {}).get('team', np.zeros(team_size))
                        team_noise = np.random.normal(0, self.noise['team'], size=team_size,)
                        obs[agent][self.env.obs_map['team']] = obs[agent][self.env.obs_map['team']] + team_noise
                        self.temp_noise[agent] = {'team': team_noise, 'team_old': team_old, 'target': target_noise}
                        if 'football' in self.name:
                            ball_noise = np.random.normal(0, self.noise['team'], size=self.dim,)
                            self.temp_noise[agent]['ball'] = ball_noise
                            obs[agent][self.env.obs_map['ball_pos']] = obs[agent][self.env.obs_map['ball_pos']] + ball_noise

        if self.belief:
            if 'fire' in self.name:
                self.obs_history.append(self._fire_history_entry(infos['__common__']['decomposed_obs']))
            else:
                self.obs_history.append(deepcopy(obs))

        self.last_raw_reward = sum(rew.values())

        #fill in observations
        if self.belief:
            converted_obs, transform_pred = self.obs_packaging_func(self.obs_history, self.env.obs_map, self.agents, min_obs=self.belief_n, noise=self.temp_noise)
            predictions = {}
            errors = []

            if self.eval:
                infos['__common__']['sampled_predictions'] = {}
            
            for agent, agent_obs in obs.items():
                #obs[agent][self.env.obs_map['target_pos']] = obs[agent][self.env.obs_map['target_pos']]
                model_input = torch.from_numpy(converted_obs[agent]['input']).to(torch.float32)
                # optional Gaussian noise on the belief model's input only (the policy's own observation is
                # untouched), for the belief-noise sweep
                input_noise = self.belief_kwargs.get('input_noise', 0.0)
                if input_noise:
                    noise = input_noise * torch.randn_like(model_input)
                    # inputs already noisy from the env (e.g. the observed prey history when noise.target is on)
                    skip = self.belief_kwargs.get('input_noise_skip')
                    if skip:
                        noise[skip] = 0.0
                    model_input = model_input + noise
                if self.belief_model.vae:
                    prediction, mu, logvar = self.belief_model.model(model_input)
                else:
                    prediction = self.belief_model.model(model_input)
                prediction = prediction.detach().cpu().numpy()

                self.obs_history[-1] = transform_pred(self.obs_history[-1],self.env.obs_map,agent,prediction)

                #if 'fire' in self.name:
                #    team_state = infos['__common__']['decomposed_obs'][agent]['team']
                #    team_size = prediction.shape[0] // 2
                #    team_curr_pred = prediction[team_size:]
                #    pred_t = torch.from_numpy(np.asarray(team_curr_pred, dtype=np.float32)).unsqueeze(0)
                #    target_t = torch.from_numpy(np.asarray(team_state, dtype=np.float32)).unsqueeze(0)
                #else:
                #    team_state = self.obs_history[-1][agent][self.env.obs_map['team']]
                #    pred_t = torch.from_numpy(np.asarray(team_state, dtype=np.float32)).unsqueeze(0)
                #    target_t = torch.from_numpy(np.asarray(agent_obs[self.env.obs_map['team']], dtype=np.float32)).unsqueeze(0)
                #if self.belief_model.vae:
                #    loss, _ = self.belief_model.model.loss(pred_t, target_t, mu.unsqueeze(0), logvar.unsqueeze(0))
                #else:
                #    loss = self.belief_model.model.loss(pred_t, target_t)
                #error = self.permutation_invariant_error(team_state, agent_obs[self.env.obs_map['team']])
                #error = loss
                #error = error.item()
                if 'fire' in self.name:
                    team_state = self.obs_history[-1][agent]['team']
                    error = self.env.team_error(team_state,agent)
                else:
                    team_state = self.obs_history[-1][agent][self.env.obs_map['team']]
                    error = self.env.team_error(self.obs_history[-1][agent], agent)
                errors.append(error)
                predictions[agent] = team_state
                #compute individual reward penalty for deviation of belief model
                #rew[agent] = rew[agent] - error * self.env.reward_cfg['belief_dev_scale']
                if self.eval:
                    if self.belief_model.vae:
                        infos['__common__']['sampled_predictions'][agent] = self.belief_model.sample_stochastic(model_input).detach().cpu().numpy()[:, obs_idxs]
                    else:
                        infos['__common__']['sampled_predictions'][agent] = team_state
            avg_error = np.average(errors)
            infos['__common__']['belief_error'] = avg_error
            self.prediction_history.append(predictions)
            if 'fire' in self.name:
                for agent in self.agents:
                    obs[agent] = self.obs_history[-1][agent]['pred']
            else:
                obs = self.obs_history[-1]
        return obs,rew,terminated,truncated,infos
    
    def _fire_history_entry(self, decomposed_obs):
        '''
        copy of the decomposed fire observation for the belief history. With belief_kwargs['image_noise'] = p (noise
        sweep), each cell of every agent's fire window is independently replaced, with probability p, by a uniformly
        random fire state (GREEN, RED, BLACK, WHITE). The belief's input and the policy's image (rebuilt from this
        history by the fire transform) both see the noisy fire; the logged decomposed_obs stays the true state.
        '''
        entry = deepcopy(decomposed_obs)
        p = self.belief_kwargs.get('image_noise', 0.0) if self.belief_kwargs else 0.0
        if p:
            for agent in self.agents:
                fire = np.array(entry[agent]['fire'])
                flip = np.random.random(fire.shape) < p
                fire[flip] = np.random.randint(0, 4, size=int(flip.sum()))
                entry[agent]['fire'] = fire
        return entry

    def reset(self, **kwargs):
        obs,infos = {},{}
        obs,infos = self.env.reset(**kwargs)

        obs.pop("target", None)
        
        infos['__common__']['raw_reward'] = 0.0
        infos['__common__']['obs_no_noise'] = deepcopy(obs)
        self.last_raw_reward = 0.0

        if self.belief:
            self.obs_history = []
            self.prediction_history = []

            predictions = {}
            for agent in self.agents:
                if 'fire' in self.name:
                    predictions[agent] = infos['__common__']['decomposed_obs'][agent]['team']
                else:
                    predictions[agent] = obs[agent][self.env.obs_map['team']]

            if 'fire' in self.name:
                self.obs_history.append(self._fire_history_entry(infos['__common__']['decomposed_obs']))
            else:
                self.obs_history.append(deepcopy(obs))
            self.prediction_history.append(predictions)

            infos['__common__']['belief_error'] = 0.0

        return obs,infos

    def close_all(self):
        """Close the wrapped environment if it supports close, then mark this wrapper closed."""
        if hasattr(self.env, "close") and callable(self.env.close):
            try:
                self.env.close()
            except Exception:
                pass
        elif hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "close") and callable(self.env.unwrapped.close):
            try:
                self.env.unwrapped.close()
            except Exception:
                pass

        super().close()

    def render_rgb(self):
        return self.env.render_rgb()

    def set_difficulty(self, difficulty):
        self.env.set_difficulty(difficulty)
        

    def permutation_invariant_error(self, pred, target):
        '''
        pred:   (N, 6)
        target: (N, 6)

        Returns:
            scalar, sum over batch of minimum assignment distances
        '''

        # Reshape to (N, 2, 3)
        pred = pred.reshape(-1, 2, 2)
        target = target.reshape(-1, 2, 2)

        # Direct assignment distances
        direct = (
            np.linalg.norm(pred[:, 0] - target[:, 0], axis=1) +
            np.linalg.norm(pred[:, 1] - target[:, 1], axis=1)
        )

        # Swapped assignment distances
        swapped = (
            np.linalg.norm(pred[:, 0] - target[:, 1], axis=1) +
            np.linalg.norm(pred[:, 1] - target[:, 0], axis=1)
        )

        # Take minimum per sample, then sum batch
        return np.minimum(direct, swapped).sum()




