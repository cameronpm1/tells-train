"""
Multi-box drone defense environment for gym-pybullet-drones.

Learner controls N protector drones.
The final drone is a scripted adversarial drone.

Task:
    Five goal boxes are represented internally as propagating point states.
    Each box state is [x, y, z, vx, vy, vz].
    Current default box velocity is zero, so the boxes remain fixed unless
    nonzero box velocities are assigned later.

    Protector drones must hover over box points to mark them as protected.
    The adversary periodically retargets an unprotected box and moves using
    a single potential-field action function. That function takes the drone
    team state, box state, and attraction/repulsion parameters as inputs.

Recommended action type:
    ActionType.VEL

For ActionType.VEL, actions are 4D:
    [vx_dir, vy_dir, vz_dir, speed_fraction]
"""

import numpy as np
import pybullet as p
from copy import deepcopy
from typing import Optional
from gymnasium import spaces

from controllers.drone_control import drone_controller
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class CaravanAviary(BaseRLAviary):
    def __init__(
        self,
        agent_list: list[str],
        learned_agent_list: list[str],
        reward_kwargs: dict,
        max_episode_length: int = 200,
        grid_size: float = 20.0,
        goal_pos: Optional[np.ndarray] = None,
        goal_line_center: Optional[np.ndarray] = None,
        num_goal_boxes: int = 5,
        num_protectors: int = 3,
        goal_box_spacing: float = 1.0,
        goal_box_half_extents: tuple[float, float, float] = (0.5, 0.5, 0.1),
        intrusion_radius: float = 0.3,
        adversary_replan_steps: int = 10,
        base_altitude: float = 0.5,
        base_speed: float = 8.0,
        speed_ratio: float = 0.4,
        protection_radius: float = 1.1,
        target_spawn_distance: float = 5.0,
        controller_kwargs: Optional[dict] = None,
        drone_model: DroneModel = DroneModel.CF2X,
        neighbourhood_radius: float = np.inf,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui: bool = False,
        record: bool = False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.VEL,
    ):
        self.full_agent_list = agent_list
        self.agents = learned_agent_list
        self.reward_cfg = reward_kwargs
        self.n_agents = len(self.agents)
        self.target_idx = len(self.agents)
        self.grid_size = grid_size
        self.base_altitude = base_altitude
        self.base_speed = base_speed
        self.speed_ratio = speed_ratio
        self.target_spawn_distance = float(target_spawn_distance)
        self.max_episode_length = max_episode_length

        self.num_goal_boxes = int(num_goal_boxes)
        self.goal_box_spacing = float(goal_box_spacing)
        self.goal_box_half_extents = tuple(float(v) for v in goal_box_half_extents)
        self.protection_radius = float(protection_radius)
        self.intrusion_radius = float(intrusion_radius)
        self.adversary_replan_steps = int(adversary_replan_steps)

        self.controller_cfg = controller_kwargs

        if goal_line_center is None:
            if goal_pos is None:
                center_xy = np.array([0.0, 0.0], dtype=np.float32)
            else:
                center_xy = np.asarray(goal_pos, dtype=np.float32)[0:2]
        else:
            center_xy = np.asarray(goal_line_center, dtype=np.float32)[0:2]

        self.goal_line_center_xy = center_xy.astype(np.float32)
        self.box_state = np.zeros((self.num_goal_boxes, 6), dtype=np.float32)
        self._initialize_box_state()

        self._rng = np.random.default_rng()
        self._goal_body_ids: list[int] = []
        self._policy_step_counter = 0
        self.current_target_box_idx = 0

        self.num_drones = self.n_agents + 1

        if initial_xyzs is None:
            initial_xyzs = self._sample_initial_xyzs(self._rng)

        super().__init__(
            drone_model=drone_model,
            num_drones=self.num_drones,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )

        self._step = 0
        self.difficulty = 1.0

        target_start = 6
        target_stop = target_start + (3 * self.num_goal_boxes) + 3
        team_stop = target_stop + 3 * (len(self.agents) - 1)
        target_obs_stop = target_stop + 3 * len(self.agents)
        self.obs_map = {
            'self': slice(0, 6),
            'target': slice(target_start, target_stop),
            'team': slice(target_stop, team_stop),
            'target_obs': slice(target_stop, target_obs_stop),
            'self_pos': slice(3, 6),
            'target_pos': slice(target_start + 3 * self.num_goal_boxes, target_start + 3 * self.num_goal_boxes + 3),
            'target_goal': slice(target_start, target_start + 3 * self.num_goal_boxes)
        }
        self.goal_rel = True

        self.action_library = {
            0:  np.array([0.0, 0.0, 0.0, 0.0]),

            1:  np.array([1.0, 0.0, 0.0, self.base_speed]),
            2:  np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0.0, self.base_speed]),
            3:  np.array([0.0, 1.0, 0.0, self.base_speed]),
            4:  np.array([-1 / np.sqrt(2), 1 / np.sqrt(2), 0.0, self.base_speed]),
            5:  np.array([-1.0, 0.0, 0.0, self.base_speed]),
            6:  np.array([-1 / np.sqrt(2), -1 / np.sqrt(2), 0.0, self.base_speed]),
            7:  np.array([0.0, -1.0, 0.0, self.base_speed]),
            8:  np.array([1 / np.sqrt(2), -1 / np.sqrt(2), 0.0, self.base_speed]),

            9:  np.array([0.0, 0.0, 1.0, self.base_speed]),
            10: np.array([0.0, 0.0, -1.0, self.base_speed]),
        }

    def _initialize_box_state(self):
        """
        Initialize box point states.

        Each row is:
            [x, y, z, vx, vy, vz]

        The default velocity is zero. Future experiments can assign nonzero
        velocities to self.box_state[:, 3:6] and the points will propagate.
        """
        center_offset = 0.5 * (self.num_goal_boxes * self.goal_box_half_extents[0] * 2 + self.goal_box_spacing * (self.num_goal_boxes - 1)) - 0.5
        
        for k in range(self.num_goal_boxes):
            x = self.goal_line_center_xy[0] + (k*(self.goal_box_half_extents[0]*2+self.goal_box_spacing) - center_offset)
            y = self.goal_line_center_xy[1]
            z = self.base_altitude
            self.box_state[k, 0:3] = np.array([x, y, z], dtype=np.float32)
            self.box_state[k, 3:6] = 0.0


    def _propagate_box_points(self, dt: float):
        """
        Propagate the box point states forward with constant velocity.

        The current default velocity is zero, so this is a no-op unless the
        box velocities are modified elsewhere.
        """
        self.box_state[:, 0:3] += self.box_state[:, 3:6] * float(dt)
        self._update_goal_visuals()

    def _update_goal_visuals(self):
        """Move the visual ground boxes to the x-y positions of the point states."""
        if not self._goal_body_ids:
            return

        z_center = self.goal_box_half_extents[2]

        for goal_idx, body_id in enumerate(self._goal_body_ids):
            point = self.box_state[goal_idx, 0:3]
            p.resetBasePositionAndOrientation(
                bodyUniqueId=body_id,
                posObj=[float(point[0]), float(point[1]), float(z_center)],
                ornObj=[0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.CLIENT,
            )

    def _get_team_state(self):
        """
        Return all drone states in a compact array.

        Rows 0 through num_agents - 1 are protector drones.
        The final row is the adversary.

        Each row is:
            [x, y, z, vx, vy, vz]
        """
        team_state = np.zeros((self.num_drones, 6), dtype=np.float32)

        for i in range(self.num_drones):
            state = self._getDroneStateVector(i)
            team_state[i, 0:3] = state[0:3]
            team_state[i, 3:6] = state[10:13]

        return team_state

    def _action_space(self,agent):
        """
        Learner only controls protector drones.
        The adversarial drone action is injected internally.
        """
        return spaces.Discrete(len(self.action_library))

    def _observation_space(self,agent):
        """
        Per-protector observation:
            own velocity, 3
            own position, 3
            relative position to each box point, 3 * num_goal_boxes
            relative position to the adversary, 3
            relative position of every other protector, 3 * (num_agents - 1)
        """
        obs_dim = (
            6
            + (3 * self.num_goal_boxes)
            + 3
            + 3 * (self.n_agents - 1)
        )

        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _computeObs(self):
        team_state = self._get_team_state()
        positions = team_state[:, 0:3]
        velocities = team_state[:, 3:6]

        obs = {}

        for i in range(self.num_drones):
            local_obs = {}
            own_pos = positions[i]
            own_vel = velocities[i]
            box_rel_positions = np.array([box[0:3] - own_pos for box in self.box_state]).flatten()
            local_obs['self'] = np.concatenate((own_vel, own_pos))

            rel_positions = []
            for j in range(self.num_drones):
                if j == self.target_idx:
                    local_obs['target'] = np.concatenate((box_rel_positions, positions[self.target_idx] - own_pos))
                elif j == i:
                    continue
                else:
                    rel_positions.append(positions[j] - own_pos)
            local_obs['team'] = np.array(rel_positions).flatten()

            obs[self.full_agent_list[i]] = np.concatenate((local_obs['self'], local_obs['target'], local_obs['team']))

        return obs
    
    def _computeInfo(self):
        protected = self._compute_protected_boxes()
        breached = bool(self._breached_box())
        oob = bool(self._out_of_bounds())
        protected_fraction = float(np.mean(protected))
        success = bool(protected_fraction >= 1.0 and not breached and not oob)

        info = {}
        for agent in self.agents:
            info[agent] = {
                'protected_boxes': int(np.sum(protected)),
                'protected_fraction': protected_fraction,
                'breached': breached,
                'oob': oob,
                'success': success,
                'target_box_idx': int(self.current_target_box_idx),
            }
        info['target'] = {
            'target_box_idx': int(self.current_target_box_idx),
            'breached': breached,
            'oob': oob,
        }

        return info

    def reset(self, seed=None, options=None):
        self._step = 0

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._policy_step_counter = 0
        self.current_target_box_idx = int(self._rng.integers(0, self.num_goal_boxes))
        self._initialize_box_state()
        self.INIT_XYZS = self._sample_initial_xyzs(self._rng)

        obs, infos = super().reset(seed=seed, options=options)
        self.obs = obs

        return deepcopy(obs), {'__common__': self._computeInfo()}

    def step(self, action_dict):
        self._step += 1

        #action_dict = drone_controller(deepcopy(self.obs), self.obs_map, self.controller_cfg)
        action = []
        for agent in self.agents:
            action.append(self.action_library[action_dict[agent]])

        #action = np.asarray(action, dtype=np.float32)
        #speed = np.ones((self.n_agents, 1), dtype=np.float32) * self.base_speed
        #action = np.hstack([action, speed])

        dt = 1.0 / float(getattr(self, "CTRL_FREQ", 30))
        self._propagate_box_points(dt)

        full_action = np.zeros((self.num_drones, 4), dtype=np.float32)
        full_action[: self.n_agents, :] = action
        full_action[self.target_idx, :] = self._scripted_target_action()

        self._policy_step_counter += 1
        self.behavior_cloning_reward(action_dict)

        for i in range(10):
            obs, rewards, terminations, truncations, infos = super().step(full_action)
        self.obs = obs

        return deepcopy(obs), rewards, terminations, truncations, {'__common__': self._computeInfo()}

    def _scripted_target_action(
            self,
            obs: Optional[dict] = None,
        ):
        """
        Compute the adversary action from the compact team state and box state.
        """
        replan = self._policy_step_counter % self.controller_cfg['adversary_replan_steps'] == 0

        if obs is None:
            action, target_box_idx = self._compute_adversary_action_from_state(
                obs=self.obs['target'],
                obs_map=self.obs_map,
                current_target_box_idx=self.current_target_box_idx,
                replan=replan,
            )
        else:
            action, target_box_idx = self._compute_adversary_action_from_state(
                obs=obs,
                obs_map=self.obs_map,
                current_target_box_idx=self.current_target_box_idx,
                replan=replan,
            )

        self.current_target_box_idx = int(target_box_idx)

        return action

    def _compute_adversary_action_from_state(
        self,
        obs,
        obs_map,
        current_target_box_idx: int,
        replan: bool = True,
    ):
        adversary_pos = obs[obs_map['self_pos']]
        team_pos = (obs[obs_map['target_obs']] + np.tile(adversary_pos, self.n_agents)).reshape(-1,3)
        box_pos = (obs[obs_map['target']][:-3] + np.tile(adversary_pos, int(len(obs[obs_map['target']])/3 - 1))).reshape(-1,3)

        protection_radius = self.controller_cfg['protection_radius']
        repulsion_radius = self.controller_cfg['adversary_repulsion_radius']
        repulsion_gain = self.controller_cfg['adversary_repulsion_gain']
        attraction_gain = self.controller_cfg['adversary_attraction_gain']

        selected_target_box_idx = int(current_target_box_idx)
        protected_boxes = self._compute_protected_boxes()
    
        if replan or not (0 <= selected_target_box_idx < self.num_goal_boxes) or protected_boxes[selected_target_box_idx] == 1:
            dists_to_adversary = np.linalg.norm(box_pos[:, 0:3] - adversary_pos[0:3], axis=1)
            dists_to_protectors = np.linalg.norm(
                box_pos[:, None, 0:3] - team_pos[None, :, 0:3], axis=2
            )
            min_dist_to_protector = np.min(dists_to_protectors, axis=1)
            guarded = min_dist_to_protector < 2.0

            candidate_dists = np.where(guarded, np.inf, dists_to_adversary)
            if np.all(np.isinf(candidate_dists)):
                candidate_dists = dists_to_adversary #every box has a protector within range, ignore guarding
            selected_target_box_idx = int(np.argmin(candidate_dists)) #closest box with no protector within 2

        target_pos = box_pos[selected_target_box_idx]
        force = np.zeros(3, dtype=np.float32)
        attraction = target_pos - adversary_pos
        force += float(attraction_gain) * attraction

        for protector in team_pos:
            diff = adversary_pos - protector
            diff[2] = 0.0

            dist = np.linalg.norm(diff) + 1e-6

            if dist <= repulsion_radius:
                force += float(repulsion_gain) * diff / (dist**2)

        speed = self.base_speed * self.speed_ratio

        if np.linalg.norm(force) < 0.5:
            action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            direction = force / np.linalg.norm(force)
            action = np.concatenate([direction, np.array([speed])])

        return action, selected_target_box_idx

    def behavior_cloning_reward(self, action_dict: dict) -> dict:
        controller_actions = drone_controller(deepcopy(self.obs), self.obs_map, self.controller_cfg)
        action_matches = []

        for agent in self.agents:
            if agent not in action_dict or agent not in controller_actions:
                continue

            learned_action = int(action_dict[agent])
            action_matches.append(float(learned_action == controller_actions[agent]))

        if len(action_matches) == 0:
            controller_action_match = 0.0
        else:
            controller_action_match = float(np.mean(action_matches))
        controller_action_error = 1.0 - controller_action_match

        self.controller_metrics = {
            'controller_action_error': controller_action_error,
            'controller_action_match': controller_action_match,
            'controller_action_reward': controller_action_match,
        }
    
    def _compute_spacing_reward(self):
        """
        Score how evenly protectors are spaced along the goal line: project
        protector positions onto the box-line axis, then map the coefficient
        of variation of the pairwise axis-distances to (0, 1], 1 being even.
        Zeroed out entirely if any neighbor pair is closer than min_spacing_dist.
        """
        axis = self.box_state[-1, 0:3] - self.box_state[0, 0:3]
        axis_norm = np.linalg.norm(axis)
        axis_unit = axis / axis_norm if axis_norm > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)

        protector_pos = self._get_team_state()[: self.n_agents, 0:3]
        protector_proj = protector_pos @ axis_unit
        protector_pair_dists = np.abs(protector_proj[:, None] - protector_proj[None, :])[np.triu_indices(self.n_agents, k=1)]

        min_spacing_dist = self.reward_cfg.get('min_spacing_dist', 0.0)
        if np.any(protector_pair_dists < min_spacing_dist):
            return 0.0

        protector_pair_dists = np.delete(protector_pair_dists, np.argmax(protector_pair_dists))
        mean_gap = np.mean(protector_pair_dists)
        return float(np.exp(-np.std(protector_pair_dists) / mean_gap)) if mean_gap > 1e-6 else 0.0


    def _computeReward(self):
        """
        Shared team reward.

        Positive component:
            fraction of boxes currently protected.

        Negative component:
            penalty if the adversary breaches any box point.
        """
        reward = 0

        reward += self.reward_cfg['step_reward']
        reward += self.controller_metrics['controller_action_reward'] * self.reward_cfg['bc_scale']
        reward += self._compute_spacing_reward() * self.reward_cfg.get('spacing_scale', 0.0)

        target_pos = self._getDroneStateVector(self.target_idx)[0:3]
        dist_to_closest_box = float(np.min(np.linalg.norm(self.box_state[:, 0:2] - target_pos[0:2], axis=1)))
        reward += min(dist_to_closest_box, 4.0) * self.reward_cfg.get('target_dist_scale', 1.0)

        if self._breached_box():
            reward -= self.reward_cfg.get('intruded_penalty', 300)

        if self._out_of_bounds():
            reward -= self.reward_cfg.get('oob_penalty', 300)

        reward_dict = {}
        for agent in self.agents:
            reward_dict[agent] = reward
        reward_dict['target'] = 0.0

        return reward_dict

    def _computeTerminated(self):

        if self._out_of_bounds():
            terminated = True
        elif self._breached_box():
            terminated = True
        else:
            terminated = False

        terminated_dict = {}
        for agent in self.agents:
            terminated_dict[agent] = terminated
        terminated_dict['target'] = False

        return terminated_dict

    def _computeTruncated(self):
        if self._step > self.max_episode_length:
            truncated = True
        else:
            truncated = False

        truncated_dict = {}
        for agent in self.agents:
            truncated_dict[agent] = truncated
        truncated_dict['target'] = False

        return truncated_dict

    def _compute_protected_boxes(self):
        team_state = self._get_team_state()
        protector_pos = team_state[: self.n_agents, 0:3]
        protected = np.zeros(self.num_goal_boxes, dtype=bool)

        for goal_idx, point in enumerate(self.box_state[:, 0:3]):
            for agent_idx in range(self.n_agents):
                pos = protector_pos[agent_idx]
                horizontal_dist = np.linalg.norm(pos - point)

                if horizontal_dist <= self.protection_radius:
                    protected[goal_idx] = 1
                    break

        return protected

    def _breached_box(self):
        target_pos = self._getDroneStateVector(self.target_idx)[0:3]

        for goal_idx, point in enumerate(self.box_state[:, 0:3]):
            horizontal_dist = np.linalg.norm(target_pos[0:2] - point[0:2])

            if horizontal_dist <= self.intrusion_radius:
                return True

        return False

    def _out_of_bounds(self):
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            x, y, z = state[0:3]

            if i == self.target_idx:
                xy_bound = self.grid_size
            else:
                xy_bound = self.grid_size * 2.0

            if abs(x) >= xy_bound or abs(y) >= xy_bound:
                return True

            if z < 0.05 or z > 3.0:
                return True

        for point in self.box_state[:, 0:3]:
            if abs(point[0]) >= self.grid_size * 2.0 or abs(point[1]) >= self.grid_size * 2.0:
                return True

        return False

    def _sample_initial_xyzs(self, rng):
        spawn_radius = 2.0
        xyzs = np.zeros((self.n_agents + 1, 3), dtype=np.float32)

        min_goal_x = float(np.min(self.box_state[:, 0])) + 1.0
        max_goal_x = float(np.max(self.box_state[:, 0])) - 1.0
        min_goal_y = float(np.min(self.box_state[:, 1]))
        max_goal_y = float(np.max(self.box_state[:, 1]))
        target_x = rng.uniform(min_goal_x, max_goal_x)
        target_y_sign = -1.0 if rng.random() < 0.5 else 1.0
        target_y = rng.uniform(min_goal_y, max_goal_y) + target_y_sign * self.target_spawn_distance

        xyzs[self.target_idx] = np.array(
            [target_x, target_y, self.base_altitude*2],
            dtype=np.float32,
        )

        if self.n_agents <= self.num_goal_boxes:
            assigned_goals = np.rint(
                np.linspace(0, self.num_goal_boxes - 1, self.n_agents)
            ).astype(int)
        else:
            assigned_goals = np.arange(self.n_agents) % self.num_goal_boxes

        spawn_x = rng.uniform(self.box_state[0, 0], self.box_state[self.num_goal_boxes - 1, 0])
        box_center = np.array(
            [spawn_x, self.box_state[1, 1], self.box_state[1, 2]],
            dtype=np.float32,
        )

        for i in range(self.n_agents):
            angle = rng.uniform(-np.pi, np.pi)
            radius = rng.uniform(0.0, spawn_radius)

            offset_xy = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            xy = box_center[0:2] + offset_xy
            xyzs[i] = np.array([xy[0],xy[1], self.base_altitude])

        return xyzs

    def _addObstacles(self):
        """
        Adds translucent goal boxes on the ground.
        The actual target/protection state is the propagating box point state.
        These visual boxes do not create collision geometry.
        """
        self._goal_body_ids = []

        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=list(self.goal_box_half_extents),
            rgbaColor=[0.2, 0.85, 0.2, 0.35],
            physicsClientId=self.CLIENT,
        )

        z_center = self.goal_box_half_extents[2]

        for point in self.box_state[:, 0:3]:
            body_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=[float(point[0]), float(point[1]), float(z_center)],
                physicsClientId=self.CLIENT,
            )
            self._goal_body_ids.append(body_id)

    def set_difficulty(self, difficulty):
        self.difficulty = float(difficulty)

    def close(self):
        """Immediately terminate the PyBullet drone environment."""
        try:
            self.close()
        except Exception:
            pass

    def team_error(
        self, 
        estimate: np.ndarray,
        agent: str
    ):

        # Reshape to (N, 2, 3)
        pred = estimate[self.obs_map['team']].reshape(-1, 2, 3)
        target = self.obs[agent][self.obs_map['team']].reshape(-1, 2, 3)

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

    def render_pybullet_rgb(
        self,
        width: int = 640,
        height: int = 480,
        camera_position: np.ndarray = None,
        target_position: np.ndarray = None,
    ):
        """
        Returns an RGB image from the PyBullet simulation without GUI rendering.

        Args:
            width, height: image resolution
            camera_position: (3,) camera position in world frame
            target_position: (3,) point camera looks at

        Returns:
            rgb image as (H, W, 3) uint8 numpy array
        """

        if camera_position is None:
            camera_position = np.array([2, 2, 1.33], dtype=np.float32) #np.array([4.5, 4.5, 2.9], dtype=np.float32)

        if target_position is None:
            target_position = np.array([0.0, 0.0, 0.5], dtype=np.float32)

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_position.tolist(),
            cameraTargetPosition=target_position.tolist(),
            cameraUpVector=[0, 0, 1],
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=float(width) / height,
            nearVal=0.1,
            farVal=100.0,
        )

        _, _, rgba, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER,  # works without GUI
        )

        rgba = np.reshape(rgba, (height, width, 4))
        rgb = rgba[:, :, :3].astype(np.uint8)

        return rgb

    def render_rgb(self, bounds: Optional[tuple] = None):
        """
        Custom 2D top-down RGB renderer.

        Shows:
            - the goal boxes
            - protector drones as blue quadrotor symbols
            - target/adversary drone as a red quadrotor symbol
            - altitude by slight drone size changes

        Args:
            bounds: optional fixed (xmin, xmax, ymin, ymax) view in world units; when None
                the view is fit to the boxes and drones every frame

        Returns
        -------
        img : np.ndarray
            RGB image with shape (H, W, 3), dtype uint8.
        """
        import numpy as np

        width = int(getattr(self, "render_width", 700))
        height = int(getattr(self, "render_height", 700))

        img = np.full((height, width, 3), 245, dtype=np.uint8)

        team_state = self._get_team_state()
        drone_pos = team_state[:, 0:3]
        box_pos = self.box_state[: self.num_goal_boxes, 0:3]

        hx, hy, _ = self.goal_box_half_extents

        # ------------------------------------------------------------
        # Dynamic square view around boxes and drones
        # ------------------------------------------------------------
        all_x = np.concatenate([
            drone_pos[:, 0],
            box_pos[:, 0] - hx,
            box_pos[:, 0] + hx,
        ])
        all_y = np.concatenate([
            drone_pos[:, 1],
            box_pos[:, 1] - hy,
            box_pos[:, 1] + hy,
        ])

        if bounds is not None:
            xmin, xmax, ymin, ymax = (float(b) for b in bounds)
        else:
            pad = max(2.0, float(self.target_spawn_distance) * 0.25)
            xmin, xmax = float(np.min(all_x) - pad), float(np.max(all_x) + pad)
            ymin, ymax = float(np.min(all_y) - pad), float(np.max(all_y) + pad)

        # Keep aspect ratio square so distances do not visually distort.
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)
        span = max(xmax - xmin, ymax - ymin, 1e-6)
        xmin, xmax = xmid - span / 2.0, xmid + span / 2.0
        ymin, ymax = ymid - span / 2.0, ymid + span / 2.0

        def world_to_pixel(x, y):
            px = int((x - xmin) / (xmax - xmin) * (width - 1))
            py = int((ymax - y) / (ymax - ymin) * (height - 1))
            return px, py

        def draw_filled_circle(cx, cy, r, color):
            x0, x1 = max(0, cx - r), min(width - 1, cx + r)
            y0, y1 = max(0, cy - r), min(height - 1, cy + r)

            if x1 < x0 or y1 < y0:
                return

            yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
            img[y0:y1 + 1, x0:x1 + 1][mask] = color

        def draw_circle_outline(cx, cy, r, color, thickness=2):
            x0, x1 = max(0, cx - r), min(width - 1, cx + r)
            y0, y1 = max(0, cy - r), min(height - 1, cy + r)

            if x1 < x0 or y1 < y0:
                return

            yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
            dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
            mask = (dist2 <= r ** 2) & (dist2 >= (r - thickness) ** 2)
            img[y0:y1 + 1, x0:x1 + 1][mask] = color

        def draw_line(x0, y0, x1, y1, color, thickness=1):
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            dx = abs(x1 - x0)
            dy = -abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx + dy

            x, y = x0, y0

            while True:
                draw_filled_circle(x, y, thickness, color)

                if x == x1 and y == y1:
                    break

                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x += sx
                if e2 <= dx:
                    err += dx
                    y += sy

        def draw_rectangle(x0, y0, x1, y1, fill, outline, thickness=2):
            left, right = sorted([x0, x1])
            top, bottom = sorted([y0, y1])

            left = max(0, min(width - 1, left))
            right = max(0, min(width - 1, right))
            top = max(0, min(height - 1, top))
            bottom = max(0, min(height - 1, bottom))

            img[top:bottom + 1, left:right + 1] = fill

            for t in range(thickness):
                if top + t <= bottom - t and left + t <= right - t:
                    img[top + t, left:right + 1] = outline
                    img[bottom - t, left:right + 1] = outline
                    img[top:bottom + 1, left + t] = outline
                    img[top:bottom + 1, right - t] = outline

        def draw_drone(cx, cy, radius, color):
            black = np.array([20, 20, 20], dtype=np.uint8)

            arm = max(8, int(radius * 2.5))
            rotor_r = max(3, int(radius * 0.65))
            line_thick = max(1, radius // 4)

            # Arm outlines
            draw_line(cx - arm, cy, cx + arm, cy, black, line_thick + 1)
            draw_line(cx, cy - arm, cx, cy + arm, black, line_thick + 1)

            # Arms
            draw_line(cx - arm, cy, cx + arm, cy, color, line_thick)
            draw_line(cx, cy - arm, cx, cy + arm, color, line_thick)

            # Rotors
            rotor_centers = [
                (cx - arm, cy),
                (cx + arm, cy),
                (cx, cy - arm),
                (cx, cy + arm),
            ]

            for rx, ry in rotor_centers:
                draw_circle_outline(rx, ry, rotor_r + 1, black, thickness=2)
                draw_circle_outline(rx, ry, rotor_r, color, thickness=2)

            # Body
            draw_filled_circle(cx, cy, radius + 1, black)
            draw_filled_circle(cx, cy, radius, color)

        # ------------------------------------------------------------
        # Background grid
        # ------------------------------------------------------------
        grid_color = np.array([225, 225, 225], dtype=np.uint8)
        for x in np.linspace(xmin, xmax, 9):
            px, _ = world_to_pixel(x, ymin)
            img[:, max(0, min(width - 1, px))] = grid_color

        for y in np.linspace(ymin, ymax, 9):
            _, py = world_to_pixel(xmin, y)
            img[max(0, min(height - 1, py)), :] = grid_color

        # ------------------------------------------------------------
        # Draw goal boxes
        # ------------------------------------------------------------
        for goal_idx, point in enumerate(box_pos):
            x, y = float(point[0]), float(point[1])

            p0 = world_to_pixel(x - hx, y - hy)
            p1 = world_to_pixel(x + hx, y + hy)

            fill = np.array([170, 225, 170], dtype=np.uint8)
            outline = np.array([40, 120, 40], dtype=np.uint8)

            # Optional: highlight the box the target is currently pursuing.
            if goal_idx == getattr(self, "current_target_box_idx", -1):
                outline = np.array([190, 60, 30], dtype=np.uint8)

            draw_rectangle(
                p0[0], p0[1],
                p1[0], p1[1],
                fill=fill,
                outline=outline,
                thickness=3,
            )

        # ------------------------------------------------------------
        # Draw drones
        # ------------------------------------------------------------
        base_altitude = max(float(getattr(self, "base_altitude", 0.5)), 1e-6)

        for i, pos in enumerate(drone_pos):
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            px, py = world_to_pixel(x, y)

            # Higher altitude appears slightly smaller.
            altitude_scale = 1.0 - 0.18 * ((z - base_altitude) / base_altitude)
            altitude_scale = float(np.clip(altitude_scale, 0.65, 1.25))

            radius = max(5, int(9 * altitude_scale))

            if i == self.target_idx:
                color = np.array([220, 35, 35], dtype=np.uint8)
            else:
                color = np.array([35, 90, 220], dtype=np.uint8)

            draw_drone(px, py, radius, color)

        return img

    def decompose_reward(
        self,
        joint_reward,
        obs,
    ):
        dist_norm = 2

        target_obs = obs['target']
        adversary_pos = target_obs[self.obs_map['self_pos']]
        team_pos = (
            target_obs[self.obs_map['target_obs']]
            + np.tile(adversary_pos, self.n_agents)
        ).reshape(-1, 3)
        box_pos = (
            target_obs[self.obs_map['target']][:-3]
            + np.tile(adversary_pos, int(len(target_obs[self.obs_map['target']]) / 3 - 1))
        ).reshape(-1, 3)

        repulsion_radius = self.controller_cfg['adversary_repulsion_radius']
        repulsion_gain = self.controller_cfg['adversary_repulsion_gain']
        attraction_gain = self.controller_cfg['adversary_attraction_gain']

        target_pos = box_pos[self.current_target_box_idx]
        attraction = float(attraction_gain) * (target_pos - adversary_pos)
        attraction_norm = np.linalg.norm(attraction)

        forces = {}
        for agent, protector in zip(self.agents, team_pos):
            diff = adversary_pos - protector
            diff[2] = 0.0

            dist = np.linalg.norm(diff) + 1e-6

            if dist <= repulsion_radius:
                forces[agent] = float(repulsion_gain) * diff / (dist ** 2)
            else:
                forces[agent] = np.zeros(3, dtype=np.float32)

        force_norms = {agent: np.linalg.norm(force) for agent, force in forces.items()}
        total_force_norm = sum(force_norms.values()) + attraction_norm + 1e-6

        decomposed_reward = {}
        for agent in self.agents:
            self_pos = obs[agent][self.obs_map['self_pos']]
            neighbor_dists = [
                np.linalg.norm(self_pos - obs[other][self.obs_map['self_pos']])
                for other in self.agents if other != agent
            ]
            nearest_dist = min(neighbor_dists) if neighbor_dists else 0.0
            nn_factor = np.clip(nearest_dist, 0.0, dist_norm) / dist_norm

            force_share = force_norms[agent] / total_force_norm
            decomposed_reward[agent] = force_share * nn_factor * joint_reward

        return decomposed_reward