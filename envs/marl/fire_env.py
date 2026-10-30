
import itertools
import gymnasium
import matplotlib
import numpy as np
from copy import deepcopy
from typing import Optional
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path

matplotlib.use("Agg")

from controllers.fire_control import probabilistic_fire_controller, extinguish_controller, nearest_cell_dist, inward_band_cost

GRID_SIZE = 60

class DroneFireEnv(gymnasium.Env):

    def __init__(
        self,
        drone_names: list[str],
        fire_controller,
        drone_start_area: list[int] = [0, 0],
        controller_kwargs: Optional[dict] = None,
        fire_controller_kwargs: Optional[dict] = None,
        reward_kwargs: Optional[dict] = None,
        observation_range: int = 30,
        local_observations: bool = False,
        warm_up_steps: int = 10,
        noise: bool = True,
        local_radius: int = 5,
        max_episode_length: int = 300,
        drone_start_noise: int = 3,
        ignition_points=None,
        n_random_fires: int = 1,
        extinguish_radius: int = 5,
        move_speed: int = 1,
    ):
        super().__init__()

        self.agents = list(drone_names)
        self.n_agents = len(self.agents)
        self.full_agent_list = self.agents + ['target']

        self.warm_up_steps = warm_up_steps
        self.noise = noise
        self.observation_range = observation_range
        self.local_observations = local_observations
        self.local_radius = int(local_radius)
        self.max_episode_length = int(max_episode_length)

        self.reward_kwargs = reward_kwargs if reward_kwargs is not None else {}
        self.reward_cfg = self.reward_kwargs
        self.controller_kwargs = controller_kwargs if controller_kwargs is not None else {}
        self.ignition_points = None if ignition_points is None else np.asarray(ignition_points, dtype=np.int32)
        self.drone_start_area = np.asarray(drone_start_area, dtype=np.int32)

        self.env = DroneFireSim(
            drone_names=drone_names,
            drone_start_area=drone_start_area,
            fire_controller=fire_controller,
            fire_controller_kwargs=fire_controller_kwargs,
            drone_start_noise=drone_start_noise,
            ignition_points=ignition_points,
            n_random_fires=n_random_fires,
            extinguish_radius=extinguish_radius,
            move_speed=move_speed,
        )

        self._step = 0
        self.obs = None
        self.obs_map = None

    def _observation_space(self, agent):

        return spaces.Box(
            low=0,
            high=4,
            shape=((self.observation_range * 2 + 1),(self.observation_range * 2 + 1), 1),
            dtype=np.int8,
        )

    def _action_space(self, agent):
        return spaces.Discrete(len(self.env.ACTION_LIBRARY))

    def _sample_center_quarter_ignition_points(self, n_points=3):
        lo, hi = int(GRID_SIZE*0.45), int(GRID_SIZE*0.55)
        pts = []
        while len(pts) < n_points:
            pt = np.array([np.random.randint(lo, hi), np.random.randint(lo, hi)], dtype=np.int32)
            if not any(np.array_equal(pt, p) for p in pts):
                pts.append(pt)
        return np.asarray(pts, dtype=np.int32)

    def _sample_drone_start_area(self, ignition_points):
        center = ignition_points.mean(axis=0)
        for _ in range(100):
            angle = np.random.uniform(0.0, 2.0 * np.pi)
            radius = np.random.uniform(10.0, 15.0)
            candidate = np.round(center + radius * np.array([np.cos(angle), np.sin(angle)])).astype(np.int32)
            if 0 <= candidate[0] < GRID_SIZE and 0 <= candidate[1] < GRID_SIZE:
                return candidate
        return np.clip(np.round(center + np.array([15.0, 15.0])).astype(np.int32), 0, GRID_SIZE - 1)

    def reset(self, *, seed=None, options=None):
        self._step = 0

        if seed is not None:
            np.random.seed(seed)

        self.ignition_points = self._sample_center_quarter_ignition_points()
        self.drone_start_area = self._sample_drone_start_area(self.ignition_points)

        obs = self.env.reset(
            drone_start_area=self.drone_start_area,
            ignition_points=self.ignition_points,
        )
        self.obs = self._compute_obs(obs)

        warm_up_noise = 0 #+ np.random.randint(0, 1) #-5-6
        for _ in range(self.warm_up_steps + warm_up_noise):
            actions = {agent: 0 for agent in self.agents}
            obs, rew, done, info = self.env.step(actions)

        self.info = {'__common__':{}}
        self.info['__common__']['decomposed_obs'] = {agent: deepcopy(self.obs[agent]) for agent in self.agents}
        pre_combine_obs = self.obs
        self.obs = self._combine_obs(self.obs)
        for agent in self.agents:
            self.info['__common__']['decomposed_obs'][agent]['noise'] = pre_combine_obs[agent]['noise']

        return deepcopy(self.obs), deepcopy(self.info)

    def step(self, action_dict: dict):
        self._step += 1

        #action_dict = encirclement_controller(self.obs)
        #action_dict = extinguish_controller(self.info['__common__']['decomposed_obs'], **self.controller_kwargs)
        filtered_actions = {}
        for agent in self.agents:
            action = action_dict.get(agent, 0)
            filtered_actions[agent] = self.boundary_safe_action(agent, action)

        controller_metrics = self.behavior_cloning_reward(action_dict)

        obs, rew, done, info = self.env.step(filtered_actions)

        controller_metrics['extinguished'] = rew
        fire_contained = np.any(self.env.fire_state == self.env.RED) and not self._fire_can_spread()
        controller_metrics['fire_contained'] = fire_contained

        truncations = self._compute_truncation()
        terminations = done or fire_contained
        #reward = self._compute_reward()

        self.info = {'__common__':{}}
        self.obs = self._compute_obs(obs)
        rew = self._compute_reward(controller_metrics)
        reward = {agent : rew for agent in self.agents}
        reward['target'] = 0.0
        truncations = {agent : truncations for agent in self.full_agent_list}
        terminations = {agent : terminations for agent in self.full_agent_list}
        self.info['__common__']['decomposed_obs'] = {agent: deepcopy(self.obs[agent]) for agent in self.agents}
        pre_combine_obs = self.obs
        self.obs = self._combine_obs(self.obs)
        for agent in self.agents:
            self.info['__common__']['decomposed_obs'][agent]['noise'] = pre_combine_obs[agent]['noise']

        return deepcopy(self.obs), reward, terminations, truncations, deepcopy(self.info)

    def _compute_obs(self, obs):
        obs_dict = {}

        for agent in self.agents:
            agent_pos = obs['drones'][agent]
            partial_obs = self._local_fire_window(obs['fire'], agent_pos)
            team = []

            for agent2 in self.agents:
                if agent2 == agent:
                    continue

                team.append(obs['drones'][agent2]-agent_pos)

            agent_obs = partial_obs
            obs_dict[agent] = {}
            obs_dict[agent]['fire'] = partial_obs
            obs_dict[agent]['team'] = team
            obs_dict[agent]['self_pos'] = agent_pos


        obs_dict['target'] = self.env.fire_state

        return obs_dict

    def _combine_obs(self, obs):
        new_obs = {}
        window_size = self.observation_range * 2 + 1

        for agent in self.agents:
            partial_obs = obs[agent]['fire']

            obs[agent]['noise'] = []

            for team_pos in obs[agent]['team']:
                if self.noise:
                    noise_r = np.random.randint(-7, 7)
                    noise_c = np.random.randint(-7, 7)
                else:
                    noise_r = 0
                    noise_c = 0
                obs[agent]['noise'].append((noise_r, noise_c))
                rel_r = team_pos[0] + noise_r
                rel_c = team_pos[1] + noise_c

                if abs(rel_r) <= self.observation_range and abs(rel_c) <= self.observation_range:
                    cross_r = rel_r + self.observation_range
                    cross_c = rel_c + self.observation_range
                    for rr, cc in [(cross_r, cross_c), (cross_r - 1, cross_c), (cross_r + 1, cross_c), (cross_r, cross_c - 1), (cross_r, cross_c + 1)]:
                        if 0 <= rr < window_size and 0 <= cc < window_size:
                            partial_obs[rr, cc] = self.env.BLUE

            new_obs[agent] = partial_obs[..., None]

        new_obs['target'] = obs['target']
        return new_obs

    def behavior_cloning_reward(self, action_dict: dict) -> dict:
        controller_actions = extinguish_controller(self.info['__common__']['decomposed_obs'])
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

        return {
            'controller_action_error': controller_action_error,
            'controller_action_match': controller_action_match,
            'controller_action_reward': controller_action_match,
        }

    def _compute_reward(
        self,
        controller_metrics,
    ):
        green_count = np.sum(self.env.fire_state == self.env.GREEN)
        non_green_ratio = 1 - green_count / GRID_SIZE**2

        red_count = np.sum(self.env.fire_state == self.env.RED)
        red_ratio = red_count / GRID_SIZE**2

        reward = non_green_ratio * self.reward_kwargs['green_scale']
        reward -= red_ratio * self.reward_kwargs['red_scale']
        reward += controller_metrics['extinguished'] * self.reward_kwargs['extinguish_scale']
        reward += controller_metrics['controller_action_reward'] * self.reward_kwargs['bc_scale']
        reward -= self._proximity_penalty_count() * self.reward_kwargs['proximity_penalty']

        if np.all(self.env.fire_state != self.env.GREEN):
            return reward + self.reward_kwargs['lose_penalty']
        if np.all(self.env.fire_state != self.env.RED):
            return reward + self.reward_kwargs['extinguish_reward']
        if controller_metrics.get('fire_contained', False):
            return reward + self.reward_kwargs['extinguish_reward']

        return reward

    def _fire_can_spread(self) -> bool:
        fire_state = self.env.fire_state
        red_mask = fire_state == self.env.RED
        green_mask = fire_state == self.env.GREEN
        rows, cols = fire_state.shape

        padded_red = np.pad(red_mask, 1, mode="constant", constant_values=False)
        neighbor_offsets = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]

        adjacent_to_red = np.zeros_like(green_mask)
        for dr, dc in neighbor_offsets:
            adjacent_to_red |= padded_red[1 + dr: 1 + dr + rows, 1 + dc: 1 + dc + cols]

        return bool(np.any(green_mask & adjacent_to_red))

    def _proximity_penalty_count(self, min_dist: float = 1.0) -> int:
        positions = [self.env.drone_states[agent] for agent in self.agents]
        count = 0

        for i, pos_i in enumerate(positions):
            for j, pos_j in enumerate(positions):
                if i == j:
                    continue
                if np.linalg.norm(pos_i - pos_j) <= min_dist:
                    count += 1
                    break

        return count

    def _compute_truncation(self):

        if self._step >= self.max_episode_length:
            return True
        else:
            return False

    def _local_fire_window(self, fire_state, agent_pos):
        r, c = agent_pos
        radius = self.observation_range

        padded_fire = np.pad(
            fire_state,
            pad_width=radius,
            mode="constant",
            constant_values=self.env.WHITE,
        )

        rp = r + radius
        cp = c + radius

        window = padded_fire[
            rp - radius: rp + radius + 1,
            cp - radius: cp + radius + 1,
        ]

        return window.astype(np.float32)

    def boundary_safe_action(self, agent_id: str, action: int):
        action = int(action)

        if action not in self.env.ACTION_LIBRARY:
            return 0

        if action == 9:
            return 9

        pos = self.env.drone_states[agent_id]
        next_pos = pos + self.env.ACTION_LIBRARY[action]

        r, c = next_pos

        if r < 0 or r >= GRID_SIZE:
            return 0

        if c < 0 or c >= GRID_SIZE:
            return 0

        return action

    def get_obs(self):
        return deepcopy(self.obs)

    def seed(self, seed: int):
        self.seed_value = seed
        np.random.seed(seed)

    def team_error(
        self, 
        estimate: np.ndarray,
        agent: str
    ):

        # Reshape to (N, 3, 2)
        pred = np.asarray(estimate).reshape(-1, 3, 2)
        target = np.asarray(
            self.info['__common__']['decomposed_obs'][agent]['team']
        ).reshape(-1, 3, 2)

        # each team estimate picks its own teammate-order assignment
        # independently, so pred and target don't have to agree on slot identity
        errors = [
            np.linalg.norm(pred - target[:, perm, :], axis=2).sum(axis=1)
            for perm in itertools.permutations(range(3))
        ]

        # take minimum per team estimate, then sum across estimates
        error = np.stack(errors, axis=0).min(axis=0).sum()
        return error

    def render_rgb(self):
        '''
        render the current fire grid + drone positions in the same visual
        style as overlay_plot_fire.py's ground-truth panel (DRONE_COLORS
        drone icons, forest-green/burning-red/burned-black grid with cell
        gridlines via _draw_fire_grid/_draw_drone_icon) instead of the
        plain COLORS pixel dump + blue-plus markers this used to draw, so
        eval gifs/videos read the same way as the static overlay plots

        drones are colored in sorted-name order (matching
        overlay_plot_fire.load_episode_positions' `sorted(...)` agent
        ordering) rather than self.agents' insertion order, so a given
        drone always gets the same DRONE_COLORS entry here as it does in
        the overlay plots
        '''
        return render_fire_frame(self.env.fire_state, self.env.drone_states)

    def decompose_reward(
        self,
        joint_reward,
        obs,
    ):
        '''
        obs:dict
            agent -> {'self_pos': ...} (as in info['__common__']['decomposed_obs'])
            plus a 'target' key holding the full (GRID_SIZE, GRID_SIZE) fire
            grid (as in info['target']) - the same shape _compute_drone_slots
            accepts for its centralized-obs branch
        '''
        min_slot_dist = 2 #self.controller_kwargs.get('min_slot_dist', 6)

        positions = np.array([obs[agent]['self_pos'] for agent in self.agents], dtype=float)
        grid = np.asarray(obs['target'])
        green_cells = np.argwhere(grid == DroneFireSim.GREEN).astype(float)
        black_cells = np.argwhere(grid == DroneFireSim.BLACK).astype(float)

        d_green = nearest_cell_dist(positions, green_cells)
        pixel_score = inward_band_cost(d_green)

        if len(black_cells) > 0:
            d_black = nearest_cell_dist(positions, black_cells)
            pixel_score += inward_band_cost(d_black)

        total_score = pixel_score.sum() + 1e-6

        decomposed_reward = {}
        for i, agent in enumerate(self.agents):
            neighbor_dists = [
                np.linalg.norm(positions[i] - positions[j])
                for j in range(len(self.agents)) if j != i
            ]
            nearest_dist = min(neighbor_dists) if neighbor_dists else 0.0
            nn_factor = np.clip(nearest_dist, 0.0, min_slot_dist) / min_slot_dist

            score_share = pixel_score[i] / total_score
            decomposed_reward[agent] = score_share * nn_factor * joint_reward

        return decomposed_reward

class DroneFireSim:
    GREEN, RED, BLACK, WHITE, BLUE = 0, 1, 2, 3, 4

    COLORS = np.array([
        [0, 255, 0],    # green
        [255, 0, 0],    # red
        [0, 0, 0],      # black
        [255, 255, 255], #white
        [0, 0, 255],    # blue (teammate marker)
    ], dtype=np.uint8)

    ACTION_LIBRARY = {
        0: np.array([0, 0]),     # stay
        1: np.array([-1, 0]),    # up
        2: np.array([1, 0]),     # down
        3: np.array([0, -1]),    # left
        4: np.array([0, 1]),     # right
        5: np.array([-1, -1]),   # up-left
        6: np.array([-1, 1]),    # up-right
        7: np.array([1, -1]),    # down-left
        8: np.array([1, 1]),     # down-right
        9: np.array([0, 0]),     #extinguish fire
    }

    def __init__(
        self,
        drone_names: list[str],
        drone_start_area: list[float],
        fire_controller,
        fire_controller_kwargs: Optional[dict] = None,
        drone_start_noise: int = 3,
        ignition_points=None,
        n_random_fires: int = 1,
        extinguish_radius: int = 5,
        move_speed: int = 1,
    ):
        self.drone_names = drone_names
        self.n_drones = len(drone_names)
        self.drone_start_area = np.array(drone_start_area, dtype=np.int32)
        self.drone_start_noise = drone_start_noise
        self.fire_controller = fire_controller
        self.fire_controller_kwargs = fire_controller_kwargs if fire_controller_kwargs is not None else {}
        self.extinguish_radius = extinguish_radius
        self.move_speed = move_speed
        self.ACTION_LIBRARY = {
            k: (v * move_speed if k != 9 else v)
            for k, v in DroneFireSim.ACTION_LIBRARY.items()
        }

        self.ignition_points = ignition_points
        self.n_random_fires = n_random_fires

        self.drone_states = {}
        self.fire_state = None
        self.t = 0

        self.reset()

    def initialize_fire(self):
        fire = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

        if self.ignition_points is None:
            points = np.random.randint(
                0,
                GRID_SIZE,
                size=(self.n_random_fires, 2),
            )
        else:
            points = np.asarray(self.ignition_points, dtype=np.int32)

            if points.ndim == 1:
                points = points[None, :]

        points = np.clip(points, 0, GRID_SIZE - 1)
        fire[points[:, 0], points[:, 1]] = self.RED

        return fire

    def reset(
        self,
        drone_start_area: Optional[list[float]] = None,
        ignition_points: Optional[list[list[float]]] = None,
    ):
        if ignition_points is not None:
            self.ignition_points = np.asarray(ignition_points, dtype=np.int32)
            if self.ignition_points.ndim == 1:
                self.ignition_points = self.ignition_points[None, :]

        if drone_start_area is not None:
            self.drone_start_area = np.asarray(drone_start_area, dtype=np.int32)

        min_drone_dist = 2.0
        placed = []
        for drone in self.drone_names:
            for _ in range(100):
                noise = np.random.randint(-self.drone_start_noise, self.drone_start_noise + 1, size=2)
                candidate = np.clip(self.drone_start_area + noise, 0, GRID_SIZE - 1)
                if all(np.linalg.norm(candidate - p) >= min_drone_dist for p in placed):
                    break
            self.drone_states[drone] = candidate
            placed.append(candidate)

        self.fire_state = self.initialize_fire()
        self.t = 0

        return self.state()

    def state(self):
        return {
            "t": self.t,
            "fire": self.fire_state.copy(),
            "drones": {k: v.copy() for k, v in self.drone_states.items()},
        }

    def extinguish_fire(
        self,
        drone,
        max_extinguish_prob=1.0,
        decay_rate=0.05,
    ):
        pos = self.drone_states[drone]
        r, c = pos

        rows, cols = self.fire_state.shape

        r_min = max(0, r - self.extinguish_radius)
        r_max = min(rows, r + self.extinguish_radius + 1)
        c_min = max(0, c - self.extinguish_radius)
        c_max = min(cols, c + self.extinguish_radius + 1)

        rr, cc = np.meshgrid(
            np.arange(r_min, r_max),
            np.arange(c_min, c_max),
            indexing="ij",
        )
        distance = np.sqrt((rr - r) ** 2 + (cc - c) ** 2)

        window = self.fire_state[r_min:r_max, c_min:c_max]
        in_range_red = (window == self.RED) & (distance <= self.extinguish_radius)

        extinguish_prob = max_extinguish_prob * np.exp(-decay_rate * distance)
        trial = np.random.random(distance.shape) < extinguish_prob

        newly_extinguished = in_range_red & trial
        window[newly_extinguished] = self.BLACK

        return int(np.sum(newly_extinguished))

    def move_drones(self, actions):
        extinguish_count = 0

        for drone in self.drone_names:

            discrete_action = int(actions[drone])
            vec_action = self.ACTION_LIBRARY[discrete_action]

            self.drone_states[drone] = np.clip(
                self.drone_states[drone] + vec_action,
                0,
                GRID_SIZE - 1,
            )

            if actions[drone] == 9:
                extinguish_count += self.extinguish_fire(drone)

        return extinguish_count

    def step(self, actions):
        extinguish_count = self.move_drones(actions)
        drone_positions = np.array(list(self.drone_states.values()), dtype=float)
        self.fire_state = self.fire_controller(
            self.fire_state,
            drone_positions=drone_positions,
            **self.fire_controller_kwargs,
        ).astype(np.int8)
        self.t += 1

        if np.all(self.fire_state != self.GREEN) or np.all(self.fire_state != self.RED):
            done = True
        else:
            done = False

        return self.state(), extinguish_count, done, None

    def image(self):
        return self.COLORS[self.fire_state]


DRONE_COLORS = ['#1f3fae', '#2f6fc9', '#4a93cf', '#6cb5d9']
UNBURNED_COLOR = '#1e4d2e'
BURNING_COLOR = '#a4231f'
BURNED_COLOR = '#101010'
GRID_LINE_COLOR = 'black'

FIRE_CMAP = ListedColormap([UNBURNED_COLOR, BURNING_COLOR, BURNED_COLOR])
FIRE_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], FIRE_CMAP.N)

def _draw_fire_grid(ax, grid, cell_size: float = 1.0):

    extent = (-0.5, GRID_SIZE - 0.5, GRID_SIZE - 0.5, -0.5)
    ax.imshow(grid, cmap=FIRE_CMAP, norm=FIRE_NORM, interpolation='nearest', extent=extent, zorder=1)

    ax.set_xticks(np.arange(-0.5, GRID_SIZE, cell_size), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_SIZE, cell_size), minor=True)
    ax.grid(which='minor', color=GRID_LINE_COLOR, linewidth=0.32, alpha=0.5, zorder=2)
    ax.tick_params(which='minor', length=0)
    ax.set_xticks([])
    ax.set_yticks([])


def _rounded_x_path(center, half_width: float, arm_length: float, corner_radius: float):

    half_width = min(half_width, arm_length * 0.9)
    corner_radius = min(corner_radius, half_width * 0.95, (arm_length - half_width) * 0.95)

    raw_points = [
        (half_width, arm_length), (half_width, half_width), (arm_length, half_width),
        (arm_length, -half_width), (half_width, -half_width), (half_width, -arm_length),
        (-half_width, -arm_length), (-half_width, -half_width), (-arm_length, -half_width),
        (-arm_length, half_width), (-half_width, half_width), (-half_width, arm_length),
    ]
    concave_idxs = {1, 4, 7, 10}

    theta = np.radians(45)
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    points = [rotation @ np.array(p) for p in raw_points]

    n = len(points)
    verts = []
    codes = []

    for i, p in enumerate(points):
        if i in concave_idxs:
            dir_in = points[i - 1] - p
            dir_in /= np.linalg.norm(dir_in)
            dir_out = points[(i + 1) % n] - p
            dir_out /= np.linalg.norm(dir_out)

            entry = p + dir_in * corner_radius
            exit_ = p + dir_out * corner_radius

            codes.append(Path.MOVETO if not codes else Path.LINETO)
            verts.append(entry)
            codes.append(Path.CURVE3)
            verts.append(p)
            codes.append(Path.CURVE3)
            verts.append(exit_)
        else:
            codes.append(Path.MOVETO if not codes else Path.LINETO)
            verts.append(p)

    verts.append(verts[0])
    codes.append(Path.CLOSEPOLY)

    verts = np.array(verts) + np.asarray(center)
    return Path(verts, codes)


def _draw_drone_icon(ax, center, color, size: float = 1.1, alpha: float = 1.0, zorder: float = 4,
                      rotor_radius: float = 0.26, rotor_linewidth: float = 1.8):

    body_reach = size * 0.36
    body_half_width = size * 0.17
    body_corner_radius = size * 0.3
    arm_length = size * 0.55
    arm_linewidth = max(size * 0.9, 1.0)

    body_path = _rounded_x_path(center, body_half_width, body_reach, body_corner_radius)
    ax.add_patch(PathPatch(body_path, facecolor=color, edgecolor='none', alpha=alpha, zorder=zorder))

    cx, cy = center
    for angle_deg in (45, 135, 225, 315):
        angle = np.radians(angle_deg)
        dx, dy = np.cos(angle), np.sin(angle)
        start = (cx + dx * body_reach, cy + dy * body_reach)
        tip = (cx + dx * arm_length, cy + dy * arm_length)

        ax.plot([start[0], tip[0]], [start[1], tip[1]], color=color, linewidth=arm_linewidth,
                 alpha=alpha, zorder=zorder, solid_capstyle='round')
        ax.add_patch(Circle(tip, rotor_radius, facecolor='none', edgecolor=color,
                             linewidth=rotor_linewidth, alpha=alpha, zorder=zorder + 0.1))


def render_fire_frame(fire_state, drone_states: dict, bounds: Optional[tuple] = None):
    '''
    draw a fire grid and drone icons (see DroneFireEnv.render_rgb) to an RGB frame

    fire_state: (GRID_SIZE, GRID_SIZE) grid of DroneFireSim cell values
    drone_states: drone name -> (row, col)
    bounds: optional (col_min, col_max, row_min, row_max) view in cell coordinates; when
        None the whole grid is shown
    '''
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

    _draw_fire_grid(ax, fire_state)

    for idx, drone in enumerate(sorted(drone_states)):
        row, col = drone_states[drone]
        color = DRONE_COLORS[idx % len(DRONE_COLORS)]
        _draw_drone_icon(ax, (col, row), color, zorder=4)

    if bounds is None:
        bounds = (-0.5, GRID_SIZE - 0.5, -0.5, GRID_SIZE - 0.5)
    col_min, col_max, row_min, row_max = bounds
    ax.set_xlim(col_min, col_max)
    ax.set_ylim(row_max, row_min)
    ax.set_aspect('equal')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)

    return frame


def render_fire_with_drones(env, save_path="test.png"):
    img = env.image().copy()

    # Overlay drones in blue
    for drone_name, pos in env.drone_states.items():
        r, c = pos
        img[r, c] = np.array([0, 0, 255], dtype=np.uint8)

        # Optional: make drone easier to see with a small 3x3 blue block
        r0, r1 = max(0, r - 1), min(GRID_SIZE, r + 2)
        c0, c1 = max(0, c - 1), min(GRID_SIZE, c + 2)
        img[r0:r1, c0:c1] = np.array([0, 0, 255], dtype=np.uint8)

    plt.imsave(save_path, img)
