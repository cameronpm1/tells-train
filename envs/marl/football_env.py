
import sys
import math
import numpy as np
import gymnasium as gym
from pathlib import Path
from copy import deepcopy
from gymnasium import spaces
from PIL import Image, ImageDraw

from controllers.football_control import compute_rondo_actions

THIS_FILE = Path(__file__).resolve()
SCENARIO_DIR = THIS_FILE.parent
PROJECT_ROOT = THIS_FILE.parents[2]
FOOTBALL_REPO = PROJECT_ROOT / "external" / "football"

if str(FOOTBALL_REPO) not in sys.path:
    sys.path.insert(0, str(FOOTBALL_REPO))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gfootball.scenarios import *
import gfootball.env as football_env

PROJECT_ROOT = Path(__file__).resolve()
FOOTBALL_REPO = PROJECT_ROOT / "external" / "football"

sys.path.insert(0, str(FOOTBALL_REPO))

'''
compute team initial positions
'''

rx = 0.21 #0.28 #0.21
ry = 0.135 #0.18 #0.135
n_passers = 5

AGENT_ANCHORS = []
for i in range(5):
    theta = 2.0 * math.pi * i / n_passers
    x = rx * math.cos(theta)
    y = ry * math.sin(theta)
    AGENT_ANCHORS.append((x, y))

AR = 2.38


def _distance_point_to_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    '''
    shortest distance from point p to the segment ab
    '''
    ab = b - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq < 1e-12:
        return float(np.linalg.norm(p - a))

    t = np.clip(np.dot(p - a, ab) / ab_len_sq, 0.0, 1.0)
    closest = a + t * ab

    return float(np.linalg.norm(p - closest))


def register_file_as_grf_scenario(scenario_name: str):
    """
    Registers this current Python module as:
        gfootball.scenarios.<scenario_name>

    This allows GRF to find build_scenario() even though the scenario is
    defined in the same file as the environment wrapper.
    """

    module_name = f"gfootball.scenarios.{scenario_name}"
    current_module = sys.modules[__name__]

    if not hasattr(current_module, "build_scenario"):
        raise RuntimeError(
            f"{__name__} does not define build_scenario(). "
            "Make sure build_scenario() is defined before creating the env."
        )

    sys.modules[module_name] = current_module

class CirclePass5v1Env(gym.Env):
    """
    Multi-agent 5v1 rondo-style drill.

    Agents:
        left_0 ... left_4: passing team
        right_0: middle defender

    Objective:
        left team keeps possession and passes around the circle.
        right defender tries to steal the ball.
    """

    def __init__(
            self, 
            agent_list: list[str],
            full_agent_list: list[str],
            adversary_controller,
            max_episode_length: int,
            reward_kwargs: dict,
            controller_kwargs: dict,
            render=False
        ):
        super().__init__()

        self.agents = agent_list
        self.n_agents = len(self.agents)
        self.adversary_controller = adversary_controller
        self.max_episode_length = max_episode_length
        self.reward_cfg = reward_kwargs
        self.controller_kwargs = controller_kwargs
        self.full_agent_list = full_agent_list
        self._step = 0

        self.env = self._init_scenario(
            scenario_name='academy_circle_pass_5v1', 
            render=render
        )

        self.prev_ball_owned_player = -1
        self.pass_count = 0

        self.action_library = {
            0: 14,  # stay / release_direction
            1: 1,   # move left
            2: 2,   # move top_left
            3: 3,   # move top
            4: 4,   # move top_right
            5: 5,   # move right
            6: 6,   # move bottom_right
            7: 7,   # move bottom
            8: 8,   # move bottom_left
            9: 11,   # long_pass
            10: 13,
        }
        ball_owner_end = 10 + 2 * (len(self.agents) - 1) + len(self.agents) + 1

        self.obs_map = {
            'self': slice(0, 4),
            'target': slice(4, 10),
            'team': slice(10, 10 + 2 * (len(self.agents) - 1)),
            'target_obs': slice(10, 10 + 2 * (len(self.agents))),
            'self_pos': slice(2, 4),
            'self_vel': slice(0, 2),
            'self_anchor': slice(4, 6),
            'ball_owner': slice(10 + 2 * (len(self.agents) - 1), ball_owner_end),
            'ball_pos': slice(ball_owner_end, ball_owner_end + 2),
            'target_ball': slice(4, 6),
            'target_pos': slice(6, 8),
            'target_vel': slice(8, 10),
            'target_ball_owned' : slice(10 + 2 * (len(self.agents)), 10 + 2 * (len(self.agents)) + 1),
            'state_space':(ball_owner_end + 2,),
        }

    def _init_scenario(self, scenario_name, render):
        register_file_as_grf_scenario(scenario_name)

        env = football_env.create_environment(
            env_name=scenario_name,
            representation="raw",
            rewards="scoring",
            render=render,
            number_of_left_players_agent_controls=self.n_agents+1,
            number_of_right_players_agent_controls=2,
        )

        return env

    def _observation_space(self, agent):
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_map['state_space']),
            dtype=np.float32,
        )

    def _action_space(self, agent):
        return spaces.Discrete(len(self.action_library))

    def reset(self, *, seed=None, options=None):

        self._step = 0
        self.prev_ball_owned_team = -1
        self.prev_ball_owned_player = -1
        self.pass_count = 0

        obs = self.env.reset()

        self.raw_obs = obs
        obs_no_gk = np.delete(obs,[0,self.n_agents+1])
        new_obs = self._convert_obs(obs_no_gk, first=True)
        self.obs = new_obs

        return deepcopy(self.obs), {'__common__': {agent: {} for agent in self.agents}}

    def step(
        self, 
        actions_dict: dict
    ):

        self._step += 1
        
        actions = []
        for agent in self.agents:
            actions.append(actions_dict[agent])
        #controller_actions = compute_rondo_actions(self.obs,self.obs_map)
        #actions = [controller_actions[agent] for agent in self.agents]
        controller_metrics = self.behavior_cloning_reward(actions_dict)
        if self._step == 1 and self.controller_kwargs['fast']:
            adversary_action = 10 #turn on sprint 
        elif self._step == 1:
            adversary_action = 0
        else:
            adversary_action = self.adversary_controller(self.obs['target'],self.obs_map,self.controller_kwargs)
        actions = np.concatenate(([0],actions,[0,adversary_action]))
        actions = [self.action_library[a] for a in actions]

        obs, base_reward, done, raw_info = self.env.step(actions)

        self.raw_obs = obs
        self.oob = done

        obs_no_gk = np.delete(obs,0)
        obs_no_gk = np.delete(obs_no_gk,self.n_agents+1)
        new_obs = self._convert_obs(obs_no_gk)
        self.obs = new_obs
        team_reward = self._compute_team_reward(self.obs,controller_metrics)
        truncation = self._compute_truncation()
        termination = self._compute_termination()

        reward = {agent : team_reward for agent in self.agents}
        reward['target'] = 0.0
        truncations = {agent : truncation for agent in self.full_agent_list}
        terminations = {agent : termination or done for agent in self.full_agent_list}

        info = {}
        info['__common__'] = {
            agent: {'team_reward': team_reward, 'oob': self.oob, **dict(raw_info)}
            for agent in self.agents
        }

        terminations["__all__"] = all(terminations.values())
        truncations["__all__"] = all(truncations.values())

        return deepcopy(self.obs), reward, terminations, truncations, info

    def _convert_obs(self, obs, first=False):

        new_obs = {}
        ball_owner = None
        ball_pos_by_agent = {}
        for i,agent_obs in enumerate(obs):
            if i < self.n_agents:
                agent_name = self.agents[i]
                team = agent_obs['left_team'][1:]
                pos = team[i]
                if first:
                    vel = np.zeros((2,))
                else:
                    vel =  pos - self.obs[agent_name][self.obs_map['self_pos']]
                team = np.delete(team,i,axis=0).flatten() - np.tile(pos,self.n_agents-1)

                if agent_obs['ball_owned_team'] == 0 and agent_obs['ball_owned_player'] == i + 1:
                    ball_owner = i
                elif np.linalg.norm(agent_obs['ball'][0:2] - pos) < 0.03 and np.linalg.norm(agent_obs['ball_direction']) < 0.001:
                    ball_owner = i

                ball_pos_by_agent[agent_name] = agent_obs['ball'][0:2] - pos

                ball = deepcopy(AGENT_ANCHORS[i])

                target_pos = agent_obs['right_team'][1] - pos
                target_pos = np.concatenate((target_pos,agent_obs['right_team_direction'][1]))
            else:
                agent_name = 'target'
                # GRF's raw observations are ego-centric: for this right-side
                # player, 'left_team' is its own team and 'right_team' is the
                # opponent, both expressed in a frame mirrored (negated)
                # relative to what every left-side (passer) observation uses.
                # Negate on the way out so pos/team/ball line up with the
                # passers' frame instead of being its point-reflection.
                self_pos_ego = agent_obs['left_team'][1]
                pos = -self_pos_ego
                team = -agent_obs['right_team'][1:]
                ball_ego = -agent_obs['ball'][0:2]

                if agent_obs['ball_owned_team'] == 0 or np.linalg.norm(ball_ego - pos) < 0.01:
                    ball_owned = [1]
                else:
                    ball_owned = [0]

                if first:
                    vel = np.zeros((2,))
                else:
                    vel =  pos - self.obs[agent_name][self.obs_map['self_pos']]
                team = team.flatten() - np.tile(pos,self.n_agents)
                team = np.concatenate((team,ball_owned))

                if agent_obs['ball_owned_player'] == (i + 1):
                    ball_owner = i

                ball_pos = ball_ego - pos
                ball = ball_pos

                # NOTE: target_pos is self-referential (own position minus
                # itself) and always evaluates to [0, 0] - a separate,
                # pre-existing bug intentionally left untouched here.
                target_pos = self_pos_ego - self_pos_ego
                target_pos = np.concatenate((target_pos,agent_obs['left_team_direction'][1]))

            new_obs[agent_name] = np.concatenate((vel,pos,ball,target_pos,team))

        for agent_name in self.agents:
            ball_owner_array = np.zeros((len(self.agents) + 1))
            if ball_owner is not None:
                ball_owner_array[ball_owner] = 1
            new_obs[agent_name] = np.concatenate((new_obs[agent_name],ball_owner_array,ball_pos_by_agent[agent_name]))


        return new_obs

    def behavior_cloning_reward(self, action_dict: dict) -> dict:
        controller_actions = compute_rondo_actions(deepcopy(self.obs), self.obs_map)
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

    def _compute_team_reward(self, obs, controller_metrics):

        reward = 0
        reward += controller_metrics['controller_action_reward'] * self.reward_cfg['controller_match_reward']
        ball_owner = np.where((obs['agent0'][self.obs_map['ball_owner']]) == 1)[0]
        target_has_ball = obs['target'][self.obs_map['target_ball_owned']][0]
        if self.oob:
            return self.reward_cfg['oob_penalty']
        elif target_has_ball:
            return self.reward_cfg['target_steal_penalty']
        else:
            if len(ball_owner) > 0:
                if self.prev_ball_owned_player == -1:
                    self.prev_ball_owned_player = ball_owner[-1]
                else:
                    if self.prev_ball_owned_player != ball_owner[-1]:
                        if (self.prev_ball_owned_player - ball_owner[-1]) % 5 in (1, 5 - 1):
                            reward += self.reward_cfg['pass_reward']
                        else:
                            #double reward for non-neighbor passes
                            reward += 2*self.reward_cfg['pass_reward']
                        self.prev_ball_owned_player = ball_owner[-1]

        dev = np.linalg.norm(self.raw_obs[0]['left_team'][1:] - AGENT_ANCHORS, axis=1)
        dev_threshold = [1 if d > self.reward_cfg['deviation_threshold'] else 0 for d in dev]
        reward += (5 - sum(dev_threshold)) * self.reward_cfg['deviation_scale']

        return reward

    def _compute_truncation(self):

        if self._step >= self.max_episode_length:
            return True
        else:
            return False

    def _compute_termination(self):

        target_has_ball = self.obs['target'][self.obs_map['target_ball_owned']][0]
        if target_has_ball:
            return True
        else:
            return False

    def team_error(
        self, 
        estimate: np.ndarray,
        agent: str
    ):

        # Reshape to (N, 2, 3)
        pred = np.concatenate([estimate[self.obs_map['team']],estimate[self.obs_map['ball_pos']]], axis=0).reshape(-1,2)
        target = np.concatenate([self.obs[agent][self.obs_map['team']],self.obs[agent][self.obs_map['ball_pos']]], axis=0).reshape(-1,2)


        return np.linalg.norm(pred-target,axis=1).sum()

    def _ball_owner_index(self, obs):
        owner = np.where(obs['agent0'][self.obs_map['ball_owner']] == 1)[0]

        return int(owner[0]) if len(owner) > 0 else None

    def decompose_reward(
        self,
        joint_reward,
        obs,
        last_obs,
    ):
        '''
        per-agent contribution scale is 1.0 for whichever agent's possession
        of the ball just changed to (a completed pass, detected by the
        ball-owner one-hot differing between obs and last_obs), 0.5 for any
        other agent that is open - the adversary is more than open_threshold
        away from the segment between the ball and that agent - and 0.0
        otherwise
        '''
        open_threshold = 0.07

        current_owner = self._ball_owner_index(obs)
        previous_owner = self._ball_owner_index(last_obs)

        pass_receiver = None
        if (
            current_owner is not None
            and current_owner != previous_owner
            and current_owner < len(self.agents)
        ):
            pass_receiver = self.agents[current_owner]

        decomposed_reward = {}
        for agent in self.agents:
            if agent == pass_receiver:
                contribution_scale = 1.0
            else:
                ball_vec = obs[agent][self.obs_map['ball_pos']]
                target_vec = obs[agent][self.obs_map['target_pos']]
                adversary_dist = _distance_point_to_segment(np.zeros(2), ball_vec, target_vec)
                contribution_scale = 0.5 if adversary_dist > open_threshold else 0.0

            decomposed_reward[agent] = contribution_scale * joint_reward

        return decomposed_reward



    def render_rgb_old(self):
        """
        Returns the current GRF frame as an RGB uint8 image.

        Output:
            np.ndarray with shape (H, W, 3), dtype=np.uint8
        """

        frame = self.env.render(mode="rgb_array")
        frame = np.asarray(frame)

        # Drop alpha channel if present.
        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]

        return frame.astype(np.uint8)

    def render_rgb(self, width=1280, height=720, obs_index=0, bounds=None):
        '''
        bounds: optional (x_min, x_max, y_min, y_max) view in pitch coordinates, drawn edge to
        edge; when None the whole pitch is shown with a pixel margin around it
        '''

        obs_i = self.raw_obs[obs_index]

        left_team = np.asarray(obs_i["left_team"], dtype=np.float32)
        right_team = np.asarray(obs_i["right_team"], dtype=np.float32)

        left_dir = np.asarray(obs_i["left_team_direction"], dtype=np.float32)
        right_dir = np.asarray(obs_i["right_team_direction"], dtype=np.float32)

        left_roles = np.asarray(obs_i["left_team_roles"])
        right_roles = np.asarray(obs_i["right_team_roles"])

        ball = np.asarray(obs_i["ball"][:2], dtype=np.float32)

        pitch_x, pitch_y = 1.05, 0.45
        if bounds is None:
            x_min, x_max = -pitch_x, pitch_x
            y_min, y_max = -pitch_y, pitch_y
            margin = 40
        else:
            x_min, x_max, y_min, y_max = (float(b) for b in bounds)
            margin = 0

        img = Image.new("RGB", (width, height), (35, 120, 45))
        draw = ImageDraw.Draw(img)

        def to_px(p):
            x, y = float(p[0]), float(p[1])

            px = margin + (x - x_min) / (x_max - x_min) * (width - 2 * margin)
            py = margin + (y - y_min) / (y_max - y_min) * (height - 2 * margin)

            return int(px), int(py)

        def in_play(p):
            return x_min <= p[0] <= x_max and y_min <= p[1] <= y_max

        def draw_heading_arrow(p, v, color=(255, 255, 255)):
            if not in_play(p):
                return

            speed = np.linalg.norm(v)

            # If nearly stationary, do not draw a noisy arrow.
            if speed < 1e-5:
                return

            px, py = to_px(p)

            # Convert velocity direction to pixel direction.
            v_unit = v / speed

            arrow_len = 35
            dx = float(v_unit[0]) * arrow_len
            dy = float(v_unit[1]) * arrow_len

            x2 = px + dx
            y2 = py + dy

            draw.line(
                [(px, py), (x2, y2)],
                fill=color,
                width=4,
            )

            # Arrowhead.
            angle = math.atan2(dy, dx)
            head_len = 10
            head_angle = math.pi / 6

            x3 = x2 - head_len * math.cos(angle - head_angle)
            y3 = y2 - head_len * math.sin(angle - head_angle)

            x4 = x2 - head_len * math.cos(angle + head_angle)
            y4 = y2 - head_len * math.sin(angle + head_angle)

            draw.polygon(
                [(x2, y2), (x3, y3), (x4, y4)],
                fill=color,
            )

        def draw_player(p, v, color, label=None):
            if not in_play(p):
                return

            px, py = to_px(p)
            r = 12

            draw.ellipse(
                [px - r, py - r, px + r, py + r],
                fill=color,
                outline=(0, 0, 0),
                width=2,
            )

            draw_heading_arrow(p, v, color=(255, 255, 255))

            if label is not None:
                draw.text((px + 14, py - 10), str(label), fill=(255, 255, 255))

        # Field boundary, in pitch coordinates so it stays put when bounds zoom the view.
        draw.rectangle(
            [to_px([-pitch_x, -pitch_y]), to_px([pitch_x, pitch_y])],
            outline=(255, 255, 255),
            width=3,
        )

        # Center line and circle. The circle radius is fixed in pitch units (70px in the default
        # 1280px full-pitch view) and scaled with the horizontal zoom.
        cx, cy = to_px([0.0, 0.0])
        draw.line(
            [to_px([0.0, -pitch_y]), to_px([0.0, pitch_y])],
            fill=(255, 255, 255),
            width=2,
        )
        circle_r = 70 * (2 * pitch_x) / (1280 - 80)
        r_px = circle_r / (x_max - x_min) * (width - 2 * margin)
        draw.ellipse(
            [cx - r_px, cy - r_px, cx + r_px, cy + r_px],
            outline=(255, 255, 255),
            width=2,
        )

        # Left team, skipping GK.
        for idx, p in enumerate(left_team):
            if left_roles[idx] == 0:
                continue

            draw_player(
                p,
                left_dir[idx],
                color=(40, 90, 255),
                label=f"L{idx}",
            )

        # Right team, skipping GK.
        for idx, p in enumerate(right_team):
            if right_roles[idx] == 0:
                continue

            draw_player(
                p,
                right_dir[idx],
                color=(255, 60, 60),
                label=f"R{idx}",
            )

        # Ball.
        if in_play(ball):
            bx, by = to_px(ball)
            r = 7

            draw.ellipse(
                [bx - r, by - r, bx + r, by + r],
                fill=(255, 255, 255),
                outline=(0, 0, 0),
                width=2,
            )

        return np.asarray(img, dtype=np.uint8)

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()

def build_scenario(builder):
    builder.config().game_duration = 1500
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False
    builder.config().end_episode_on_score = True

    ball_offset_x = np.random.random()*0.1 + 0.97
    ball_offset_y = np.random.random()*0.1 + 0.97
    start_agent = np.random.choice([0,1,2,3,4])

    builder.SetBallPosition(AGENT_ANCHORS[start_agent][0]*ball_offset_x, AGENT_ANCHORS[start_agent][1]*ball_offset_y)

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-2.0, 0.0, e_PlayerRole_GK, lazy=True, controllable=True)
    for x, y in AGENT_ANCHORS:
        x_offset = ((np.random.random() * 2) -1) * 0.07
        y_offset = ((np.random.random() * 2) -1) * 0.05
        builder.AddPlayer(x + x_offset, y + y_offset, e_PlayerRole_CM, lazy=False, controllable=True)

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(2.0, 0.0, e_PlayerRole_GK, lazy=True, controllable=True)
    builder.AddPlayer(0.0, 0.0, e_PlayerRole_CM, lazy=False, controllable=True)
