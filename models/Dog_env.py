import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium import Env
class DogEnv(Env):
    """
    2D Dog Navigation Environment (Stable-Baselines3 + Gymnasium compatible)

    State:
        [x, y, vx, vy, dx, dy]

    Actions:
        Continuous: acceleration (ax, ay)
        Discrete:
            0 = noop
            1 = move toward target
            2 = turn left
            3 = turn right
            4 = sit (stop)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        action_space_type="continuous",
        target_distance=3.0,
        target_angle_noise=0.0,
        max_steps=200,
        success_radius=0.2,
        action_max=1.0,
        dt=0.2,
        seed=None,
        max_vel=5.0,
        world_size=20.0
    ):
        super().__init__()

        assert action_space_type in ("continuous", "discrete")

        self.action_space_type = action_space_type
        self.target_distance = float(target_distance)
        self.target_angle_noise = float(target_angle_noise)
        self.max_steps = int(max_steps)
        self.success_radius = float(success_radius)
        self.action_max = float(action_max)
        self.dt = float(dt)
        self.max_vel = float(max_vel)
        self.world_size = float(world_size)

        # ---------- Observation Space ----------
        # [x, y, vx, vy, dx, dy]
        low = np.array(
            [-world_size, -world_size, -max_vel, -max_vel, -world_size, -world_size],
            dtype=np.float32
        )
        high = np.array(
            [world_size, world_size, max_vel, max_vel, world_size, world_size],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # ---------- Action Space ----------
        if self.action_space_type == "continuous":
            self.action_space = spaces.Box(
                low=-action_max,
                high=action_max,
                shape=(2,),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(5)

        self._rng = np.random.default_rng(seed)

        self.pos = None
        self.vel = None
        self.target = None
        self.step_count = 0
        self.episode_reward = 0.0

    # -------------------------------------------------
    # Utilities
    # -------------------------------------------------

    def _sample_target(self):
        base_angle = self._rng.uniform(0, 2 * np.pi)
        angle = base_angle + self._rng.normal(scale=self.target_angle_noise)
        tx = np.cos(angle) * self.target_distance
        ty = np.sin(angle) * self.target_distance
        return np.array([tx, ty], dtype=np.float32)

    def _get_obs(self):
        dxdy = self.target - self.pos
        obs = np.concatenate([self.pos, self.vel, dxdy]).astype(np.float32)
        return obs

    # -------------------------------------------------
    # Gymnasium API
    # -------------------------------------------------

    def reset(self,seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.pos = np.zeros(2, dtype=np.float32)
        self.vel = self._rng.normal(scale=0.01, size=2).astype(np.float32)
        self.target = self._sample_target()

        self.step_count = 0
        self.episode_reward = 0.0

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        self.step_count += 1

        # ---------- Action Processing ----------
        if self.action_space_type == "continuous":
            acc = np.clip(np.array(action, dtype=np.float32), -self.action_max, self.action_max)
        else:
            acc = self._discrete_to_acc(action)

        # ---------- Physics ----------
        self.vel = self.vel + acc * self.dt
        self.vel = np.clip(self.vel, -self.max_vel, self.max_vel)
        self.pos = self.pos + self.vel * self.dt

        # ---------- Distance ----------
        vec = self.target - self.pos
        dist = np.linalg.norm(vec)

        # ---------- Reward Shaping ----------
        # bounded distance reward (stable for PPO)
        distance_reward = -np.tanh(dist)

        # smooth control
        action_penalty = -0.02 * np.sum(acc ** 2)

        # time pressure
        step_penalty = -0.005

        # boundary penalty
        out_of_bounds = np.any(np.abs(self.pos) > self.world_size)
        safety_penalty = -5.0 if out_of_bounds else 0.0

        reward = distance_reward + action_penalty + step_penalty + safety_penalty

        # ---------- Termination ----------
        terminated = False
        truncated = False
        info = {}

        # success condition
        if dist <= self.success_radius:
            reward += 10.0
            terminated = True
            info["success"] = True

        # time limit
        if self.step_count >= self.max_steps:
            truncated = True

        # world boundary hard stop
        if out_of_bounds:
            terminated = True
            info["out_of_bounds"] = True

        self.episode_reward += reward

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, info

    # -------------------------------------------------
    # Discrete Action Mapping
    # -------------------------------------------------

    def _discrete_to_acc(self, act):
        if act == 0:  # noop (drag)
            return -0.2 * self.vel

        elif act == 1:  # forward to target
            vec = self.target - self.pos
            norm = vec / (np.linalg.norm(vec) + 1e-8)
            return norm * 0.8

        elif act == 2:  # left turn
            rot = np.array([[0, -1], [1, 0]])
            dir = rot @ self.vel
            norm = dir / (np.linalg.norm(dir) + 1e-8)
            return norm * 0.5

        elif act == 3:  # right turn
            rot = np.array([[0, 1], [-1, 0]])
            dir = rot @ self.vel
            norm = dir / (np.linalg.norm(dir) + 1e-8)
            return norm * 0.5

        elif act == 4:  # sit
            self.vel = np.zeros_like(self.vel)
            return np.zeros_like(self.vel)

        return np.zeros(2, dtype=np.float32)

    # -------------------------------------------------
    # Rendering
    # -------------------------------------------------

    def render(self):
        print(
            f"Step {self.step_count} | "
            f"pos={self.pos} | vel={self.vel} | target={self.target}"
        )

    def close(self):
        pass
    # Curriculum Interface-
    def set_difficulty(self, target_distance=None, target_angle_noise=None, success_radius=None):
        if target_distance is not None:
            self.target_distance = float(target_distance)
        if target_angle_noise is not None:
            self.target_angle_noise = float(target_angle_noise)
        if success_radius is not None:
            self.success_radius = float(success_radius)

    def get_difficulty(self):
        return {
            "target_distance": float(self.target_distance),
            "target_angle_noise": float(self.target_angle_noise),
            "success_radius": float(self.success_radius),
        }
