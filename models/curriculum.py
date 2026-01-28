import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
class CurriculumCallback(BaseCallback):
    """
    Curriculum Learning Callback

    Increases environment difficulty when agent performance exceeds a threshold.
    Compatible with:
        - DummyVecEnv
        - VecNormalize
        - Gymnasium API
        - Stable-Baselines3 callbacks
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 50000,
        threshold: float = -5.0,
        scale_increase: float = 1.2,
        noise_increase: float = 0.05,
        radius_decay: float = 0.95,
        min_success_radius: float = 0.05,
        n_eval_episodes: int = 5,
        verbose: int = 1,
    ):
        super().__init__(verbose)

        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.threshold = float(threshold)

        self.scale_increase = float(scale_increase)
        self.noise_increase = float(noise_increase)
        self.radius_decay = float(radius_decay)
        self.min_success_radius = float(min_success_radius)

        self.n_eval_episodes = int(n_eval_episodes)

        self.last_eval_step = 0
        self.curriculum_level = 0

    # -------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------

    def _get_base_env(self):
        """Unwrap VecEnv/VecNormalize to get real environment"""
        env = self.eval_env
        if isinstance(env, VecNormalize):
            env = env.venv
        if isinstance(env, VecEnv):
            env = env.envs[0]
        return env

    # -------------------------------------------------
    # Main Callback
    # -------------------------------------------------

    def _on_step(self) -> bool:
        # Run only every eval_freq steps
        if (self.num_timesteps - self.last_eval_step) < self.eval_freq:
            return True

        self.last_eval_step = self.num_timesteps

        mean_reward = self._evaluate_policy()

        if self.verbose:
            print(f"\n[Curriculum] Mean eval reward: {mean_reward:.3f}")

        if mean_reward > self.threshold:
            self._increase_difficulty(mean_reward)

        return True

    # -------------------------------------------------
    # Evaluation
    # -------------------------------------------------

    def _evaluate_policy(self):
        returns = []

        base_env = self._get_base_env()

        for _ in range(self.n_eval_episodes):
            obs, _ = base_env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0

            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = base_env.step(action)
                total_reward += reward
                steps += 1
                if steps > base_env.max_steps:
                    break

            returns.append(total_reward)

        return float(np.mean(returns))

    # -------------------------------------------------
    # Curriculum Logic
    # -------------------------------------------------

    def _increase_difficulty(self, mean_reward):
        env = self._get_base_env()

        cur = env.get_difficulty()

        new_dist = cur["target_distance"] * self.scale_increase
        new_noise = cur["target_angle_noise"] + self.noise_increase
        new_radius = max(
            self.min_success_radius,
            cur["success_radius"] * self.radius_decay
        )

        env.set_difficulty(
            target_distance=new_dist,
            target_angle_noise=new_noise,
            success_radius=new_radius
        )

        self.curriculum_level += 1
        self.threshold = mean_reward * 1.1  # adaptive threshold

        if self.verbose:
            print(
                f"[Curriculum] Level {self.curriculum_level} ↑\n"
                f"  target_distance: {cur['target_distance']:.2f} → {new_dist:.2f}\n"
                f"  angle_noise:      {cur['target_angle_noise']:.3f} → {new_noise:.3f}\n"
                f"  success_radius:   {cur['success_radius']:.3f} → {new_radius:.3f}\n"
                f"  new threshold:    {self.threshold:.3f}\n"
            )
