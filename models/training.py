import argparse
import os
import gymnasium as gym
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold
)

from Dog_env import DogEnv
from curriculum import CurriculumCallback


# -------------------------------------------------
# Env Factory
# -------------------------------------------------

def make_env(action_space_type, difficulty_cfg, seed=None):
    def _init():
        env = DogEnv(
            action_space_type=action_space_type,
            target_distance=difficulty_cfg["target_distance"],
            target_angle_noise=difficulty_cfg["target_angle_noise"],
            max_steps=difficulty_cfg["max_steps"],
            success_radius=difficulty_cfg["success_radius"],
            seed=seed,
        )
        env = Monitor(env)
        return env
    return _init


# -------------------------------------------------
# Main Training Pipeline
# -------------------------------------------------

def main(args):
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    difficulty_cfg = {
        "target_distance": args.init_target_distance,
        "target_angle_noise": args.init_angle_noise,
        "max_steps": args.max_steps,
        "success_radius": args.success_radius,
    }

    # =========================
    # Training Environment
    # =========================
    train_env = DummyVecEnv([
        make_env(args.action_space, difficulty_cfg, seed=42)
    ])

    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0
    )

    # =========================
    # Evaluation Environment
    # (SAME WRAPPING!)
    # =========================
    eval_env = DummyVecEnv([
        make_env(args.action_space, difficulty_cfg, seed=12345)
    ])

    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0
    )

    eval_env.training = False
    eval_env.norm_reward = False

    # =========================
    # Algorithm Selection
    # =========================
    algo = args.algo.lower()

    if algo == "ppo":
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log="./logs",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
        )

    elif algo == "sac":
        if args.action_space != "continuous":
            raise ValueError("SAC requires continuous action space")
        model = SAC(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log="./logs",
            learning_rate=3e-4,
            gamma=0.99,
            batch_size=256,
        )

    elif algo == "dqn":
        if args.action_space != "discrete":
            raise ValueError("DQN requires discrete action space")
        model = DQN(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log="./logs",
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=10_000,
            batch_size=64,
            gamma=0.99,
        )

    else:
        raise ValueError("Unknown algorithm: ppo | sac | dqn")

    # =========================
    # Callbacks
    # =========================

    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path="./models/",
        name_prefix=f"{algo}_checkpoint"
    )

    stop_cb = StopTrainingOnRewardThreshold(
        reward_threshold=args.stop_reward,
        verbose=1
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model",
        log_path="./logs/eval",
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
        callback_on_new_best=stop_cb,
    )

    curriculum_cb = CurriculumCallback(
        eval_env=eval_env,
        eval_freq=args.curriculum_eval_freq,
        threshold=args.curriculum_threshold,
        verbose=1
    )

    # =========================
    # Training
    # =========================

    model.learn(
        total_timesteps=int(args.total_timesteps),
        callback=[checkpoint_cb, eval_cb, curriculum_cb]
    )
    # Save Final Model

    model_path = os.path.join("models", f"{algo}_final.zip")
    model.save(model_path)
    train_env.save(os.path.join("models", "vecnormalize.pkl"))

    print("\nTraining complete")
    print("Model saved to:", model_path)
# CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn", "sac"])
    parser.add_argument("--action-space", type=str, default="continuous", choices=["continuous", "discrete"])
    parser.add_argument("--total-timesteps", type=int, default=300_000)

    parser.add_argument("--init-target-distance", type=float, default=2.5)
    parser.add_argument("--init-angle-noise", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--success-radius", type=float, default=0.2)
    args=parser.parse_args()
    main(args)
