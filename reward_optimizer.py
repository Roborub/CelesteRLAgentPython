import os
import time
import numpy as np
import gymnasium as gym
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from pipe_manager import PipeManager
from celeste_env import CelesteEnvironment
from celeste_frame_skip_env import CelesteFrameSkipEnvironment

from config import (
    EnvironmentConfig,
    LevelConfig,
    ServerConfig,
    ObservationConfig,
)

from celeste_extractor import CelesteFeatureExtractor


# ---------------------------------------------------------
# PPO Policy Configuration
# ---------------------------------------------------------
policy_kwargs = dict(
    features_extractor_class=CelesteFeatureExtractor,
    features_extractor_kwargs=dict(
        grid_channels=ObservationConfig.CATEGORY_COUNT,
        grid_size=ObservationConfig.GRID_SIZE,
        static_size=ObservationConfig.STATIC_FEATURE_COUNT,
    ),
)


# ---------------------------------------------------------
# Build a single Celeste environment instance
# ---------------------------------------------------------
def make_env(level_id: int, instance_index: int, reward_weights: dict):
    def _init():
        port = 5000 + instance_index
        pipe = PipeManager(
            host=ServerConfig.HOST,
            port=port,
            instance_index=instance_index,
        )

        base_env = CelesteEnvironment(
            pipe_manager=pipe,
            env_config=EnvironmentConfig,
            reward_weights=reward_weights,
            level_id=level_id,
        )

        env = CelesteFrameSkipEnvironment(base_env)
        return Monitor(env)

    return _init


# ---------------------------------------------------------
# Evaluate success rate for promotion
# ---------------------------------------------------------
def evaluate_success_rate(model, env, episodes=30):
    successes = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if info.get("terminal_signal") == "LEVEL_CHANGED":
            successes += 1

    return successes / episodes


# ---------------------------------------------------------
# Train PPO on a single level until mastery
# ---------------------------------------------------------
def train_level(level_id, previous_actor_state, reward_weights):
    print(f"\n=== TRAINING LEVEL {level_id} ===")

    # Build environment
    env_fns = [make_env(level_id, 0, reward_weights)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)

    # Fresh PPO instance (fresh critic)
    model = PPO(
        policy=EnvironmentConfig.Policy,
        env=vec_env,
        learning_rate=EnvironmentConfig.LEARNING_RATE,
        n_steps=EnvironmentConfig.N_STEPS,
        batch_size=EnvironmentConfig.BATCH_SIZE,
        n_epochs=EnvironmentConfig.N_EPOCHS,
        gamma=EnvironmentConfig.GAMMA,
        gae_lambda=EnvironmentConfig.GAE_LAMBDA,
        ent_coef=EnvironmentConfig.ENTROPY_COEF,
        verbose=EnvironmentConfig.VERBOSE,
        tensorboard_log=EnvironmentConfig.TENSOR_LOG,
        policy_kwargs=policy_kwargs,
    )

    # Warm‑start actor if available
    if previous_actor_state is not None:
        print("Loading previous actor weights...")
        model.policy.load_state_dict(previous_actor_state)

    # Curriculum loop: train → evaluate → promote
    MAX_STEPS = 1_000_000
    STEPS_PER_ITER = 200_000
    total_steps = 0
    consecutive_successes = 0

    while total_steps < MAX_STEPS:
        print(f"Training batch ({STEPS_PER_ITER} steps)...")
        model.learn(total_timesteps=STEPS_PER_ITER, reset_num_timesteps=False)
        total_steps += STEPS_PER_ITER

        print("Evaluating success rate...")
        success_rate = evaluate_success_rate(model, vec_env, episodes=30)
        print(f"Success rate: {success_rate:.2f}")

        if success_rate >= 0.80:
            consecutive_successes += 1
            print(f"Success streak: {consecutive_successes}/3")
        else:
            consecutive_successes = 0

        if consecutive_successes >= 3:
            print(f"Level {level_id} mastered early!")
            break

    # Save actor weights only
    actor_state = model.policy.state_dict()

    # Cleanup
    vec_env.close()

    return actor_state


# ---------------------------------------------------------
# Full Curriculum Trainer
# ---------------------------------------------------------
def train_curriculum(reward_weights):
    previous_actor_state = None

    for level_id in LevelConfig.LEVEL_ID_MAP.keys():
        previous_actor_state = train_level(
            level_id=level_id,
            previous_actor_state=previous_actor_state,
            reward_weights=reward_weights,
        )

    print("\n=== CURRICULUM COMPLETE ===")
    return previous_actor_state


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    reward_weights = {
        "LEVEL_COMPLETE": 500.0,
        "DEAD": -5.0,
        "DISTANCE": 0.1,
        "FRONTIER_COLLECT": 50.0,
        "FRONTIER_PROGRESS_WEIGHT": 0.05,
        "STAGNATION": -0.05,
        "ALIVE": 0.0,
        "LEAP": 0.0,
        "GAP_JUMPED": 0.0,
        "EXPLORATION": 0.0,
    }

    final_actor = train_curriculum(reward_weights)

    os.makedirs("models", exist_ok=True)
    final_path = "models/celeste_curriculum_final_actor.pth"

    torch.save(final_actor, final_path)

    print(f"Saved final actor to {final_path}")
