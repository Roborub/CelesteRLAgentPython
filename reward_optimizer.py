
import gymnasium as gym
import optuna
import os
import glob
import numpy as np
import time
import psutil
from typing import Callable, List, Optional, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback 
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from pipe_manager import PipeManager
from celeste_env import CelesteEnvironment
from celeste_frame_skip_env import CelesteFrameSkipEnvironment
from config import OptunaConfig, RewardWeights, EnvironmentConfig, ServerConfig, ModelConfig, ObservationConfig
from callbacks import CelesteLoggingCallback, HParamCallback, CelestePruningCallback
from celeste_extractor import CelesteFeatureExtractor

custom_layers = [EnvironmentConfig.LAYER_NEURONS] * EnvironmentConfig.LAYERS

policy_kwargs = dict(
    features_extractor_class=CelesteFeatureExtractor,
    features_extractor_kwargs=dict(
        grid_channels=ObservationConfig.CATEGORY_COUNT,
        grid_size=ObservationConfig.GRID_SIZE,
        static_size=ObservationConfig.STATIC_FEATURE_COUNT
    )
)


def get_latest_checkpoint(checkpoint_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Finds the latest .zip model and .pkl normalization stats in a folder."""
    if not os.path.exists(checkpoint_dir):
        return None, None
    
    # Get all .zip files and sort by modification time (latest first)
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    if not checkpoint_files:
        return None, None
    
    latest_model = max(checkpoint_files, key=os.path.getmtime)
    
    # Derive the matching .pkl filename (CheckpointCallback naming convention)
    # Model: name_prefix_XXXX_steps.zip -> Stats: name_prefix_vecnormalize_XXXX_steps.pkl
    base_name = os.path.basename(latest_model).replace(".zip", "")
    step_count = base_name.split("_")[-2] # Extract 'XXXX'
    stats_name = f"celeste_model_vecnormalize_{step_count}_steps.pkl"
    latest_stats = os.path.join(checkpoint_dir, stats_name)
    
    if not os.path.exists(latest_stats):
        # Fallback: check for the most recent .pkl file generally
        stats_files = glob.glob(os.path.join(checkpoint_dir, "*.pkl"))
        latest_stats = max(stats_files, key=os.path.getmtime) if stats_files else None

    return latest_model, latest_stats

def make_celeste_env(instance_index: int, suggested_weights: dict) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        unique_port = 5000 + instance_index
        pm = PipeManager(host=ServerConfig.HOST, port=unique_port, instance_index=instance_index)
        base_env = CelesteEnvironment(
            pipe_manager=pm,
            env_config=EnvironmentConfig,
            reward_weights=suggested_weights
        )
        env = CelesteFrameSkipEnvironment(env=base_env)
        return Monitor(env)
    return _init

def objective(trial):
    env = None
    try:
        suggested_weights = {}
        for attr in dir(RewardWeights):
            if not attr.startswith("__"):
                val = getattr(RewardWeights, attr)
                if isinstance(val, tuple) and len(val) == 2:
                    suggested_weights[attr] = trial.suggest_float(attr, val[0], val[1])

        num_instances = OptunaConfig.CELESTE_INSTANCE_COUNT
        env_fns = [make_celeste_env(i, suggested_weights) for i in range(num_instances)]

        # 1. Initialize Environment
        env = DummyVecEnv(env_fns)
        env = VecNormalize(env, norm_obs=False, norm_reward=True)

        # 2. Check for Checkpoints to Resume
        checkpoint_dir = f"./models/checkpoints/trial_0/"
        latest_model_path, latest_stats_path = get_latest_checkpoint(checkpoint_dir)

        if latest_model_path and latest_stats_path:
            print(f"--- RESUMING TRIAL {trial.number} FROM: {latest_model_path} ---")
            env = VecNormalize.load(latest_stats_path, env)
            model = PPO.load(
                latest_model_path,
                env=env,
                custom_objects={
                    "features_extractor_class": CelesteFeatureExtractor
                }
            )
            
        else:
            print(f"--- STARTING FRESH TRIAL {trial.number} ---")
            model = PPO(
                policy=EnvironmentConfig.Policy,
                policy_kwargs=policy_kwargs,
                env=env,
                learning_rate=EnvironmentConfig.LEARNING_RATE,
                n_steps=EnvironmentConfig.N_STEPS,
                batch_size=EnvironmentConfig.BATCH_SIZE,
                n_epochs=EnvironmentConfig.N_EPOCHS,
                gamma=EnvironmentConfig.GAMMA,
                gae_lambda=EnvironmentConfig.GAE_LAMBDA,
                verbose=EnvironmentConfig.VERBOSE,
                ent_coef=EnvironmentConfig.ENTROPY_COEF,
                tensorboard_log=EnvironmentConfig.TENSOR_LOG
            )

        time.sleep(5)
        set_high_priority()

        # 3. Setup Callbacks (Including Mid-Trial Saving)
        checkpoint_callback = CheckpointCallback(
            save_freq=max(20000 // num_instances, 1),
            save_path=checkpoint_dir,
            name_prefix="celeste_model",
            save_vecnormalize=True 
        )

        callbacks = CallbackList([
            CelesteLoggingCallback(), 
            checkpoint_callback,
            HParamCallback(trial_weights=suggested_weights),
            CelestePruningCallback(trial=trial, env=env, check_freq=OptunaConfig.PRUNING_CHECK_RATE)
        ])

        # 4. Train
        model.learn(
            total_timesteps=OptunaConfig.TIMESTEPS, 
            callback=callbacks, 
            tb_log_name=f"trial_{trial.number}",
            reset_num_timesteps=False # IMPORTANT: Keeps the global step count if resuming
        )

        # 5. Final Evaluation & Save Best
        all_reward_calculators = env.get_attr("reward_calculator")
        exploration_scores = [calc.reward_totals.get("exploration", 0.0) for calc in all_reward_calculators]
        avg_score = float(np.mean(exploration_scores))

        if trial.number == 0 or avg_score > trial.study.best_value:
            base_path = f"models/v{ModelConfig.VERSION}_best"
            model.save(base_path)
            env.save(f"{base_path}_vec_normalize.pkl")

        return avg_score

    except optuna.TrialPruned:
        raise 
    except Exception as e:
        print(f"Trial #{trial.number} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return -1.0
    finally:
        if env is not None:
            env.close()

def set_high_priority():
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == 'Celeste.exe':
            try:
                proc.nice(psutil.HIGH_PRIORITY_CLASS)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

