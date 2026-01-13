import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from reward_calculator import RewardCalculator
from config import ObservationConfig


class CelesteEnvironment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, pipe_manager, env_config, reward_weights=None):
        super().__init__()
        self.spawns = None
        self.pipe = pipe_manager
        self.reward_calculator = RewardCalculator(reward_weights, self)

        self.first_step_obs = None
        
        # Action space unchanged
        self.action_space = spaces.MultiDiscrete([3, 3, 2, 2, 2])

        # --- CNN OBSERVATION SPACE ---
        grid_shape = (ObservationConfig.CATEGORY_COUNT,
                      ObservationConfig.GRID_SIZE,
                      ObservationConfig.GRID_SIZE)

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=grid_shape,
                dtype=np.float32
            ),
            "static": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(ObservationConfig.STATIC_FEATURE_COUNT,),
                dtype=np.float32
            )
        })

        self.step_count = 0
        self.episode_count = 0
        self.max_steps = env_config.MAX_EPISODE_STEPS

        # Store raw flat obs until reshaped
        self.current_obs_raw = np.zeros(
            ObservationConfig.TOTAL_FEATURE_COUNT, dtype=np.float32
        )

    # ---------------------------------------------------------
    # Helper: Convert flat vector → Dict(grid, static)
    # ---------------------------------------------------------
    def _reshape_observation(self, flat):
        static = flat[:ObservationConfig.STATIC_FEATURE_COUNT]

        grid_flat = flat[ObservationConfig.STATIC_FEATURE_COUNT:]
        grid = grid_flat.reshape(
            ObservationConfig.CATEGORY_COUNT,
            ObservationConfig.GRID_SIZE,
            ObservationConfig.GRID_SIZE
        )

        return {
            "grid": grid.astype(np.float32),
            "static": static.astype(np.float32)
        }

    # ---------------------------------------------------------
    # RESET
    # ---------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        self.pipe.send_reset()
        self.reward_calculator.reset()
        
        while True:
            obs, status = self.pipe.receive_observation()

            if status == "READY":
                self.pipe.send_action([1, 1, 0, 0, 0])
                continue

            if status in ["TICK", "SYNC"] and obs is not None:
                self.episode_count += 1
                self.current_obs_raw = obs.astype(np.float32)
                return self._reshape_observation(self.current_obs_raw), {}

            if status == "CONNECTION_CLOSED":
                raise ConnectionError("Lost connection to Celeste.")

            time.sleep(0.001)

    # ---------------------------------------------------------
    # STEP
    # ---------------------------------------------------------
    def step(self, action):
        self.step_count += 1
        last_obs_raw = self.current_obs_raw
        
        self.pipe.send_action(action)

        new_state_data, signal = self.pipe.receive_observation()

        terminated = False
        truncated = False

        if signal in ['ERROR', 'CONNECTION_CLOSED']:
            new_raw = last_obs_raw
            terminated = True

        elif signal == 'DEAD':
            new_raw = last_obs_raw
            terminated = True

        elif signal == 'LEVEL_COMPLETE':
            new_raw = new_state_data if new_state_data is not None else last_obs_raw
            terminated = True

        else:
            new_raw = new_state_data if new_state_data is not None else last_obs_raw
            truncated = (self.step_count >= self.max_steps)

        reward_dict = self.reward_calculator.calculate(
            last_obs_raw, new_raw, signal
        )

        total_reward = float(sum(reward_dict.values()))

        self.current_obs_raw = new_raw.astype(np.float32)

        info = {f"reward/{k}": v for k, v in reward_dict.items()}
        info["terminal_signal"] = signal
        info["step_count"] = self.step_count
        
        return self._reshape_observation(self.current_obs_raw), total_reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        self.pipe.close()

