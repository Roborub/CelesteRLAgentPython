import gymnasium as gym

from config import EnvironmentConfig

class CelesteFrameSkipEnvironment(gym.Wrapper):
    def __init__(self, env, skip=EnvironmentConfig.FRAME_SKIP):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = {}

        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            total_reward += float(reward)
            
            if terminated or truncated:
                break
        
        return obs, total_reward, terminated, truncated, info
