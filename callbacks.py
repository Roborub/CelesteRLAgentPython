import optuna
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.vec_env import VecNormalize

import numpy as np

from config import OptunaConfig

class CelesteLoggingCallback(BaseCallback):
    def _on_step(self) -> bool:
        if "infos" in self.locals:
            info = self.locals["infos"][0]
            for key, value in info.items():
                if key.startswith("reward/") or key == "max_dist":
                    self.logger.record_mean(f"breakdown/{key}", value)
        return True

class HParamCallback(BaseCallback):
    def __init__(self, trial_weights):
        super().__init__()
        self.trial_weights = trial_weights

    def _on_training_start(self) -> None:
        hparam_dict = {k: v for k, v in self.trial_weights.items()}
        metric_dict = {
            "rollout/ep_rew_mean": 0,
            "train/value_loss": 0
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv")
        )

    def _on_step(self) -> bool:
        return True


class CelestePruningCallback(BaseCallback):
    def __init__(self, trial: optuna.Trial, env: VecNormalize, check_freq: int = OptunaConfig.PRUNING_CHECK_RATE):
        super().__init__()
        self.trial = trial
        self.check_freq = check_freq
        self.step_count = 0
        self.env = env
        self.best_so_far = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            all_reward_calculators = self.env.get_attr("reward_calculator")
            
            current_max = max([calc.reward_totals.get("exploration", 0.0) for calc in all_reward_calculators])
            self.best_so_far = max(self.best_so_far, current_max)
            
            self.step_count += 1

            self.trial.report(self.best_so_far, self.step_count)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                    
        return True
