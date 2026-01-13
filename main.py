import time
import optuna
import subprocess
import os
import torch

from reward_optimizer import objective
from config import ModelConfig, OptunaConfig, EnvironmentConfig, CustomRewardWeights

def global_cleanup():
    print("--- Performing Global Cleanup ---")
    if os.name == 'nt':
        try:
            subprocess.run(['taskkill', '/F', '/IM', 'Celeste.exe', '/T'], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Cleanup: All previous Celeste instances terminated.")
        except Exception as e:
            print(f"Cleanup Note: {e}")
    time.sleep(2.0)

def main():
    global_cleanup()
    torch.set_num_threads(1)

    study_name = f"celeste_reward_optimization_v{ModelConfig.VERSION}"
    storage_name = f"sqlite:///{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True
    )

    if EnvironmentConfig.RunFinalModel:
        print(f"--- Running Final Model with Best Params from {study_name} ---")
        trial = optuna.trial.FixedTrial(study.best_params)
        objective(trial)

    elif EnvironmentConfig.UseCustomWeights:
        print(f"--- Enqueuing Custom Weights for Trial #1 ---")
        study.enqueue_trial(CustomRewardWeights)
        study.optimize(objective, n_trials=OptunaConfig.TRIALS)

    else:
        try:
            study.optimize(objective, n_trials=OptunaConfig.TRIALS)
        except KeyboardInterrupt:
            print("Optimization interrupted by user.")

if __name__ == "__main__":
    main()
