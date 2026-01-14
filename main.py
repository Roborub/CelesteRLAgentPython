import time
import subprocess
import os
import torch

from config import (
    ModelConfig,
    EnvironmentConfig,
    CustomRewardWeights,
)

from reward_optimizer import train_curriculum   # NEW: curriculum trainer


# ---------------------------------------------------------
# Kill all Celeste instances before starting
# ---------------------------------------------------------
def global_cleanup():
    print("--- Performing Global Cleanup ---")
    if os.name == 'nt':
        try:
            subprocess.run(
                ['taskkill', '/F', '/IM', 'Celeste.exe', '/T'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("Cleanup: All previous Celeste instances terminated.")
        except Exception as e:
            print(f"Cleanup Note: {e}")
    time.sleep(2.0)


# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
def main():
    global_cleanup()
    torch.set_num_threads(1)

    print(f"=== CELESTE CURRICULUM TRAINER v{ModelConfig.VERSION} ===")

    # Choose reward weights
    if EnvironmentConfig.UseCustomWeights:
        reward_weights = CustomRewardWeights
        print("--- Using Custom Reward Weights ---")
    else:
        raise RuntimeError(
            "Optuna-based reward tuning is no longer supported in curriculum mode. "
            "Set UseCustomWeights=True in config."
        )

    # Run curriculum training
    final_actor_state = train_curriculum(reward_weights)

    # Save final actor
    os.makedirs("models", exist_ok=True)
    final_path = f"models/celeste_curriculum_final_actor_v{ModelConfig.VERSION}.pth"
    torch.save(final_actor_state, final_path)

    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Final actor saved to: {final_path}")


if __name__ == "__main__":
    main()
