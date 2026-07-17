import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "python" / "envs"))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from lane_keeping_env import LaneKeepingEnv

class ViolationCounter(BaseCallback):
    """Count violations across all training steps via the env's info dict"""
    def __init__(self):
        super().__init__()
        self.total = 0
        self.log = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if info.get("violation"):
                self.total += 1
        self.log.append((self.num_timesteps, self.total))
        return True
    
def main(total_timestep=100_000):
    env = LaneKeepingEnv()
    model = SAC("MlpPolicy", env, seed=0, verbose=1)

    violation_counter = ViolationCounter()
    model.learn(total_timesteps=total_timestep, callback=violation_counter)

    out = REPO_ROOT / "output" / "baseline_violations.csv"
    np.savetxt(out, np.array(violation_counter.log, dtype=int), delimiter=",",
               header="step,cum_violations", comments="", fmt="%d")
    model.save(REPO_ROOT / "python" / "runs" / "sac_baseline")

    print(f"\ntotal violations during training: {violation_counter.total}")
    print(f"violation log -> {out}")

if __name__ == "__main__":
    main()