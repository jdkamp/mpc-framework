import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "python" / "envs"))
sys.path.append(str(REPO_ROOT / "build"))

from stable_baselines3 import SAC
from lane_keeping_env import LaneKeepingEnv
from filtered_env import SafetyFilterWrapper
import mpc_py
from train_baseline import ViolationCounter

def make_filter():
    gp = mpc_py.GPResidualModel(str(REPO_ROOT / "data" / "gp_params.json"), 2.7)
    cfg = mpc_py.MPCConfig(); cfg.N = 30; cfg.dt = 0.1
    w = mpc_py.MPCWeights(gp)
    lim = mpc_py.MPCLimits(gp)
    lim.u_min = np.array([-0.5, -5.0])
    lim.u_max = np.array([0.5, 5.0])
    x_max = lim.x_max.copy(); x_max[1] = 2.0; lim.x_max = x_max
    x_min = lim.x_min.copy(); x_min[1] = -2.0; lim.x_min = x_min
    lim.delta_u_min = np.array([-1.0, -5.0])
    lim.delta_u_max = np.array([ 1.0,  5.0])
    return mpc_py.SafetyFilter(gp, cfg, w, lim, 0.95)

def main(total_timesteps=20_000):
    env = SafetyFilterWrapper(LaneKeepingEnv(), make_filter())
    model = SAC("MlpPolicy", env, seed=0, verbose=1)

    violation_counter = ViolationCounter()
    model.learn(total_timesteps=total_timesteps, callback=violation_counter)


    out = REPO_ROOT / "output" / "filtered_violations.csv"
    np.savetxt(out, np.array(violation_counter.log, dtype=int), delimiter=",",
               header="step,cum_violations", comments="", fmt="%d")
    model.save(REPO_ROOT / "python" / "runs" / "sac_filtered")

    print(f"\ntotal violations during training: {violation_counter.total}")
    print(f"violation log -> {out}")


if __name__ == "__main__":
    main()