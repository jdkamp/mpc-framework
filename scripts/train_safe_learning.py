import sys
from pathlib import Path
import numpy as np
import json
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "python" / "envs"))   # lane_keeping_env, filtered_env
sys.path.append(str(REPO_ROOT / "build"))             # mpc_py*.so

from stable_baselines3 import SAC
import mpc_py
from lane_keeping_env import LaneKeepingEnv
from filtered_env import SafetyFilterWrapper
from gp_retrain import retrain_gp


def build_filter(params_path, wheelbase = 2.7):
    gp = mpc_py.GPResidualModel(str(params_path), wheelbase)
    cfg = mpc_py.MPCConfig(); cfg.N = 30; cfg.dt = 0.1
    w = mpc_py.MPCWeights(gp)
    lim = mpc_py.MPCLimits(gp)
    lim.u_min = np.array([-0.5, -5.0]); lim.u_max = np.array([0.5, 5.0])
    x_min= lim.x_min.copy(); x_min[1] = -2.0; lim.x_min = x_min
    x_max = lim.x_max.copy(); x_max[1] = 2.0; lim.x_max = x_max
    lim.delta_u_min = np.array([-1.0, -5.0]); lim.delta_u_max = np.array([1.0, 5.0])

    return mpc_py.SafetyFilter(gp, cfg, w, lim, 0.95)

def build_cold_start_params(out_path, output_scale=0.05, n_anchors=20, length_scale=(3.0, 0.2), seed=0):
    # Cold start: a few real training points keep the covariance tube stable (a real,
    # nonzero GP mean -> sound linearization), while an inflated output_scale makes the
    # filter conservative where those sparse anchors leave the operating region uncertain.
    df = pd.read_csv(REPO_ROOT / "data" / "gp_training_data.csv")
    rows = df[["v", "delta", "residual"]].to_numpy()
    rng = np.random.default_rng(seed)
    sub = rows[rng.choice(len(rows), size=min(n_anchors, len(rows)), replace=False)]
    params = {
        "kernel": {"output_scale": output_scale, "length_scale": list(length_scale), "noise_variance": 1e-2},
        "X_train": sub[:, :2].tolist(),
        "y_train": sub[:, 2].tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(params, f)
    return out_path



def measure_effective_bound(safety_filter, steps=40):
    # Drive full-lock steering through the filter on a determinstic (noise-off) plant
    # return the max py it permits = the effective safe boundary
    probe = LaneKeepingEnv(k_true=0.0, sigma_scale=0.0)     # isolate the filter: no noise, no residual
    probe.reset(seed=0)
    probe.x = np.array([0.0, 0.0, 0.0, 10.0])               # center of lane, cruising
    max_py = 0.0
    for _ in range(steps):
        x = probe.x
        u = safety_filter.filter(np.asarray(x), np.array([0.5, 0.0])) # full lock towards py
        probe.x = probe._dynamics(x, u)
        max_py = max(max_py, probe.x[1])
    return max_py


def main(generations=12, steps_per_gen=50):
    gen_dir = REPO_ROOT / "output" / "safe_learning"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # clear params from any previous run
    for old in gen_dir.glob("gp_gen*.json"):
        old.unlink()

    # Generation 0: cold-start filter (deliberately uncertain)
    param_path = build_cold_start_params(gen_dir / "gp_gen0.json")
    bicycle = mpc_py.BicycleModel(2.7)
    env = SafetyFilterWrapper(LaneKeepingEnv(), build_filter(param_path), bicycle)
    model = SAC("MlpPolicy", env, seed=0, verbose=0)

    effective_bounds = []
    for gen in range(generations):
        model.learn(steps_per_gen, reset_num_timesteps=False)

        eff = measure_effective_bound(env.filter)
        effective_bounds.append(eff)
        print(f"gen {gen}: effective bound = {eff:.3f} m, data rows = {len(env.data)}")

        # retrain on all data so far, swap the new filter in
        new_path = gen_dir / f"gp_gen{gen+1}.json"
        n = retrain_gp(env.data, new_path)
        env.filter = build_filter(new_path)
        print(f"retrained on {n} cells -> {new_path.name}")

    # effective bounds[] rises towards bound across generations
    np.savetxt(gen_dir / "effective_bounds.csv", effective_bounds)


if __name__ == "__main__":
    main()