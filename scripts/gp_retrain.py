import numpy as np
import pandas as pd
import json
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


def retrain_gp(data, out_path, n_v=15, n_delta=15, v_range=(0.0, 15.0), delta_range=(-0.5, 0.5), min_count=3):
    """ - Bin (v, delta. residual) rows onto a grid 
        - Average per cell 
        - Fit a GP 
        - Export params"""

    data = np.asarray(data)
    v, delta, resid = data[:, 0], data[:, 1], data[:, 2]

    # Which grid cell does each row fall in
    iv      = np.clip(((v     - v_range[0])     / (v_range[1]     - v_range[0]))     * n_v,       0, n_v-1).astype(int)
    idelta  = np.clip(((delta - delta_range[0]) / (delta_range[1] - delta_range[0])) * n_delta,   0, n_delta-1).astype(int)

    # Group by cell, average v / delta / residual within each occupied cell
    df = pd.DataFrame({"iv": iv, "idelta": idelta, "v": v, "delta": delta, "resid": resid})
    cells = df.groupby(["iv", "idelta"]).agg(
        v=("v", "mean"), delta=("delta", "mean"),
        resid=("resid", "mean"), count=("resid", "size")
    )
    cells = cells[cells["count"] >= min_count]     # drop under sampled (still noisy cells)
    X_train = cells[["v", "delta"]].to_numpy()
    y_train = cells["resid"].to_numpy()

    # Fit the GP
    kernel = (ConstantKernel(0.05, constant_value_bounds="fixed") *
          RBF(length_scale=[3.0, 0.2], length_scale_bounds="fixed"))
    # Higher noise floor than the offline trainer (1e-5): the online-collected residuals
    # (observed yaw rate - nominal) carry the plant's process noise and are far noisier,
    # so the GP should not over-trust them. This floor is also the filter's retained
    # safety margin.
    noise = 1e-2
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise, n_restarts_optimizer=5, random_state=0)
    gpr.fit(X_train, y_train)

    # Export
    k = gpr.kernel_
    sigma_f_sq = float(k.k1.constant_value)
    length_scale = np.asarray(k.k2.length_scale).ravel().tolist()

    params = {
        'kernel': {
            "output_scale": sigma_f_sq,
            "length_scale": length_scale,
            "noise_variance": noise
        },
        "X_train": X_train.tolist(),
        "y_train": y_train.tolist(),
    }

    with open(out_path, 'w') as f:
        json.dump(params, f)


    return len(X_train)
