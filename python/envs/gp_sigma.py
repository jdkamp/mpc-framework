import json
import numpy as np
from pathlib import Path

_DEFAULT_PARAMS = Path(__file__).resolve().parents[2] / "data" / "gp_params.json"

class GPSigma:
    """Posterior std-dev of the psi-residual GP, sigma(v, delta).
    Mirrors src/models/GPResidualModel.cpp — same json, same kernel."""

    def __init__(self, params_path=_DEFAULT_PARAMS):
        with open(params_path) as f:
            p = json.load(f)
        self.X = np.array(p["X_train"])
        self.sigma_f_sq = p["kernel"]["output_scale"]
        self.l_v        = p["kernel"]["length_scale"][0]
        self.l_delta    = p["kernel"]["length_scale"][1]
        self.noise      = p["kernel"]["noise_variance"]

        v = self.X[:, 0]
        d = self.X[:, 1]
        dv = v[:, None] - v[None, :]
        dd = d[:, None] - d [None, :]
        K = self.sigma_f_sq * np.exp(-0.5 * (dv**2 / self.l_v**2 + dd**2 / self.l_delta**2))
        K += self.noise * np.eye(len(v))
        self.K_inv = np.linalg.inv(K)


    def _k_star(self, v, delta):
        # kernel vector between the query (v, delta) and every training point
        return self.sigma_f_sq * np.exp(
            -0.5 * ((v - self.X[:, 0])**2 / self.l_v**2
                    + (delta - self.X[:, 1])**2 / self.l_delta**2))

    def sigma(self, v, delta):
        k = self._k_star(v, delta)
        var = self.sigma_f_sq - k @ self.K_inv @ k
        return np.sqrt(max(var, 0.0))   # clip: numerics can make it -1e-12

