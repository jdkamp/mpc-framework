import sys
from pathlib import Path
import numpy as np
import pytest
import json

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "build")) # where CMake puts mpc_py.*.so

mpc_missing = not any((REPO_ROOT / "build").glob("mpc_py*.so"))
pytestmark = pytest.mark.skipif(mpc_missing, reason="mpc_py not built (run cmake --build build)")

import mpc_py


def make_filter():
    gp = mpc_py.GPResidualModel(str(REPO_ROOT / "data" / "gp_params.json"), 2.7)
    cfg = mpc_py.MPCConfig(); cfg.N = 30; cfg.dt = 0.1
    w = mpc_py.MPCWeights(gp)
    lim = mpc_py.MPCLimits(gp)
    lim.u_min = np.array([-0.5, -3.0]); lim.u_max = np.array([0.5, 3.0])
    x_max = lim.x_max.copy(); x_max[1] = 9.0; lim.x_max = x_max # copy-modify-assign
    return mpc_py.SafetyFilter(gp, cfg, w, lim, 0.95), gp


def test_gp_variance_matches_reference():
    gp = mpc_py.GPResidualModel(str(REPO_ROOT / "data" / "gp_params.json"), 2.7)
    params = json.load(open(REPO_ROOT / "data" / "gp_params.json"))
    chk = next(c for c in params["reference_checks"] if c["v"] == 10.0 and c["delta"] == 0.3)
    var = gp.variance(np.array([0.0, 0, 0, 10.0]), np.array([0.3, 0.0]))[2]
    assert abs(var - chk["var"]) < 1e-5


def test_filter_pass_through():
    f, gp = make_filter()
    u = f.filter(np.array([0.0, 0, 0, 10.0]), np.array([0.1, 0.0]))
    assert abs(u[0] - 0.1) < 1e-4 and abs(u[1]) < 1e-4


def test_filter_blocks_unsafe():
    f, gp = make_filter()
    u = f.filter(np.array([0.0, 8.0, 0.3, 10.0]), np.array([0.4, 0.0]))
    assert u[0] < 0.1 # mirror of the C++ BlocksUnsafeAction test
