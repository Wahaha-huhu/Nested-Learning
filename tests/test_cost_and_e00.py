"""Torch-free tests: planning arithmetic and the E00 toy problem."""
import numpy as np

from nlrepro.cost.estimator import cost_range, state_gib, train_flops
from nlrepro.optim.toy_momentum import Landscape, OptConfig, numeric_grad, run


def test_review_reference_numbers():
    lo_h, hi_h, lo_usd, hi_usd = cost_range(train_flops(100e6, 1e9), "rtx4090")
    assert round(lo_h, 1) == 1.7 and round(hi_h, 1) == 4.2
    assert abs(state_gib(8e9) - 119.2) < 0.1


def test_e00_gradient_and_reduction():
    land = Landscape()
    for p in ([-3.5, 2.0], [0.3, -0.7], [1.2, 1.1]):
        p = np.array(p)
        assert np.max(np.abs(land.grad(p) - numeric_grad(land, p))) < 1e-4
    cfg = OptConfig(beta=0.0, convention="normalized")
    a, b = run(land, cfg, False), run(land, cfg, True)     # beta=0 -> identical to standard
    assert np.allclose(a["trajectory"], b["trajectory"])
