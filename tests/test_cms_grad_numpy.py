"""Torch-free finite-difference check of the analytic inner gradients in cms.incontext_mlp_scan."""
import numpy as np


def silu(z):
    return z / (1 + np.exp(-z))


def dsilu(z):
    s = 1 / (1 + np.exp(-z))
    return s * (1 + z * (1 - s))


def loss(W1, W2, A, V, et):
    m = A + silu(A @ W1.T) @ W2.T
    return 0.5 * (et[:, None] * (m - V) ** 2).sum()


def analytic(W1, W2, A, V, et):
    z = A @ W1.T
    g = silu(z)
    m = A + g @ W2.T
    e = (m - V) * et[:, None]
    gW2 = e.T @ g
    gW1 = ((e @ W2) * dsilu(z)).T @ A
    return gW1, gW2


def test_incontext_mlp_gradients():
    r = np.random.default_rng(0)
    c, d, h = 9, 4, 8
    A, V = r.normal(size=(c, d)), r.normal(size=(c, d))
    et = r.uniform(0.1, 1.0, size=c)
    W1, W2 = r.normal(size=(h, d)), r.normal(size=(d, h))
    gW1, gW2 = analytic(W1, W2, A, V, et)
    eps = 1e-6
    for W, G, which in [(W1, gW1, 1), (W2, gW2, 2)]:
        num = np.zeros_like(W)
        for idx in np.ndindex(W.shape):
            Wp, Wm = W.copy(), W.copy()
            Wp[idx] += eps; Wm[idx] -= eps
            lp = loss(Wp, W2, A, V, et) if which == 1 else loss(W1, Wp, A, V, et)
            lm = loss(Wm, W2, A, V, et) if which == 1 else loss(W1, Wm, A, V, et)
            num[idx] = (lp - lm) / (2 * eps)
        assert np.max(np.abs(num - G)) < 1e-5, which
