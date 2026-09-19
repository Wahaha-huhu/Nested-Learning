"""Torch-free check of the chunk-parallel algebra used in nlrepro/models/titans_selfmod.py.

This mirrors selfmod_titans_scan line by line in numpy and compares it with a token-by-token
loop that has the same mini-batch semantics. The torch implementation itself is covered by
tests/test_models_torch.py (runs where torch is installed).
"""
import numpy as np


def l2n(x):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6)


def scan_np(u, q, eta, la, M0, C, rho, selfmod, selfval):
    H, L, d = u.shape
    Dk, Dv, Dm = (np.zeros((H, d, d)) for _ in range(3))
    ys = []
    for c0 in range(0, L, C):
        sl = slice(c0, min(c0 + C, L))
        U, Q, et = u[:, sl], q[:, sl], eta[:, sl]
        Lam = np.cumsum(la[:, sl], -1)
        c = Lam.shape[-1]
        diff = Lam[..., :, None] - Lam[..., None, :]
        mask = np.tril(np.ones((c, c), bool))
        P = np.where(mask, np.exp(np.where(mask, diff, 0.0)), 0.0)
        w_end = np.exp(Lam[..., -1:] - Lam)
        dec = np.exp(Lam[..., -1])[..., None, None]
        Mk, Mv, Mm = M0[0] + Dk, M0[1] + Dv, M0[2] + Dm
        T = lambda M: np.swapaxes(M, -1, -2)
        K = l2n(U @ T(Mk))
        V = U @ T(Mv)
        Vh = V @ T(Mm) if selfval else V
        E = K @ T(Mm) - Vh
        A = (Q @ T(K)) * P
        Y = Q @ T(M0[2]) + np.exp(Lam)[..., None] * (Q @ T(Dm)) - A @ (et[..., None] * E)
        ys.append(Y)
        w = (w_end * et)[..., None]
        Dm = dec * Dm - T(w * E) @ K
        if selfmod:
            KV = K - V
            Dk = dec * Dk - rho * (T(w * (KV @ T(Mk))) @ K)
            Dv = dec * Dv - rho * (T(w * (KV @ T(Mv))) @ K)
    return np.concatenate(ys, 1)


def ref_np(u, q, eta, la, M0, C, rho, selfmod, selfval):
    H, L, d = u.shape
    out = np.zeros_like(u)
    for h in range(H):
        Dk, Dv, Dm = (np.zeros((d, d)) for _ in range(3))
        for c0 in range(0, L, C):
            Mk, Mv, Mm = M0[0, h] + Dk, M0[1, h] + Dv, M0[2, h] + Dm
            for t in range(c0, min(c0 + C, L)):
                a = np.exp(la[h, t])
                k = l2n(Mk @ u[h, t])
                v = Mv @ u[h, t]
                vh = Mm @ v if selfval else v
                e = Mm @ k - vh
                Dm = a * Dm - eta[h, t] * np.outer(e, k)
                out[h, t] = (M0[2, h] + Dm) @ q[h, t]
                if selfmod:
                    Dk = a * Dk - rho * eta[h, t] * np.outer(Mk @ (k - v), k)
                    Dv = a * Dv - rho * eta[h, t] * np.outer(Mv @ (k - v), k)
    return out


def _inputs(seed, H=2, L=23, d=5):
    r = np.random.default_rng(seed)
    u = r.normal(size=(H, L, d))
    q = l2n(r.normal(size=(H, L, d)))
    eta = 0.3 / (1 + np.exp(-r.normal(size=(H, L))))
    la = -np.log1p(np.exp(-(3 + r.normal(size=(H, L)))))  # logsigmoid
    M0 = np.eye(d)[None, None] + 0.1 * r.normal(size=(3, H, d, d))
    return u, q, eta, la, M0


def test_chunk_dual_form_matches_token_loop():
    for seed, C, selfmod, selfval in [(0, 4, True, False), (1, 7, True, True),
                                      (2, 23, False, False), (3, 1, True, False)]:
        args = _inputs(seed)
        a = scan_np(*args, C, 0.1, selfmod, selfval)
        b = ref_np(*args, C, 0.1, selfmod, selfval)
        assert np.max(np.abs(a - b)) < 1e-9, (seed, C, np.max(np.abs(a - b)))


def test_causality():
    u, q, eta, la, M0 = _inputs(5)
    y1 = scan_np(u, q, eta, la, M0, 6, 0.1, True, False)
    u2 = u.copy(); u2[:, 15:] += 10.0
    y2 = scan_np(u2, q, eta, la, M0, 6, 0.1, True, False)
    assert np.allclose(y1[:, :15], y2[:, :15])
