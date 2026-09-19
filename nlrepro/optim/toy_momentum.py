"""E00 / Figure 4: standard momentum vs delta momentum on psi(r, theta).

Paper facts (Sec. 4.4, Eq. 53):
    psi(r, theta) = r^2 + k * (r - theta + a * sin(w * r))^2,  start (r0, theta0) = (-3.5, 2).
    Delta momentum (Eq. 48-49):  W <- W + m,  m <- m (alpha I - g g^T) - eta P g.

Everything else (k, a, w, eta, alpha, the scale of the g g^T term, the stopping rule) is NOT
given in the paper. Our reconstruction choices are declared in DeltaMomentumConfig and
docs/DEVIATIONS.md (D-E00-*). The global minimiser is (0, 0) with psi = 0.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class Landscape:
    k: float = 10.0      # coupling strength (not in paper)
    a: float = 1.0       # sine amplitude (not in paper)
    w: float = 2.0       # sine frequency (not in paper)

    def value(self, p: np.ndarray) -> float:
        r, th = p
        s = r - th + self.a * np.sin(self.w * r)
        return float(r * r + self.k * s * s)

    def grad(self, p: np.ndarray) -> np.ndarray:
        r, th = p
        s = r - th + self.a * np.sin(self.w * r)
        ds_dr = 1.0 + self.a * self.w * np.cos(self.w * r)
        return np.array([2.0 * r + 2.0 * self.k * s * ds_dr, -2.0 * self.k * s])


@dataclass(frozen=True)
class OptConfig:
    eta: float = 0.004          # step size (not in paper)
    alpha: float = 0.9          # momentum retention (not in paper)
    beta: float = 0.5           # strength for the "normalized" convention only
    convention: str = "normalized"  # "normalized" (D-E00-2a) or "gd_l2" (D-E00-2b)
    tol: float = 1e-6           # converged when psi < tol ...
    patience: int = 10          # ... for `patience` consecutive steps
    max_steps: int = 20000
    start: tuple = (-3.5, 2.0)  # paper


def run(land: Landscape, cfg: OptConfig, delta: bool) -> dict:
    """Run one optimiser. Returns trajectory, objective curve and steps-to-convergence.

    Standard momentum (heavy ball, paper Eq. 33 sign convention):
        m <- alpha m - eta g ;  p <- p + m
    Delta momentum (paper Eq. 49 with P = I). The printed rule m(alpha - g^T g) has no usable
    scale (with ||g|| ~ 300 at the start it explodes), so two conventions are provided:
      "normalized": m <- (alpha I - beta g_hat g_hat^T) m - eta g,   g_hat = g / ||g||
                    (unit-norm key, as in DeltaNet-style delta rules; decay independent of ||g||)
      "gd_l2":      m <- (alpha I - c g g^T) m - eta g,   c = min(eta, alpha / ||g||^2)
                    (one GD step with the same step size eta on an L2 memory objective; the
                    decay grows with ||g||^2 and is capped so the along-g eigenvalue stays >= 0)
    """
    p = np.array(cfg.start, dtype=np.float64)
    m = np.zeros(2, dtype=np.float64)
    traj, vals = [p.copy()], [land.value(p)]
    below, converged_at = 0, None
    for step in range(1, cfg.max_steps + 1):
        g = land.grad(p)
        if delta and cfg.convention == "normalized":
            gn = np.linalg.norm(g)
            gh = g / gn if gn > 0 else np.zeros(2)
            m = cfg.alpha * m - cfg.beta * gh * (gh @ m)   # (alpha I - beta gh gh^T) m
        elif delta and cfg.convention == "gd_l2":
            gg = float(g @ g)
            c = min(cfg.eta, cfg.alpha / gg) if gg > 0 else 0.0
            m = cfg.alpha * m - c * g * (g @ m)
        elif delta:
            raise ValueError(f"unknown convention {cfg.convention}")
        else:
            m = cfg.alpha * m
        m = m - cfg.eta * g
        p = p + m
        v = land.value(p)
        if not np.isfinite(v):
            break
        traj.append(p.copy())
        vals.append(v)
        below = below + 1 if v < cfg.tol else 0
        if below >= cfg.patience:
            converged_at = step - cfg.patience + 1
            break
    return {
        "method": "delta" if delta else "standard",
        "converged_at": converged_at,
        "final_value": vals[-1],
        "diverged": not np.isfinite(vals[-1]) or vals[-1] > 1e6,
        "trajectory": np.array(traj),
        "values": np.array(vals),
    }


def numeric_grad(land: Landscape, p: np.ndarray, h: float = 1e-6) -> np.ndarray:
    g = np.zeros(2)
    for i in range(2):
        e = np.zeros(2)
        e[i] = h
        g[i] = (land.value(p + e) - land.value(p - e)) / (2 * h)
    return g


def describe(land: Landscape, cfg: OptConfig) -> dict:
    return {"landscape": asdict(land), "optimizer": asdict(cfg)}
