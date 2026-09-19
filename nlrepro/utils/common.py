from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time

import yaml


def load_yaml(path: str, overrides: list[str] | None = None) -> dict:
    """Load YAML; apply dotted overrides like `train.max_steps=100`."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = parse_scalar(val)
    return cfg


def parse_scalar(val: str):
    """yaml.safe_load, but '5e-4' (a string in YAML 1.1) becomes a float."""
    v = yaml.safe_load(val)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            pass
    return v


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class JsonlLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = open(path, "a", buffering=1)

    def log(self, **kw) -> None:
        kw.setdefault("time", time.time())
        self.f.write(json.dumps(kw) + "\n")
        print(" ".join(f"{k}={_fmt(v)}" for k, v in kw.items() if k != "time"), flush=True)


def _fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else v


def environment() -> dict:
    env = {"python": platform.python_version(), "host": platform.node()}
    try:
        env["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        env["git_commit"] = None
    try:
        import torch
        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["gpu_mem_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except ImportError:
        pass
    for mod in ("transformers", "datasets", "lm_eval"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    return env


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    """YAML with optional `base: other.yaml` inheritance (resolved relative to the file)."""
    cfg = load_yaml(path)
    if "base" in cfg:
        parent = load_config(os.path.join(os.path.dirname(os.path.abspath(path)), cfg.pop("base")))
        cfg = deep_merge(parent, cfg)
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = parse_scalar(val)
    return cfg
