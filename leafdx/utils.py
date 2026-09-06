"""Helpers: exact reproducibility (seeding) and inference-latency measurement."""
import os
import random
import time

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False):
    """Fixes the Python/NumPy/Torch global RNGs (per-run reproducibility)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True
        # TF32 on Ampere+ (RTX 30/40, A100, H100): a clear speed-up in matmul/conv with no
        # measurable loss in classification accuracy. Disabled under --deterministic.
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass


def fmt_duration(sec: float) -> str:
    """Formats seconds for humans: '1h 23m 45s' / '2m 5s' / '9.3s'."""
    sec = float(sec)
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{sec:.1f}s"


def capture_environment() -> dict:
    """Hardware/software environment (reported in the paper for reproducibility)."""
    import platform
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    for pkg in ("timm", "numpy", "sklearn", "scipy", "torchvision"):
        try:
            env[pkg] = __import__(pkg).__version__
        except Exception:
            env[pkg] = None
    return env


@torch.no_grad()
def measure_latency(model, img_size: int, device: str,
                    warmup: int = 10, iters: int = 50, n_views: int = 1) -> dict:
    """Single-image inference latency (ms) and throughput (FPS). A deployment metric.

    `n_views` is the number of TTA views. Since the REPORTED accuracy of the proposed
    model is obtained with TTA (4 forward passes), measuring the latency with a single
    pass would under-state the deployment cost by a factor of 4 — an obvious
    inconsistency. Hence both the real (TTA-inclusive) and the single-pass value are
    reported.
    """
    was_training = model.training
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)

    def _time_once():
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters

    try:
        dt1 = _time_once()
        dt = dt1 * max(1, int(n_views))
        out = {"latency_ms": dt * 1000.0, "fps": 1.0 / dt,
               "latency_ms_single_pass": dt1 * 1000.0, "tta_views": int(n_views)}
    except Exception:
        out = {"latency_ms": float("nan"), "fps": float("nan"),
               "latency_ms_single_pass": float("nan"), "tta_views": int(n_views)}
    if was_training:
        model.train()
    return out
