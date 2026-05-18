from __future__ import annotations

import platform
import shutil

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def detect_compute_environment() -> dict[str, bool | str]:
    """Detect the current compute environment in a simple, explicit way."""
    has_nvidia_gpu = shutil.which("nvidia-smi") is not None
    is_macos = platform.system() == "Darwin"
    is_apple_silicon = is_macos and platform.machine() == "arm64"

    mps_built = False
    mps_available = False
    if torch is not None:
        mps_built = torch.backends.mps.is_built()
        mps_available = torch.backends.mps.is_available()

    if has_nvidia_gpu:
        preferred_device = "cuda"
    elif is_apple_silicon and mps_available:
        preferred_device = "mps"
    else:
        preferred_device = "cpu"

    return {
        "preferred_device": preferred_device,
        "has_nvidia_gpu": has_nvidia_gpu,
        "is_macos": is_macos,
        "is_apple_silicon": is_apple_silicon,
        "mps_built": mps_built,
        "mps_available": mps_available,
    }


def build_xgboost_runtime_params() -> dict[str, str]:
    """Build device-related parameters for XGBoost."""
    environment = detect_compute_environment()

    # XGBoost 这里优先走 CUDA；Apple MPS 仅保留环境判断，不强行开启不兼容参数。
    if environment["preferred_device"] == "cuda":
        return {
            "tree_method": "hist",
            "device": "cuda",
        }

    return {
        "tree_method": "hist",
    }


def build_lightgbm_runtime_params() -> dict[str, str | int]:
    """Build device-related parameters for LightGBM."""
    environment = detect_compute_environment()

    # LightGBM 的 GPU 路线主要还是 CUDA/OpenCL；当前项目只在 NVIDIA 分支下显式开启。
    if environment["preferred_device"] == "cuda":
        return {
            "device_type": "gpu",
            "gpu_device_id": 0,
        }

    return {
        "device_type": "cpu",
    }


def build_catboost_runtime_params() -> dict[str, str]:
    """Build device-related parameters for CatBoost."""
    environment = detect_compute_environment()

    # CatBoost 在这个项目里也只对 CUDA 分支启用 GPU。
    if environment["preferred_device"] == "cuda":
        return {
            "task_type": "GPU",
            "devices": "0",
        }

    return {
        "task_type": "CPU",
    }
