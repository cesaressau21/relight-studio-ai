"""Compute device selection and GPU diagnostics."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import torch

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    name: str
    cuda: bool
    capability: tuple[int, int] | None
    total_memory: int  # bytes, 0 on CPU
    warning: str = ""
    problem: bool = False  # a GPU is there but unusable (not just a computer without CUDA)

    @property
    def label(self) -> str:
        return self.name if self.cuda else "CPU"


@lru_cache(maxsize=1)
def select_device() -> DeviceInfo:
    """Use CUDA when available and functional, else fall back to the CPU.

    ``RELIGHT_DEVICE=cpu`` forces CPU processing.
    """
    if os.environ.get("RELIGHT_DEVICE", "").lower() == "cpu":
        return DeviceInfo(torch.device("cpu"), "CPU", False, None, 0, "CPU forced by RELIGHT_DEVICE")
    if not torch.cuda.is_available():
        return DeviceInfo(
            torch.device("cpu"), "CPU", False, None, 0,
            "CUDA is not available: processing runs on the CPU and will be slower.",
        )
    try:
        cap = torch.cuda.get_device_capability(0)
        arch = f"sm_{cap[0]}{cap[1]}"
        if arch not in torch.cuda.get_arch_list():
            raise RuntimeError(f"this PyTorch build has no kernels for {arch}")
        probe = torch.ones(8, device="cuda") * 2
        if float(probe.sum()) != 16.0:
            raise RuntimeError("CUDA self-test returned a wrong result")
        props = torch.cuda.get_device_properties(0)
        return DeviceInfo(torch.device("cuda"), props.name, True, cap, props.total_memory)
    except Exception as exc:  # broken driver / unsupported architecture
        log.exception("CUDA probe failed")
        return DeviceInfo(
            torch.device("cpu"), "CPU", False, None, 0,
            f"CUDA present but unusable ({exc}); falling back to the CPU.", problem=True,
        )


def vram_usage() -> tuple[int, int] | None:
    """(used, total) bytes of device memory, or None on CPU."""
    info = select_device()
    if not info.cuda:
        return None
    free, total = torch.cuda.mem_get_info()
    return total - free, total
