import torch
import torch.nn as nn
from typing import List, Tuple
import math

class FilterParams:
    """Parameters for a single filter in the filterbank."""
    def __init__(self, center_freq: float, gain: float, q_factor: float, fs: float, filter_type: str = 'peaking'):
        self.center_freq = center_freq
        self.gain = gain
        self.q_factor = q_factor
        self.fs = fs
        self.filter_type = filter_type


def design_biquad_coefficients(params: FilterParams) -> Tuple[float, float, float, float, float]:
    omega = 2 * math.pi * params.center_freq / params.fs
    sin_omega = math.sin(omega)
    cos_omega = math.cos(omega)
    alpha = sin_omega / (2 * params.q_factor)
    A = 10 ** (params.gain / 40)
    if params.filter_type == 'peaking':
        b0 = 1 + alpha * A
        b1 = -2 * cos_omega
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_omega
        a2 = 1 - alpha / A
    elif params.filter_type == 'lowpass':
        b0 = (1 - cos_omega) / 2
        b1 = 1 - cos_omega
        b2 = (1 - cos_omega) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_omega
        a2 = 1 - alpha
    elif params.filter_type == 'highpass':
        b0 = (1 + cos_omega) / 2
        b1 = -(1 + cos_omega)
        b2 = (1 + cos_omega) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_omega
        a2 = 1 - alpha
    elif params.filter_type == 'bandpass':
        b0 = alpha
        b1 = 0
        b2 = -alpha
        a0 = 1 + alpha
        a1 = -2 * cos_omega
        a2 = 1 - alpha
    else:
        raise ValueError(f"Unsupported filter type: {params.filter_type}")
    b0, b1, b2, a1, a2 = b0/a0, b1/a0, b2/a0, a1/a0, a2/a0
    return b0, b1, b2, a1, a2


def biquad_to_lattice(b0, b1, b2, a1, a2):
    k2 = a2
    k1 = a1 / (a2 + 1)
    v2 = b2
    v1 = b1 - a1 * b2
    v0 = b0 - (a1 / (a2 + 1)) * b1 + ((a1*a1 / (a2 + 1)) - a2) * b2
    return k1, k2, v0, v1, v2


class LatticeFilterTorch(nn.Module):
    def __init__(self, k1, k2, v0, v1, v2):
        super().__init__()
        self.k1 = nn.Parameter(torch.tensor(k1, dtype=torch.float32))
        self.k2 = nn.Parameter(torch.tensor(k2, dtype=torch.float32))
        self.v0 = nn.Parameter(torch.tensor(v0, dtype=torch.float32))
        self.v1 = nn.Parameter(torch.tensor(v1, dtype=torch.float32))
        self.v2 = nn.Parameter(torch.tensor(v2, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,) or (batch, N)
        s1 = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device) if x.ndim == 2 else 0.0
        s2 = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device) if x.ndim == 2 else 0.0
        y = []
        for i in range(x.shape[-1]):
            xi = x[:, i] if x.ndim == 2 else x[i]
            e0 = xi
            e1 = e0 - self.k1 * s1
            e2 = e1 - self.k2 * s2
            b1 = self.k1 * e1 + s1
            b2 = self.k2 * e2 + s2
            yi = self.v0 * e0 + self.v1 * e1 + self.v2 * e2
            s2 = s1
            s1 = b1
            y.append(yi)
        y = torch.stack(y, dim=-1)
        return y


class LatticeFilterbankTorch(nn.Module):
    def __init__(self, filter_params: List[FilterParams]):
        super().__init__()
        self.filters = nn.ModuleList()
        for params in filter_params:
            b0, b1, b2, a1, a2 = design_biquad_coefficients(params)
            k1, k2, v0, v1, v2 = biquad_to_lattice(b0, b1, b2, a1, a2)
            self.filters.append(LatticeFilterTorch(k1, k2, v0, v1, v2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,) or (batch, N)
        outputs = [filt(x) for filt in self.filters]
        return torch.stack(outputs, dim=0).sum(dim=0)