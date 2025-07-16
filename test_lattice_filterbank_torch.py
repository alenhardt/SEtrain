import numpy as np
import torch
import pytest
from lattice_filterbank import FilterParams, LatticeFilterbank
from lattice_filterbank_torch import FilterParams as TorchFilterParams, LatticeFilterbankTorch

def test_torch_vs_numpy_equivalence():
    fs = 48000
    duration = 0.5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    test_signal = (
        np.sin(2 * np.pi * 440 * t) +
        0.5 * np.sin(2 * np.pi * 1000 * t) +
        0.3 * np.sin(2 * np.pi * 2000 * t) +
        0.1 * np.random.randn(len(t))
    )
    filter_params = [
        FilterParams(center_freq=500, gain=6, q_factor=1.0, fs=fs, filter_type='peaking'),
        FilterParams(center_freq=1500, gain=-3, q_factor=2.0, fs=fs, filter_type='peaking'),
        FilterParams(center_freq=3000, gain=4, q_factor=0.7, fs=fs, filter_type='peaking'),
    ]
    torch_params = [
        TorchFilterParams(center_freq=500, gain=6, q_factor=1.0, fs=fs, filter_type='peaking'),
        TorchFilterParams(center_freq=1500, gain=-3, q_factor=2.0, fs=fs, filter_type='peaking'),
        TorchFilterParams(center_freq=3000, gain=4, q_factor=0.7, fs=fs, filter_type='peaking'),
    ]
    # Numpy version
    lattice_fb = LatticeFilterbank(filter_params)
    np_out = lattice_fb.filter_signal(test_signal)
    # Torch version
    torch_signal = torch.tensor(test_signal, dtype=torch.float32)
    torch_fb = LatticeFilterbankTorch(torch_params)
    torch_out = torch_fb(torch_signal).detach().cpu().numpy()
    # Compare
    diff = np_out - torch_out
    max_diff = np.max(np.abs(diff))
    rms_diff = np.sqrt(np.mean(diff**2))
    print(f"Max diff: {max_diff:.2e}")
    print(f"RMS diff: {rms_diff:.2e}")
    assert max_diff < 1e-5
    assert rms_diff < 1e-6

if __name__ == "__main__":
    test_torch_vs_numpy_equivalence()