# 16kHz Lattice Filterbank IRM Approximation Guide

## Overview

This guide demonstrates how to use the lattice filterbank system to approximate Ideal Ratio Masks (IRMs) for 16kHz audio processing. The implementation successfully adapts the 48kHz system to work optimally with 16kHz audio through appropriate parameter adjustments.

## Quick Start

### 1. Basic 16kHz Demo

```python
python3 test_16khz_demo.py
```

This creates synthetic 16kHz audio and demonstrates:
- IRM computation from noisy/clean audio pairs
- Lattice filterbank approximation
- Real-time audio processing
- Comprehensive visualization

### 2. Generated Files

- `clean_16khz.wav` - Original clean signal
- `noisy_16khz.wav` - Noisy input signal  
- `processed_16khz.wav` - Lattice filterbank output
- `16khz_lattice_demo.png` - Visualization plots

## Key Adaptations for 16kHz

### 1. Sample Rate Configuration

```python
# Default parameters adjusted for 16kHz
fs = 16000                    # Sample rate
freq_range = (100, 7000)      # Frequency range (up to ~44% of Nyquist)
nfft = 256                    # FFT size (gives 129 frequency bins)
```

### 2. Frequency Range Considerations

- **Nyquist frequency**: 8000 Hz
- **Usable range**: 100-7000 Hz (avoids aliasing issues)
- **FFT bins**: 129 bins from 0-8000 Hz (62.5 Hz per bin)

### 3. Time-Frequency Analysis

```python
# Spectrogram parameters for 16kHz
nperseg = 256          # Window length (16ms at 16kHz)
noverlap = 128         # 50% overlap (8ms hop)
```

## IRM Input Format

The lattice filterbank expects a 2D IRM array with shape `(time_frames, frequency_bins)`:

```python
# Compute STFT spectrograms
f_clean, t_clean, S_clean = spectrogram(clean, fs=16000, nperseg=256, 
                                       noverlap=128, return_onesided=True)
f_noisy, t_noisy, S_noisy = spectrogram(noisy, fs=16000, nperseg=256,
                                       noverlap=128, return_onesided=True)

# Compute magnitude IRM
mag_clean = np.abs(S_clean)
mag_noisy = np.abs(S_noisy)
mag_noisy_safe = np.maximum(mag_noisy, 1e-10)
irm_full = mag_clean / mag_noisy_safe

# Transpose to get (time × frequency) format
H_mag = irm_full.T  # Shape: (time_frames, frequency_bins)
```

## Lattice Filterbank Configuration

### 1. Basic Configuration

```python
from lattice_filterbank import approximate_irm_with_lattice_filterbank, TimeVaryingLatticeFilterbank

irm_config = approximate_irm_with_lattice_filterbank(
    H_mag,                    # IRM array (time_frames, freq_bins)
    fs=16000,                # Sample rate
    nfft=256,                # FFT size
    num_filters=6,           # Number of filters per time frame
    filter_type='peaking',   # Filter type
    freq_range=(100, 7000),  # Frequency range for 16kHz
    max_gain_db=15.0         # Maximum filter gain
)
```

### 2. Processing Audio

```python
# Create filterbank
filterbank = TimeVaryingLatticeFilterbank(irm_config)

# Process audio buffer
processed_audio = filterbank.process_buffer(noisy_audio)
```

## Configuration Parameters

### Essential Parameters for 16kHz

| Parameter | 16kHz Value | 48kHz Value | Notes |
|-----------|-------------|-------------|-------|
| `fs` | 16000 | 48000 | Sample rate |
| `freq_range` | (100, 7000) | (100, 16000) | Frequency range |
| `nfft` | 256 | 256 | FFT size |
| `num_filters` | 4-8 | 6-12 | Fewer filters for narrower band |
| `max_gain_db` | 15.0 | 20.0 | Slightly conservative |

### Filter Types Supported

- `'peaking'` - Parametric EQ filters (most common for IRM)
- `'lowpass'` - Low-pass filters
- `'highpass'` - High-pass filters  
- `'bandpass'` - Band-pass filters

## Performance Characteristics

### Typical Results with 16kHz Audio

```
✨ Approximation Results:
   Time frames processed: 374
   Filters per frame: 6
   Average error: 0.0000
   Max error: 0.0000
```

### Filter Configuration Example

```
🎛️ Filter Configuration (First Time Frame):
   Filter 1: 100 Hz, Q=2.00, Gain=10.9 dB
   Filter 2: 234 Hz, Q=2.00, Gain=15.0 dB
   Filter 3: 547 Hz, Q=2.00, Gain=15.0 dB
   Filter 4: 1280 Hz, Q=2.00, Gain=15.0 dB
   Filter 5: 2993 Hz, Q=2.00, Gain=-15.0 dB
   Filter 6: 7000 Hz, Q=2.00, Gain=-15.0 dB
```

## Real-World Usage Example

### With Actual Audio Files

```python
import librosa
import numpy as np
from scipy.signal import spectrogram
from lattice_filterbank import approximate_irm_with_lattice_filterbank, TimeVaryingLatticeFilterbank

# Load audio files (forced to 16kHz)
clean_audio, sr = librosa.load('clean_speech.wav', sr=16000)
noisy_audio, sr = librosa.load('noisy_speech.wav', sr=16000)

# Align lengths
min_length = min(len(clean_audio), len(noisy_audio))
clean_audio = clean_audio[:min_length]
noisy_audio = noisy_audio[:min_length]

# Compute IRM
f_clean, t_clean, S_clean = spectrogram(clean_audio, fs=16000, nperseg=256, 
                                       noverlap=128, return_onesided=True)
f_noisy, t_noisy, S_noisy = spectrogram(noisy_audio, fs=16000, nperseg=256,
                                       noverlap=128, return_onesided=True)

mag_clean = np.abs(S_clean)
mag_noisy = np.abs(S_noisy)
mag_noisy_safe = np.maximum(mag_noisy, 1e-10)
irm_full = mag_clean / mag_noisy_safe
H_mag = irm_full.T

# Create lattice filterbank
irm_config = approximate_irm_with_lattice_filterbank(
    H_mag, fs=16000, nfft=256, num_filters=6, 
    filter_type='peaking', freq_range=(100, 7000), max_gain_db=15.0
)

# Process audio
filterbank = TimeVaryingLatticeFilterbank(irm_config)
enhanced_audio = filterbank.process_buffer(noisy_audio)

# Save result
import soundfile as sf
sf.write('enhanced_16khz.wav', enhanced_audio, 16000)
```

## Advantages of 16kHz Processing

### 1. Computational Efficiency
- Lower sample rate = faster processing
- Fewer frequency bins to optimize
- Reduced memory requirements

### 2. Speech Processing Focus
- 16kHz captures full speech bandwidth (0-8kHz)
- Most speech energy below 4kHz
- Sufficient for most speech enhancement tasks

### 3. Real-Time Feasibility
- Lower latency due to smaller FFT windows
- Faster filter updates
- More suitable for real-time applications

## Troubleshooting

### Common Issues

1. **Wrong IRM shape**: Ensure `H_mag.shape = (time_frames, frequency_bins)`
2. **Frequency range**: Keep `freq_range[1] < fs/2` (< 8000 Hz for 16kHz)
3. **FilterParams access**: Use `.center_freq`, `.gain`, `.q_factor` attributes
4. **Method names**: Use `process_buffer()` not `process_audio()`

### Verification

```python
print(f"IRM shape: {H_mag.shape}")
print(f"Frequency range: {f_clean[0]:.1f} - {f_clean[-1]:.1f} Hz")
print(f"Sample rate: {sr} Hz")
print(f"Time frames: {H_mag.shape[0]}")
print(f"Frequency bins: {H_mag.shape[1]}")
```

## Integration with Existing Systems

The 16kHz lattice filterbank can be easily integrated into existing audio processing pipelines:

- **Speech enhancement systems**
- **Noise reduction applications**  
- **Real-time audio processing**
- **Hearing aid processing**
- **Voice communication systems**

The RBJ lattice structure provides glitch-free parameter modulation, making it ideal for time-varying spectral processing tasks where smooth transitions are critical.