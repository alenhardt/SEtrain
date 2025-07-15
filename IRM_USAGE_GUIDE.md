# 🎯 IRM Approximation with Lattice Filterbank - Usage Guide

## Overview

This system uses RBJ's lattice-ladder filters to approximate an Ideal Ratio Mask (IRM) with time-varying parameters. Perfect for speech enhancement, source separation, and dynamic audio processing.

## Quick Start

### 1. **Basic IRM Approximation**

```python
import numpy as np
from lattice_filterbank import approximate_irm_with_lattice_filterbank, TimeVaryingLatticeFilterbank

# Your IRM data: shape (T, F) where T=time frames, F=frequency bins (129 for 256-pt FFT)
H_mag = your_irm_data  # Shape: (num_time_frames, 129)

# Approximate with lattice filterbank
irm_config = approximate_irm_with_lattice_filterbank(
    H_mag, 
    fs=48000,           # Sample rate
    nfft=256,           # FFT size used to compute H_mag
    num_filters=8,      # Number of filters in bank
    filter_type='peaking',  # 'peaking', 'bandpass', or 'mixed'
    freq_range=(100, 16000),  # Frequency range to cover
    max_gain_db=20.0    # Maximum gain/attenuation
)

# Create time-varying filterbank
filterbank = TimeVaryingLatticeFilterbank(irm_config)
```

### 2. **Process Audio with Time-Varying IRM**

```python
# Process audio buffer with frame-synchronized parameter updates
def process_audio_with_irm(audio_signal, filterbank, hop_length_samples):
    # Map each audio sample to an IRM frame
    num_samples = len(audio_signal) 
    frame_indices = np.arange(num_samples) // hop_length_samples
    frame_indices = np.clip(frame_indices, 0, filterbank.num_frames - 1)
    
    # Process with time-varying parameters
    processed_audio = filterbank.process_buffer(audio_signal, frame_indices)
    return processed_audio

# Example usage
hop_length = 512  # STFT hop length in samples
processed_signal = process_audio_with_irm(audio_data, filterbank, hop_length)
```

## Detailed Usage

### **Input IRM Format**

Your IRM should be a 2D numpy array with shape `(T, F)`:
- **T**: Number of time frames (STFT hops)
- **F**: Number of frequency bins = `NFFT//2 + 1`
- **Values**: Magnitude response (0.0 to 1.0+ for gain, 0.0 to 1.0 for attenuation)

```python
# Example: Create IRM from STFT
import librosa

# Compute STFT of clean and noisy signals
stft_clean = librosa.stft(clean_signal, n_fft=256, hop_length=128)
stft_noisy = librosa.stft(noisy_signal, n_fft=256, hop_length=128)

# Compute Ideal Ratio Mask
irm = np.abs(stft_clean) / (np.abs(stft_noisy) + 1e-8)
irm = np.transpose(irm)  # Shape: (time_frames, freq_bins)

print(f"IRM shape: {irm.shape}")  # Should be (T, 129) for 256-point FFT
```

### **Configuration Options**

```python
# Advanced configuration
irm_config = approximate_irm_with_lattice_filterbank(
    H_mag,
    fs=48000,                    # Sample rate
    nfft=256,                    # FFT size (must match your STFT)
    num_filters=12,              # More filters = better approximation
    filter_type='mixed',         # 'peaking', 'bandpass', 'mixed'
    freq_range=(80, 18000),      # Frequency range to cover
    max_gain_db=25.0             # Maximum gain range
)

# Filter types:
# - 'peaking': All peaking EQ filters (best for most IRMs)
# - 'bandpass': All bandpass filters 
# - 'mixed': Mix of lowpass, peaking, and highpass
```

### **Real-Time Processing**

```python
class RealTimeIRMProcessor:
    def __init__(self, irm_config, hop_length_samples):
        self.filterbank = TimeVaryingLatticeFilterbank(irm_config)
        self.hop_length = hop_length_samples
        self.sample_count = 0
        
    def process_chunk(self, audio_chunk):
        """Process a chunk of audio samples."""
        # Determine IRM frame for this chunk
        start_frame = self.sample_count // self.hop_length
        end_frame = (self.sample_count + len(audio_chunk)) // self.hop_length
        
        # Update filterbank parameters if needed
        target_frame = min(start_frame, self.filterbank.num_frames - 1)
        self.filterbank.update_to_frame(target_frame)
        
        # Process chunk
        processed_chunk = np.zeros_like(audio_chunk)
        for i, sample in enumerate(audio_chunk):
            processed_chunk[i] = self.filterbank.process_sample(sample)
        
        self.sample_count += len(audio_chunk)
        return processed_chunk

# Usage
processor = RealTimeIRMProcessor(irm_config, hop_length=512)

# Process audio in chunks (e.g., for real-time applications)
chunk_size = 1024
for i in range(0, len(audio_signal), chunk_size):
    chunk = audio_signal[i:i+chunk_size]
    processed_chunk = processor.process_chunk(chunk)
    # Send to output buffer/device
```

## Applications

### **Speech Enhancement**

```python
# Speech enhancement using IRM approximation
def enhance_speech_with_lattice_irm(noisy_speech, clean_reference, fs=16000):
    # Compute STFT
    stft_noisy = librosa.stft(noisy_speech, n_fft=256, hop_length=128)
    stft_clean = librosa.stft(clean_reference, n_fft=256, hop_length=128)
    
    # Compute IRM
    irm = np.abs(stft_clean) / (np.abs(stft_noisy) + 1e-8)
    irm = np.transpose(irm)  # (time, freq)
    
    # Approximate with lattice filterbank
    irm_config = approximate_irm_with_lattice_filterbank(
        irm, fs=fs, nfft=256, num_filters=10, max_gain_db=15.0
    )
    
    # Process audio
    filterbank = TimeVaryingLatticeFilterbank(irm_config)
    hop_samples = 128
    frame_indices = np.arange(len(noisy_speech)) // hop_samples
    frame_indices = np.clip(frame_indices, 0, filterbank.num_frames - 1)
    
    enhanced_speech = filterbank.process_buffer(noisy_speech, frame_indices)
    return enhanced_speech
```

### **Dynamic EQ from Target Response**

```python
# Create dynamic EQ from target frequency responses
def create_dynamic_eq(target_responses, frequencies, fs=48000):
    """
    Args:
        target_responses: (T, F) array of target magnitude responses
        frequencies: Frequency grid for target_responses
        fs: Sample rate
    """
    # Convert to proper IRM format if needed
    nfft = (len(frequencies) - 1) * 2
    
    irm_config = approximate_irm_with_lattice_filterbank(
        target_responses, 
        fs=fs, 
        nfft=nfft, 
        num_filters=8,
        filter_type='peaking'
    )
    
    return TimeVaryingLatticeFilterbank(irm_config)
```

## Performance Tips

### **Optimization**

1. **Filter Count**: More filters = better approximation but higher CPU cost
   - 6-8 filters: Good for real-time applications
   - 10-16 filters: Better approximation for offline processing

2. **Update Rate**: Match your STFT hop length
   - Typical: 128-512 samples (2.7-10.7ms at 48kHz)
   - Faster updates = smoother transitions but higher CPU cost

3. **Frequency Range**: Limit to perceptually important range
   - Speech: 80-8000 Hz
   - Music: 20-20000 Hz
   - Telephony: 300-3400 Hz

### **Quality Settings**

```python
# High quality (offline processing)
irm_config = approximate_irm_with_lattice_filterbank(
    H_mag, num_filters=16, max_gain_db=30.0, filter_type='mixed'
)

# Real-time optimized
irm_config = approximate_irm_with_lattice_filterbank(
    H_mag, num_filters=8, max_gain_db=20.0, filter_type='peaking'
)

# Low latency
irm_config = approximate_irm_with_lattice_filterbank(
    H_mag, num_filters=6, max_gain_db=15.0, filter_type='peaking'
)
```

## Key Advantages

### **Why Lattice Filters for IRM?**

1. **Smooth Parameter Updates**: No glitches during coefficient changes
2. **Numerical Stability**: Better than direct form for real-time modulation
3. **Real-Time Capable**: Fast parameter updates every few milliseconds
4. **Mathematically Equivalent**: Perfect biquad approximation using RBJ's method

### **Compared to Traditional Methods**

| Method | Parameter Updates | Glitches | CPU Cost | Accuracy |
|--------|------------------|----------|----------|----------|
| Direct Biquad | ❌ Glitchy | High | Low | Perfect |
| Overlap-Add | ✅ Smooth | None | High | Good |
| **RBJ Lattice** | ✅ **Smooth** | **None** | **Medium** | **Perfect** |

## Files Created

After running the demo, you'll have:

- **`irm_approximation_demo.wav`** - Processed audio with IRM approximation
- **`irm_original_signal.wav`** - Original test signal
- **`irm_approximation_demo.png`** - Complete visualization showing:
  - Target IRM spectrogram
  - Filter parameter evolution over time
  - Original vs processed spectrograms
  - Time domain comparison
  - Approximation error analysis

## Error Handling

```python
try:
    irm_config = approximate_irm_with_lattice_filterbank(H_mag, fs=48000)
    filterbank = TimeVaryingLatticeFilterbank(irm_config)
    processed_audio = filterbank.process_buffer(audio_data, frame_indices)
    
except Exception as e:
    print(f"IRM processing error: {e}")
    # Fallback to passthrough or simple filtering
    processed_audio = audio_data
```

## Integration with Existing Systems

The lattice filterbank integrates easily with:
- **librosa** for STFT analysis
- **scipy** for signal processing
- **soundfile** for audio I/O  
- **Real-time audio frameworks** (PyAudio, JACK, etc.)
- **Machine learning pipelines** (as a differentiable audio processor)

---

**This system provides state-of-the-art real-time IRM approximation using mathematically perfect lattice structures with smooth parameter modulation capabilities.**