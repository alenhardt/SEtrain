#!/usr/bin/env python3
"""
Simple example showing how to use the lattice filterbank for IRM approximation.

This script demonstrates how to:
1. Load your own IRM data (T, F) format
2. Approximate it with a lattice filterbank  
3. Process audio with time-varying parameters
"""

import numpy as np
from lattice_filterbank import approximate_irm_with_lattice_filterbank, TimeVaryingLatticeFilterbank

def example_with_your_irm():
    """
    Example function showing how to use your own IRM data.
    Replace 'your_irm_data' with your actual IRM array.
    """
    
    print("🎯 IRM Lattice Filterbank Example")
    print("=" * 40)
    
    # ============================================
    # STEP 1: Prepare your IRM data
    # ============================================
    
    # Your IRM should have shape (T, F) where:
    # T = number of time frames (STFT hops)  
    # F = number of frequency bins (NFFT//2 + 1)
    
    # Example: Load your actual IRM data
    # H_mag = np.load('your_irm_file.npy')  # Shape: (T, 129) for 256-point FFT
    
    # For this example, create synthetic IRM data
    T, F = 30, 129  # 30 time frames, 129 frequency bins
    H_mag = create_example_irm(T, F)
    
    print(f"IRM shape: {H_mag.shape}")
    print(f"IRM range: {H_mag.min():.3f} to {H_mag.max():.3f}")
    
    # ============================================  
    # STEP 2: Configure lattice filterbank
    # ============================================
    
    # Basic configuration - adjust these for your application
    irm_config = approximate_irm_with_lattice_filterbank(
        H_mag,
        fs=48000,              # Your audio sample rate
        nfft=256,              # FFT size used to compute H_mag  
        num_filters=8,         # Number of filters (6-16 typical)
        filter_type='peaking', # 'peaking', 'bandpass', or 'mixed'
        freq_range=(100, 16000), # Frequency range to cover
        max_gain_db=20.0       # Maximum gain/attenuation
    )
    
    print(f"✅ Lattice filterbank configured")
    print(f"Average approximation error: {np.mean(irm_config['approximation_error']):.4f}")
    
    # ============================================
    # STEP 3: Create time-varying filterbank
    # ============================================
    
    filterbank = TimeVaryingLatticeFilterbank(irm_config)
    
    # ============================================
    # STEP 4: Process audio with your IRM
    # ============================================
    
    # Create example audio signal (replace with your actual audio)
    duration = 1.5  # seconds
    fs = 48000
    t = np.linspace(0, duration, int(fs * duration))
    
    # Example signal: chirp + noise
    f_start, f_end = 300, 3000
    test_signal = 0.7 * np.sin(2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * duration)))
    test_signal += 0.2 * np.random.randn(len(t))
    
    print(f"Processing {len(test_signal)} audio samples...")
    
    # Map audio samples to IRM frames
    # This depends on your STFT hop length
    hop_length_samples = 512  # Match your STFT analysis
    frame_indices = np.arange(len(test_signal)) // hop_length_samples
    frame_indices = np.clip(frame_indices, 0, filterbank.num_frames - 1)
    
    # Process with time-varying IRM approximation
    processed_signal = filterbank.process_buffer(test_signal, frame_indices)
    
    print(f"✅ Audio processing complete")
    
    # ============================================
    # STEP 5: Save results
    # ============================================
    
    from scipy.io import wavfile
    
    # Convert to 16-bit integer for WAV files
    test_int16 = (test_signal * 32767).astype(np.int16)
    processed_int16 = (processed_signal * 32767).astype(np.int16)
    
    # Save audio files
    wavfile.write('example_original.wav', fs, test_int16)
    wavfile.write('example_processed_with_irm.wav', fs, processed_int16)
    
    print(f"📁 Saved: example_original.wav")
    print(f"📁 Saved: example_processed_with_irm.wav")
    
    return irm_config, filterbank


def create_example_irm(T: int, F: int) -> np.ndarray:
    """
    Create example IRM data for demonstration.
    Replace this with your actual IRM loading code.
    """
    
    # Create frequency grid (matches 256-point FFT)
    frequencies = np.fft.fftfreq(256, 1/48000)[:F]
    
    # Create time-varying spectral mask
    H_mag = np.zeros((T, F))
    
    for t in range(T):
        time_factor = t / T
        
        # Example: Time-varying bandpass mask
        center_freq = 800 + 2000 * np.sin(2 * np.pi * time_factor * 1.5)
        bandwidth = 600 + 400 * np.cos(2 * np.pi * time_factor * 2.0)
        
        for f_idx, freq in enumerate(frequencies):
            if freq <= 0:
                H_mag[t, f_idx] = 0.5
            else:
                # Distance from center frequency
                freq_distance = abs(freq - center_freq)
                
                # Bandpass-like response
                if freq_distance < bandwidth / 2:
                    gain = 1.0 - 0.3 * (freq_distance / (bandwidth / 2))
                else:
                    gain = 0.2 * np.exp(-(freq_distance - bandwidth/2) / 800)
                
                H_mag[t, f_idx] = np.clip(gain, 0.1, 1.5)
    
    return H_mag


def real_time_processing_example():
    """
    Example of real-time processing with chunk-based updates.
    """
    
    print("\n🎛️ Real-Time Processing Example")
    print("=" * 40)
    
    # Create IRM configuration
    T, F = 50, 129
    H_mag = create_example_irm(T, F)
    
    irm_config = approximate_irm_with_lattice_filterbank(
        H_mag, fs=48000, num_filters=6, max_gain_db=15.0
    )
    
    # Real-time processor class
    class RealTimeIRMProcessor:
        def __init__(self, irm_config, hop_length_samples):
            self.filterbank = TimeVaryingLatticeFilterbank(irm_config)
            self.hop_length = hop_length_samples
            self.sample_count = 0
            
        def process_chunk(self, audio_chunk):
            # Determine current IRM frame
            current_frame = self.sample_count // self.hop_length
            current_frame = min(current_frame, self.filterbank.num_frames - 1)
            
            # Update filterbank parameters
            self.filterbank.update_to_frame(current_frame)
            
            # Process samples
            processed_chunk = np.zeros_like(audio_chunk)
            for i, sample in enumerate(audio_chunk):
                processed_chunk[i] = self.filterbank.process_sample(sample)
            
            self.sample_count += len(audio_chunk)
            return processed_chunk
    
    # Initialize processor
    processor = RealTimeIRMProcessor(irm_config, hop_length_samples=512)
    
    # Simulate real-time processing with chunks
    chunk_size = 1024  # Process 1024 samples at a time
    total_samples = 48000  # 1 second of audio
    
    all_processed = []
    
    print(f"Processing {total_samples} samples in chunks of {chunk_size}...")
    
    for i in range(0, total_samples, chunk_size):
        # Create audio chunk (in real app, this would come from audio input)
        t_chunk = np.linspace(i/48000, (i+chunk_size)/48000, chunk_size)
        audio_chunk = 0.5 * np.sin(2 * np.pi * 1000 * t_chunk)  # 1kHz tone
        
        # Process chunk
        processed_chunk = processor.process_chunk(audio_chunk)
        all_processed.append(processed_chunk)
        
        if i % (chunk_size * 10) == 0:  # Progress update
            progress = 100 * i / total_samples
            frame = processor.sample_count // 512
            print(f"  Progress: {progress:.1f}% (IRM frame {frame})")
    
    # Combine all chunks
    processed_signal = np.concatenate(all_processed)
    
    print(f"✅ Real-time processing simulation complete")
    print(f"Processed {len(processed_signal)} samples with {T} parameter updates")
    
    return processed_signal


if __name__ == "__main__":
    print("🎯 LATTICE FILTERBANK IRM APPROXIMATION EXAMPLES")
    print("=" * 60)
    
    # Run basic example
    irm_config, filterbank = example_with_your_irm()
    
    # Run real-time processing example  
    real_time_processed = real_time_processing_example()
    
    print("\n" + "=" * 60)
    print("✅ EXAMPLES COMPLETE")
    print("=" * 60)
    print("\nTo use with your own data:")
    print("1. Replace create_example_irm() with your IRM loading code")
    print("2. Ensure your IRM has shape (T, F) with F = NFFT//2 + 1")
    print("3. Adjust fs, nfft, and hop_length to match your STFT parameters")
    print("4. Tune num_filters and max_gain_db for your application")
    print("\nSee IRM_USAGE_GUIDE.md for detailed documentation!")