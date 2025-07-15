#!/usr/bin/env python3
"""
Demo script for 16kHz lattice filterbank IRM approximation with synthetic data.
This works without requiring external audio files.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import soundfile as sf
from lattice_filterbank import approximate_irm_with_lattice_filterbank, TimeVaryingLatticeFilterbank

def create_synthetic_audio_16khz(duration=3.0, fs=16000):
    """Create synthetic noisy and clean audio for testing."""
    print(f"🎵 Creating synthetic 16kHz audio ({duration}s)...")
    
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Clean signal: combination of tones
    clean = (0.3 * np.sin(2 * np.pi * 440 * t) +        # A4 note
             0.2 * np.sin(2 * np.pi * 880 * t) +        # A5 note  
             0.15 * np.sin(2 * np.pi * 1320 * t) +      # E6 note
             0.1 * np.sin(2 * np.pi * 2200 * t))        # Higher frequency
    
    # Add some time-varying component
    clean += 0.1 * np.sin(2 * np.pi * 200 * t * (1 + 0.5 * np.sin(2 * np.pi * 0.5 * t)))
    
    # Noise: colored noise that affects different frequencies differently
    noise = np.random.randn(len(t))
    
    # Apply frequency-dependent noise (more noise at high frequencies)
    from scipy.signal import butter, filtfilt
    
    # Low-frequency noise component
    b_low, a_low = butter(4, 1000 / (fs/2), 'low')
    low_noise = filtfilt(b_low, a_low, noise) * 0.8
    
    # High-frequency noise component  
    b_high, a_high = butter(4, 2000 / (fs/2), 'high')
    high_noise = filtfilt(b_high, a_high, noise) * 1.5
    
    total_noise = low_noise + high_noise
    
    # Noisy signal
    noisy = clean + total_noise
    
    # Normalize
    clean = clean / np.max(np.abs(clean)) * 0.8
    noisy = noisy / np.max(np.abs(noisy)) * 0.8
    
    print(f"   Clean signal: {len(clean)} samples")
    print(f"   Noisy signal: {len(noisy)} samples") 
    print(f"   SNR: {20 * np.log10(np.std(clean) / np.std(total_noise)):.1f} dB")
    
    return clean, noisy, fs

def compute_synthetic_irm(clean, noisy, nfft=256, fs=16000):
    """Compute IRM from synthetic audio."""
    print(f"🔬 Computing IRM from synthetic audio...")
    
    # Compute spectrograms
    f_clean, t_clean, S_clean = spectrogram(clean, fs=fs, nperseg=nfft, 
                                           noverlap=nfft//2, return_onesided=True)
    f_noisy, t_noisy, S_noisy = spectrogram(noisy, fs=fs, nperseg=nfft,
                                           noverlap=nfft//2, return_onesided=True)
    
    # Compute magnitude IRM
    mag_clean = np.abs(S_clean)
    mag_noisy = np.abs(S_noisy)
    
    # Avoid division by zero
    mag_noisy_safe = np.maximum(mag_noisy, 1e-10)
    irm_full = mag_clean / mag_noisy_safe
    
    # IRM is already in the correct format (frequency × time)
    # Transpose to get (time × frequency) as expected by lattice filterbank
    H_mag = irm_full.T  # Shape: (time_frames, frequency_bins)
    
    print(f"   Frequency bins: {len(f_clean)} (up to {f_clean[-1]:.0f} Hz)")
    print(f"   Time frames: {H_mag.shape[0]}")
    print(f"   IRM shape: {H_mag.shape}")
    print(f"   IRM range: {H_mag.min():.3f} to {H_mag.max():.3f}")
    
    return H_mag, f_clean, irm_full

def demo_16khz_lattice_filterbank():
    """Main demo function."""
    print("🎯 16kHz Lattice Filterbank IRM Demo")
    print("=" * 50)
    
    # Create synthetic data
    clean_audio, noisy_audio, fs = create_synthetic_audio_16khz()
    
    # Compute IRM
    H_mag, frequencies, irm_full = compute_synthetic_irm(clean_audio, noisy_audio, fs=fs)
    
    # Create lattice filterbank approximation
    print(f"\n🎛️ Creating lattice filterbank approximation...")
    irm_config = approximate_irm_with_lattice_filterbank(
        H_mag, 
        fs=fs,
        nfft=256,
        num_filters=6,
        filter_type='peaking',
        freq_range=(100, 7000),  # Appropriate for 16kHz
        max_gain_db=15.0
    )
    
    print(f"\n✨ Approximation Results:")
    print(f"   Time frames processed: {len(irm_config['filter_params'])}")
    print(f"   Filters per frame: {len(irm_config['filter_params'][0])}")
    print(f"   Average error: {np.mean(irm_config['approximation_error']):.4f}")
    print(f"   Max error: {np.max(irm_config['approximation_error']):.4f}")
    
    # Create filterbank and process audio
    filterbank = TimeVaryingLatticeFilterbank(irm_config)
    
    print(f"\n🔊 Processing audio through filterbank...")
    processed_audio = filterbank.process_buffer(noisy_audio)
    
    # Save audio files
    sf.write('clean_16khz.wav', clean_audio, fs)
    sf.write('noisy_16khz.wav', noisy_audio, fs)
    sf.write('processed_16khz.wav', processed_audio, fs)
    
    print(f"\n💾 Saved audio files:")
    print(f"   clean_16khz.wav - Original clean signal")
    print(f"   noisy_16khz.wav - Noisy input")
    print(f"   processed_16khz.wav - Lattice filterbank output")
    
    # Create visualization
    plot_results(H_mag, frequencies, irm_config, clean_audio, noisy_audio, 
                processed_audio, fs)
    
    return irm_config, filterbank

def plot_results(H_mag, frequencies, irm_config, clean_audio, noisy_audio, 
                processed_audio, fs):
    """Create visualization plots."""
    print(f"\n📊 Creating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('16kHz Lattice Filterbank IRM Approximation Demo', fontsize=16)
    
    # Plot 1: IRM and approximation
    ax1 = axes[0, 0]
    # Use time-averaged IRM for visualization
    H_mag_avg = np.mean(H_mag, axis=0)
    ax1.semilogx(frequencies, 20*np.log10(H_mag_avg + 1e-10), 'b-', linewidth=2, 
                 label='Target IRM (time-averaged)', alpha=0.7)
    # For visualization, use the first frame's approximation
    # Get the filter response for the first time frame
    first_frame_filters = irm_config['filter_params'][0]
    # Compute combined response (simplified)
    freq_response = np.ones(len(frequencies))
    for filt in first_frame_filters:
        # Simple approximation of filter response for visualization
        fc, Q, gain = filt.center_freq, filt.q_factor, filt.gain
        for i, f in enumerate(frequencies):
            if f > 0:
                # Simplified peaking filter response
                freq_ratio = f / fc
                if abs(freq_ratio - 1) < 0.5:  # Near center frequency
                    freq_response[i] *= 10**(gain/20)
    
    ax1.semilogx(frequencies, 20*np.log10(freq_response + 1e-10), 
                 'r--', linewidth=2, label='Lattice Approximation (Frame 1)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('IRM vs Lattice Filterbank Approximation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(100, 8000)
    
    # Plot 2: Filter parameters (first frame)
    ax2 = axes[0, 1]
    first_frame_filters = irm_config['filter_params'][0]
    filter_freqs = [p.center_freq for p in first_frame_filters]
    filter_gains = [p.gain for p in first_frame_filters]
    filter_qs = [p.q_factor for p in first_frame_filters]
    
    bars = ax2.bar(range(len(filter_freqs)), filter_gains, alpha=0.7)
    ax2.set_xlabel('Filter Index')
    ax2.set_ylabel('Gain (dB)')
    ax2.set_title('Individual Filter Gains')
    ax2.grid(True, alpha=0.3)
    
    # Add frequency labels
    for i, (freq, gain) in enumerate(zip(filter_freqs, filter_gains)):
        ax2.text(i, gain + 0.5, f'{freq:.0f}Hz', ha='center', va='bottom', 
                rotation=45, fontsize=8)
    
    # Plot 3: Time domain comparison
    ax3 = axes[1, 0]
    t = np.linspace(0, len(clean_audio)/fs, len(clean_audio))
    ax3.plot(t[:8000], clean_audio[:8000], 'g-', alpha=0.7, label='Clean')
    ax3.plot(t[:8000], noisy_audio[:8000], 'b-', alpha=0.5, label='Noisy')
    ax3.plot(t[:8000], processed_audio[:8000], 'r-', alpha=0.8, label='Processed')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Time Domain Signals (first 0.5s)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Approximation error over time
    ax4 = axes[1, 1]
    time_frames = np.arange(len(irm_config['approximation_error']))
    ax4.plot(time_frames, irm_config['approximation_error'], 
             'purple', linewidth=2)
    ax4.set_xlabel('Time Frame')
    ax4.set_ylabel('Approximation Error')
    ax4.set_title('Lattice Filterbank Approximation Error Over Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('16khz_lattice_demo.png', dpi=150, bbox_inches='tight')
    print(f"   Saved plot: 16khz_lattice_demo.png")
    
    # Print filter details (first frame)
    print(f"\n🎛️ Filter Configuration (First Time Frame):")
    for i, params in enumerate(first_frame_filters):
        print(f"   Filter {i+1}: {params.center_freq:.0f} Hz, Q={params.q_factor:.2f}, "
              f"Gain={params.gain:.1f} dB")

if __name__ == "__main__":
    try:
        irm_config, filterbank = demo_16khz_lattice_filterbank()
        print(f"\n✅ Demo completed successfully!")
        print(f"   Check the generated audio files and plot.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()