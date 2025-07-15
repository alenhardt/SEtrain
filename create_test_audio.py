#!/usr/bin/env python3
"""
Create test audio files for IRM demonstration.

This script generates:
1. A clean "speech-like" signal with multiple tones
2. A noisy version with added background noise
3. These can be used to test the IRM lattice filterbank
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def create_clean_speech_signal(duration=3.0, fs=48000):
    """
    Create a clean speech-like signal with multiple harmonic components.
    """
    t = np.linspace(0, duration, int(fs * duration))
    
    # Speech-like signal with time-varying fundamental frequency
    f0_base = 120  # Base fundamental frequency (Hz)
    f0_variation = 30 * np.sin(2 * np.pi * 0.8 * t)  # Vibrato
    f0 = f0_base + f0_variation
    
    # Create harmonics (like voice formants)
    clean_signal = np.zeros_like(t)
    
    # Fundamental frequency
    clean_signal += 0.3 * np.sin(2 * np.pi * f0 * t)
    
    # Harmonics with different envelopes
    for harmonic in [2, 3, 4, 5]:
        amplitude = 0.2 / harmonic  # Decreasing amplitude with harmonic number
        
        # Add some time-varying formant-like resonances
        formant_freq = f0 * harmonic + 50 * np.sin(2 * np.pi * 0.3 * t)
        envelope = 1.0 + 0.3 * np.sin(2 * np.pi * 0.5 * t)
        
        clean_signal += amplitude * envelope * np.sin(2 * np.pi * formant_freq * t)
    
    # Add some higher frequency content (consonant-like)
    for freq in [2000, 3500, 5000]:
        burst_envelope = np.exp(-((t % 0.5) / 0.1)**2)  # Periodic bursts
        amplitude = 0.1 * np.random.uniform(0.5, 1.0)
        clean_signal += amplitude * burst_envelope * np.sin(2 * np.pi * freq * t)
    
    # Apply overall envelope (speech-like amplitude modulation)
    speech_envelope = (1.0 + 0.4 * np.sin(2 * np.pi * 1.2 * t)) * \
                     (1.0 + 0.2 * np.sin(2 * np.pi * 3.7 * t))
    clean_signal *= speech_envelope
    
    # Normalize
    clean_signal = clean_signal / np.max(np.abs(clean_signal)) * 0.7
    
    return clean_signal, t

def create_background_noise(duration, fs, noise_type='colored'):
    """
    Create background noise with spectral coloring.
    """
    t = np.linspace(0, duration, int(fs * duration))
    
    if noise_type == 'white':
        noise = np.random.randn(len(t))
    
    elif noise_type == 'colored':
        # Create colored noise with spectral shaping
        noise = np.random.randn(len(t))
        
        # Apply spectral coloring in frequency domain
        noise_fft = np.fft.fft(noise)
        freqs = np.fft.fftfreq(len(noise), 1/fs)
        
        # Create a colored noise spectrum (pink-ish noise)
        coloring = 1.0 / (1.0 + (np.abs(freqs) / 1000)**0.8)
        noise_fft *= coloring
        
        noise = np.fft.ifft(noise_fft).real
    
    elif noise_type == 'environmental':
        # Simulate environmental noise with multiple components
        noise = np.zeros_like(t)
        
        # Low-frequency rumble
        noise += 0.5 * np.sin(2 * np.pi * 60 * t + np.random.randn() * 0.1)
        noise += 0.3 * np.sin(2 * np.pi * 120 * t + np.random.randn() * 0.1)
        
        # Mid-frequency hum/buzz
        for freq in [300, 500, 800]:
            amplitude = 0.2 * np.random.uniform(0.5, 1.5)
            phase_noise = np.cumsum(np.random.randn(len(t)) * 0.01)
            noise += amplitude * np.sin(2 * np.pi * freq * t + phase_noise)
        
        # High-frequency hiss
        hiss = np.random.randn(len(t))
        # High-pass filter the hiss
        from scipy import signal
        b, a = signal.butter(4, 2000/(fs/2), 'high')
        hiss_filtered = signal.filtfilt(b, a, hiss)
        noise += 0.3 * hiss_filtered
        
        # Add some random colored noise
        colored_noise = np.random.randn(len(t))
        noise_fft = np.fft.fft(colored_noise)
        freqs = np.fft.fftfreq(len(colored_noise), 1/fs)
        coloring = 1.0 / (1.0 + (np.abs(freqs) / 500)**1.2)
        noise_fft *= coloring
        colored_noise = np.fft.ifft(noise_fft).real
        noise += 0.4 * colored_noise
    
    # Normalize noise
    noise = noise / np.max(np.abs(noise))
    
    return noise

def create_test_files():
    """Create test audio files for IRM demonstration."""
    
    print("🎵 Creating test audio files for IRM demonstration...")
    
    # Parameters
    duration = 4.0  # seconds
    fs = 48000  # Hz
    
    # Create clean signal
    print("   Creating clean speech-like signal...")
    clean_signal, t = create_clean_speech_signal(duration, fs)
    
    # Create background noise
    print("   Creating environmental background noise...")
    noise = create_background_noise(duration, fs, 'environmental')
    
    # Create noisy mixture
    snr_db = 5.0  # Signal-to-noise ratio in dB
    snr_linear = 10**(snr_db / 20)
    
    # Scale noise relative to signal
    signal_power = np.mean(clean_signal**2)
    noise_power = np.mean(noise**2)
    noise_scale = np.sqrt(signal_power / (snr_linear**2 * noise_power))
    
    scaled_noise = noise * noise_scale
    noisy_signal = clean_signal + scaled_noise
    
    print(f"   SNR: {snr_db} dB")
    print(f"   Signal RMS: {np.sqrt(np.mean(clean_signal**2)):.3f}")
    print(f"   Noise RMS: {np.sqrt(np.mean(scaled_noise**2)):.3f}")
    
    # Save audio files
    print("💾 Saving audio files...")
    
    # Convert to appropriate range for 16-bit audio
    clean_int16 = (clean_signal * 32767 * 0.9).astype(np.int16)
    noisy_int16 = (noisy_signal * 32767 * 0.9).astype(np.int16)
    noise_int16 = (scaled_noise * 32767 * 0.9).astype(np.int16)
    
    sf.write('test_clean_speech.wav', clean_signal, fs)
    sf.write('test_noisy_mixture.wav', noisy_signal, fs)
    sf.write('test_noise_only.wav', scaled_noise, fs)
    
    print("   📁 test_clean_speech.wav")
    print("   📁 test_noisy_mixture.wav") 
    print("   📁 test_noise_only.wav")
    
    # Create visualization
    print("📊 Creating visualization...")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    
    # Time domain plots
    time_window = slice(0, int(fs * 1.0))  # First 1 second
    t_plot = t[time_window]
    
    axes[0, 0].plot(t_plot, clean_signal[time_window])
    axes[0, 0].set_title('Clean Speech Signal')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(t_plot, scaled_noise[time_window], alpha=0.7)
    axes[0, 1].set_title('Background Noise')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(t_plot, noisy_signal[time_window], alpha=0.7)
    axes[1, 0].set_title('Noisy Mixture (SNR = 5 dB)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Frequency domain plots
    from scipy import signal as sp_signal
    
    # Compute spectrograms
    f_clean, t_clean, Sxx_clean = sp_signal.spectrogram(clean_signal, fs, nperseg=1024)
    f_noise, t_noise, Sxx_noise = sp_signal.spectrogram(scaled_noise, fs, nperseg=1024)
    f_noisy, t_noisy, Sxx_noisy = sp_signal.spectrogram(noisy_signal, fs, nperseg=1024)
    
    # Plot spectrograms
    im1 = axes[1, 1].imshow(10*np.log10(Sxx_clean + 1e-10), aspect='auto', origin='lower',
                           extent=[t_clean[0], t_clean[-1], f_clean[0], f_clean[-1]])
    axes[1, 1].set_title('Clean Speech Spectrogram')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Frequency (Hz)')
    axes[1, 1].set_ylim(0, 8000)
    plt.colorbar(im1, ax=axes[1, 1], label='Power (dB)')
    
    im2 = axes[2, 0].imshow(10*np.log10(Sxx_noise + 1e-10), aspect='auto', origin='lower',
                           extent=[t_noise[0], t_noise[-1], f_noise[0], f_noise[-1]])
    axes[2, 0].set_title('Noise Spectrogram')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Frequency (Hz)')
    axes[2, 0].set_ylim(0, 8000)
    plt.colorbar(im2, ax=axes[2, 0], label='Power (dB)')
    
    im3 = axes[2, 1].imshow(10*np.log10(Sxx_noisy + 1e-10), aspect='auto', origin='lower',
                           extent=[t_noisy[0], t_noisy[-1], f_noisy[0], f_noisy[-1]])
    axes[2, 1].set_title('Noisy Mixture Spectrogram')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Frequency (Hz)')
    axes[2, 1].set_ylim(0, 8000)
    plt.colorbar(im3, ax=axes[2, 1], label='Power (dB)')
    
    plt.tight_layout()
    plt.savefig('test_audio_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("   📁 test_audio_analysis.png")
    
    print("\n✅ Test files created successfully!")
    print("Now you can run:")
    print("python3 irm_example.py test_noisy_mixture.wav test_clean_speech.wav")
    
    return clean_signal, noisy_signal, scaled_noise, fs

if __name__ == "__main__":
    create_test_files()