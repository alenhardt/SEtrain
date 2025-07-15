#!/usr/bin/env python3
"""
IRM Lattice Filterbank Example - Real Audio Files

This script:
1. Takes two WAV files: noisy mixture and clean (denoised) version
2. Computes the complex IRM: mask = STFT(clean) / STFT(mixture)  
3. Approximates the magnitude mask with a lattice filterbank
4. Processes the noisy file and saves the result

Usage:
    python irm_example.py noisy_mixture.wav clean_denoised.wav [options]

Options:
    --output OUTPUT_FILE    Output filename (default: filtered_output.wav)
    --num-filters N         Number of filters in bank (default: 8)
    --nfft N               FFT size (default: 512)
    --hop-length N         STFT hop length (default: 256)
    --max-gain-db DB       Maximum gain in dB (default: 20.0)
    --filter-type TYPE     Filter type: peaking, bandpass, mixed (default: peaking)
    --freq-range F1,F2     Frequency range in Hz (default: 80,16000)
    --magnitude-only       Use magnitude-only mask (default: True)
    --include-phase        Try to include phase information (experimental)
"""

import argparse
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
from lattice_filterbank import approximate_irm_with_lattice_filterbank, TimeVaryingLatticeFilterbank


def load_audio_files(noisy_file, clean_file):
    """Load and align two audio files."""
    print(f"📁 Loading audio files...")
    
    # Load audio files
    noisy_audio, sr_noisy = librosa.load(noisy_file, sr=None)
    clean_audio, sr_clean = librosa.load(clean_file, sr=None)
    
    print(f"   Noisy: {noisy_file} - {len(noisy_audio)} samples, {sr_noisy} Hz")
    print(f"   Clean: {clean_file} - {len(clean_audio)} samples, {sr_clean} Hz")
    
    # Check sample rates match
    if sr_noisy != sr_clean:
        print(f"⚠️  Sample rate mismatch! Resampling clean to {sr_noisy} Hz")
        clean_audio = librosa.resample(clean_audio, orig_sr=sr_clean, target_sr=sr_noisy)
    
    # Align lengths (trim to shorter)
    min_length = min(len(noisy_audio), len(clean_audio))
    noisy_audio = noisy_audio[:min_length]
    clean_audio = clean_audio[:min_length]
    
    print(f"✅ Aligned to {min_length} samples at {sr_noisy} Hz ({min_length/sr_noisy:.2f} seconds)")
    
    return noisy_audio, clean_audio, sr_noisy


def compute_irm_mask(noisy_audio, clean_audio, nfft=512, hop_length=256, magnitude_only=True):
    """
    Compute the Ideal Ratio Mask from noisy and clean audio.
    
    mask = STFT(clean) / STFT(noisy)
    """
    print(f"🔬 Computing IRM mask...")
    print(f"   STFT parameters: nfft={nfft}, hop_length={hop_length}")
    
    # Compute STFTs
    stft_noisy = librosa.stft(noisy_audio, n_fft=nfft, hop_length=hop_length)
    stft_clean = librosa.stft(clean_audio, n_fft=nfft, hop_length=hop_length)
    
    print(f"   STFT shape: {stft_noisy.shape} (freq x time)")
    
    # Compute complex mask: mask = clean / noisy
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    complex_mask = stft_clean / (stft_noisy + epsilon)
    
    if magnitude_only:
        # Use magnitude-only mask
        magnitude_mask = np.abs(complex_mask)
        
        # Transpose to (time, freq) format for filterbank
        H_mag = magnitude_mask.T  # Shape: (time_frames, freq_bins)
        
        print(f"✅ Magnitude mask computed: shape {H_mag.shape}")
        print(f"   Mask range: {H_mag.min():.3f} to {H_mag.max():.3f}")
        
        return H_mag, None
    
    else:
        # Experimental: try to handle phase information
        magnitude_mask = np.abs(complex_mask)
        phase_mask = np.angle(complex_mask)
        
        H_mag = magnitude_mask.T
        H_phase = phase_mask.T
        
        print(f"✅ Complex mask computed: shape {H_mag.shape}")
        print(f"   Magnitude range: {H_mag.min():.3f} to {H_mag.max():.3f}")
        print(f"   Phase range: {H_phase.min():.3f} to {H_phase.max():.3f} rad")
        
        return H_mag, H_phase


def apply_minimum_phase_conversion(H_mag):
    """
    Convert magnitude response to minimum phase using cepstrum method.
    This is experimental for handling phase information.
    """
    print("🔄 Applying minimum phase conversion...")
    
    # Work in log domain
    log_H = np.log(H_mag + 1e-8)
    
    # Apply cepstrum method for each time frame
    H_complex = np.zeros(H_mag.shape, dtype=complex)
    
    for t in range(H_mag.shape[0]):
        # Real cepstrum
        log_spectrum = log_H[t, :]
        
        # Symmetrize for real cepstrum (assume conjugate symmetry)
        log_spectrum_full = np.concatenate([log_spectrum, log_spectrum[-2:0:-1]])
        
        # Cepstrum
        cepstrum = np.fft.ifft(log_spectrum_full).real
        
        # Minimum phase cepstrum (causal)
        cepstrum_min_phase = cepstrum.copy()
        cepstrum_min_phase[1:len(cepstrum)//2] *= 2
        cepstrum_min_phase[len(cepstrum)//2+1:] = 0
        
        # Back to frequency domain
        log_H_min_phase = np.fft.fft(cepstrum_min_phase)
        H_min_phase = np.exp(log_H_min_phase)
        
        # Take first half (positive frequencies)
        H_complex[t, :] = H_min_phase[:len(log_spectrum)]
    
    print(f"✅ Minimum phase conversion complete")
    
    return np.abs(H_complex), np.angle(H_complex)


def create_filterbank_approximation(H_mag, fs, nfft, num_filters=8, max_gain_db=20.0, 
                                   filter_type='peaking', freq_range=(80, 16000)):
    """Create lattice filterbank approximation of the IRM."""
    print(f"🎛️ Creating lattice filterbank approximation...")
    print(f"   Filters: {num_filters}, Type: {filter_type}")
    print(f"   Frequency range: {freq_range[0]}-{freq_range[1]} Hz")
    print(f"   Max gain: ±{max_gain_db} dB")
    
    # Create lattice filterbank configuration
    irm_config = approximate_irm_with_lattice_filterbank(
        H_mag,
        fs=fs,
        nfft=nfft,
        num_filters=num_filters,
        filter_type=filter_type,
        freq_range=freq_range,
        max_gain_db=max_gain_db
    )
    
    # Create time-varying filterbank
    filterbank = TimeVaryingLatticeFilterbank(irm_config)
    
    avg_error = np.mean(irm_config['approximation_error'])
    max_error = np.max(irm_config['approximation_error'])
    
    print(f"✅ Filterbank created")
    print(f"   Average approximation error: {avg_error:.4f}")
    print(f"   Maximum approximation error: {max_error:.4f}")
    
    return filterbank, irm_config


def process_audio_with_filterbank(noisy_audio, filterbank, hop_length, fs):
    """Process noisy audio with time-varying lattice filterbank."""
    print(f"🎵 Processing audio with filterbank...")
    
    # Map audio samples to IRM time frames
    frame_indices = np.arange(len(noisy_audio)) // hop_length
    frame_indices = np.clip(frame_indices, 0, filterbank.num_frames - 1)
    
    print(f"   Processing {len(noisy_audio)} samples ({len(noisy_audio)/fs:.2f}s)")
    print(f"   Using {filterbank.num_frames} IRM frames with {hop_length} sample hops")
    
    # Process with time-varying parameters
    processed_audio = filterbank.process_buffer(noisy_audio, frame_indices)
    
    print(f"✅ Audio processing complete")
    
    return processed_audio


def create_visualization(noisy_audio, clean_audio, processed_audio, H_mag, irm_config, 
                        fs, nfft, hop_length, output_prefix):
    """Create comprehensive visualization of the IRM approximation process."""
    print(f"📊 Creating visualization...")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Time domain signals
    time_axis = np.arange(len(noisy_audio)) / fs
    
    axes[0, 0].plot(time_axis, noisy_audio, alpha=0.7, label='Noisy')
    axes[0, 0].plot(time_axis, clean_audio, alpha=0.7, label='Clean')
    axes[0, 0].set_title('Input Signals')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(time_axis, noisy_audio, alpha=0.7, label='Noisy')
    axes[0, 1].plot(time_axis, processed_audio, alpha=0.7, label='Filtered')
    axes[0, 1].set_title('Output Comparison')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # IRM magnitude spectrogram
    frequencies = np.fft.fftfreq(nfft, 1/fs)[:nfft//2 + 1]
    time_frames = np.arange(H_mag.shape[0]) * hop_length / fs
    
    im1 = axes[1, 0].imshow(H_mag.T, aspect='auto', origin='lower', 
                           extent=[time_frames[0], time_frames[-1], 
                                  frequencies[0], frequencies[-1]])
    axes[1, 0].set_title('Target IRM Magnitude')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    axes[1, 0].set_ylim(0, fs//4)  # Show up to Nyquist/2
    plt.colorbar(im1, ax=axes[1, 0], label='Magnitude')
    
    # Reconstruct filterbank approximation from filter parameters
    # Create frequency grid
    freq_bins = np.fft.fftfreq(nfft, 1/fs)[:nfft//2 + 1]
    H_approx = np.ones((H_mag.shape[0], len(freq_bins)))
    
    # For each time frame, compute combined filter response
    for t in range(H_mag.shape[0]):
        if t < len(irm_config['filter_params']):
            frame_params = irm_config['filter_params'][t]
            for param in frame_params:
                # Simple approximation: Gaussian-like response around center frequency
                center_f = param.center_freq
                gain_linear = 10**(param.gain / 20)
                bandwidth = center_f / param.q_factor
                
                # Apply filter response to frequency grid
                for f_idx, freq in enumerate(freq_bins):
                    if freq > 0:
                        # Gaussian-like filter response
                        response = gain_linear * np.exp(-((freq - center_f) / bandwidth)**2)
                        H_approx[t, f_idx] *= response
    
    im2 = axes[1, 1].imshow(H_approx.T, aspect='auto', origin='lower',
                           extent=[time_frames[0], time_frames[-1], 
                                  frequencies[0], frequencies[-1]])
    axes[1, 1].set_title('Filterbank Approximation')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Frequency (Hz)')
    axes[1, 1].set_ylim(0, fs//4)
    plt.colorbar(im2, ax=axes[1, 1], label='Magnitude')
    
    # Approximation error
    error = np.abs(H_mag - H_approx)
    im3 = axes[2, 0].imshow(error.T, aspect='auto', origin='lower',
                           extent=[time_frames[0], time_frames[-1], 
                                  frequencies[0], frequencies[-1]])
    axes[2, 0].set_title('Approximation Error')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Frequency (Hz)')
    axes[2, 0].set_ylim(0, fs//4)
    plt.colorbar(im3, ax=axes[2, 0], label='|Error|')
    
    # Filter parameter evolution
    filter_params = irm_config['filter_params']
    
    # Plot gain values for first few filters
    for filt_idx in range(min(3, irm_config['config']['num_filters'])):  # Show first 3 filters
        gain_values = []
        for t in range(len(filter_params)):
            if filt_idx < len(filter_params[t]):
                gain_values.append(filter_params[t][filt_idx].gain)
            else:
                gain_values.append(0.0)
        
        axes[2, 1].plot(time_frames[:len(gain_values)], gain_values, 
                       label=f'Filter {filt_idx+1} Gain (dB)', alpha=0.7)
    
    axes[2, 1].set_title('Filter Parameters Over Time')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Parameter Value')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    plot_filename = f"{output_prefix}_analysis.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📁 Saved: {plot_filename}")


def main():
    parser = argparse.ArgumentParser(description='IRM Lattice Filterbank Approximation')
    parser.add_argument('noisy_file', help='Noisy mixture WAV file')
    parser.add_argument('clean_file', help='Clean (denoised) WAV file')
    parser.add_argument('--output', default='filtered_output.wav', 
                       help='Output filename (default: filtered_output.wav)')
    parser.add_argument('--num-filters', type=int, default=8,
                       help='Number of filters in bank (default: 8)')
    parser.add_argument('--nfft', type=int, default=512,
                       help='FFT size (default: 512)')
    parser.add_argument('--hop-length', type=int, default=256,
                       help='STFT hop length (default: 256)')
    parser.add_argument('--max-gain-db', type=float, default=20.0,
                       help='Maximum gain in dB (default: 20.0)')
    parser.add_argument('--filter-type', choices=['peaking', 'bandpass', 'mixed'], 
                       default='peaking', help='Filter type (default: peaking)')
    parser.add_argument('--freq-range', type=str, default='80,16000',
                       help='Frequency range in Hz as F1,F2 (default: 80,16000)')
    parser.add_argument('--magnitude-only', action='store_true', default=True,
                       help='Use magnitude-only mask (default: True)')
    parser.add_argument('--include-phase', action='store_true',
                       help='Try to include phase information (experimental)')
    
    args = parser.parse_args()
    
    # Parse frequency range
    freq_range = tuple(map(int, args.freq_range.split(',')))
    
    print("🎯 IRM LATTICE FILTERBANK - REAL AUDIO FILES")
    print("=" * 60)
    
    try:
        # Load audio files
        noisy_audio, clean_audio, fs = load_audio_files(args.noisy_file, args.clean_file)
        
        # Compute IRM mask
        magnitude_only = args.magnitude_only and not args.include_phase
        H_mag, H_phase = compute_irm_mask(noisy_audio, clean_audio, 
                                         args.nfft, args.hop_length, magnitude_only)
        
        # Optional: minimum phase conversion
        if args.include_phase and H_phase is None:
            print("🔄 Phase information requested - applying minimum phase conversion")
            H_mag, H_phase = apply_minimum_phase_conversion(H_mag)
        
        # Create filterbank approximation
        filterbank, irm_config = create_filterbank_approximation(
            H_mag, fs, args.nfft, args.num_filters, args.max_gain_db,
            args.filter_type, freq_range
        )
        
        # Process audio
        processed_audio = process_audio_with_filterbank(
            noisy_audio, filterbank, args.hop_length, fs
        )
        
        # Save output
        output_prefix = args.output.rsplit('.', 1)[0]  # Remove extension
        sf.write(args.output, processed_audio, fs)
        print(f"📁 Saved: {args.output}")
        
        # Save original files for comparison
        sf.write(f"{output_prefix}_original_noisy.wav", noisy_audio, fs)
        sf.write(f"{output_prefix}_original_clean.wav", clean_audio, fs)
        print(f"📁 Saved: {output_prefix}_original_noisy.wav")
        print(f"📁 Saved: {output_prefix}_original_clean.wav")
        
        # Create visualization
        create_visualization(noisy_audio, clean_audio, processed_audio, 
                           H_mag, irm_config, fs, args.nfft, args.hop_length,
                           output_prefix)
        
        # Print summary
        print("\n" + "=" * 60)
        print("✅ PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Input files: {args.noisy_file}, {args.clean_file}")
        print(f"Output: {args.output}")
        print(f"Filterbank: {args.num_filters} {args.filter_type} filters")
        print(f"IRM frames: {H_mag.shape[0]}, Frequency bins: {H_mag.shape[1]}")
        print(f"Audio duration: {len(noisy_audio)/fs:.2f} seconds")
        avg_error = np.mean(irm_config['approximation_error'])
        print(f"Average approximation error: {avg_error:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())