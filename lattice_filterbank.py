import numpy as np
import scipy.signal as signal
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class FilterParams:
    """Parameters for a single filter in the filterbank."""
    center_freq: float  # Center frequency in Hz
    gain: float         # Gain in dB
    q_factor: float     # Q factor
    fs: float          # Sampling frequency in Hz
    filter_type: str = 'peaking'  # 'peaking', 'lowpass', 'highpass', 'bandpass'


class LatticeFilter:
    """Single lattice-ladder filter implementation."""
    
    def __init__(self, k1: float, k2: float, v0: float, v1: float, v2: float):
        """
        Initialize lattice-ladder filter with coefficients.
        
        Args:
            k1, k2: Lattice coefficients
            v0, v1, v2: Ladder coefficients
        """
        self.k1 = k1
        self.k2 = k2
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        
        # Internal state variables
        self.s1 = 0.0
        self.s2 = 0.0
        
    def reset_state(self):
        """Reset filter internal state."""
        self.s1 = 0.0
        self.s2 = 0.0
        
    def process_sample(self, x: float) -> float:
        """
        Process a single sample through the lattice-ladder filter.
        
        Args:
            x: Input sample
            
        Returns:
            Filtered output sample
        """
        # Lattice structure
        e0 = x
        e1 = e0 - self.k1 * self.s1
        e2 = e1 - self.k2 * self.s2
        
        # Update delays
        self.s2 = self.s1 + self.k2 * e2
        self.s1 = e0 + self.k1 * e1
        
        # Ladder structure (output combination)
        y = self.v0 * e0 + self.v1 * e1 + self.v2 * e2
        
        return y


class DirectForm1Biquad:
    """Direct Form 1 biquad filter implementation for comparison."""
    
    def __init__(self, b0: float, b1: float, b2: float, a1: float, a2: float):
        """
        Initialize direct form 1 biquad with coefficients.
        
        Args:
            b0, b1, b2: Numerator coefficients
            a1, a2: Denominator coefficients (a0 is assumed to be 1)
        """
        self.b0, self.b1, self.b2 = b0, b1, b2
        self.a1, self.a2 = a1, a2
        
        # Delay lines
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0
        
    def reset_state(self):
        """Reset filter internal state."""
        self.x1 = self.x2 = 0.0
        self.y1 = self.y2 = 0.0
        
    def process_sample(self, x: float) -> float:
        """
        Process a single sample through the direct form 1 biquad.
        
        Args:
            x: Input sample
            
        Returns:
            Filtered output sample
        """
        y = (self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 
             - self.a1 * self.y1 - self.a2 * self.y2)
        
        # Update delays
        self.x2 = self.x1
        self.x1 = x
        self.y2 = self.y1
        self.y1 = y
        
        return y


def design_biquad_coefficients(params: FilterParams) -> Tuple[float, float, float, float, float]:
    """
    Design biquad coefficients from filter parameters.
    
    Args:
        params: Filter parameters
        
    Returns:
        Tuple of (b0, b1, b2, a1, a2) coefficients
    """
    omega = 2 * np.pi * params.center_freq / params.fs
    sin_omega = np.sin(omega)
    cos_omega = np.cos(omega)
    alpha = sin_omega / (2 * params.q_factor)
    A = 10 ** (params.gain / 40)  # Convert dB to linear amplitude
    
    if params.filter_type == 'peaking':
        # Peaking EQ filter
        b0 = 1 + alpha * A
        b1 = -2 * cos_omega
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_omega
        a2 = 1 - alpha / A
        
    elif params.filter_type == 'lowpass':
        # Low-pass filter
        b0 = (1 - cos_omega) / 2
        b1 = 1 - cos_omega
        b2 = (1 - cos_omega) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_omega
        a2 = 1 - alpha
        
    elif params.filter_type == 'highpass':
        # High-pass filter
        b0 = (1 + cos_omega) / 2
        b1 = -(1 + cos_omega)
        b2 = (1 + cos_omega) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_omega
        a2 = 1 - alpha
        
    elif params.filter_type == 'bandpass':
        # Band-pass filter
        b0 = alpha
        b1 = 0
        b2 = -alpha
        a0 = 1 + alpha
        a1 = -2 * cos_omega
        a2 = 1 - alpha
        
    else:
        raise ValueError(f"Unsupported filter type: {params.filter_type}")
    
    # Normalize by a0
    b0, b1, b2, a1, a2 = b0/a0, b1/a0, b2/a0, a1/a0, a2/a0
    
    return b0, b1, b2, a1, a2


def biquad_to_lattice(b0: float, b1: float, b2: float, a1: float, a2: float) -> Tuple[float, float, float, float, float]:
    """
    Convert biquad coefficients to lattice-ladder form.
    
    Args:
        b0, b1, b2: Numerator coefficients
        a1, a2: Denominator coefficients
        
    Returns:
        Tuple of (k1, k2, v0, v1, v2) lattice-ladder coefficients
    """
    # Convert denominator to lattice coefficients
    k2 = a2
    k1 = a1 / (1 - k2) if abs(1 - k2) > 1e-10 else 0
    
    # Convert numerator to ladder coefficients
    # This transformation maintains the same transfer function
    v0 = b0
    v1 = b1 - k1 * b0
    v2 = b2 - k1 * v1 - k2 * b0
    
    return k1, k2, v0, v1, v2


class LatticeFilterbank:
    """Filterbank implementation using lattice-ladder filters."""
    
    def __init__(self, filter_params: List[FilterParams]):
        """
        Initialize filterbank with list of filter parameters.
        
        Args:
            filter_params: List of FilterParams for each filter in the bank
        """
        self.filters = []
        self.biquad_coeffs = []
        
        for params in filter_params:
            # Design biquad coefficients
            b0, b1, b2, a1, a2 = design_biquad_coefficients(params)
            self.biquad_coeffs.append((b0, b1, b2, a1, a2))
            
            # Convert to lattice-ladder form
            k1, k2, v0, v1, v2 = biquad_to_lattice(b0, b1, b2, a1, a2)
            
            # Create lattice filter
            lattice_filter = LatticeFilter(k1, k2, v0, v1, v2)
            self.filters.append(lattice_filter)
            
    def reset_states(self):
        """Reset all filter states."""
        for filt in self.filters:
            filt.reset_state()
            
    def process_sample(self, x: float) -> float:
        """
        Process a single sample through the entire filterbank.
        
        Args:
            x: Input sample
            
        Returns:
            Filtered output sample (sum of all filter outputs)
        """
        output = 0.0
        for filt in self.filters:
            output += filt.process_sample(x)
        return output
        
    def filter_signal(self, signal_in: np.ndarray) -> np.ndarray:
        """
        Filter a complete signal through the lattice filterbank.
        
        Args:
            signal_in: Input signal array
            
        Returns:
            Filtered output signal array
        """
        self.reset_states()
        output = np.zeros_like(signal_in)
        
        for i, sample in enumerate(signal_in):
            output[i] = self.process_sample(sample)
            
        return output


class DirectForm1Filterbank:
    """Filterbank implementation using direct form 1 biquads for comparison."""
    
    def __init__(self, biquad_coeffs: List[Tuple[float, float, float, float, float]]):
        """
        Initialize filterbank with biquad coefficients.
        
        Args:
            biquad_coeffs: List of (b0, b1, b2, a1, a2) coefficient tuples
        """
        self.filters = []
        for b0, b1, b2, a1, a2 in biquad_coeffs:
            self.filters.append(DirectForm1Biquad(b0, b1, b2, a1, a2))
            
    def reset_states(self):
        """Reset all filter states."""
        for filt in self.filters:
            filt.reset_state()
            
    def process_sample(self, x: float) -> float:
        """Process a single sample through the filterbank."""
        output = 0.0
        for filt in self.filters:
            output += filt.process_sample(x)
        return output
        
    def filter_signal(self, signal_in: np.ndarray) -> np.ndarray:
        """Filter a complete signal through the direct form 1 filterbank."""
        self.reset_states()
        output = np.zeros_like(signal_in)
        
        for i, sample in enumerate(signal_in):
            output[i] = self.process_sample(sample)
            
        return output


def latcfilt(signal_in: np.ndarray, filter_params: List[FilterParams]) -> np.ndarray:
    """
    Convenience function to filter a 1D signal using lattice-ladder filterbank.
    
    Args:
        signal_in: Input 1D signal
        filter_params: List of filter parameters
        
    Returns:
        Filtered output signal
    """
    filterbank = LatticeFilterbank(filter_params)
    return filterbank.filter_signal(signal_in)


def test_equivalence():
    """Test that lattice-ladder and direct form 1 implementations produce identical results."""
    print("Testing equivalence between lattice-ladder and direct form 1 implementations...")
    
    # Test parameters
    fs = 48000  # Sampling frequency
    duration = 1.0  # Signal duration in seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Create test signal (sum of sinusoids + noise)
    test_signal = (np.sin(2 * np.pi * 440 * t) +  # 440 Hz
                   0.5 * np.sin(2 * np.pi * 1000 * t) +  # 1 kHz
                   0.3 * np.sin(2 * np.pi * 2000 * t) +  # 2 kHz
                   0.1 * np.random.randn(len(t)))  # Noise
    
    # Define filter parameters
    filter_params = [
        FilterParams(center_freq=500, gain=6, q_factor=1.0, fs=fs, filter_type='peaking'),
        FilterParams(center_freq=1500, gain=-3, q_factor=2.0, fs=fs, filter_type='peaking'),
        FilterParams(center_freq=3000, gain=4, q_factor=0.7, fs=fs, filter_type='peaking'),
    ]
    
    # Create lattice filterbank
    lattice_fb = LatticeFilterbank(filter_params)
    
    # Create equivalent direct form 1 filterbank
    df1_fb = DirectForm1Filterbank(lattice_fb.biquad_coeffs)
    
    # Filter the signal with both implementations
    lattice_output = lattice_fb.filter_signal(test_signal)
    df1_output = df1_fb.filter_signal(test_signal)
    
    # Calculate difference
    difference = lattice_output - df1_output
    max_diff = np.max(np.abs(difference))
    rms_diff = np.sqrt(np.mean(difference**2))
    
    print(f"Maximum absolute difference: {max_diff:.2e}")
    print(f"RMS difference: {rms_diff:.2e}")
    
    # Check if they're practically identical (within numerical precision)
    tolerance = 1e-10
    if max_diff < tolerance:
        print("✅ PASS: Lattice-ladder and direct form 1 implementations are equivalent!")
    else:
        print("❌ FAIL: Implementations differ beyond numerical precision")
        
    return max_diff < tolerance


def test_filter_types():
    """Test different filter types."""
    print("\nTesting different filter types...")
    
    fs = 48000
    duration = 0.1
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # White noise test signal
    test_signal = np.random.randn(len(t))
    
    filter_types = ['peaking', 'lowpass', 'highpass', 'bandpass']
    
    for ftype in filter_types:
        print(f"\nTesting {ftype} filter...")
        
        params = [FilterParams(center_freq=1000, gain=6, q_factor=1.0, fs=fs, filter_type=ftype)]
        
        try:
            lattice_fb = LatticeFilterbank(params)
            df1_fb = DirectForm1Filterbank(lattice_fb.biquad_coeffs)
            
            lattice_out = lattice_fb.filter_signal(test_signal)
            df1_out = df1_fb.filter_signal(test_signal)
            
            diff = np.max(np.abs(lattice_out - df1_out))
            print(f"  Max difference: {diff:.2e}")
            
            if diff < 1e-10:
                print(f"  ✅ {ftype} filter test PASSED")
            else:
                print(f"  ❌ {ftype} filter test FAILED")
                
        except Exception as e:
            print(f"  ❌ {ftype} filter test FAILED with error: {e}")


def plot_frequency_response():
    """Plot frequency response of the filterbank."""
    print("\nGenerating frequency response plot...")
    
    fs = 48000
    
    # Define a multi-band EQ
    filter_params = [
        FilterParams(center_freq=100, gain=3, q_factor=1.0, fs=fs, filter_type='peaking'),
        FilterParams(center_freq=500, gain=-2, q_factor=2.0, fs=fs, filter_type='peaking'),
        FilterParams(center_freq=2000, gain=4, q_factor=1.5, fs=fs, filter_type='peaking'),
        FilterParams(center_freq=8000, gain=-1, q_factor=0.8, fs=fs, filter_type='peaking'),
    ]
    
    # Create filterbank
    filterbank = LatticeFilterbank(filter_params)
    
    # Calculate frequency response
    frequencies = np.logspace(1, 4, 1000)  # 10 Hz to 10 kHz
    omega = 2 * np.pi * frequencies / fs
    
    # Calculate overall response by summing individual filter responses
    H_total = np.zeros(len(frequencies), dtype=complex)
    
    for i, (b0, b1, b2, a1, a2) in enumerate(filterbank.biquad_coeffs):
        # Calculate frequency response for each biquad
        b = [b0, b1, b2]
        a = [1, a1, a2]
        w, h = signal.freqz(b, a, omega)
        H_total += h
    
    # Convert to magnitude and phase
    magnitude_db = 20 * np.log10(np.abs(H_total))
    phase_deg = np.angle(H_total) * 180 / np.pi
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.semilogx(frequencies, magnitude_db)
    plt.title('Lattice-Ladder Filterbank Frequency Response')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.xlim([10, 10000])
    
    plt.subplot(2, 1, 2)
    plt.semilogx(frequencies, phase_deg)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (degrees)')
    plt.grid(True)
    plt.xlim([10, 10000])
    
    plt.tight_layout()
    plt.savefig('filterbank_response.png', dpi=300, bbox_inches='tight')
    print("Frequency response plot saved as 'filterbank_response.png'")


if __name__ == "__main__":
    print("Lattice-Ladder Filterbank Implementation")
    print("=" * 50)
    
    # Run tests
    test_equivalence()
    test_filter_types()
    
    # Generate frequency response plot
    plot_frequency_response()
    
    print("\n" + "=" * 50)
    print("All tests completed!")