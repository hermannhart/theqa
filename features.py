"""
Feature extraction methods for TheQA package.

This module provides various features that can be extracted from sequences
to detect phase transitions under noise.
"""

import numpy as np
from scipy import signal, stats
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple


class BaseFeature(ABC):
    """Abstract base class for feature extractors."""
    
    def __init__(self, transform: Optional[str] = None):
        """
        Parameters
        ----------
        transform : str, optional
            Transformation to apply before feature extraction:
            'log', 'sqrt', 'abs', 'normalize', None
        """
        self.transform_type = transform
    
    def transform(self, sequence: np.ndarray) -> np.ndarray:
        """Apply transformation to sequence."""
        if self.transform_type is None:
            return sequence
        elif self.transform_type == 'log':
            return np.log(np.abs(sequence) + 1)
        elif self.transform_type == 'sqrt':
            return np.sqrt(np.abs(sequence))
        elif self.transform_type == 'abs':
            return np.abs(sequence)
        elif self.transform_type == 'normalize':
            return (sequence - np.mean(sequence)) / (np.std(sequence) + 1e-10)
        else:
            raise ValueError(f"Unknown transform: {self.transform_type}")
    
    @abstractmethod
    def extract(self, sequence: np.ndarray) -> float:
        """Extract feature value from sequence."""
        pass
    
    def __call__(self, sequence: np.ndarray) -> float:
        """Convenience method for extraction."""
        return self.extract(sequence)


class PeakCounter(BaseFeature):
    """
    Count local maxima in sequence.
    
    Parameters
    ----------
    transform : str, optional
        Transformation to apply before counting
    prominence : float or str, default='adaptive'
        Minimum prominence of peaks. If 'adaptive', uses std/4
    distance : int, optional
        Minimum distance between peaks
    """
    
    def __init__(self, transform: Optional[str] = 'log', 
                 prominence: Union[float, str] = 'adaptive',
                 distance: Optional[int] = None):
        super().__init__(transform)
        self.prominence = prominence
        self.distance = distance
    
    def extract(self, sequence: np.ndarray) -> float:
        """Count number of peaks."""
        if len(sequence) < 3:
            return 0
        
        # Determine prominence
        if self.prominence == 'adaptive':
            prom = np.std(sequence) / 4
        else:
            prom = self.prominence
        
        # Find peaks
        peaks, _ = signal.find_peaks(
            sequence, 
            prominence=prom,
            distance=self.distance
        )
        
        return len(peaks)


class EntropyCalculator(BaseFeature):
    """
    Calculate Shannon entropy of sequence.
    
    Parameters
    ----------
    bins : int, default=20
        Number of bins for histogram
    normalize : bool, default=True
        Normalize entropy by log(bins)
    """
    
    def __init__(self, transform: Optional[str] = None,
                 bins: int = 20, normalize: bool = True):
        super().__init__(transform)
        self.bins = bins
        self.normalize = normalize
    
    def extract(self, sequence: np.ndarray) -> float:
        """Calculate entropy."""
        if len(sequence) < 2:
            return 0
        
        # Create histogram
        hist, _ = np.histogram(sequence, bins=self.bins)
        
        # Normalize to probabilities
        p = hist / np.sum(hist)
        
        # Remove zeros
        p = p[p > 0]
        
        # Calculate entropy
        entropy = -np.sum(p * np.log(p))
        
        # Normalize if requested
        if self.normalize and self.bins > 1:
            entropy /= np.log(self.bins)
        
        return entropy


class SpectralAnalyzer(BaseFeature):
    """
    Extract spectral features from sequence.
    
    Parameters
    ----------
    feature_type : str, default='dominant_freq'
        Type of spectral feature:
        - 'dominant_freq': Frequency with maximum power
        - 'spectral_entropy': Entropy of power spectrum
        - 'n_peaks': Number of spectral peaks
    n_peaks : int, default=1
        Number of peaks to find (for 'n_peaks' type)
    """
    
    def __init__(self, transform: Optional[str] = None,
                 feature_type: str = 'dominant_freq',
                 n_peaks: int = 1):
        super().__init__(transform)
        self.feature_type = feature_type
        self.n_peaks = n_peaks
    
    def extract(self, sequence: np.ndarray) -> float:
        """Extract spectral feature."""
        if len(sequence) < 4:
            return 0
        
        # Compute power spectrum
        freqs = np.fft.fftfreq(len(sequence))
        fft = np.fft.fft(sequence)
        power = np.abs(fft)**2
        
        # Only positive frequencies
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        power = power[pos_mask]
        
        if len(power) == 0:
            return 0
        
        if self.feature_type == 'dominant_freq':
            # Find dominant frequency
            idx = np.argmax(power)
            return freqs[idx]
        
        elif self.feature_type == 'spectral_entropy':
            # Normalize power spectrum
            power_norm = power / np.sum(power)
            # Calculate entropy
            entropy = -np.sum(power_norm * np.log(power_norm + 1e-10))
            return entropy
        
        elif self.feature_type == 'n_peaks':
            # Find spectral peaks
            peaks, _ = signal.find_peaks(power, prominence=np.max(power)/10)
            return min(len(peaks), self.n_peaks)
        
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")


class AutoCorrelation(BaseFeature):
    """
    Autocorrelation-based features.
    
    Parameters
    ----------
    lag : int or str, default='adaptive'
        Lag for autocorrelation. If 'adaptive', finds first minimum
    feature_type : str, default='value'
        - 'value': Autocorrelation at specified lag
        - 'decay_rate': Exponential decay rate
        - 'first_zero': Lag of first zero crossing
    """
    
    def __init__(self, transform: Optional[str] = None,
                 lag: Union[int, str] = 'adaptive',
                 feature_type: str = 'value'):
        super().__init__(transform)
        self.lag = lag
        self.feature_type = feature_type
    
    def extract(self, sequence: np.ndarray) -> float:
        """Extract autocorrelation feature."""
        if len(sequence) < 3:
            return 0
        
        # Normalize sequence
        seq_norm = sequence - np.mean(sequence)
        
        # Compute autocorrelation
        autocorr = np.correlate(seq_norm, seq_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags
        autocorr = autocorr / autocorr[0]  # Normalize
        
        if self.feature_type == 'value':
            if self.lag == 'adaptive':
                # Find first local minimum
                mins = signal.argrelmin(autocorr)[0]
                lag_idx = mins[0] if len(mins) > 0 else len(autocorr)//4
            else:
                lag_idx = min(self.lag, len(autocorr)-1)
            
            return autocorr[lag_idx]
        
        elif self.feature_type == 'decay_rate':
            # Fit exponential decay
            x = np.arange(1, min(len(autocorr), 20))
            y = np.abs(autocorr[1:len(x)+1])
            
            if np.any(y > 0):
                # Log-linear fit
                log_y = np.log(y + 1e-10)
                slope, _ = np.polyfit(x, log_y, 1)
                return -slope
            else:
                return 0
        
        elif self.feature_type == 'first_zero':
            # Find first zero crossing
            zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
            if len(zero_crossings) > 0:
                return zero_crossings[0]
            else:
                return len(autocorr)
        
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")


class ZeroCrossings(BaseFeature):
    """
    Count zero crossings in sequence.
    
    Parameters
    ----------
    normalize : bool, default=True
        Normalize by sequence length
    detrend : bool, default=True
        Remove linear trend before counting
    """
    
    def __init__(self, transform: Optional[str] = None,
                 normalize: bool = True, detrend: bool = True):
        super().__init__(transform)
        self.normalize = normalize
        self.detrend = detrend
    
    def extract(self, sequence: np.ndarray) -> float:
        """Count zero crossings."""
        if len(sequence) < 2:
            return 0
        
        seq = sequence.copy()
        
        # Detrend if requested
        if self.detrend:
            x = np.arange(len(seq))
            z = np.polyfit(x, seq, 1)
            trend = np.polyval(z, x)
            seq = seq - trend
        
        # Count crossings
        crossings = np.sum(np.diff(np.sign(seq)) != 0)
        
        # Normalize if requested
        if self.normalize:
            crossings = crossings / (len(sequence) - 1)
        
        return crossings


class WaveletFeatures(BaseFeature):
    """
    Wavelet-based features (requires pywt).
    
    Parameters
    ----------
    wavelet : str, default='db4'
        Wavelet type
    level : int, default=3
        Decomposition level
    feature_type : str, default='energy'
        - 'energy': Energy in detail coefficients
        - 'entropy': Wavelet entropy
    """
    
    def __init__(self, transform: Optional[str] = None,
                 wavelet: str = 'db4', level: int = 3,
                 feature_type: str = 'energy'):
        super().__init__(transform)
        self.wavelet = wavelet
        self.level = level
        self.feature_type = feature_type
        
        # Check if pywt is available
        try:
            import pywt
            self.pywt = pywt
        except ImportError:
            raise ImportError("pywt required for wavelet features. "
                            "Install with: pip install PyWavelets")
    
    def extract(self, sequence: np.ndarray) -> float:
        """Extract wavelet feature."""
        if len(sequence) < 2**self.level:
            return 0
        
        # Perform wavelet decomposition
        coeffs = self.pywt.wavedec(sequence, self.wavelet, level=self.level)
        
        if self.feature_type == 'energy':
            # Calculate energy in detail coefficients
            energy = 0
            for i in range(1, len(coeffs)):  # Skip approximation
                energy += np.sum(coeffs[i]**2)
            return energy
        
        elif self.feature_type == 'entropy':
            # Calculate wavelet entropy
            # Flatten all coefficients
            all_coeffs = np.concatenate([c.flatten() for c in coeffs])
            
            # Calculate relative energy
            energy = all_coeffs**2
            if np.sum(energy) > 0:
                p = energy / np.sum(energy)
                # Shannon entropy
                entropy = -np.sum(p * np.log(p + 1e-10))
                return entropy
            else:
                return 0
        
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")


# Composite features
class FeatureVector(BaseFeature):
    """
    Combine multiple features into a vector.
    
    Parameters
    ----------
    features : list of BaseFeature
        Features to combine
    aggregation : str, default='concatenate'
        How to combine features:
        - 'concatenate': Return all values
        - 'mean': Average of all features
        - 'weighted': Weighted average
    weights : array-like, optional
        Weights for weighted average
    """
    
    def __init__(self, features: list, 
                 aggregation: str = 'concatenate',
                 weights: Optional[np.ndarray] = None):
        self.features = features
        self.aggregation = aggregation
        self.weights = weights
    
    def extract(self, sequence: np.ndarray) -> Union[float, np.ndarray]:
        """Extract all features."""
        values = [f.extract(sequence) for f in self.features]
        
        if self.aggregation == 'concatenate':
            return np.array(values)
        elif self.aggregation == 'mean':
            return np.mean(values)
        elif self.aggregation == 'weighted':
            if self.weights is None:
                raise ValueError("Weights required for weighted aggregation")
            return np.average(values, weights=self.weights)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
