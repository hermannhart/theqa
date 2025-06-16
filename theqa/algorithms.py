"""
Efficient algorithms for computing critical noise thresholds.

This module provides O(n log n) and faster algorithms as alternatives
to the O(n²) empirical method.
"""

import numpy as np
from scipy import signal, optimize
from typing import Optional, Tuple, Callable
from joblib import Parallel, delayed
import warnings


def spectral_sigma_c(sequence: np.ndarray,
                    feature: Optional[Callable] = None,
                    criterion: Optional[Callable] = None,
                    epsilon: float = 0.1) -> float:
    """
    Compute σc using spectral method - O(n log n).
    
    Based on the principle that Gaussian noise has predictable
    effects on the Fourier transform of a signal.
    
    Parameters
    ----------
    sequence : array-like
        Input sequence
    feature : callable, optional
        Feature extractor (default: peak counting)
    criterion : callable, optional
        Statistical criterion (default: variance)
    epsilon : float, default=0.1
        Detection threshold
    
    Returns
    -------
    float
        Critical noise threshold
    """
    sequence = np.asarray(sequence)
    n = len(sequence)
    
    if n < 4:
        return 0.001
    
    # Default feature: spectral power
    if feature is None:
        from .features import SpectralAnalyzer
        feature = SpectralAnalyzer(feature_type='dominant_freq')
    
    # Transform sequence
    if hasattr(feature, 'transform'):
        seq_transformed = feature.transform(sequence)
    else:
        seq_transformed = np.log(np.abs(sequence) + 1)
    
    # Compute power spectrum
    fft = np.fft.fft(seq_transformed)
    power = np.abs(fft)**2
    freqs = np.fft.fftfreq(n)
    
    # Find dominant frequency
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_power = power[pos_mask]
    
    if len(pos_power) == 0:
        return np.pi/2 - 0.1
    
    dom_idx = np.argmax(pos_power)
    f_dom = pos_freqs[dom_idx]
    
    # Theoretical formula for critical threshold
    # Based on: noise power equals signal power at dominant frequency
    sigma_c = np.sqrt(-np.log(epsilon) / (2 * np.pi**2 * f_dom**2))
    
    # Enforce bounds
    sigma_c = np.clip(sigma_c, 0.001, np.pi/2 - 0.1)
    
    return sigma_c


def gradient_sigma_c(sequence: np.ndarray,
                    feature: Optional[Callable] = None,
                    criterion: Optional[Callable] = None,
                    max_iter: int = 20,
                    tol: float = 1e-4) -> float:
    """
    Compute σc using information gradient method - O(n).
    
    Finds the critical threshold by maximizing the rate of
    information loss with respect to noise level.
    
    Parameters
    ----------
    sequence : array-like
        Input sequence
    feature : callable, optional
        Feature extractor
    criterion : callable, optional
        Statistical criterion
    max_iter : int, default=20
        Maximum iterations
    tol : float, default=1e-4
        Convergence tolerance
    
    Returns
    -------
    float
        Critical noise threshold
    """
    sequence = np.asarray(sequence)
    n = len(sequence)
    
    if n < 2:
        return 0.001
    
    # Normalize sequence
    seq_norm = (sequence - np.mean(sequence)) / (np.std(sequence) + 1e-10)
    
    # Define mutual information approximation
    def mutual_info(sigma):
        """Approximate mutual information between clean and noisy signal."""
        if sigma < 1e-10:
            return np.log(n)
        snr = 1 / sigma**2
        return 0.5 * np.log(1 + snr)
    
    # Define gradient
    def grad_mi(sigma):
        """Gradient of mutual information."""
        if sigma < 1e-10:
            return -1e10
        return -1 / (sigma * (1 + sigma**2))
    
    # Find maximum gradient (steepest descent)
    sigma = 0.1  # Initial guess
    
    for _ in range(max_iter):
        g = grad_mi(sigma)
        h = 2 / (sigma**2 * (1 + sigma**2)**2)  # Hessian
        
        if abs(h) < 1e-10:
            break
        
        # Newton step
        delta = -g / h
        sigma_new = sigma + delta
        
        # Ensure positive
        sigma_new = max(sigma_new, 1e-4)
        
        if abs(delta) < tol:
            break
        
        sigma = sigma_new
    
    # Scale by sequence properties
    scale_factor = 1 / (1 + np.log(n))
    sigma_c = sigma * scale_factor
    
    # Enforce bounds
    sigma_c = np.clip(sigma_c, 0.001, np.pi/2 - 0.1)
    
    return sigma_c


def analytical_sigma_c(sequence: np.ndarray,
                      system_type: str = 'general') -> float:
    """
    Analytical approximation of σc - O(1) after O(n) preprocessing.
    
    Uses closed-form formulas based on system classification.
    
    Parameters
    ----------
    sequence : array-like
        Input sequence
    system_type : str, default='general'
        Type of system:
        - 'collatz_like': qn+1 systems
        - 'fibonacci_like': Exponential growth
        - 'chaotic': Chaotic maps
        - 'periodic': Periodic sequences
        - 'random': Near-random
        - 'general': Auto-detect
    
    Returns
    -------
    float
        Critical noise threshold
    """
    sequence = np.asarray(sequence)
    n = len(sequence)
    
    if n < 3:
        return 0.001
    
    # Auto-detect system type if needed
    if system_type == 'general':
        system_type = _detect_system_type(sequence)
    
    # Apply appropriate formula
    if system_type == 'collatz_like':
        # Estimate multiplier q
        q = _estimate_collatz_multiplier(sequence)
        sigma_c = 0.043 * (np.log(q) / np.log(2))**0.65 + 0.10
    
    elif system_type == 'fibonacci_like':
        # Estimate growth rate
        growth_rate = _estimate_growth_rate(sequence)
        sigma_c = 0.01 / growth_rate
    
    elif system_type == 'chaotic':
        # Estimate Lyapunov exponent
        lyapunov = _estimate_lyapunov(sequence)
        sigma_c = 0.15 * np.sqrt(lyapunov)
    
    elif system_type == 'periodic':
        # Estimate period
        period = _estimate_period(sequence)
        sigma_c = 0.05 * np.sqrt(period) / np.log(n + 2)
    
    elif system_type == 'random':
        # High threshold for random sequences
        sigma_c = 0.8 + 0.2 * np.random.rand()
    
    else:
        # General approximation based on entropy
        entropy = _estimate_entropy(sequence)
        sigma_c = 0.1 * np.sqrt(entropy) / np.log(n + 2)
    
    # Enforce bounds
    sigma_c = np.clip(sigma_c, 0.001, np.pi/2 - 0.1)
    
    return sigma_c


def adaptive_sigma_c(sequence: np.ndarray,
                    methods: list = None,
                    weights: Optional[np.ndarray] = None) -> float:
    """
    Adaptive combination of multiple methods for highest accuracy.
    
    Parameters
    ----------
    sequence : array-like
        Input sequence
    methods : list, optional
        List of methods to use (default: all three)
    weights : array-like, optional
        Weights for combining estimates
    
    Returns
    -------
    float
        Critical noise threshold
    """
    if methods is None:
        methods = ['spectral', 'gradient', 'analytical']
    
    estimates = []
    
    for method in methods:
        try:
            if method == 'spectral':
                est = spectral_sigma_c(sequence)
            elif method == 'gradient':
                est = gradient_sigma_c(sequence)
            elif method == 'analytical':
                est = analytical_sigma_c(sequence)
            else:
                continue
            
            estimates.append(est)
        except Exception:
            # Skip failed methods
            pass
    
    if len(estimates) == 0:
        warnings.warn("All methods failed; returning default")
        return 0.1
    
    # Combine estimates
    if len(estimates) == 1:
        return estimates[0]
    
    # Check for outliers
    median = np.median(estimates)
    mad = np.median(np.abs(estimates - median))
    
    # Remove outliers (> 3 MAD from median)
    valid = [e for e in estimates if abs(e - median) <= 3 * mad]
    
    if len(valid) == 0:
        valid = estimates
    
    # Weighted average or median
    if weights is not None and len(weights) == len(valid):
        return np.average(valid, weights=weights)
    else:
        return np.median(valid)


def parallel_sigma_c(sequences: list,
                    method: str = 'auto',
                    n_jobs: int = -1,
                    **kwargs) -> list:
    """
    Compute σc for multiple sequences in parallel.
    
    Parameters
    ----------
    sequences : list of array-like
        List of sequences to analyze
    method : str, default='auto'
        Method to use
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = all CPUs)
    **kwargs
        Additional arguments passed to computation
    
    Returns
    -------
    list
        Critical thresholds for each sequence
    """
    def compute_single(seq):
        try:
            if method == 'spectral':
                return spectral_sigma_c(seq, **kwargs)
            elif method == 'gradient':
                return gradient_sigma_c(seq, **kwargs)
            elif method == 'analytical':
                return analytical_sigma_c(seq, **kwargs)
            elif method == 'adaptive':
                return adaptive_sigma_c(seq, **kwargs)
            else:  # auto
                return adaptive_sigma_c(seq, **kwargs)
        except Exception as e:
            warnings.warn(f"Computation failed: {e}")
            return np.nan
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_single)(seq) for seq in sequences
    )
    
    return results


# Helper functions for system detection and parameter estimation

def _detect_system_type(sequence: np.ndarray) -> str:
    """Detect the type of dynamical system from sequence properties."""
    n = len(sequence)
    
    if n < 10:
        return 'general'
    
    # Check for periodicity
    autocorr = np.correlate(sequence, sequence, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    if np.max(autocorr[1:n//2]) > 0.8 * autocorr[0]:
        return 'periodic'
    
    # Check ratios for classification
    ratios = sequence[1:] / (sequence[:-1] + 1e-10)
    
    # Collatz-like: mix of growth and decay
    if np.any(ratios > 2.5) and np.any(ratios < 0.6):
        return 'collatz_like'
    
    # Fibonacci-like: consistent growth
    if np.median(ratios) > 1.3 and np.std(ratios) < 0.5:
        return 'fibonacci_like'
    
    # Chaotic: high variability
    if np.std(ratios) > 0.5:
        return 'chaotic'
    
    # Random: no clear pattern
    return 'random'


def _estimate_collatz_multiplier(sequence: np.ndarray) -> float:
    """Estimate the multiplier q in qn+1 systems."""
    # Find large jumps
    ratios = sequence[1:] / (sequence[:-1] + 1e-10)
    large_jumps = ratios[ratios > 2]
    
    if len(large_jumps) > 0:
        # Estimate q from median large jump
        # For qn+1: ratio ≈ q when n is small
        return np.median(large_jumps)
    else:
        return 3  # Default


def _estimate_growth_rate(sequence: np.ndarray) -> float:
    """Estimate exponential growth rate."""
    if len(sequence) < 3:
        return 1.0
    
    # Log-linear fit
    x = np.arange(len(sequence))
    y = np.log(np.abs(sequence) + 1)
    
    try:
        slope, _ = np.polyfit(x, y, 1)
        return np.exp(slope)
    except:
        return 1.0


def _estimate_lyapunov(sequence: np.ndarray) -> float:
    """Estimate Lyapunov exponent."""
    if len(sequence) < 10:
        return 0.1
    
    # Finite difference approximation
    diffs = np.abs(np.diff(sequence))
    
    # Average log of stretching
    log_stretch = np.log(diffs + 1e-10)
    lyapunov = np.mean(log_stretch)
    
    return max(0, lyapunov)


def _estimate_period(sequence: np.ndarray) -> int:
    """Estimate period of sequence."""
    n = len(sequence)
    
    # Try different periods
    for p in range(2, min(n//2, 50)):
        # Check if sequence repeats with period p
        if n >= 2*p:
            seg1 = sequence[:p]
            seg2 = sequence[p:2*p]
            
            if np.allclose(seg1, seg2, rtol=0.1):
                return p
    
    return n  # No period found


def _estimate_entropy(sequence: np.ndarray) -> float:
    """Estimate Shannon entropy."""
    if len(sequence) < 2:
        return 0
    
    # Histogram
    hist, _ = np.histogram(sequence, bins=20)
    
    # Probabilities
    p = hist / np.sum(hist)
    p = p[p > 0]
    
    # Entropy
    return -np.sum(p * np.log(p))
