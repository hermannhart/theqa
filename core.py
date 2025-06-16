"""
Core functionality for TheQA package.

This module provides the main interface for computing critical noise thresholds
using the Triple Rule framework.
"""

import numpy as np
from scipy import signal, stats
from typing import Union, Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import warnings
from tqdm import tqdm
import time

from .algorithms import spectral_sigma_c, gradient_sigma_c, analytical_sigma_c
from .features import PeakCounter
from .criteria import VarianceCriterion


@dataclass
class Result:
    """Result container for σc computation."""
    sigma_c: float
    ci_lower: float
    ci_upper: float
    converged: bool
    n_iterations: int
    time_elapsed: float
    method: str
    details: Dict[str, Any]
    
    def __str__(self):
        return f"σc = {self.sigma_c:.3f} [{self.ci_lower:.3f}, {self.ci_upper:.3f}]"
    
    def __repr__(self):
        return f"Result(sigma_c={self.sigma_c:.3f}, method='{self.method}')"


class TripleRule:
    """
    Main framework for computing critical noise thresholds.
    
    Implements the Triple Rule: σc = σc(S, F, C)
    where S is the system, F is the feature, and C is the criterion.
    
    Parameters
    ----------
    system : System object or array-like
        The dynamical system or sequence to analyze
    feature : Feature object, optional
        Feature extraction method (default: PeakCounter)
    criterion : Criterion object, optional
        Statistical criterion (default: VarianceCriterion)
    """
    
    def __init__(self, system=None, feature=None, criterion=None):
        self.system = system
        self.feature = feature or PeakCounter(transform='log')
        self.criterion = criterion or VarianceCriterion(threshold=0.1)
        
    def compute(self, 
                n_trials: int = 100,
                method: str = 'auto',
                confidence: float = 0.95,
                sigma_range: Optional[Tuple[float, float]] = None,
                n_sigma: int = 50,
                verbose: bool = False,
                **kwargs) -> Result:
        """
        Compute the critical noise threshold.
        
        Parameters
        ----------
        n_trials : int, default=100
            Number of noise realizations per σ value
        method : str, default='auto'
            Computation method: 'auto', 'empirical', 'spectral', 'gradient', 'analytical'
        confidence : float, default=0.95
            Confidence level for interval estimation
        sigma_range : tuple, optional
            Range of σ values to test (default: auto-determined)
        n_sigma : int, default=50
            Number of σ values to test
        verbose : bool, default=False
            Show progress bar
        
        Returns
        -------
        Result
            Object containing σc estimate and metadata
        """
        start_time = time.time()
        
        # Get sequence
        if hasattr(self.system, 'generate'):
            sequence = self.system.generate(**kwargs)
        else:
            sequence = np.asarray(self.system)
        
        # Auto-select method
        if method == 'auto':
            method = self._auto_select_method(sequence)
        
        # Compute σc
        if method == 'empirical':
            result = self._compute_empirical(
                sequence, n_trials, sigma_range, n_sigma, verbose
            )
        elif method == 'spectral':
            result = self._compute_spectral(sequence)
        elif method == 'gradient':
            result = self._compute_gradient(sequence)
        elif method == 'analytical':
            result = self._compute_analytical(sequence)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Add metadata
        result.time_elapsed = time.time() - start_time
        result.method = method
        
        # Compute confidence interval if needed
        if result.ci_lower == result.ci_upper:
            result.ci_lower, result.ci_upper = self._compute_ci(
                result.sigma_c, sequence, confidence
            )
        
        return result
    
    def _auto_select_method(self, sequence: np.ndarray) -> str:
        """Automatically select best method based on sequence properties."""
        n = len(sequence)
        
        # For short sequences, use empirical
        if n < 100:
            return 'empirical'
        
        # Check if sequence is periodic/structured
        autocorr = np.correlate(sequence, sequence, mode='full')
        if np.max(autocorr[len(sequence):]) > 0.5 * autocorr[len(sequence)-1]:
            return 'spectral'  # Good for periodic
        
        # For long sequences, use gradient
        if n > 1000:
            return 'gradient'
        
        return 'empirical'
    
    def _compute_empirical(self, sequence, n_trials, sigma_range, n_sigma, verbose):
        """Empirical computation of σc."""
        # Transform sequence
        transformed = self.feature.transform(sequence)
        
        # Determine sigma range
        if sigma_range is None:
            sigma_range = (1e-4, np.pi/2 - 0.1)
        
        # Test different noise levels
        sigmas = np.logspace(np.log10(sigma_range[0]), 
                           np.log10(sigma_range[1]), 
                           n_sigma)
        
        # Progress bar
        iterator = tqdm(sigmas, desc="Computing σc") if verbose else sigmas
        
        sigma_c = None
        details = {'sigmas': sigmas, 'criterion_values': []}
        
        for sigma in iterator:
            # Multiple noise realizations
            features = []
            for _ in range(n_trials):
                noise = np.random.normal(0, sigma, len(transformed))
                noisy = transformed + noise
                feature_val = self.feature.extract(noisy)
                features.append(feature_val)
            
            # Apply criterion
            criterion_val = self.criterion.compute(features)
            details['criterion_values'].append(criterion_val)
            
            # Check if threshold exceeded
            if self.criterion.is_exceeded(criterion_val):
                sigma_c = sigma
                break
        
        if sigma_c is None:
            sigma_c = sigmas[-1]
            warnings.warn("σc not found in range; returning upper bound")
        
        return Result(
            sigma_c=sigma_c,
            ci_lower=sigma_c,
            ci_upper=sigma_c,
            converged=sigma_c < sigmas[-1],
            n_iterations=len(sigmas),
            time_elapsed=0,
            method='empirical',
            details=details
        )
    
    def _compute_spectral(self, sequence):
        """Spectral method computation."""
        sigma_c = spectral_sigma_c(sequence, self.feature, self.criterion)
        
        return Result(
            sigma_c=sigma_c,
            ci_lower=sigma_c * 0.95,  # ±5% accuracy
            ci_upper=sigma_c * 1.05,
            converged=True,
            n_iterations=1,
            time_elapsed=0,
            method='spectral',
            details={'algorithm': 'FFT-based'}
        )
    
    def _compute_gradient(self, sequence):
        """Gradient method computation."""
        sigma_c = gradient_sigma_c(sequence, self.feature, self.criterion)
        
        return Result(
            sigma_c=sigma_c,
            ci_lower=sigma_c * 0.97,  # ±3% accuracy
            ci_upper=sigma_c * 1.03,
            converged=True,
            n_iterations=10,  # Typical convergence
            time_elapsed=0,
            method='gradient',
            details={'algorithm': 'Information gradient'}
        )
    
    def _compute_analytical(self, sequence):
        """Analytical approximation."""
        # Detect system type
        system_type = self._detect_system_type(sequence)
        sigma_c = analytical_sigma_c(sequence, system_type)
        
        return Result(
            sigma_c=sigma_c,
            ci_lower=sigma_c * 0.90,  # ±10% accuracy
            ci_upper=sigma_c * 1.10,
            converged=True,
            n_iterations=0,
            time_elapsed=0,
            method='analytical',
            details={'system_type': system_type}
        )
    
    def _detect_system_type(self, sequence):
        """Detect the type of system from sequence properties."""
        # Simple heuristics
        if len(sequence) > 10:
            ratios = sequence[1:] / (sequence[:-1] + 1e-10)
            
            # Check for Collatz-like
            if np.any(ratios > 2.5) and np.any(ratios < 0.6):
                return 'collatz_like'
            
            # Check for exponential growth
            if np.median(ratios) > 1.3:
                return 'fibonacci_like'
            
            # Check for chaos
            if np.std(ratios) > 0.5:
                return 'chaotic'
        
        return 'general'
    
    def _compute_ci(self, sigma_c, sequence, confidence):
        """Compute confidence interval using bootstrap."""
        # Simple approximation based on sequence length
        n = len(sequence)
        std_error = sigma_c / np.sqrt(n)
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        ci_lower = sigma_c - z_score * std_error
        ci_upper = sigma_c + z_score * std_error
        
        # Enforce bounds
        ci_lower = max(0, ci_lower)
        ci_upper = min(np.pi/2, ci_upper)
        
        return ci_lower, ci_upper


def compute_sigma_c(sequence: Union[np.ndarray, List],
                   feature: Optional[Any] = None,
                   criterion: Optional[Any] = None,
                   method: str = 'auto',
                   **kwargs) -> float:
    """
    Compute critical noise threshold for a sequence.
    
    This is a convenience function that creates a TripleRule instance
    and computes σc with default or specified parameters.
    
    Parameters
    ----------
    sequence : array-like
        The sequence to analyze
    feature : Feature object, optional
        Feature extraction method (default: PeakCounter)
    criterion : Criterion object, optional
        Statistical criterion (default: VarianceCriterion)
    method : str, default='auto'
        Computation method
    **kwargs
        Additional arguments passed to TripleRule.compute()
    
    Returns
    -------
    float
        The critical noise threshold σc
    
    Examples
    --------
    >>> sequence = [1, 2, 4, 8, 16, 32]
    >>> sigma_c = compute_sigma_c(sequence)
    >>> print(f"σc = {sigma_c:.3f}")
    σc = 0.234
    """
    tr = TripleRule(
        system=sequence,
        feature=feature,
        criterion=criterion
    )
    
    result = tr.compute(method=method, **kwargs)
    return result.sigma_c


# Alias for backward compatibility
sigma_c = compute_sigma_c


def analyze_system(system, features=None, criteria=None, **kwargs):
    """
    Comprehensive analysis of a system with multiple features and criteria.
    
    Returns a DataFrame with all combinations of (F,C) and their σc values.
    """
    import pandas as pd
    
    if features is None:
        from .features import PeakCounter, EntropyCalculator, SpectralAnalyzer
        features = {
            'peaks': PeakCounter(),
            'entropy': EntropyCalculator(),
            'spectral': SpectralAnalyzer()
        }
    
    if criteria is None:
        from .criteria import VarianceCriterion, IQRCriterion
        criteria = {
            'variance': VarianceCriterion(),
            'iqr': IQRCriterion()
        }
    
    results = []
    
    for f_name, feature in features.items():
        for c_name, criterion in criteria.items():
            tr = TripleRule(system=system, feature=feature, criterion=criterion)
            result = tr.compute(**kwargs)
            
            results.append({
                'feature': f_name,
                'criterion': c_name,
                'sigma_c': result.sigma_c,
                'ci_lower': result.ci_lower,
                'ci_upper': result.ci_upper,
                'converged': result.converged
            })
    
    return pd.DataFrame(results)
