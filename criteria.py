"""
Statistical criteria for detecting phase transitions.

This module provides various criteria for determining when
a system has transitioned under noise perturbation.
"""

import numpy as np
from scipy import stats
from abc import ABC, abstractmethod
from typing import List, Union, Optional


class BaseCriterion(ABC):
    """Abstract base class for statistical criteria."""
    
    def __init__(self, threshold: float):
        """
        Parameters
        ----------
        threshold : float
            Threshold value for detection
        """
        self.threshold = threshold
    
    @abstractmethod
    def compute(self, values: List[float]) -> float:
        """Compute criterion value from feature values."""
        pass
    
    def is_exceeded(self, value: float) -> bool:
        """Check if threshold is exceeded."""
        return value > self.threshold
    
    def __call__(self, values: List[float]) -> float:
        """Convenience method."""
        return self.compute(values)


class VarianceCriterion(BaseCriterion):
    """
    Variance-based detection criterion.
    
    The most common criterion - detects when variance of features
    exceeds a threshold.
    
    Parameters
    ----------
    threshold : float, default=0.1
        Variance threshold
    normalize : bool, default=False
        Normalize by mean squared
    """
    
    def __init__(self, threshold: float = 0.1, normalize: bool = False):
        super().__init__(threshold)
        self.normalize = normalize
    
    def compute(self, values: List[float]) -> float:
        """Compute variance of values."""
        if len(values) < 2:
            return 0
        
        variance = np.var(values)
        
        if self.normalize:
            mean_sq = np.mean(values)**2
            if mean_sq > 0:
                variance /= mean_sq
        
        return variance


class IQRCriterion(BaseCriterion):
    """
    Interquartile range criterion.
    
    More robust to outliers than variance.
    
    Parameters
    ----------
    threshold : float, default=0.2
        IQR threshold
    normalize : bool, default=False
        Normalize by median
    """
    
    def __init__(self, threshold: float = 0.2, normalize: bool = False):
        super().__init__(threshold)
        self.normalize = normalize
    
    def compute(self, values: List[float]) -> float:
        """Compute IQR of values."""
        if len(values) < 4:
            return 0
        
        q75 = np.percentile(values, 75)
        q25 = np.percentile(values, 25)
        iqr = q75 - q25
        
        if self.normalize:
            median = np.median(values)
            if abs(median) > 0:
                iqr /= abs(median)
        
        return iqr


class EntropyCriterion(BaseCriterion):
    """
    Entropy-based criterion.
    
    Detects when distribution of features becomes more random.
    
    Parameters
    ----------
    threshold : float, default=0.1
        Entropy threshold
    bins : int, default=10
        Number of bins for histogram
    """
    
    def __init__(self, threshold: float = 0.1, bins: int = 10):
        super().__init__(threshold)
        self.bins = bins
    
    def compute(self, values: List[float]) -> float:
        """Compute entropy of value distribution."""
        if len(values) < self.bins:
            return 0
        
        # Create histogram
        hist, _ = np.histogram(values, bins=self.bins)
        
        # Convert to probabilities
        p = hist / np.sum(hist)
        p = p[p > 0]  # Remove zeros
        
        # Calculate entropy
        entropy = -np.sum(p * np.log(p))
        
        # Normalize by maximum entropy
        max_entropy = np.log(self.bins)
        if max_entropy > 0:
            entropy /= max_entropy
        
        return entropy


class MADCriterion(BaseCriterion):
    """
    Median Absolute Deviation criterion.
    
    Very robust to outliers.
    
    Parameters
    ----------
    threshold : float, default=0.15
        MAD threshold
    scale : float, default=1.4826
        Scale factor for consistency with standard deviation
    """
    
    def __init__(self, threshold: float = 0.15, scale: float = 1.4826):
        super().__init__(threshold)
        self.scale = scale
    
    def compute(self, values: List[float]) -> float:
        """Compute scaled MAD."""
        if len(values) < 3:
            return 0
        
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        
        return self.scale * mad


class ThresholdCriterion(BaseCriterion):
    """
    Simple threshold crossing criterion.
    
    Detects when a certain fraction of values exceed a threshold.
    
    Parameters
    ----------
    threshold : float, default=0.5
        Fraction threshold
    value_threshold : float, default=1.0
        Value threshold
    """
    
    def __init__(self, threshold: float = 0.5, value_threshold: float = 1.0):
        super().__init__(threshold)
        self.value_threshold = value_threshold
    
    def compute(self, values: List[float]) -> float:
        """Compute fraction exceeding value threshold."""
        if len(values) == 0:
            return 0
        
        fraction = np.sum(np.array(values) > self.value_threshold) / len(values)
        return fraction


class CVCriterion(BaseCriterion):
    """
    Coefficient of Variation criterion.
    
    Ratio of standard deviation to mean.
    
    Parameters
    ----------
    threshold : float, default=0.3
        CV threshold
    """
    
    def __init__(self, threshold: float = 0.3):
        super().__init__(threshold)
    
    def compute(self, values: List[float]) -> float:
        """Compute coefficient of variation."""
        if len(values) < 2:
            return 0
        
        mean = np.mean(values)
        if abs(mean) < 1e-10:
            return float('inf')
        
        std = np.std(values)
        return std / abs(mean)


class SkewnessCriterion(BaseCriterion):
    """
    Skewness-based criterion.
    
    Detects asymmetry in feature distribution.
    
    Parameters
    ----------
    threshold : float, default=0.5
        Absolute skewness threshold
    """
    
    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold)
    
    def compute(self, values: List[float]) -> float:
        """Compute absolute skewness."""
        if len(values) < 3:
            return 0
        
        return abs(stats.skew(values))


class KurtosisCriterion(BaseCriterion):
    """
    Kurtosis-based criterion.
    
    Detects heavy tails in feature distribution.
    
    Parameters
    ----------
    threshold : float, default=1.0
        Excess kurtosis threshold
    """
    
    def __init__(self, threshold: float = 1.0):
        super().__init__(threshold)
    
    def compute(self, values: List[float]) -> float:
        """Compute excess kurtosis."""
        if len(values) < 4:
            return 0
        
        return stats.kurtosis(values)


class CompositeCriterion(BaseCriterion):
    """
    Combine multiple criteria.
    
    Parameters
    ----------
    criteria : list of BaseCriterion
        Criteria to combine
    aggregation : str, default='any'
        How to combine:
        - 'any': Exceeded if any criterion exceeded
        - 'all': Exceeded if all criteria exceeded
        - 'weighted': Weighted combination
    weights : array-like, optional
        Weights for weighted combination
    threshold : float, default=1.0
        Threshold for weighted combination
    """
    
    def __init__(self, criteria: List[BaseCriterion],
                 aggregation: str = 'any',
                 weights: Optional[np.ndarray] = None,
                 threshold: float = 1.0):
        super().__init__(threshold)
        self.criteria = criteria
        self.aggregation = aggregation
        self.weights = weights
    
    def compute(self, values: List[float]) -> float:
        """Compute composite criterion."""
        criterion_values = [c.compute(values) for c in self.criteria]
        
        if self.aggregation == 'any':
            # Return max normalized value
            normalized = [cv / c.threshold for cv, c in zip(criterion_values, self.criteria)]
            return max(normalized)
        
        elif self.aggregation == 'all':
            # Return min normalized value
            normalized = [cv / c.threshold for cv, c in zip(criterion_values, self.criteria)]
            return min(normalized)
        
        elif self.aggregation == 'weighted':
            if self.weights is None:
                weights = np.ones(len(self.criteria)) / len(self.criteria)
            else:
                weights = self.weights
            
            normalized = [cv / c.threshold for cv, c in zip(criterion_values, self.criteria)]
            return np.average(normalized, weights=weights)
        
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
    
    def is_exceeded(self, value: float) -> bool:
        """Check if composite threshold exceeded."""
        if self.aggregation in ['any', 'all']:
            return value > 1.0
        else:
            return value > self.threshold


class AdaptiveCriterion(BaseCriterion):
    """
    Adaptive criterion that adjusts threshold based on sequence properties.
    
    Parameters
    ----------
    base_criterion : BaseCriterion
        Underlying criterion to use
    adaptation : str, default='length'
        How to adapt threshold:
        - 'length': Scale with sequence length
        - 'range': Scale with value range
        - 'std': Scale with standard deviation
    scale_factor : float, default=1.0
        Scaling factor for adaptation
    """
    
    def __init__(self, base_criterion: BaseCriterion,
                 adaptation: str = 'length',
                 scale_factor: float = 1.0):
        self.base_criterion = base_criterion
        self.adaptation = adaptation
        self.scale_factor = scale_factor
        self._adapted_threshold = base_criterion.threshold
    
    def compute(self, values: List[float]) -> float:
        """Compute with adapted threshold."""
        # Adapt threshold based on values
        if self.adaptation == 'length':
            factor = np.sqrt(len(values) / 100)  # Normalize to 100
        elif self.adaptation == 'range':
            if len(values) > 1:
                factor = (np.max(values) - np.min(values)) / 10
            else:
                factor = 1.0
        elif self.adaptation == 'std':
            if len(values) > 1:
                factor = np.std(values) / 0.1  # Normalize to 0.1
            else:
                factor = 1.0
        else:
            raise ValueError(f"Unknown adaptation: {self.adaptation}")
        
        # Update threshold
        self._adapted_threshold = self.base_criterion.threshold * factor * self.scale_factor
        
        # Compute criterion value
        return self.base_criterion.compute(values)
    
    def is_exceeded(self, value: float) -> bool:
        """Check against adapted threshold."""
        return value > self._adapted_threshold
    
    @property
    def threshold(self):
        """Return current adapted threshold."""
        return self._adapted_threshold
