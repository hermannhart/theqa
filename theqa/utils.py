"""Utility functions for TheQA."""

import numpy as np
from typing import Union, List


def ensure_array(x: Union[List, np.ndarray]) -> np.ndarray:
    """Ensure input is numpy array."""
    return np.asarray(x)


def check_sequence_validity(sequence: np.ndarray) -> bool:
    """Check if sequence is valid for analysis."""
    if len(sequence) < 3:
        return False
    if np.any(np.isnan(sequence)):
        return False
    if np.any(np.isinf(sequence)):
        return False
    return True


def bootstrap_ci(data: np.ndarray, confidence: float = 0.95, n_bootstrap: int = 1000):
    """Compute bootstrap confidence intervals."""
    # Implementation
    pass
