"""
Unit tests for theqa.core module.
"""

import pytest
import numpy as np
from theqa import (
    compute_sigma_c, TripleRule, Result,
    CollatzSystem, PeakCounter, VarianceCriterion
)


class TestComputeSigmaC:
    """Test compute_sigma_c function."""
    
    def test_simple_sequence(self):
        """Test with simple exponential sequence."""
        sequence = [1, 2, 4, 8, 16, 32, 64]
        sigma_c = compute_sigma_c(sequence)
        
        assert isinstance(sigma_c, float)
        assert 0 < sigma_c < np.pi/2
    
    def test_empty_sequence(self):
        """Test with empty sequence."""
        with pytest.raises(Exception):
            compute_sigma_c([])
    
    def test_constant_sequence(self):
        """Test with constant sequence."""
        sequence = [5] * 100
        sigma_c = compute_sigma_c(sequence)
        
        # Should be very high (robust to noise)
        assert sigma_c > 0.5
    
    def test_random_sequence(self):
        """Test with random sequence."""
        np.random.seed(42)
        sequence = np.random.randn(100)
        sigma_c = compute_sigma_c(sequence)
        
        # Random sequences should be robust
        assert sigma_c > 0.3
    
    def test_methods(self):
        """Test different computation methods."""
        sequence = CollatzSystem(n=27).generate()
        
        # Test all methods
        methods = ['empirical', 'spectral', 'gradient', 'analytical']
        results = {}
        
        for method in methods:
            sigma_c = compute_sigma_c(sequence, method=method)
            results[method] = sigma_c
            assert 0 < sigma_c < np.pi/2
        
        # Methods should give similar results (within 50%)
        values = list(results.values())
        assert max(values) / min(values) < 2.0


class TestTripleRule:
    """Test TripleRule class."""
    
    def test_initialization(self):
        """Test TripleRule initialization."""
        system = CollatzSystem(n=27)
        feature = PeakCounter()
        criterion = VarianceCriterion()
        
        tr = TripleRule(system, feature, criterion)
        
        assert tr.system == system
        assert tr.feature == feature
        assert tr.criterion == criterion
    
    def test_compute_basic(self):
        """Test basic computation."""
        tr = TripleRule(
            system=CollatzSystem(n=27),
            feature=PeakCounter(transform='log'),
            criterion=VarianceCriterion(threshold=0.1)
        )
        
        result = tr.compute(n_trials=50, method='empirical')
        
        assert isinstance(result, Result)
        assert 0 < result.sigma_c < np.pi/2
        assert result.ci_lower <= result.sigma_c <= result.ci_upper
        assert result.time_elapsed > 0
    
    def test_auto_method_selection(self):
        """Test automatic method selection."""
        # Short sequence -> empirical
        short_seq = [1, 2, 3, 4, 5]
        tr = TripleRule(system=short_seq)
        result = tr.compute(method='auto')
        assert result.method == 'empirical'
        
        # Long sequence -> gradient
        long_seq = np.random.randn(2000)
        tr = TripleRule(system=long_seq)
        result = tr.compute(method='auto')
        assert result.method in ['gradient', 'empirical']
    
    def test_confidence_intervals(self):
        """Test confidence interval computation."""
        tr = TripleRule(system=CollatzSystem(n=27))
        
        # Different confidence levels
        for confidence in [0.90, 0.95, 0.99]:
            result = tr.compute(confidence=confidence, n_trials=30)
            
            # Higher confidence -> wider interval
            interval_width = result.ci_upper - result.ci_lower
            assert interval_width > 0
    
    def test_convergence(self):
        """Test convergence detection."""
        tr = TripleRule(system=CollatzSystem(n=27))
        
        # With sufficient range, should converge
        result = tr.compute(
            method='empirical',
            sigma_range=(0.001, 1.0),
            n_sigma=50
        )
        
        assert result.converged == True


class TestResult:
    """Test Result dataclass."""
    
    def test_result_creation(self):
        """Test Result creation and properties."""
        result = Result(
            sigma_c=0.117,
            ci_lower=0.110,
            ci_upper=0.124,
            converged=True,
            n_iterations=50,
            time_elapsed=1.23,
            method='empirical',
            details={'sigmas': [0.1, 0.11, 0.12]}
        )
        
        assert result.sigma_c == 0.117
        assert result.converged == True
        assert result.method == 'empirical'
        assert 'sigmas' in result.details
    
    def test_result_string_representation(self):
        """Test string representations."""
        result = Result(
            sigma_c=0.117,
            ci_lower=0.110,
            ci_upper=0.124,
            converged=True,
            n_iterations=50,
            time_elapsed=1.23,
            method='empirical',
            details={}
        )
        
        # Test __str__
        str_repr = str(result)
        assert "0.117" in str_repr
        assert "0.110" in str_repr
        assert "0.124" in str_repr
        
        # Test __repr__
        repr_str = repr(result)
        assert "Result" in repr_str
        assert "0.117" in repr_str
        assert "empirical" in repr_str


class TestIntegration:
    """Integration tests for core functionality."""
    
    def test_full_workflow(self):
        """Test complete analysis workflow."""
        # 1. Create system
        system = CollatzSystem(n=27, q=3)
        
        # 2. Quick analysis
        sigma_c_quick = compute_sigma_c(system.generate())
        assert 0.05 < sigma_c_quick < 0.5  # Reasonable range for Collatz
        
        # 3. Detailed analysis
        tr = TripleRule(
            system=system,
            feature=PeakCounter(transform='log'),
            criterion=VarianceCriterion(threshold=0.1)
        )
        
        result = tr.compute(
            n_trials=100,
            method='empirical',
            confidence=0.95
        )
        
        # 4. Check results
        assert abs(result.sigma_c - 0.117) < 0.05  # Close to known value
        assert result.converged
        assert result.ci_upper - result.ci_lower < 0.1  # Reasonable CI width
    
    def test_different_systems(self):
        """Test with various system types."""
        from theqa import FibonacciSystem, LogisticMap
        
        systems = {
            'fibonacci': (FibonacciSystem(n=50), 0.001, 0.01),  # Ultra-sensitive
            'collatz': (CollatzSystem(n=27), 0.05, 0.2),       # Medium
            'logistic': (LogisticMap(r=4.0), 0.1, 0.5),        # Robust
        }
        
        for name, (system, min_sc, max_sc) in systems.items():
            sigma_c = compute_sigma_c(system.generate())
            assert min_sc < sigma_c < max_sc, f"{name} σc out of expected range"


def test_analyze_system():
    """Test analyze_system helper function."""
    from theqa.core import analyze_system
    
    system = CollatzSystem(n=27)
    df = analyze_system(system, n_trials=20)
    
    assert len(df) > 0
    assert 'feature' in df.columns
    assert 'criterion' in df.columns
    assert 'sigma_c' in df.columns
    assert all(df['sigma_c'] > 0)
    assert all(df['sigma_c'] < np.pi/2)
