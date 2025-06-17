"""
Optimization framework for finding optimal (F,C) pairs.

This module provides functions to optimize the choice of feature extractor
and statistical criterion for specific applications.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from itertools import product
import warnings

from .core import TripleRule
from .features import *
from .criteria import *
from .algorithms import adaptive_sigma_c


def optimize_for_sensitivity(system,
                           test_features: Optional[Dict] = None,
                           test_criteria: Optional[Dict] = None,
                           n_trials: int = 50,
                           method: str = 'adaptive') -> Dict[str, Any]:
    """
    Find (F,C) pair that minimizes σc for maximum sensitivity.
    
    Parameters
    ----------
    system : System object or array-like
        The system to analyze
    test_features : dict, optional
        Features to test (default: common sensitive features)
    test_criteria : dict, optional
        Criteria to test (default: common sensitive criteria)
    n_trials : int, default=50
        Number of trials for σc computation
    method : str, default='adaptive'
        Method for σc computation
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'feature': Best feature name
        - 'criterion': Best criterion name
        - 'sigma_c': Achieved σc value
        - 'feature_obj': Feature object
        - 'criterion_obj': Criterion object
    """
    if test_features is None:
        test_features = {
            'peaks_log': PeakCounter(transform='log'),
            'peaks_sqrt': PeakCounter(transform='sqrt'),
            'entropy_fine': EntropyCalculator(bins=50),
            'spectral_freq': SpectralAnalyzer(feature_type='dominant_freq'),
            'zero_cross': ZeroCrossings(detrend=True),
            'autocorr': AutoCorrelation(feature_type='first_zero'),
        }
    
    if test_criteria is None:
        test_criteria = {
            'variance_0.05': VarianceCriterion(threshold=0.05),
            'variance_0.1': VarianceCriterion(threshold=0.1),
            'iqr_0.1': IQRCriterion(threshold=0.1),
            'entropy_0.05': EntropyCriterion(threshold=0.05),
        }
    
    best_sigma_c = float('inf')
    best_result = None
    
    for f_name, feature in test_features.items():
        for c_name, criterion in test_criteria.items():
            try:
                tr = TripleRule(
                    system=system,
                    feature=feature,
                    criterion=criterion
                )
                
                result = tr.compute(
                    n_trials=n_trials,
                    method=method,
                    verbose=False
                )
                
                if result.sigma_c < best_sigma_c:
                    best_sigma_c = result.sigma_c
                    best_result = {
                        'feature': f_name,
                        'criterion': c_name,
                        'sigma_c': result.sigma_c,
                        'feature_obj': feature,
                        'criterion_obj': criterion,
                        'full_result': result
                    }
            except Exception as e:
                warnings.warn(f"Failed for {f_name} + {c_name}: {str(e)}")
                continue
    
    if best_result is None:
        raise ValueError("No valid (F,C) pair found")
    
    return best_result


def optimize_for_robustness(system,
                           test_features: Optional[Dict] = None,
                           test_criteria: Optional[Dict] = None,
                           n_trials: int = 50,
                           method: str = 'adaptive') -> Dict[str, Any]:
    """
    Find (F,C) pair that maximizes σc for maximum robustness.
    
    Parameters
    ----------
    system : System object or array-like
        The system to analyze
    test_features : dict, optional
        Features to test (default: common robust features)
    test_criteria : dict, optional
        Criteria to test (default: common robust criteria)
    n_trials : int, default=50
        Number of trials for σc computation
    method : str, default='adaptive'
        Method for σc computation
    
    Returns
    -------
    dict
        Dictionary with optimal configuration
    """
    if test_features is None:
        test_features = {
            'mean': lambda s: np.mean(s),
            'median': lambda s: np.median(s),
            'range': lambda s: np.max(s) - np.min(s),
            'std': lambda s: np.std(s),
            'iqr': lambda s: np.percentile(s, 75) - np.percentile(s, 25),
            'entropy_coarse': EntropyCalculator(bins=5),
        }
    
    if test_criteria is None:
        test_criteria = {
            'variance_1.0': VarianceCriterion(threshold=1.0),
            'variance_2.0': VarianceCriterion(threshold=2.0),
            'threshold_0.8': ThresholdCriterion(threshold=0.8),
            'mad_0.5': MADCriterion(threshold=0.5),
        }
    
    best_sigma_c = 0
    best_result = None
    
    for f_name, feature in test_features.items():
        for c_name, criterion in test_criteria.items():
            try:
                # Handle callable features
                if callable(feature) and not hasattr(feature, 'extract'):
                    # Wrap in a simple feature class
                    class CallableFeature:
                        def __init__(self, func):
                            self.func = func
                        def transform(self, seq):
                            return seq
                        def extract(self, seq):
                            return self.func(seq)
                    feature_obj = CallableFeature(feature)
                else:
                    feature_obj = feature
                
                tr = TripleRule(
                    system=system,
                    feature=feature_obj,
                    criterion=criterion
                )
                
                result = tr.compute(
                    n_trials=n_trials,
                    method=method,
                    verbose=False
                )
                
                if result.sigma_c > best_sigma_c:
                    best_sigma_c = result.sigma_c
                    best_result = {
                        'feature': f_name,
                        'criterion': c_name,
                        'sigma_c': result.sigma_c,
                        'feature_obj': feature_obj,
                        'criterion_obj': criterion,
                        'full_result': result
                    }
            except Exception as e:
                continue
    
    if best_result is None:
        raise ValueError("No valid (F,C) pair found")
    
    return best_result


def optimize_for_discrimination(systems: List,
                              test_features: Optional[Dict] = None,
                              test_criteria: Optional[Dict] = None,
                              n_trials: int = 30,
                              method: str = 'adaptive') -> Dict[str, Any]:
    """
    Find (F,C) pair that best discriminates between different systems.
    
    Parameters
    ----------
    systems : list of System objects
        Systems to discriminate between
    test_features : dict, optional
        Features to test
    test_criteria : dict, optional
        Criteria to test
    n_trials : int, default=30
        Number of trials for σc computation
    method : str, default='adaptive'
        Method for σc computation
    
    Returns
    -------
    dict
        Dictionary with optimal discriminating configuration
    """
    if test_features is None:
        test_features = {
            'peaks': PeakCounter(transform='log'),
            'entropy': EntropyCalculator(bins=20),
            'spectral': SpectralAnalyzer(),
            'autocorr': AutoCorrelation(),
        }
    
    if test_criteria is None:
        test_criteria = {
            'variance': VarianceCriterion(threshold=0.1),
            'iqr': IQRCriterion(threshold=0.2),
        }
    
    best_score = 0
    best_result = None
    
    for f_name, feature in test_features.items():
        for c_name, criterion in test_criteria.items():
            sigma_c_values = []
            
            for system in systems:
                try:
                    tr = TripleRule(
                        system=system,
                        feature=feature,
                        criterion=criterion
                    )
                    
                    result = tr.compute(
                        n_trials=n_trials,
                        method=method,
                        verbose=False
                    )
                    
                    sigma_c_values.append(result.sigma_c)
                except:
                    sigma_c_values.append(np.nan)
            
            # Remove NaN values
            sigma_c_values = [x for x in sigma_c_values if not np.isnan(x)]
            
            if len(sigma_c_values) >= 2:
                # Discrimination score = coefficient of variation
                mean_sc = np.mean(sigma_c_values)
                std_sc = np.std(sigma_c_values)
                
                if mean_sc > 0:
                    score = std_sc / mean_sc
                    
                    if score > best_score:
                        best_score = score
                        best_result = {
                            'feature': f_name,
                            'criterion': c_name,
                            'score': score,
                            'sigma_c_values': sigma_c_values,
                            'feature_obj': feature,
                            'criterion_obj': criterion,
                            'mean_sigma_c': mean_sc,
                            'std_sigma_c': std_sc
                        }
    
    if best_result is None:
        raise ValueError("No valid discriminating (F,C) pair found")
    
    return best_result


def pareto_optimal_design(objectives: List[str],
                         system_or_systems: Union[Any, List],
                         test_features: Optional[Dict] = None,
                         test_criteria: Optional[Dict] = None,
                         constraints: Optional[Dict] = None,
                         n_trials: int = 30) -> List[Dict[str, Any]]:
    """
    Find Pareto-optimal (F,C) pairs for multiple objectives.
    
    Parameters
    ----------
    objectives : list of str
        Objectives to optimize: 'sensitivity', 'robustness', 'discrimination'
    system_or_systems : System object or list of Systems
        System(s) to analyze
    test_features : dict, optional
        Features to test
    test_criteria : dict, optional
        Criteria to test
    constraints : dict, optional
        Constraints on computation (e.g., {'max_time': 1.0})
    n_trials : int, default=30
        Number of trials
    
    Returns
    -------
    list of dict
        Pareto-optimal solutions
    """
    if test_features is None:
        test_features = {
            'peaks': PeakCounter(transform='log'),
            'entropy': EntropyCalculator(),
            'spectral': SpectralAnalyzer(),
        }
    
    if test_criteria is None:
        test_criteria = {
            'variance': VarianceCriterion(threshold=0.1),
            'iqr': IQRCriterion(threshold=0.2),
        }
    
    # Evaluate all combinations
    solutions = []
    
    for f_name, feature in test_features.items():
        for c_name, criterion in test_criteria.items():
            solution = {
                'feature': f_name,
                'criterion': c_name,
                'feature_obj': feature,
                'criterion_obj': criterion,
                'objectives': {}
            }
            
            # Evaluate each objective
            if 'sensitivity' in objectives:
                if isinstance(system_or_systems, list):
                    system = system_or_systems[0]
                else:
                    system = system_or_systems
                
                try:
                    tr = TripleRule(system, feature, criterion)
                    result = tr.compute(n_trials=n_trials, method='adaptive')
                    solution['objectives']['sensitivity'] = -result.sigma_c  # Negative for maximization
                except:
                    solution['objectives']['sensitivity'] = -float('inf')
            
            if 'robustness' in objectives:
                if isinstance(system_or_systems, list):
                    system = system_or_systems[0]
                else:
                    system = system_or_systems
                
                try:
                    tr = TripleRule(system, feature, criterion)
                    result = tr.compute(n_trials=n_trials, method='adaptive')
                    solution['objectives']['robustness'] = result.sigma_c
                except:
                    solution['objectives']['robustness'] = 0
            
            if 'discrimination' in objectives:
                if isinstance(system_or_systems, list):
                    systems = system_or_systems
                else:
                    raise ValueError("Discrimination requires multiple systems")
                
                sigma_c_values = []
                for sys in systems:
                    try:
                        tr = TripleRule(sys, feature, criterion)
                        result = tr.compute(n_trials=n_trials, method='adaptive')
                        sigma_c_values.append(result.sigma_c)
                    except:
                        pass
                
                if len(sigma_c_values) >= 2:
                    solution['objectives']['discrimination'] = np.std(sigma_c_values) / (np.mean(sigma_c_values) + 1e-10)
                else:
                    solution['objectives']['discrimination'] = 0
            
            solutions.append(solution)
    
    # Find Pareto front
    pareto_front = []
    
    for i, sol1 in enumerate(solutions):
        is_dominated = False
        
        for j, sol2 in enumerate(solutions):
            if i == j:
                continue
            
            # Check if sol2 dominates sol1
            dominates = True
            for obj in objectives:
                if sol1['objectives'].get(obj, -float('inf')) > sol2['objectives'].get(obj, -float('inf')):
                    dominates = False
                    break
            
            if dominates and any(sol2['objectives'].get(obj, -float('inf')) > sol1['objectives'].get(obj, -float('inf')) 
                                for obj in objectives):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_front.append(sol1)
    
    return pareto_front


def auto_design(requirements: Dict[str, Any]) -> Dict[str, Any]:
    """
    Automatically select optimal (F,C) pair based on requirements.
    
    Parameters
    ----------
    requirements : dict
        Requirements dictionary containing:
        - 'task': 'detect' | 'resist' | 'classify'
        - 'sensitivity': 'high' | 'medium' | 'low'
        - 'speed': 'realtime' | 'fast' | 'accurate'
        - 'system': System object
        - 'systems': List of systems (for classification)
    
    Returns
    -------
    dict
        Optimal configuration
    """
    task = requirements.get('task', 'detect')
    sensitivity = requirements.get('sensitivity', 'medium')
    speed = requirements.get('speed', 'fast')
    
    # Choose method based on speed requirement
    if speed == 'realtime':
        method = 'analytical'
        n_trials = 10
    elif speed == 'fast':
        method = 'adaptive'
        n_trials = 30
    else:  # accurate
        method = 'empirical'
        n_trials = 100
    
    if task == 'detect':
        # Optimize for sensitivity
        if 'system' not in requirements:
            raise ValueError("System required for detection task")
        
        return optimize_for_sensitivity(
            requirements['system'],
            n_trials=n_trials,
            method=method
        )
    
    elif task == 'resist':
        # Optimize for robustness
        if 'system' not in requirements:
            raise ValueError("System required for resistance task")
        
        return optimize_for_robustness(
            requirements['system'],
            n_trials=n_trials,
            method=method
        )
    
    elif task == 'classify':
        # Optimize for discrimination
        if 'systems' not in requirements:
            raise ValueError("Multiple systems required for classification task")
        
        return optimize_for_discrimination(
            requirements['systems'],
            n_trials=n_trials,
            method=method
        )
    
    else:
        raise ValueError(f"Unknown task: {task}")


def sensitivity_analysis(system,
                        feature,
                        criterion,
                        parameter_name: str,
                        parameter_values: List[float],
                        n_trials: int = 50) -> Dict[str, Any]:
    """
    Analyze how σc changes with a parameter.
    
    Parameters
    ----------
    system : System object
        System to analyze
    feature : Feature object
        Feature extractor
    criterion : Criterion object
        Statistical criterion
    parameter_name : str
        Name of parameter to vary
    parameter_values : list
        Values to test
    n_trials : int
        Number of trials per value
    
    Returns
    -------
    dict
        Results of sensitivity analysis
    """
    results = []
    
    for value in parameter_values:
        # Set parameter
        if hasattr(system, parameter_name):
            setattr(system, parameter_name, value)
        elif hasattr(feature, parameter_name):
            setattr(feature, parameter_name, value)
        elif hasattr(criterion, parameter_name):
            setattr(criterion, parameter_name, value)
        else:
            raise ValueError(f"Parameter {parameter_name} not found")
        
        # Compute σc
        tr = TripleRule(system, feature, criterion)
        result = tr.compute(n_trials=n_trials, method='adaptive')
        
        results.append({
            'parameter_value': value,
            'sigma_c': result.sigma_c,
            'ci_lower': result.ci_lower,
            'ci_upper': result.ci_upper
        })
    
    return {
        'parameter_name': parameter_name,
        'parameter_values': parameter_values,
        'results': results
    }
