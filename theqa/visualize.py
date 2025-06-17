"""
Visualization functions for TheQA package.

This module provides functions for creating various plots and visualizations
related to critical noise thresholds and system analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from typing import List, Optional, Dict, Any, Union, Tuple
import warnings

from .core import TripleRule, compute_sigma_c
from .features import PeakCounter
from .criteria import VarianceCriterion


def plot_phase_transition(sequence: Union[np.ndarray, Any],
                         feature: Optional[Any] = None,
                         criterion: Optional[Any] = None,
                         sigma_range: Optional[Tuple[float, float]] = None,
                         n_sigma: int = 50,
                         n_trials: int = 100,
                         save_path: Optional[str] = None,
                         figsize: Tuple[float, float] = (10, 6),
                         show_examples: bool = True) -> plt.Figure:
    """
    Plot phase transition curve showing variance vs noise level.
    
    Parameters
    ----------
    sequence : array-like or System object
        Sequence or system to analyze
    feature : Feature object, optional
        Feature extractor (default: PeakCounter)
    criterion : Criterion object, optional
        Statistical criterion (default: VarianceCriterion)
    sigma_range : tuple, optional
        Range of sigma values (default: auto)
    n_sigma : int, default=50
        Number of sigma values to test
    n_trials : int, default=100
        Number of trials per sigma
    save_path : str, optional
        Path to save figure
    figsize : tuple, default=(10, 6)
        Figure size
    show_examples : bool, default=True
        Show example sequences at different noise levels
    
    Returns
    -------
    plt.Figure
        The created figure
    """
    # Set defaults
    if feature is None:
        feature = PeakCounter(transform='log')
    if criterion is None:
        criterion = VarianceCriterion(threshold=0.1)
    
    # Get sequence
    if hasattr(sequence, 'generate'):
        seq = sequence.generate()
    else:
        seq = np.asarray(sequence)
    
    # Transform sequence
    if hasattr(feature, 'transform'):
        seq_transformed = feature.transform(seq)
    else:
        seq_transformed = np.log(np.abs(seq) + 1)
    
    # Determine sigma range
    if sigma_range is None:
        sigma_range = (1e-4, 1.0)
    
    # Test different noise levels
    sigmas = np.logspace(np.log10(sigma_range[0]), np.log10(sigma_range[1]), n_sigma)
    variances = []
    mean_features = []
    
    for sigma in sigmas:
        features = []
        for _ in range(n_trials):
            noise = np.random.normal(0, sigma, len(seq_transformed))
            noisy = seq_transformed + noise
            
            if hasattr(feature, 'extract'):
                f_val = feature.extract(noisy)
            else:
                # Default: count peaks
                peaks = len(np.where(np.diff(np.sign(np.diff(noisy))) == -2)[0])
                f_val = peaks
            
            features.append(f_val)
        
        variances.append(np.var(features))
        mean_features.append(np.mean(features))
    
    # Find critical threshold
    sigma_c = None
    for i, (sigma, var) in enumerate(zip(sigmas, variances)):
        if criterion.is_exceeded(var):
            sigma_c = sigma
            break
    
    # Create figure
    if show_examples:
        fig = plt.figure(figsize=(figsize[0], figsize[1] * 1.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
        ax1 = fig.add_subplot(gs[0])
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
    
    # Main plot
    ax1.semilogx(sigmas, variances, 'b-', linewidth=2, label='Variance')
    
    if sigma_c is not None:
        ax1.axvline(x=sigma_c, color='r', linestyle='--', linewidth=2,
                   label=f'σc = {sigma_c:.3f}')
        ax1.axhline(y=criterion.threshold, color='g', linestyle=':', 
                   label=f'Threshold = {criterion.threshold}')
    
    ax1.set_xlabel('Noise Level (σ)')
    ax1.set_ylabel('Feature Variance')
    ax1.set_title('Phase Transition Analysis')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add second y-axis for mean
    ax2 = ax1.twinx()
    ax2.semilogx(sigmas, mean_features, 'orange', linestyle='--', alpha=0.5,
                label='Mean feature')
    ax2.set_ylabel('Mean Feature Value', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    # Show examples
    if show_examples:
        ax3 = fig.add_subplot(gs[1])
        
        # Select example noise levels
        if sigma_c is not None:
            example_sigmas = [sigma_c/10, sigma_c/2, sigma_c, sigma_c*2]
        else:
            example_sigmas = [sigmas[0], sigmas[len(sigmas)//3], 
                            sigmas[2*len(sigmas)//3], sigmas[-1]]
        
        colors = ['blue', 'green', 'orange', 'red']
        
        for sigma, color in zip(example_sigmas, colors):
            noise = np.random.normal(0, sigma, len(seq_transformed))
            noisy = seq_transformed + noise
            
            ax3.plot(noisy[:min(100, len(noisy))], color=color, alpha=0.7,
                    label=f'σ = {sigma:.3f}')
        
        ax3.set_xlabel('Position')
        ax3.set_ylabel('Value')
        ax3.set_title('Example Sequences with Different Noise Levels')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_sigma_landscape(systems: Union[List, Dict],
                        feature: Optional[Any] = None,
                        criterion: Optional[Any] = None,
                        save_path: Optional[str] = None,
                        figsize: Tuple[float, float] = (12, 8),
                        show_classes: bool = True) -> plt.Figure:
    """
    Plot landscape of σc values across different systems.
    
    Parameters
    ----------
    systems : list or dict
        Systems to analyze
    feature : Feature object, optional
        Feature extractor
    criterion : Criterion object, optional
        Statistical criterion
    save_path : str, optional
        Path to save figure
    figsize : tuple, default=(12, 8)
        Figure size
    show_classes : bool, default=True
        Show universality class boundaries
    
    Returns
    -------
    plt.Figure
        The created figure
    """
    # Convert to dict if list
    if isinstance(systems, list):
        systems = {f"System {i}": s for i, s in enumerate(systems)}
    
    # Compute σc for each system
    results = []
    
    for name, system in systems.items():
        try:
            sigma_c = compute_sigma_c(system, feature=feature, 
                                    criterion=criterion, method='adaptive')
            
            # Classify
            if sigma_c < 0.01:
                class_name = "Ultra-sensitive"
                color = '#2E86AB'
            elif sigma_c < 0.1:
                class_name = "Sensitive"
                color = '#A23B72'
            elif sigma_c < 0.3:
                class_name = "Medium"
                color = '#F18F01'
            else:
                class_name = "Robust"
                color = '#C73E1D'
            
            results.append({
                'name': name,
                'sigma_c': sigma_c,
                'class': class_name,
                'color': color
            })
        except Exception as e:
            warnings.warn(f"Failed to compute σc for {name}: {str(e)}")
    
    # Sort by σc
    results.sort(key=lambda x: x['sigma_c'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    names = [r['name'] for r in results]
    values = [r['sigma_c'] for r in results]
    colors = [r['color'] for r in results]
    
    bars = ax1.bar(range(len(results)), values, color=colors, alpha=0.7)
    
    # Add class boundaries
    if show_classes:
        ax1.axhline(y=0.01, color='black', linestyle='--', alpha=0.5)
        ax1.axhline(y=0.1, color='black', linestyle='--', alpha=0.5)
        ax1.axhline(y=0.3, color='black', linestyle='--', alpha=0.5)
        ax1.axhline(y=np.pi/2, color='red', linestyle='--', alpha=0.5,
                   label='π/2 bound')
    
    ax1.set_xlabel('System')
    ax1.set_ylabel('Critical Threshold (σc)')
    ax1.set_title('σc Landscape Across Systems')
    ax1.set_xticks(range(len(results)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    
    # Scatter plot in log space
    log_values = [np.log10(r['sigma_c']) for r in results]
    
    ax2.scatter(range(len(results)), log_values, 
               c=colors, s=100, alpha=0.7, edgecolors='black')
    
    # Add class regions
    if show_classes:
        ax2.axhspan(np.log10(1e-10), np.log10(0.01), alpha=0.2, color='#2E86AB')
        ax2.axhspan(np.log10(0.01), np.log10(0.1), alpha=0.2, color='#A23B72')
        ax2.axhspan(np.log10(0.1), np.log10(0.3), alpha=0.2, color='#F18F01')
        ax2.axhspan(np.log10(0.3), np.log10(np.pi/2), alpha=0.2, color='#C73E1D')
    
    ax2.set_xlabel('System Index')
    ax2.set_ylabel('log₁₀(σc)')
    ax2.set_title('Logarithmic View of σc Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Add class labels
    ax2.text(len(results) + 0.5, np.log10(0.005), 'Ultra-sensitive', 
            rotation=90, va='center')
    ax2.text(len(results) + 0.5, np.log10(0.05), 'Sensitive', 
            rotation=90, va='center')
    ax2.text(len(results) + 0.5, np.log10(0.2), 'Medium', 
            rotation=90, va='center')
    ax2.text(len(results) + 0.5, np.log10(0.5), 'Robust', 
            rotation=90, va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_system_fingerprint(system,
                           features: Optional[Dict] = None,
                           criteria: Optional[Dict] = None,
                           save_path: Optional[str] = None,
                           figsize: Tuple[float, float] = (10, 8),
                           cmap: str = 'viridis') -> plt.Figure:
    """
    Plot system fingerprint showing σc for different (F,C) combinations.
    
    Parameters
    ----------
    system : System object
        System to analyze
    features : dict, optional
        Features to test
    criteria : dict, optional
        Criteria to test
    save_path : str, optional
        Path to save figure
    figsize : tuple, default=(10, 8)
        Figure size
    cmap : str, default='viridis'
        Colormap
    
    Returns
    -------
    plt.Figure
        The created figure
    """
    # Default features and criteria
    if features is None:
        from .features import EntropyCalculator, SpectralAnalyzer, AutoCorrelation
        features = {
            'Peaks': PeakCounter(transform='log'),
            'Entropy': EntropyCalculator(),
            'Spectral': SpectralAnalyzer(),
            'Autocorr': AutoCorrelation(),
        }
    
    if criteria is None:
        from .criteria import IQRCriterion, EntropyCriterion
        criteria = {
            'Variance': VarianceCriterion(threshold=0.1),
            'IQR': IQRCriterion(threshold=0.2),
            'Entropy': EntropyCriterion(threshold=0.1),
        }
    
    # Compute σc for all combinations
    results = np.zeros((len(features), len(criteria)))
    
    for i, (f_name, feature) in enumerate(features.items()):
        for j, (c_name, criterion) in enumerate(criteria.items()):
            try:
                tr = TripleRule(system, feature, criterion)
                result = tr.compute(n_trials=50, method='adaptive')
                results[i, j] = result.sigma_c
            except:
                results[i, j] = np.nan
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Heatmap
    im = ax.imshow(results, cmap=cmap, aspect='auto')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Critical Threshold (σc)', rotation=270, labelpad=20)
    
    # Labels
    ax.set_xticks(range(len(criteria)))
    ax.set_xticklabels(list(criteria.keys()))
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(list(features.keys()))
    
    ax.set_xlabel('Criterion')
    ax.set_ylabel('Feature')
    ax.set_title('System Fingerprint: σc Values')
    
    # Add text annotations
    for i in range(len(features)):
        for j in range(len(criteria)):
            if not np.isnan(results[i, j]):
                text = ax.text(j, i, f'{results[i, j]:.3f}',
                             ha="center", va="center", color="white" 
                             if results[i, j] < np.nanmean(results) else "black")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_optimization_tradeoff(pareto_solutions: List[Dict],
                             objectives: List[str],
                             save_path: Optional[str] = None,
                             figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
    """
    Plot Pareto frontier for multi-objective optimization.
    
    Parameters
    ----------
    pareto_solutions : list of dict
        Pareto-optimal solutions from pareto_optimal_design
    objectives : list of str
        Objectives to plot (must be 2 for 2D plot)
    save_path : str, optional
        Path to save figure
    figsize : tuple, default=(10, 6)
        Figure size
    
    Returns
    -------
    plt.Figure
        The created figure
    """
    if len(objectives) != 2:
        raise ValueError("Exactly 2 objectives required for 2D plot")
    
    # Extract objective values
    obj1_name, obj2_name = objectives
    obj1_values = [sol['objectives'].get(obj1_name, 0) for sol in pareto_solutions]
    obj2_values = [sol['objectives'].get(obj2_name, 0) for sol in pareto_solutions]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot Pareto front
    ax.scatter(obj1_values, obj2_values, s=100, c='red', 
              edgecolors='black', alpha=0.7, label='Pareto optimal')
    
    # Connect points
    sorted_indices = np.argsort(obj1_values)
    sorted_obj1 = [obj1_values[i] for i in sorted_indices]
    sorted_obj2 = [obj2_values[i] for i in sorted_indices]
    ax.plot(sorted_obj1, sorted_obj2, 'r--', alpha=0.5)
    
    # Labels
    for i, sol in enumerate(pareto_solutions):
        label = f"{sol['feature']}\n{sol['criterion']}"
        ax.annotate(label, (obj1_values[i], obj2_values[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)
    
    ax.set_xlabel(obj1_name.title())
    ax.set_ylabel(obj2_name.title())
    ax.set_title('Pareto Frontier: Trade-off Between Objectives')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_quantum_comparison(classical_results: Dict[str, float],
                           quantum_results: Optional[Dict[str, float]] = None,
                           save_path: Optional[str] = None,
                           figsize: Tuple[float, float] = (12, 6)) -> plt.Figure:
    """
    Compare classical and quantum critical thresholds.
    
    Parameters
    ----------
    classical_results : dict
        Classical σc values {system_name: σc}
    quantum_results : dict, optional
        Quantum σc values (if None, computed from classical)
    save_path : str, optional
        Path to save figure
    figsize : tuple, default=(12, 6)
        Figure size
    
    Returns
    -------
    plt.Figure
        The created figure
    """
    # Compute quantum results if not provided
    if quantum_results is None:
        try:
            from .quantum import classical_to_quantum_bound
            quantum_results = {
                name: classical_to_quantum_bound(sigma_c)
                for name, sigma_c in classical_results.items()
            }
        except ImportError:
            warnings.warn("Quantum module not available")
            quantum_results = {
                name: 2.0 * sigma_c + 0.1  # Simple approximation
                for name, sigma_c in classical_results.items()
            }
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot comparison
    names = list(classical_results.keys())
    x = np.arange(len(names))
    width = 0.35
    
    classical_vals = list(classical_results.values())
    quantum_vals = [quantum_results.get(name, 0) for name in names]
    
    bars1 = ax1.bar(x - width/2, classical_vals, width, 
                    label='Classical', color='blue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, quantum_vals, width,
                    label='Quantum', color='red', alpha=0.7)
    
    # Add bounds
    ax1.axhline(y=np.pi/2, color='blue', linestyle='--', alpha=0.5,
               label='Classical bound (π/2)')
    ax1.axhline(y=np.pi, color='red', linestyle='--', alpha=0.5,
               label='Quantum bound (π)')
    
    ax1.set_xlabel('System')
    ax1.set_ylabel('Critical Threshold')
    ax1.set_title('Classical vs Quantum Critical Thresholds')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Enhancement plot
    enhancements = [q/c if c > 0 else 0 
                   for c, q in zip(classical_vals, quantum_vals)]
    
    bars3 = ax2.bar(names, enhancements, color='green', alpha=0.7)
    ax2.axhline(y=2.0, color='black', linestyle='--', alpha=0.5,
               label='2x enhancement')
    
    ax2.set_xlabel('System')
    ax2.set_ylabel('Quantum Enhancement Factor')
    ax2.set_title('Quantum Advantage in Critical Thresholds')
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    # Add value labels
    for bar, val in zip(bars3, enhancements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.2f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_convergence_analysis(system,
                            feature=None,
                            criterion=None,
                            sequence_lengths: List[int] = None,
                            n_trials: int = 50,
                            save_path: Optional[str] = None,
                            figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
    """
    Plot how σc converges with sequence length.
    
    Parameters
    ----------
    system : System object
        System to analyze
    feature : Feature object, optional
        Feature extractor
    criterion : Criterion object, optional
        Statistical criterion
    sequence_lengths : list, optional
        Sequence lengths to test
    n_trials : int, default=50
        Number of trials
    save_path : str, optional
        Path to save figure
    figsize : tuple, default=(10, 6)
        Figure size
    
    Returns
    -------
    plt.Figure
        The created figure
    """
    if feature is None:
        feature = PeakCounter(transform='log')
    if criterion is None:
        criterion = VarianceCriterion(threshold=0.1)
    
    if sequence_lengths is None:
        sequence_lengths = [50, 100, 200, 500, 1000, 2000]
    
    results = []
    
    for length in sequence_lengths:
        # Generate sequence of specific length
        if hasattr(system, 'generate'):
            seq = system.generate(max_steps=length)[:length]
        else:
            seq = system[:length]
        
        if len(seq) < 10:
            continue
        
        # Compute σc
        tr = TripleRule(seq, feature, criterion)
        result = tr.compute(n_trials=n_trials, method='adaptive')
        
        results.append({
            'length': length,
            'sigma_c': result.sigma_c,
            'ci_lower': result.ci_lower,
            'ci_upper': result.ci_upper
        })
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    lengths = [r['length'] for r in results]
    sigma_c_values = [r['sigma_c'] for r in results]
    ci_lower = [r['ci_lower'] for r in results]
    ci_upper = [r['ci_upper'] for r in results]
    
    # Plot with confidence intervals
    ax.errorbar(lengths, sigma_c_values, 
               yerr=[np.array(sigma_c_values) - np.array(ci_lower),
                     np.array(ci_upper) - np.array(sigma_c_values)],
               fmt='bo-', capsize=5, capthick=2, linewidth=2)
    
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Critical Threshold (σc)')
    ax.set_title('Convergence of σc with Sequence Length')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add convergence line if stable
    if len(sigma_c_values) > 3:
        final_value = np.mean(sigma_c_values[-2:])
        ax.axhline(y=final_value, color='red', linestyle='--', alpha=0.5,
                  label=f'Converged value ≈ {final_value:.3f}')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
