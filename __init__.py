"""
TheQA: Critical Noise Thresholds in Discrete Dynamical Systems

A comprehensive framework for computing and analyzing critical noise thresholds
(σc) in discrete mathematical systems using the Triple Rule: σc = σc(S, F, C).
"""

__version__ = "1.0.0"
__author__ = "Matthias and Arti Cyan"
__email__ = "theqa@posteo.com"

# Core imports
from .core import (
    compute_sigma_c,
    TripleRule,
    Result,
    sigma_c,  # Alias for backward compatibility
)

# System imports
from .systems import (
    CollatzSystem,
    FibonacciSystem,
    LogisticMap,
    TentMap,
    JosephusSystem,
    PrimeGapSystem,
    CustomSystem,
)

# Feature imports
from .features import (
    PeakCounter,
    EntropyCalculator,
    SpectralAnalyzer,
    AutoCorrelation,
    ZeroCrossings,
    WaveletFeatures,
)

# Criterion imports
from .criteria import (
    VarianceCriterion,
    IQRCriterion,
    EntropyCriterion,
    MADCriterion,
    ThresholdCriterion,
)

# Algorithm imports
from .algorithms import (
    spectral_sigma_c,
    gradient_sigma_c,
    analytical_sigma_c,
    adaptive_sigma_c,
    parallel_sigma_c,
)

# Optimization imports
from .optimize import (
    optimize_for_sensitivity,
    optimize_for_robustness,
    optimize_for_discrimination,
    pareto_optimal_design,
    auto_design,
)

# Visualization imports
from .visualize import (
    plot_phase_transition,
    plot_sigma_landscape,
    plot_system_fingerprint,
    plot_optimization_tradeoff,
    plot_quantum_comparison,
)

# Quantum module (optional)
try:
    from .quantum import (
        QuantumWalk,
        quantum_sigma_c,
        classical_to_quantum_bound,
        QuantumSystem,
    )
    HAS_QUANTUM = True
except ImportError:
    HAS_QUANTUM = False

# Constants
PI_HALF = 1.5707963267948966  # π/2 - Classical bound
PI = 3.141592653589793  # π - Quantum bound

# Universality classes
ULTRA_SENSITIVE = (0, 0.01)
SENSITIVE = (0.01, 0.1)
MEDIUM = (0.1, 0.3)
ROBUST = (0.3, float('inf'))

# All public API
__all__ = [
    # Core
    'compute_sigma_c',
    'TripleRule',
    'Result',
    'sigma_c',
    
    # Systems
    'CollatzSystem',
    'FibonacciSystem',
    'LogisticMap',
    'TentMap',
    'JosephusSystem',
    'PrimeGapSystem',
    'CustomSystem',
    
    # Features
    'PeakCounter',
    'EntropyCalculator',
    'SpectralAnalyzer',
    'AutoCorrelation',
    'ZeroCrossings',
    'WaveletFeatures',
    
    # Criteria
    'VarianceCriterion',
    'IQRCriterion',
    'EntropyCriterion',
    'MADCriterion',
    'ThresholdCriterion',
    
    # Algorithms
    'spectral_sigma_c',
    'gradient_sigma_c',
    'analytical_sigma_c',
    'adaptive_sigma_c',
    'parallel_sigma_c',
    
    # Optimization
    'optimize_for_sensitivity',
    'optimize_for_robustness',
    'optimize_for_discrimination',
    'pareto_optimal_design',
    'auto_design',
    
    # Visualization
    'plot_phase_transition',
    'plot_sigma_landscape',
    'plot_system_fingerprint',
    'plot_optimization_tradeoff',
    'plot_quantum_comparison',
    
    # Constants
    'PI_HALF',
    'PI',
    'ULTRA_SENSITIVE',
    'SENSITIVE',
    'MEDIUM',
    'ROBUST',
]

# Add quantum exports if available
if HAS_QUANTUM:
    __all__.extend([
        'QuantumWalk',
        'quantum_sigma_c',
        'classical_to_quantum_bound',
        'QuantumSystem',
    ])


def get_version():
    """Return the current version of TheQA."""
    return __version__


def check_installation():
    """Check if TheQA is properly installed with all dependencies."""
    import importlib
    
    required = ['numpy', 'scipy', 'matplotlib', 'pandas']
    optional = {
        'quantum': ['qiskit'],
        'ml': ['sklearn', 'tensorflow', 'torch'],
        'advanced_vis': ['plotly', 'bokeh'],
    }
    
    print(f"TheQA version {__version__}")
    print("\nRequired dependencies:")
    
    for module in required:
        try:
            importlib.import_module(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} (MISSING)")
    
    print("\nOptional dependencies:")
    for category, modules in optional.items():
        print(f"  {category}:")
        for module in modules:
            try:
                importlib.import_module(module)
                print(f"    ✓ {module}")
            except ImportError:
                print(f"    ✗ {module}")
    
    print("\nInstallation check complete.")


# Package metadata
__metadata__ = {
    'name': 'theqa',
    'version': __version__,
    'author': __author__,
    'email': __email__,
    'description': 'Critical noise thresholds in discrete dynamical systems',
    'url': 'https://github.com/hermannhart/theqa',
    'license': 'MIT',
}
