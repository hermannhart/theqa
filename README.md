# TheQA: Critical Noise Thresholds in Discrete Dynamical Systems

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![License: Elastic License 2.0](https://img.shields.io/badge/Commercial%20License-ELv2-orange)](LICENSE-COMMERCIAL.txt)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A Python package for computing critical noise thresholds (σc) in discrete dynamical systems, based on the Triple Rule framework.

## Installation

### From PyPI (recommended)
```bash
pip install theqa
```

### From source
```bash
git clone https://github.com/hermannhart/theqa.git
cd theqa
pip install -e .
```

## Quick Start

### Basic Usage

```python
from theqa import compute_sigma_c, TripleRule

# Simple usage with automatic detection
sequence = [1, 2, 4, 8, 16, 32, 64, 128]
sigma_c = compute_sigma_c(sequence)
print(f"Critical threshold: {sigma_c:.3f}")

# Advanced usage with Triple Rule
from theqa.systems import CollatzSystem
from theqa.features import PeakCounter
from theqa.criteria import VarianceCriterion

# Create system
system = CollatzSystem(n=27)

# Define Triple Rule
tr = TripleRule(
    system=system,
    feature=PeakCounter(transform='log'),
    criterion=VarianceCriterion(threshold=0.1)
)

# Compute critical threshold
result = tr.compute()
print(f"σc = {result.sigma_c:.3f}")
print(f"Confidence interval: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
```

### Available Systems

```python
from theqa.systems import (
    CollatzSystem,      # Collatz conjecture sequences
    FibonacciSystem,    # Fibonacci sequences
    LogisticMap,        # Logistic chaos
    TentMap,            # Tent map
    JosephusSystem,     # Josephus problem
    PrimeGapSystem      # Prime gap sequences
)

# Example: Analyze different systems
for SystemClass in [CollatzSystem, FibonacciSystem, LogisticMap]:
    system = SystemClass()
    sigma_c = compute_sigma_c(system.generate())
    print(f"{SystemClass.__name__}: σc = {sigma_c:.3f}")
```

### Feature Extraction

```python
from theqa.features import (
    PeakCounter,        # Count local maxima
    EntropyCalculator,  # Shannon entropy
    SpectralAnalyzer,   # Frequency domain features
    AutoCorrelation,    # Autocorrelation features
    ZeroCrossings      # Zero crossing rate
)

# Custom feature extraction
feature = EntropyCalculator(bins=20)
transformed_seq = feature.transform(sequence)
```

### Statistical Criteria

```python
from theqa.criteria import (
    VarianceCriterion,   # Variance-based detection
    IQRCriterion,        # Interquartile range
    EntropyCriterion,    # Entropy-based
    MADCriterion         # Median absolute deviation
)

# Custom criterion
criterion = IQRCriterion(threshold=0.2)
```

## Advanced Features

### Efficient Algorithms

```python
from theqa.algorithms import (
    spectral_sigma_c,    # O(n log n) spectral method
    gradient_sigma_c,    # Information gradient method
    analytical_sigma_c   # O(1) analytical approximation
)

# Compare methods
sequence = CollatzSystem(n=27).generate()

# Fast spectral method
sigma_spectral = spectral_sigma_c(sequence)

# Gradient-based method
sigma_gradient = gradient_sigma_c(sequence)

# Analytical approximation
sigma_analytical = analytical_sigma_c(sequence, system_type='collatz_like')

print(f"Spectral: {sigma_spectral:.3f}")
print(f"Gradient: {sigma_gradient:.3f}")
print(f"Analytical: {sigma_analytical:.3f}")
```

### Quantum Extensions

```python
from theqa.quantum import (
    QuantumWalk,
    quantum_sigma_c,
    classical_to_quantum_bound
)

# Analyze quantum walk
qw = QuantumWalk(n_qubits=8)
sigma_c_quantum = quantum_sigma_c(qw)
print(f"Quantum σc: {sigma_c_quantum:.3f}")

# Convert classical to quantum bound
classical_sc = 0.117
quantum_bound = classical_to_quantum_bound(classical_sc)
print(f"Classical: {classical_sc:.3f} → Quantum: {quantum_bound:.3f}")
```

### Optimization Framework

```python
from theqa.optimize import (
    optimize_for_sensitivity,
    optimize_for_robustness,
    optimize_for_discrimination,
    pareto_optimal_design
)

# Find most sensitive (F,C) pair
best_fc = optimize_for_sensitivity(system=CollatzSystem())
print(f"Most sensitive: Feature={best_fc.feature}, Criterion={best_fc.criterion}")
print(f"Achieves σc = {best_fc.sigma_c:.4f}")

# Multi-objective optimization
pareto_front = pareto_optimal_design(
    objectives=['sensitivity', 'robustness', 'discrimination'],
    constraints={'computational_cost': 1000}
)
```

### Visualization

```python
from theqa.visualize import (
    plot_phase_transition,
    plot_sigma_landscape,
    plot_system_fingerprint,
    plot_optimization_tradeoff
)

# Visualize phase transition
fig = plot_phase_transition(sequence, save_path='phase_transition.png')

# Create σc landscape
systems = [CollatzSystem(), FibonacciSystem(), LogisticMap()]
fig = plot_sigma_landscape(systems, save_path='landscape.png')
```

## Complete Example: Research Workflow

```python
import numpy as np
from theqa import TripleRule, compute_sigma_c
from theqa.systems import CollatzSystem
from theqa.features import PeakCounter, EntropyCalculator
from theqa.criteria import VarianceCriterion
from theqa.visualize import plot_phase_transition

# 1. Generate system
system = CollatzSystem(n=27)
sequence = system.generate(max_steps=1000)

# 2. Quick analysis
sigma_c_quick = compute_sigma_c(sequence)
print(f"Quick estimate: σc = {sigma_c_quick:.3f}")

# 3. Detailed analysis with Triple Rule
tr = TripleRule(
    system=system,
    feature=PeakCounter(transform='log'),
    criterion=VarianceCriterion(threshold=0.1)
)

result = tr.compute(
    n_trials=500,        # More trials for accuracy
    method='empirical',  # Use empirical method
    confidence=0.95      # 95% confidence interval
)

print("\nDetailed Analysis:")
print(f"σc = {result.sigma_c:.3f} [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
print(f"Convergence achieved: {result.converged}")
print(f"Computation time: {result.time_elapsed:.2f}s")

# 4. Compare different features
features = {
    'peaks': PeakCounter(),
    'entropy': EntropyCalculator(),
    'spectral': SpectralAnalyzer()
}

print("\nSystem Fingerprint:")
for name, feature in features.items():
    tr.feature = feature
    sigma_c = tr.compute().sigma_c
    print(f"  {name}: σc = {sigma_c:.3f}")

# 5. Visualize
fig = plot_phase_transition(
    sequence,
    feature=PeakCounter(),
    criterion=VarianceCriterion(),
    save_path='collatz_transition.png'
)
```

## API Reference

### Core Functions

- `compute_sigma_c(sequence, **kwargs)`: Compute critical threshold with automatic method selection
- `TripleRule(system, feature, criterion)`: Main framework for σc computation

### Systems
- `CollatzSystem(n, q=3)`: Generalized Collatz sequences
- `FibonacciSystem(n)`: Fibonacci-like sequences
- `LogisticMap(r, x0, length)`: Discrete logistic map
- `TentMap(r, x0, length)`: Tent map sequences

### Features
- `PeakCounter(transform=None)`: Count local maxima
- `EntropyCalculator(bins=20)`: Shannon entropy
- `SpectralAnalyzer(n_peaks=1)`: Frequency domain analysis
- `AutoCorrelation(max_lag=50)`: Autocorrelation features

### Criteria
- `VarianceCriterion(threshold=0.1)`: Variance-based detection
- `IQRCriterion(threshold=0.2)`: Interquartile range
- `EntropyCriterion(threshold=0.1)`: Entropy change detection

### Algorithms
- `spectral_sigma_c(sequence)`: FFT-based O(n log n) method
- `gradient_sigma_c(sequence)`: Information gradient method
- `analytical_sigma_c(sequence, system_type)`: O(1) approximation

## Performance Tips

1. **For real-time applications**: Use `analytical_sigma_c()` with system type hints
2. **For unknown systems**: Start with `spectral_sigma_c()` for good accuracy/speed tradeoff
3. **For research**: Use `TripleRule` with `method='empirical'` for highest accuracy
4. **For large datasets**: Use parallel processing with `parallel_sigma_c()`

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Support

- Documentation: [https://theqa.readthedocs.io](https://theqa.readthedocs.io)
- Issues: [GitHub Issues](https://github.com/hermannhart/theqa/issues)
- Discussions: [GitHub Discussions](https://github.com/hermannhart/theqa/discussions)

## Acknowledgments

Special thanks to the NumPy, SciPy, and matplotlib communities for their excellent tools.


## **Features**

🧠 TheQA builds on established methods like Monte Carlo, Metropolis algorithms, and random projections, but its innovation lies in:

🚀 Tailored sample metric selection and aggregation.

📊 Creative application to novel mathematical domains (e.g., Collatz, dimensional bridges).

🔬 Empirical validation through bootstrapping and cross-platform reproducibility.
 

### **License**
- This project follows a dual-license model:

- For Personal & Research Use: CC BY-NC 4.0 → Free for non-commercial use only.
- For Commercial Use: Companies must obtain a commercial license (Elastic License 2.0).

📜 For details, see the LICENSE file.


### ***Contributors***

- Matthias - Human resources
- Arti Cyan - Artificial  resources


### ***Contact & Support***

- For inquiries regarding commercial licensing or support, please contact:📧 theqa@posteo.com 🌐 www.theqa.space 🚀🚀🚀

- 🚀 Get started with TheQA and explore new frontiers in optimization! 🚀

---

**Enjoy exploring stochastic resonance and phase transitions in discrete dynamical systems!**



