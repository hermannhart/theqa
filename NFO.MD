# TheQA Package - Complete Structure and Installation Guide

## Package Structure

```
theqa/
├── README.md                    # Main documentation
├── setup.py                     # Installation configuration
├── requirements.txt             # Dependencies
├── LICENSE                      # MIT License
├── CONTRIBUTING.md             # Contribution guidelines
├── MANIFEST.in                 # Package manifest
├── .gitignore                  # Git ignore file
│
├── theqa/                      # Main package directory
│   ├── __init__.py            # Package initialization
│   ├── core.py                # Core functionality (TripleRule, compute_sigma_c)
│   ├── systems.py             # Dynamical systems
│   ├── features.py            # Feature extraction methods
│   ├── criteria.py            # Statistical criteria
│   ├── algorithms.py          # Efficient algorithms
│   ├── optimize.py            # Optimization framework
│   ├── visualize.py           # Visualization functions
│   ├── quantum.py             # Quantum extensions (optional)
│   ├── utils.py               # Utility functions
│   └── cli.py                 # Command-line interface
│
├── examples/                   # Example scripts
│   ├── example_usage.py       # Complete usage example
│   ├── collatz_analysis.py    # Collatz-specific analysis
│   ├── comparison.py          # System comparison
│   └── notebooks/             # Jupyter notebooks
│       ├── tutorial.ipynb     # Interactive tutorial
│       └── advanced.ipynb     # Advanced features
│
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_core.py
│   ├── test_systems.py
│   ├── test_features.py
│   ├── test_criteria.py
│   ├── test_algorithms.py
│   └── test_integration.py
│
├── docs/                       # Documentation
│   ├── conf.py                # Sphinx configuration
│   ├── index.rst              # Documentation index
│   ├── installation.rst       # Installation guide
│   ├── quickstart.rst         # Quick start guide
│   ├── api/                   # API documentation
│   └── examples/              # Example gallery
│
└── data/                       # Sample data
    ├── systems.csv            # Pre-computed system data
    └── benchmarks.json        # Performance benchmarks
```

## Installation Instructions

### 1. Create Package Directory

```bash
mkdir theqa
cd theqa
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Create Package Files

Create all the files from the artifacts above in their respective locations:

```bash
# Create directory structure
mkdir -p theqa examples tests docs data

# Create main package files
touch theqa/__init__.py
touch theqa/core.py
touch theqa/systems.py
touch theqa/features.py
touch theqa/criteria.py
touch theqa/algorithms.py

# Copy content from artifacts into these files
```

### 4. Create Additional Required Files

#### LICENSE (Elastic 2.0 License)
```
# License Notice

This software is licensed under a dual-license model:

1. **For Non-Commercial and Personal Use**  
   - This software is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.  
   - Home users and researchers may use, modify, and share this software **for non-commercial purposes only**.  
   - See `LICENSE-CCBYNC.txt` for full details.

2. **For Commercial Use**  
   - Companies, organizations, and any commercial entities must acquire a **commercial license**.  
   - This commercial license follows the **Elastic License 2.0 (ELv2)** model.  
   - See `LICENSE-COMMERCIAL.txt` for details on permitted commercial usage and restrictions.

By using this software, you agree to these terms. If you are a company or organization, please contact **[www.theqa.space]** for licensing inquiries.

---

## Summary of Key Terms

| License Type         | Permitted Uses | Restrictions |
|----------------------|---------------|-------------|
| **CC BY-NC 4.0** (for non-commercial users) | Private research, personal projects, open-source contributions | No commercial use |
| **Elastic License 2.0 (ELv2)** (for companies) | Internal business use, integration into private services | No resale, no redistribution as a SaaS, requires a paid license |

For more details, please refer to the full license files:
- `LICENSE-CCBYNC.txt` (for non-commercial users)
- `LICENSE-COMMERCIAL.txt` (for companies)

Copyright (c) 2024 Matthias and Arti Cyan


#### MANIFEST.in
```
include README.md
include LICENSE
include requirements.txt
recursive-include theqa/data *.csv *.json
recursive-include docs *.rst *.py
recursive-include examples *.py *.ipynb
```

#### .gitignore
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
ENV/
env/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/

# Documentation
docs/_build/

# OS
.DS_Store
Thumbs.db

# Project specific
*.png
*.jpg
*.pdf
results/
output/
```

### 5. Create Missing Module Files

#### theqa/optimize.py
```python
"""Optimization framework for finding optimal (F,C) pairs."""

import numpy as np
from typing import List, Tuple, Dict, Any
from .core import TripleRule
from .features import *
from .criteria import *


def optimize_for_sensitivity(system, test_features=None, test_criteria=None):
    """Find (F,C) pair that minimizes σc."""
    # Implementation from oc4.py
    pass


def optimize_for_robustness(system, test_features=None, test_criteria=None):
    """Find (F,C) pair that maximizes σc."""
    # Implementation from oc4.py
    pass


def optimize_for_discrimination(systems, test_features=None, test_criteria=None):
    """Find (F,C) pair that best discriminates between systems."""
    # Implementation from oc4.py
    pass


def pareto_optimal_design(objectives, constraints):
    """Multi-objective optimization."""
    # Implementation from oc4.py
    pass


def auto_design(requirements):
    """Automatically select optimal (F,C) pair."""
    # Implementation from oc4.py
    pass
```

#### theqa/visualize.py
```python
"""Visualization functions for TheQA."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_phase_transition(sequence, feature=None, criterion=None, save_path=None):
    """Plot phase transition curve."""
    # Implementation
    pass


def plot_sigma_landscape(systems, save_path=None):
    """Plot σc landscape across systems."""
    # Implementation
    pass


def plot_system_fingerprint(system, features=None, criteria=None, save_path=None):
    """Plot system fingerprint."""
    # Implementation
    pass


def plot_optimization_tradeoff(results, save_path=None):
    """Plot Pareto frontier."""
    # Implementation
    pass


def plot_quantum_comparison(classical_results, quantum_results, save_path=None):
    """Compare classical and quantum bounds."""
    # Implementation
    pass
```

#### theqa/utils.py
```python
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
```

#### theqa/cli.py
```python
"""Command-line interface for TheQA."""

import argparse
import sys
from .core import compute_sigma_c
from .systems import *


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute critical noise thresholds for discrete dynamical systems"
    )
    
    parser.add_argument(
        "system",
        help="System type or file path containing sequence"
    )
    
    parser.add_argument(
        "--method",
        choices=["auto", "empirical", "spectral", "gradient", "analytical"],
        default="auto",
        help="Computation method"
    )
    
    parser.add_argument(
        "--output",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Process arguments and compute σc
    # Implementation details...
    
    print(f"σc = {sigma_c:.3f}")


if __name__ == "__main__":
    main()
```

### 6. Install Package

#### For Development
```bash
pip install -e .
```

#### For Production
```bash
pip install .
```

### 7. Verify Installation

```python
# Test basic import
import theqa
print(theqa.__version__)

# Test functionality
from theqa import compute_sigma_c
sequence = [1, 2, 4, 8, 16, 32]
sigma_c = compute_sigma_c(sequence)
print(f"σc = {sigma_c:.3f}")

# Check installation
theqa.check_installation()
```

### 8. Run Examples

```bash
cd examples
python example_usage.py
```

## Publishing to PyPI

When ready to publish:

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ theqa

# Upload to PyPI
twine upload dist/*
```

## Development Workflow

1. **Clone repository**
   ```bash
   git clone https://github.com/hermannhart/theqa.git
   cd theqa
   ```

2. **Create branch**
   ```bash
   git checkout -b feature/new-feature
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests**
   ```bash
   pytest tests/
   ```

5. **Build documentation**
   ```bash
   cd docs
   make html
   ```

## Support

- **Issues**: https://github.com/hermannhart/theqa/issues
- **Discussions**: https://github.com/hermannhart/theqa/discussions
- **Email**: theqa@posteo.com
