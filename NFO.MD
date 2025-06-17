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


### 5. Install Package

#### For Development
```bash
pip install -e .
```

#### For Production
```bash
pip install .
```

### 6. Verify Installation

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

### 7. Run Examples

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
