# Contributing to TheQA

We love your input! We want to make contributing to TheQA as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with Github

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## We Use [Github Flow](https://guides.github.com/introduction/flow/index.html)

Pull requests are the best way to propose changes to the codebase:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issues](https://github.com/hermannhart/theqa/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/hermannhart/theqa/issues/new); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Development Process

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/hermannhart/theqa.git
cd theqa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all extras
pip install -e ".[dev,quantum,ml]"
```

### Code Style

We use [Black](https://github.com/psf/black) for code formatting and [flake8](https://flake8.pycqa.org/) for linting:

```bash
# Format code
black theqa/

# Check linting
flake8 theqa/
```

### Testing

We use [pytest](https://pytest.org/) for testing:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=theqa tests/

# Run specific test file
pytest tests/test_core.py

# Run tests in parallel
pytest -n auto
```

### Documentation

We use [Sphinx](https://www.sphinx-doc.org/) for documentation:

```bash
cd docs
make clean
make html
# View at docs/_build/html/index.html
```

### Type Hints

We encourage the use of type hints for better code clarity:

```python
from typing import List, Optional, Tuple

def compute_features(sequence: List[float], 
                    method: Optional[str] = None) -> Tuple[float, float]:
    """Example with type hints."""
    pass
```

## Adding New Features

### 1. Adding a New System

Create a new class in `theqa/systems.py`:

```python
class MyNewSystem(BaseSystem):
    """
    Brief description of the system.
    
    Parameters
    ----------
    param1 : type
        Description
    """
    
    def __init__(self, param1: int = 10):
        self.param1 = param1
    
    def generate(self, **kwargs) -> np.ndarray:
        """Generate sequence from the system."""
        # Implementation
        pass
```

### 2. Adding a New Feature

Create a new class in `theqa/features.py`:

```python
class MyNewFeature(BaseFeature):
    """
    Brief description of the feature.
    """
    
    def extract(self, sequence: np.ndarray) -> float:
        """Extract feature value from sequence."""
        # Implementation
        pass
```

### 3. Adding a New Algorithm

Add function to `theqa/algorithms.py`:

```python
def my_new_algorithm(sequence: np.ndarray, **kwargs) -> float:
    """
    Compute σc using my new method.
    
    Time complexity: O(?)
    """
    # Implementation
    pass
```

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the documentation with any new functionality
3. Add tests for any new functionality
4. Ensure all tests pass
5. Update the version number in `theqa/__init__.py` if appropriate
6. Create a pull request with a clear title and description

## Code Review Process

The maintainers will review your pull request and either merge it, request changes, or close it with an explanation. We aim to review all pull requests within a week.

## Community Guidelines

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## Questions?

Feel free to contact the maintainers:
- Email: theqa@posteo.com
- GitHub Issues: https://github.com/hermannhart/theqa/issues

## Recognition

Contributors will be recognized in:
- The AUTHORS file
- Release notes
- Academic publications (for significant contributions)

Thank you for contributing to TheQA! 🎉
