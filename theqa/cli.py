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
