"""
TheQA Package Setup Configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="theqa",
    version="1.0.0",
    author="Matthias C. Wurm, Arti Cyan",
    author_email="theqa@posteo.com",
    description="A Python package for computing critical noise thresholds in discrete dynamical systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hermannhart/theqa",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
        "quantum": [
            "qiskit>=0.36",
            "qiskit-aer>=0.10",
        ],
        "ml": [
            "scikit-learn>=0.24",
            "tensorflow>=2.8",
            "torch>=1.10",
        ],
    },
    entry_points={
        "console_scripts": [
            "theqa=theqa.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "theqa": ["data/*.csv", "data/*.json"],
    },
    keywords="critical noise thresholds, discrete dynamical systems, stochastic resonance, "
             "collatz conjecture, chaos theory, quantum computing, triple rule",
    project_urls={
        "Bug Reports": "https://github.com/hermannhart/theqa/issues",
        "Source": "https://github.com/hermannhart/theqa",
        "Documentation": "https://theqa.readthedocs.io",
    },
)
