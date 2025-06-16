.. TheQA documentation master file

TheQA: Critical Noise Thresholds in Discrete Dynamical Systems
===============================================================

.. image:: https://img.shields.io/pypi/v/theqa.svg
   :target: https://pypi.org/project/theqa/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/theqa.svg
   :target: https://pypi.org/project/theqa/
   :alt: Python versions

TheQA is a Python package for computing critical noise thresholds (σ\ :sub:`c`) in discrete dynamical systems. Based on the Triple Rule framework, it provides tools to analyze how mathematical sequences respond to stochastic perturbations.

Key Features
------------

* **Triple Rule Framework**: σ\ :sub:`c` = σ\ :sub:`c`\ (S, F, C) - unified theory
* **Efficient Algorithms**: 100-1000× speedup over empirical methods
* **Extensive System Library**: Collatz, Fibonacci, chaotic maps, cellular automata
* **Quantum Extensions**: Support for quantum systems with extended bounds
* **Optimization Tools**: Find optimal (F,C) pairs for your application

Quick Example
-------------

.. code-block:: python

   from theqa import compute_sigma_c, CollatzSystem

   # Analyze Collatz sequence
   system = CollatzSystem(n=27)
   sigma_c = compute_sigma_c(system.generate())
   print(f"Critical threshold: σc = {sigma_c:.3f}")

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   theory
   api/index
   examples/index
   contributing

Background
----------

The discovery of critical noise thresholds emerged from studying the Collatz conjecture. We found that at σ = 0.117, Collatz sequences undergo a sharp phase transition from deterministic to random behavior. This led to the discovery of universal phase transitions across all discrete dynamical systems.

The Triple Rule
---------------

The critical threshold depends on three components:

1. **System (S)**: The mathematical rule generating sequences
2. **Feature (F)**: What property we measure (peaks, entropy, etc.)
3. **Criterion (C)**: How we detect changes (variance, threshold, etc.)

This context-dependence is fundamental - the same system can have different thresholds depending on how we measure it.

Installation
------------

.. code-block:: bash

   pip install theqa

For development:

.. code-block:: bash

   git clone https://github.com/hermannhart/theqa.git
   cd theqa
   pip install -e ".[dev]"

License
-------

TheQA is released under the Elastic 2.0 License. See the LICENSE file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
