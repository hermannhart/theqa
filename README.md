# TheQA Research for Computational Problems

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![License: Elastic License 2.0](https://img.shields.io/badge/Commercial%20License-ELv2-orange)](LICENSE-COMMERCIAL.txt)

# special issue – Universal Information Phase Transitions

This repository contains research code and data for identifying and quantifying phase transitions in discrete and quantum systems through noise-induced changes in information metrics.

## Overview

TheQA is a quantum-inspired framework for analyzing the interplay between deterministic structure and stochastic noise in sequences, networks, and quantum systems. By tuning the noise level, we can empirically and theoretically identify the "information extraction transition": the point where intrinsic structure is overwhelmed by external randomness.

The project combines empirical benchmarking, scaling analysis, and connection to statistical physics (including DQCP – Decoherence-Driven Quantum Criticality).

---

## Script Index

### 1. `dancer2.py` – Expanded Dataset & Ensemble Stabilization

**Purpose:**  
Systematically benchmark σc (critical noise threshold) across a wide variety of sequences and categories using ensemble averaging.

**Inputs:**  
- 110 sequences from 20 distinct categories (deterministic, random, quantum, mathematical, etc.)
- User-selectable statistical metrics (e.g., sensitivity, entropy)
- Ensemble size parameter

**Outputs:**  
- CSV/JSON summary of σc values, method CV, scaling CV
- Plots showing improvements in robustness and reproducibility

**Scientific Context:**  
Addresses reproducibility and statistical stability. Ensemble averaging reduces variance in σc measurements and highlights persistent scaling issues across categories.

---

### 2. `dancer3_skalierungsdiagnostik.py` – Scaling Diagnostics

**Purpose:**  
Systematic scaling analysis: how does σc scale with sequence length and type?

**Inputs:**  
- 31 scaling factors, 11 sequence types (random, deterministic, quasi-periodic, etc.)
- Noise parameter grid
- Feature extraction parameters (e.g., peak counting, entropy threshold)

**Outputs:**  
- Power-law fits for σc vs. scale
- R² values, scaling exponents per system
- Visualizations: log-log plots, scaling fingerprints

**Scientific Context:**  
Tests universality and the Central Limit Theorem (CLT) prediction: random sequences scale as √n, deterministic sequences show anomalously weak scaling. Reveals the absence of simple universal scaling laws.

---

### 3. `edge_of_chaos.py` – Empirical Metrics at the Transition

**Purpose:**  
Quantitative analysis of information loss and sensitivity at the phase boundary.

**Inputs:**  
- Sequence data (from dancer2 or other sources)
- σc values (from previous analyses)
- Parameters for information loss, spectral entropy, sensitivity calculation

**Outputs:**  
- Correlation coefficients (e.g., between σc and Fisher Information)
- Diagnostic plots: information loss, sensitivity, entropy across noise levels
- Tabulated results for deterministic vs. chaotic systems

**Scientific Context:**  
Demonstrates that σc marks a sharp transition: maximum sensitivity, significant information loss (~0.5), and diverging susceptibility. Strong empirical connection to Fisher Information.

---

### 4. `quantum_classic.py` – Universality Testing

**Purpose:**  
Test whether σc is a universal marker or just category-specific.

**Inputs:**  
- Sequence sets grouped by mathematical and quantum properties
- ANOVA and chi-square test parameters
- Scaling and distribution metrics

**Outputs:**  
- p-values for ANOVA and chi-square tests
- Category comparison plots
- Universality class labels

**Scientific Context:**  
Statistical tests show no significant differences across categories: σc behaves universally for mathematical and quantum systems, supporting the concept of a general information extraction transition.

---

### 5. `conn.py` – Statistical Physics Connection

**Purpose:**  
Connect empirical sensitivity metrics to Fisher Information and known universality classes.

**Inputs:**  
- Sequence or system data
- Calculated sensitivity and Fisher Information
- Reference universal class datasets (for matching)

**Outputs:**  
- Correlation plots and statistics: Sensitivity² ∝ Fisher Information
- Universal class matching summary (percentage)
- Phase diagrams showing divergence of susceptibility

**Scientific Context:**  
Establishes theoretical grounding: observed phase transitions in σc align with genuine statistical physics transitions (DQCP, Fisher Information divergence). Quantifies the match to established universality classes.

---

## Requirements

- Python 3.8+
- numpy, matplotlib, scipy, pandas, scikit-learn, sympy

All scripts are self-contained and can be run from the command line.  
For full dependencies:  
```bash
pip install -r requirements.txt
```

## Usage Example

```bash
python dancer2.py
python dancer3_skalierungsdiagnostik.py
python edge_of_chaos.py
python quantum_classic.py
python conn.py
```

## License

- Non-commercial: CC BY-NC 4.0
- Commercial: Elastic License 2.0 (see LICENSE for details)

## Contact

For scientific discussion or commercial licensing just contact Matthias:  
📧 theqa@posteo.com  
🌐 https://www.theqa.space

---

