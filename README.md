# TheQA Research for Computational Problems

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![License: Elastic License 2.0](https://img.shields.io/badge/Commercial%20License-ELv2-orange)](LICENSE-COMMERCIAL.txt)

## **Overview**
This repository contains research projects utilizing **TheQA**, a quantum-inspired computational framework designed for optimization problems, quantum simulations, and complexity analysis.
TheQA leverages probability theory—laws of large numbers, central limit theorems, and concentration inequalities—to ensure stable, objective estimates of system structures. By tuning the stochastic "noise" level, TheQA amplifies weak signals via stochastic resonance, maximizing information extraction. Resonances are statistically significant patterns, distinguishable from random noise through robust metrics.

# Stochastic Phase Transitions in Discrete Dynamical Systems

This repository explores the concept of **critical noise thresholds (σ<sub>c</sub>)** in discrete deterministic sequences. It provides code, data, and theoretical background for detecting and analyzing **stochastic phase transitions** under Gaussian noise perturbations.

---

## 🔬 What is σ<sub>c</sub>?

The **critical noise threshold** σ<sub>c</sub> is the smallest standard deviation of Gaussian noise at which a deterministic system transitions from structureless behavior to measurable statistical complexity.

### Mathematical Definition

Let **S** = {s₁, s₂, ..., sₙ} be a deterministic sequence, and **T** a transformation (e.g., log, sqrt, identity). Let **F<sub>σ</sub>**(S) be a feature extractor (e.g., peak count) applied to **T(S)** + Gaussian noise of std. dev. σ.

The **critical threshold** is:


σ<sub>c</sub> = inf{ σ > 0 : Var[F<sub>σ</sub>(T(S))] > ε }


with ε ≈ 0.1 (typical).

---

### oc.py: “Why does σc vary between systems?”
#### Systematic investigation of σc variations
#### Goal: Understand why σc varies, but the method always works
Main findings:

σc correlates with log_range and std of the sequence
Prediction model: σc = k₁ - (std/√n)^α - (1/f)^β + k₂
“The variation in σc is a FEATURE, not a BUG!”

### oc2.py: “What IS σc really?”
#### Fundamental investigation: What IS σc really?
##### Goal: Understanding the microscopic nature of the phase transition
Main findings:

Microscopic analysis of the transition
σc is continuous in system parameters
Transformation scaling: σc depends on the transformation (log, sqrt, etc.)

### oc3.py: “Rigorous proof framework”
#### Systematic examination of all necessary proof components
10 proofs for:

Existence: σc exists for all discrete systems ✓
Uniqueness: There is only one σc per system ✓
Mathematical definition: σc = inf{σ > 0 : Var[F_σ(S)] > ε}
Continuity: σc changes continuously
Analytical formula: σc ≈ k-(σ/√n)^α-(1/f)^β
Universality: Works for all systems ✓
Mathematical connections: σc ≈ arg max I(S; F_σ(S))
Limit value theorems: σc ~ n^(-γ)
Physical interpretation: sin(σc) ≈ σc
Calculability: Polynomial complexity

## The most important factors influencing σc:
### 1. intrinsic properties:

log_range: Dynamic range of the sequence
std: standard deviation
Growth rate: mean(Δlog S)
Dominant frequency: f_dom

### 2nd system parameter:

q in qn+1: σc ~ log(q)/log(2)
Sequence length n: σc ~ n^(-0.55)
Entropy H: σc ~ exp(-H/T)

### 3. transformation:

Log space: σc different from linear space
Different transformations → different σc

## The theoretical model:
σc = k₁ - (σ_intrinsic/√n)^α - (1/f_dominant)^β - exp(-H/T) + k₂
Where:

σ_intrinsic: Intrinsic variation of the system
n: System size
f_dominant: Characteristic frequency
H: Entropy
T: “Temperature” (energy scale)
k₁, k₂, α, β: System-dependent constants

## The fundamental realization:
σc encodes the “sensitivity” of a discrete system to disturbances!

Small σc: System is “stiff”, needs little noise for transition
Large σc: System is “soft”, needs a lot of noise

It is like a “fingerprint” of the system:

Every system has its characteristic σc
It depends on the internal structure
It can be predicted!


### **Projects Included** - see Branches!
- **📊 Part I: Foundation - the core concept
- **📊 Part II: Discrete Phase Transitions - feature of a broad class of mathematical systems
- **📊 Part III: Theory - the Theory
- **📊 Part III: Goldbach - the Theory of σ<sub>c</sub>
  

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

## **Installation**
### **Requirements**
- **Python 3.8+**
- 🚀 numpy
- 🚀 matplotlib
- 🚀 scipy
- 🚀 pandas
- 🚀 scikit-learn
- 🚀 sympy

### **Run a script**

For example, to run the first analysis:
```bash
python 1.py
```
or, in the `sequel` branch:
```bash
python 7.py
```

### **(Optional) Requirements file**

You can also install all dependencies via `requirements.txt`:

```bash
pip install -r requirements.txt
```

### **Notes**

- All scripts are self-contained and runnable from the command line.
- For large number analyses or extensive visualizations, ensure your system has adequate RAM and CPU.
- All scripts use only standard Python and open-source scientific packages.

---

**Enjoy exploring stochastic resonance and phase transitions in discrete dynamical systems!**




