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

The **critical noise threshold** σ<sub>c</sub> is the minimal standard deviation of Gaussian noise at which a deterministic system, for a given transformation and feature extraction, transitions from deterministic to measurable statistical complexity according to a chosen statistical criterion.

### Mathematical Definition

Let **S** = {s₁, s₂, ..., sₙ} be a deterministic sequence, **T** a transformation (e.g., log, sqrt, identity), and **F<sub>σ</sub>**(S) a feature extractor (e.g., peak count) applied to **T(S)** plus Gaussian noise of std. dev. σ.

The **critical threshold** (for statistical criterion **C** and threshold ε) is:

```math
σ_c(S, T, F, C, ε) = inf{ σ > 0 : C[F_σ(T(S))] > ε }
```

Typical choice: C = variance, ε ≈ 0.1.

---

# The Evolution of the Critical Noise Threshold: From Single Values to the Triple Rule in Discrete Entropy Analysis

## Abstract

We systematically analyze the emergence and meaning of the critical noise threshold (σ₍c₎, OC) in discrete dynamical systems, with a focus on entropy-based methods. Tracing the development from our initial approaches to the most recent, we show how our understanding evolved from seeking a unique critical value to formulating the “Triple Rule,” which recognizes the context-dependence of σ₍c₎/OC on system, feature extraction, and statistical criterion. We argue that this perspective is both scientifically robust and practically fruitful, and we provide a framework for future entropy-based research in discrete systems.

---

## 1. Introduction

The concept of a **critical noise threshold** (σ₍c₎ or OC) has become central in the study of stochastic resonance and phase transitions in discrete mathematical systems. Traditionally, researchers aimed to assign a unique value to σ₍c₎ for a given system, analogous to physical constants like the melting point of a material. However, our research has revealed that this view is incomplete. Here, we document our journey from early single-value approaches to the comprehensive “Triple Rule” perspective, with entropy as a guiding example.

---

## 2. Early Approaches: Paper 1 (Foundation)

### 2.1. Motivation & Methodology

In our first analyses (see `foundation/2.py`, `foundation/4.py`), the goal was to **identify a unique σ₍c₎ for systems such as the Collatz sequence**. We used entropy and related information measures:
- **Transforming sequences** (typically via `log(x+1)`).
- **Adding Gaussian noise** with varying σ.
- **Counting features** (e.g., peaks), and
- **Measuring entropy** and mutual information as functions of σ.

### 2.2. Results

- We observed a sharp increase in entropy or feature variance at a certain σ: **σ₍c₎ ≈ 0.117** for Collatz.
- We interpreted this as a “phase transition,” similar to those found in physics.

### 2.3. Limitations

- Different features (peaks, crossings, etc.) led to different σ₍c₎ values.
- Changing the statistical criterion (variance, MI, entropy threshold) shifted σ₍c₎.
- Fixing the random seed (as in early scripts) could suppress stochastic effects.

**Conclusion:** The “unique value” for σ₍c₎ was sensitive to experimental choices.

---

## 3. Intermediate Insights: Paper 2 (Discrete Phase Transitions)

### 3.1. Deepening the Analysis

Moving to the `discrete-phase-transitions` folder (`7.py`, `9.py`, `12.py`), we broadened our investigation:
- Tested many features and criteria (entropy, MI, minimal distance in log-space).
- Compared different systems (Collatz, qn+1, Fibonacci, etc.).
- Systematically varied the parameters for feature extraction and statistics.

### 3.2. Key Findings

- The “critical” σ depended strongly on the **feature** (what is measured) and **threshold** (how significance is defined).
- For some features, the minimal observable σ₍c₎ was extremely small (e.g., when based on minimal log-distance).
- **Different systems** showed different σ₍c₎ “fingerprints”—not a single number but a set of values.

### 3.3. Toward a General Principle

We recognized an **analogy to physics**: just as the melting point of a material depends on pressure, σ₍c₎ in discrete systems depends on how and what we measure.

---

## 4. Theoretical Synthesis: Paper 3 (Theory & Goldbach)

### 4.1. Analytical Models

In the `theory` and `goldbach` folders (see `b1.py`, `b5.py`, `oc.py`, `oc3.py`), we sought deeper understanding:
- Developed models relating σ₍c₎ to system properties (e.g., entropy, log-ratio, step size, spectral properties).
- Explored universal scaling laws (σ₍c₎ ~ log(q)/log(2), dependence on entropy).
- Performed cross-system analyses and clustering to reveal systematic patterns.


## **Features**

🧠 TheQA builds on established methods like Monte Carlo, Metropolis algorithms, and random projections, but its innovation lies in:

🚀 Tailored sample metric selection and aggregation.

📊 Creative application to novel mathematical domains (e.g., Collatz, dimensional bridges).

🔬 Empirical validation through bootstrapping and cross-platform reproducibility.



### **Projects Included** - see Branches!
- **📊 Part I: Foundation - the core concept
- **📊 Part II: Discrete Phase Transitions - feature of a broad class of mathematical systems
- **📊 Part III: Theory - the Theory
- **📊 Part IV: Goldbach - the Theory of σ<sub>c</sub>
  

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



