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

## ⚙️ Methodology

1. Choose a transformation **T** (log, identity, sqrt, ...).
2. Select a feature extractor **F**.
3. For σ ∈ [10⁻⁵, 10¹]:
    - Add Gaussian noise (no fixed seed!)
    - Apply F to the noisy sequence
    - Compute variance across trials
4. Identify σ<sub>c</sub> as the point where Var exceeds ε.

---

## 📊 Universality Classes

| Class        | Range             | Examples                  | Characteristics                    |
|--------------|-------------------|---------------------------|-------------------------------------|
| Ultra-low    | σ<sub>c</sub> < 0.01 | Chaos maps, Fibonacci      | High sensitivity, exponential       |
| Low          | 0.01–0.1          | Prime gaps, 3n−1           | Mixed dynamics                      |
| Medium       | 0.1–0.3           | Collatz family             | Number-theoretic, sin(σ) ≈ σ        |
| High         | > 0.3             | Goldbach (raw)            | Power law scaling, size dependence  |

---

## 📐 Example Phase Transition Behavior

```
         ⎧ 0                    for σ < σ_c
Var[F_σ(S)] = ⎨
         ⎩ V₀(σ - σ_c)^γ       for σ ≥ σ_c
```

Also: σ<sub>c</sub> = arg maxₛ I(S; F<sub>σ</sub>(S)) (mutual information)

---

## 🧪 Sample Code

```python
def measure_sigma_c(sequence, transformation='log'):
    seq = transform(sequence, method=transformation)
    for sigma in np.logspace(-5, 1, 100):
        variances = []
        for _ in range(200):
            noise = np.random.normal(0, sigma, len(seq))
            noisy = seq + noise
            features = extract_features(noisy, sigma)
            variances.append(features)
        if np.var(variances) > 0.1:
            return sigma
```

---

## 📈 Empirical Laws

- For `qn+1` systems:




with k₁ = 0.002, α ≈ 1.98, k₂ = 0.155, R² = 0.92.

- Goldbach σ<sub>c</sub>(n) ~ n⁻¹·¹¹⁷ → σ<sub>c</sub> → 0 for large n.

---

## 📚 References

- [Stochastic Resonance in Discrete Systems (Paper 1)](https://github.com/hermannhart/theqa)
- [Critical Threshold Universality (Paper 2)](https://github.com/hermannhart/theqa)
- [sin(σ<sub>c</sub>) = σ<sub>c</sub> for Medium Systems (Paper 3)](https://github.com/hermannhart/theqa)
- [Goldbach Phase Transitions (Paper 4)](https://github.com/hermannhart/theqa)

---

## 🌐 Key Insight

All discrete systems exhibit a phase transition under noise — but **σ<sub>c</sub> is not universal**. It varies across transformations, systems, and scales.

This framework provides a unified toolset to **detect**, **classify**, and **analyze** complexity emergence in deterministic sequences.

--
## Rule for the interpretation of OC/σ_c

    The value of OC/σ_c is always defined relative to the selected feature extraction and measurement method.

    It answers the question:
    “From which noise level σ does the statistic of the selected feature in the selected system differ significantly from the deterministic case according to the selected criterion?”
```
Formal:
σ c

resp.
O C
```
### Practical consequence

    OC is context-dependent! There is not “the” one OC value of a system, but always a whole family of OC values - depending on how you measure.
    Questions like “What is Collatz's OC?” are incomplete. Only the complete specification of feature/method/threshold makes the value meaningfully comparable.
    Comparisons between systems are only meaningful if the methodology is identical.
    Different OC values are not an error, but provide information on how robust/sensitive a system is to noise - in relation to the feature under consideration.

### Example formulation
```
    “The OC value is the answer to the question:
    'At what σ does a phase transition occur for feature X in system Y according to criterion Z?”
```

### Rule of thumb

“OC is not a natural constant - but a function of the measurement question!”


## **Features**

🧠 TheQA builds on established methods like Monte Carlo, Metropolis algorithms, and random projections, but its innovation lies in:

🚀 Tailored sample metric selection and aggregation.

📊 Creative application to novel mathematical domains (e.g., Collatz, dimensional bridges).

🔬 Empirical validation through bootstrapping and cross-platform reproducibility.



### **Projects Included** - see Branches!
- **📊 Part I: Foundation - the core concept
- **📊 Part II: Discrete Phase Transitions - feature of a broad class of mathematical systems
- **📊 Part III: Theory - the Theory
  

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



