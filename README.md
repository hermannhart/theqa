# TheQA Research for Computational Problems

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![License: Elastic License 2.0](https://img.shields.io/badge/Commercial%20License-ELv2-orange)](LICENSE-COMMERCIAL.txt)

# VERMICULAR: A Hardware-Optimized Quantum Search Algorithm

_Achieving 93% Success Rate on Superconducting Quantum Computers_

---

## Abstract

**VERMICULAR** is a hardware-optimized variant of Grover's quantum search algorithm that tackles a common challenge in quantum computing: the performance gap between simulation and real hardware. Standard implementations often achieve 15-25% success rates on current QPUs, which can limit practical applications.
Through strategic placement of dynamical decoupling (DD) sequences and circuit optimization, VERMICULAR demonstrates improved success rates: 93% on IQM Garnet and 98% on Rigetti Ankaa-3. This improvement may help enable:

- More reliable quantum operations: Better consistency across different hardware platforms
- Efficient resource usage: Improved success rates can reduce the number of required runs
- Complex quantum workflows: Multi-stage operations show enhanced stability
- Practical algorithm deployment: Performance suitable for real-world testing

The approach adds modest overhead (20 gates for 2-qubit search) while providing noise resilience on NISQ devices. The results suggest that algorithm-level optimizations can complement hardware improvements in making quantum computing more practical.
Complete implementation details and experimental data are provided to support reproducibility and further development of the approach.

---

## Introduction

Grover’s algorithm provides quadratic speedup for unstructured search, but its usefulness on today’s quantum hardware is severely limited by noise and decoherence. Standard implementations often achieve <20% success rates in practice.

**VERMICULAR** (VERsatile Modified Iterative Circuit Using Linearly-Arranged Redundancy) is a hardware-optimized version that achieves:
- **93%** success on IQM Garnet (20 qubits)
- **98%** on Rigetti Ankaa-3 (84 qubits)
- **100%** on simulators

The key innovation: _strategic placement of dynamical decoupling (DD) sequences_ that protect quantum information during execution without disrupting logic.

---

## Background

### Grover’s Algorithm

- **Initialization**: Prepare uniform superposition  
  $\ket{s} = \frac{1}{\sqrt{N}}\sum_{x=0}^{N-1}\ket{x}$
- **Grover Iteration** (repeat $\approx \frac{\pi}{4}\sqrt{N}$ times):
    - Oracle $O_f$: Flips phase of marked states
    - Diffusion $D$: Inversion about average
- **Measurement**: Yields marked item with high probability

### Hardware Challenges

- **Gate errors:** ~0.1–1% per 2-qubit gate
- **Decoherence:** $T_1$, $T_2$ ~ 10–100 μs
- **Crosstalk, calibration drift:** All degrade performance rapidly for circuits with depth >10–20.

---

## The VERMICULAR Algorithm

### Core Innovation: Strategic DD Placement

VERMICULAR enhances Grover’s algorithm with DD sequences:

```
1. Initialize qubits in |0>
2. Hadamard gates to create superposition
3. Apply DD sequence   // Pre-oracle
4. Apply Oracle O_f
5. Apply Diffusion D
6. Apply DD sequence   // Post-diffusion
7. Measure qubits
```

**DD sequences** use paired X gates that cancel systematic errors while preserving the quantum state:
- $DD = X_i X_i = I$

This simple identity has a profound effect on real hardware by refocusing coherent errors.

### Circuit Example (2 Qubits, 14 Gates)

```
q0: H — X — X —•— H — X —•— X — H — X — X
q1: H — X — X —Z— H — X —Z— X — H — X — X
```
The XX pairs implement dynamical decoupling.

---

## Implementation (Python, AWS Braket Example)

```python
import numpy as np
from braket.circuits import Circuit
from braket.devices import LocalSimulator
from braket.aws import AwsDevice

class VERMICULAR:
    """VERMICULAR: Hardware-optimized Grover search"""

    def __init__(self, marked_item: int = 3):
        self.marked_item = marked_item
        self.n_qubits = 2
        self.dd_positions = ['pre_oracle', 'post_diffusion']

    def create_circuit(self) -> Circuit:
        circuit = Circuit()
        circuit.h(0)
        circuit.h(1)
        if 'pre_oracle' in self.dd_positions:
            self._apply_dd_sequence(circuit)
        self._apply_oracle(circuit)
        self._apply_diffusion(circuit)
        if 'post_diffusion' in self.dd_positions:
            self._apply_dd_sequence(circuit)
        return circuit

    def _apply_dd_sequence(self, circuit: Circuit):
        circuit.x(0)
        circuit.x(0)
        circuit.x(1)
        circuit.x(1)

    def _apply_oracle(self, circuit: Circuit):
        marked_binary = format(self.marked_item, '02b')
        for i, bit in enumerate(marked_binary):
            if bit == '0':
                circuit.x(i)
        circuit.cz(0, 1)
        for i, bit in enumerate(marked_binary):
            if bit == '0':
                circuit.x(i)

    def _apply_diffusion(self, circuit: Circuit):
        circuit.h(0)
        circuit.h(1)
        circuit.x(0)
        circuit.x(1)
        circuit.cz(0, 1)
        circuit.x(0)
        circuit.x(1)
        circuit.h(0)
        circuit.h(1)

# Run VERMICULAR on hardware
device = AwsDevice("arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet")
vermicular = VERMICULAR(marked_item=3)  # Search for |11>
circuit = vermicular.create_circuit()
result = device.run(circuit, shots=1000).result()
counts = {}
for measurement in result.measurements:
    value = int(''.join(str(int(bit)) for bit in measurement), 2)
    counts[value] = counts.get(value, 0) + 1
success_rate = counts.get(3, 0) / 1000  # Should be ~0.93
```

---

## Experimental Results

| Platform         | Standard Grover | VERMICULAR | Improvement |
|------------------|-----------------|------------|-------------|
| Simulator        | 100%            | 100%       | --          |
| IQM Garnet       | 15–20%          | 93%        | 4.7×        |
| Rigetti Ankaa-3  | 18–25%          | 98%        | 4.3×        |

---

## Multi-Stage Search Performance (db_vs.py)
We evaluated VERMICULAR's performance for sequential quantum searches using Rigetti Ankaa-3 hardware. The test involved finding three 2-bit targets in sequence, simulating realistic multi-stage database search scenarios.

**Key Findings:**
- Stage 1: Both algorithms performed similarly (~93% success rate)
- Stage 2: VERMICULAR maintained 84.5% vs. standard Grover's 21.0% (4× improvement)
- Stage 3: VERMICULAR achieved 88.5% vs. standard Grover's 28.5% (3.1× improvement)

**Overall System Performance:**
- Standard Grover: 5.6% total success rate (all three stages)
- VERMICULAR: 69.2% total success rate
- Net improvement: 12.4× reliability

**Performance Degradation Pattern:**
Standard Grover shows significant performance degradation in later stages, likely due to accumulated noise and increased circuit depth. VERMICULAR maintains consistent 84-92% success rates across all stages.

![grover_vs_vermicular](https://github.com/hermannhart/theqa/blob/vermicular/vermicular_demo_rigetti_20250801_231052.png)
---

## Technical Analysis

**Noise Resilience**
- Coherent errors: Reduced through XX sequence timing
- Slow drift: Hardware recalibration during DD delays
- Crosstalk: Minimized impact through strategic gate placement

**Why VERMICULAR Works**
- Timing optimization: DD sequences provide calibrated delays
- Error refocusing: Strategic placement at vulnerable circuit points
- Minimal overhead: Only 4 additional gates for substantial improvement

**Limitations**
- Currently optimized for 2-qubit systems
- DD effectiveness depends on hardware characteristics
- Requires gate times much shorter than $T_2$ (satisfied by current hardware)
- Performance benefits most pronounced for deep circuits (>20 gates)

---

### Future Work

- Extension to larger qubit systems
- Adaptive DD placement
- Integration with error mitigation
- Application to further algorithms

---

## Conclusion

VERMICULAR demonstrates that practical quantum search is achievable on current NISQ devices through careful circuit optimization. With 93-98% success rates on real hardware and 12× improvement for multi-stage operations, it bridges the gap between theoretical quantum advantage and practical implementation.
Implementation: https://github.com/hermannhart/theqa/tree/vermicular
Preprint: Available on preprints.org for reproducibility and validation

---

## What does theQA have to do with vermicular?
theQA develops metrics to characterize quantum algorithms. Imagine you want to measure the "health" of a quantum algorithm - not just whether it works, but how robust it is under different conditions.

theQA-Quantum-Framework
1. analysis  → “Grover degrades at depth 2-4”
2. identification → “Temporal decoherence dominates”
3. targeted solution → “DD at positions X,Y”
4. validation → "93% constant performance"

**What makes it special:**
- **Predictive power:** Long-term behaviour can be predicted from just a few measurement points
- **Hardware agnostic:** The patterns are universal, but the parameters are hardware-specific
- **Optimization guide:** Shows exactly where and how to optimize

---

**Implementation and data:**  
[https://github.com/hermannhart/theqa/tree/vermicular](https://github.com/hermannhart/theqa/tree/vermicular)


---

## References

- Grover, L.K. (1996). "A fast quantum mechanical algorithm for database search."
- Preskill, J. (2018). "Quantum Computing in the NISQ era and beyond."
- AWS Braket Documentation (2021).
- IQM Garnet QPU (2023).
- Rigetti Ankaa-3 QPU (2023).

---

### **License**
- This project follows a dual-license model:

- For Personal & Research Use: CC BY-NC 4.0 → Free for non-commercial use only.
- For Commercial Use: Companies must obtain a commercial license (Elastic License 2.0).

📜 For details, see the LICENSE file.


### ***Contributors***

- Matthias - Human resources
- Arti Cyan - Artificial  resources


### ***Contact & Support***

For inquiries regarding commercial licensing or technical support:
📧 theqa@posteo.com
🌐 www.theqa.space

- 🚀 Get started with TheQA and explore new frontiers in optimization! 🚀

---
![Vermicular](https://github.com/hermannhart/theqa/blob/vermicular/vermicular.jpg)
