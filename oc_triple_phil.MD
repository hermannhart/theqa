# The Triple Rule for the Critical Noise Value (σ₍c₎/OC)

## Scientific Rule

**The critical noise value (σ₍c₎, OC) of a discrete system is not an absolute system constant, but is always defined as a function of a triple:**

- **System S** (e.g., Collatz, Fibonacci, ...)
- **Feature extraction F** (e.g., LogPeaks, Relative Differences, Complexity, ...)
- **Test criterion T** (e.g., Variance > threshold, MI maximum, Entropy transition, ...)

**Formally:**
```math
σ_c = σ_c(S, F, T)
```

## Consequences

- There is **no single “true” value** for σ₍c₎ of a system.
- Every measured value is **only complete and comparable in the context of its triple**.
- A **σ₍c₎-fingerprint** of a system is the set of all values for various (F, T).
- The choice of triple determines how sensitive or robust your analysis is.

## Analogy to Physics

> **Just as melting point and critical temperature depend on external parameters, σ₍c₎ depends on the feature and threshold criterion.**

## Practical Recommendation

- For every publication/measurement, always specify the triple (S, F, T) explicitly:
  - Example:  
    `σc(Collatz, LogPeaks, Var>0.1) = 0.117`
- For comparisons and benchmarks, a **standard triple** can be defined, e.g.:
  - `(System, LogPeaks, Var>0.1)`
- Deviating triples can be called “modified σ₍c₎” or be labeled with a superscript/index.

## Example (Python Dict):

```python
CollatzFingerprint = {
    ("LogPeaks", "Var>0.1"): 0.117,
    ("RelDiff", "MI_max"): 0.089,
    ("Turning", "Entropy"): 0.042,
    # further combinations...
}
```

## Further Questions

- **Standardization:** Which triples should serve as benchmarks?
- **Comparability:** How can σ₍c₎-fingerprints be systematically compared between systems?
- **Theoretical formulas:** How are σ₍c₎, feature sensitivity, and system complexity related?
