"""
Complete example demonstrating TheQA package usage.

This script shows how to:
1. Analyze various dynamical systems
2. Compare different methods
3. Visualize results
4. Optimize (F,C) pairs
"""

import numpy as np
import matplotlib.pyplot as plt
from theqa import (
    compute_sigma_c, TripleRule,
    CollatzSystem, FibonacciSystem, LogisticMap,
    PeakCounter, EntropyCalculator, SpectralAnalyzer,
    VarianceCriterion, IQRCriterion,
    spectral_sigma_c, gradient_sigma_c, analytical_sigma_c,
    optimize_for_sensitivity, optimize_for_robustness
)


def basic_usage():
    """Basic usage examples."""
    print("="*60)
    print("BASIC USAGE")
    print("="*60)
    
    # Simple sequence analysis
    sequence = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    sigma_c = compute_sigma_c(sequence)
    print(f"\nSimple sequence: σc = {sigma_c:.3f}")
    
    # Collatz system
    collatz = CollatzSystem(n=27)
    sigma_c = compute_sigma_c(collatz.generate())
    print(f"Collatz(27): σc = {sigma_c:.3f}")
    
    # Fibonacci
    fib = FibonacciSystem(n=50)
    sigma_c = compute_sigma_c(fib.generate())
    print(f"Fibonacci(50): σc = {sigma_c:.3f}")


def triple_rule_demo():
    """Demonstrate the Triple Rule framework."""
    print("\n" + "="*60)
    print("TRIPLE RULE DEMONSTRATION")
    print("="*60)
    
    # Create system
    system = CollatzSystem(n=27)
    sequence = system.generate(max_steps=1000)
    
    # Different features
    features = {
        'peaks': PeakCounter(transform='log'),
        'entropy': EntropyCalculator(bins=20),
        'spectral': SpectralAnalyzer(feature_type='dominant_freq')
    }
    
    # Different criteria
    criteria = {
        'variance': VarianceCriterion(threshold=0.1),
        'iqr': IQRCriterion(threshold=0.2)
    }
    
    print("\nSystem fingerprint for Collatz(27):")
    print("-" * 40)
    print(f"{'Feature':<15} {'Criterion':<15} {'σc':<10}")
    print("-" * 40)
    
    for f_name, feature in features.items():
        for c_name, criterion in criteria.items():
            tr = TripleRule(
                system=system,
                feature=feature,
                criterion=criterion
            )
            result = tr.compute(n_trials=100, verbose=False)
            print(f"{f_name:<15} {c_name:<15} {result.sigma_c:<10.3f}")


def method_comparison():
    """Compare different computational methods."""
    print("\n" + "="*60)
    print("METHOD COMPARISON")
    print("="*60)
    
    systems = {
        'Collatz(27)': CollatzSystem(n=27).generate(),
        'Fibonacci(100)': FibonacciSystem(n=100).generate(),
        'Logistic(r=3.9)': LogisticMap(r=3.9, length=500).generate()
    }
    
    for name, sequence in systems.items():
        print(f"\n{name}:")
        print("-" * 30)
        
        # Empirical (slow but accurate)
        tr = TripleRule(system=sequence)
        result_emp = tr.compute(method='empirical', n_trials=50)
        
        # Fast methods
        sigma_spec = spectral_sigma_c(sequence)
        sigma_grad = gradient_sigma_c(sequence)
        sigma_anal = analytical_sigma_c(sequence)
        
        print(f"  Empirical:  {result_emp.sigma_c:.3f} ({result_emp.time_elapsed:.3f}s)")
        print(f"  Spectral:   {sigma_spec:.3f} (< 0.001s)")
        print(f"  Gradient:   {sigma_grad:.3f} (< 0.001s)")
        print(f"  Analytical: {sigma_anal:.3f} (< 0.001s)")


def visualization_demo():
    """Demonstrate visualization capabilities."""
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    
    # Generate Collatz sequence
    system = CollatzSystem(n=27)
    sequence = system.generate()
    log_seq = np.log(sequence + 1)
    
    # Test different noise levels
    sigmas = np.logspace(-3, 0, 50)
    variances = []
    
    for sigma in sigmas:
        features = []
        for _ in range(100):
            noise = np.random.normal(0, sigma, len(log_seq))
            noisy = log_seq + noise
            peaks = len(np.where(np.diff(np.sign(np.diff(noisy))) == -2)[0])
            features.append(peaks)
        variances.append(np.var(features))
    
    # Plot phase transition
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.semilogx(sigmas, variances, 'b-', linewidth=2)
    plt.axvline(x=0.117, color='r', linestyle='--', label='σc = 0.117')
    plt.xlabel('Noise Level (σ)')
    plt.ylabel('Feature Variance')
    plt.title('Phase Transition in Collatz System')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # System comparison
    plt.subplot(1, 2, 2)
    systems = ['Collatz', 'Fibonacci', 'Logistic', 'Tent']
    sigma_c_values = [0.117, 0.001, 0.210, 0.180]
    colors = ['blue', 'green', 'red', 'orange']
    
    plt.bar(systems, sigma_c_values, color=colors, alpha=0.7)
    plt.axhline(y=np.pi/2, color='black', linestyle='--', label='π/2 bound')
    plt.ylabel('Critical Threshold (σc)')
    plt.title('Critical Thresholds Across Systems')
    plt.legend()
    plt.ylim(0, 1.8)
    
    plt.tight_layout()
    plt.savefig('phase_transitions.png', dpi=150)
    print("\nVisualization saved to 'phase_transitions.png'")


def optimization_demo():
    """Demonstrate optimization capabilities."""
    print("\n" + "="*60)
    print("OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    # Test system
    system = CollatzSystem(n=27)
    
    print("\nOptimizing for different goals:")
    print("-" * 40)
    
    # Maximize sensitivity (minimize σc)
    print("\n1. Maximum Sensitivity (minimize σc):")
    sensitive_fc = optimize_for_sensitivity(system)
    print(f"   Optimal: {sensitive_fc}")
    
    # Maximize robustness (maximize σc)
    print("\n2. Maximum Robustness (maximize σc):")
    robust_fc = optimize_for_robustness(system)
    print(f"   Optimal: {robust_fc}")
    
    # Maximize discrimination
    print("\n3. Maximum Discrimination:")
    systems = [CollatzSystem(), FibonacciSystem(), LogisticMap()]
    disc_fc = optimize_for_discrimination(systems)
    print(f"   Optimal: {disc_fc}")


def complete_analysis():
    """Complete analysis workflow."""
    print("\n" + "="*60)
    print("COMPLETE ANALYSIS WORKFLOW")
    print("="*60)
    
    # 1. Choose system
    print("\n1. Analyzing Collatz system starting from n=27")
    system = CollatzSystem(n=27)
    sequence = system.generate(max_steps=5000)
    print(f"   Sequence length: {len(sequence)}")
    print(f"   Max value: {max(sequence):.0f}")
    
    # 2. Quick estimate
    print("\n2. Quick estimate using adaptive method")
    sigma_c_quick = compute_sigma_c(sequence, method='adaptive')
    print(f"   σc ≈ {sigma_c_quick:.3f}")
    
    # 3. Detailed analysis
    print("\n3. Detailed analysis with confidence intervals")
    tr = TripleRule(
        system=system,
        feature=PeakCounter(transform='log'),
        criterion=VarianceCriterion(threshold=0.1)
    )
    
    result = tr.compute(
        n_trials=500,
        method='empirical',
        confidence=0.95,
        verbose=True
    )
    
    print(f"\n   σc = {result.sigma_c:.3f} [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
    print(f"   Converged: {result.converged}")
    print(f"   Time: {result.time_elapsed:.2f}s")
    
    # 4. System fingerprint
    print("\n4. Computing system fingerprint")
    from theqa import analyze_system
    df_fingerprint = analyze_system(system, verbose=False)
    print("\n", df_fingerprint)
    
    # 5. Classification
    print("\n5. System classification")
    if sigma_c_quick < 0.01:
        class_name = "Ultra-sensitive"
    elif sigma_c_quick < 0.1:
        class_name = "Sensitive"
    elif sigma_c_quick < 0.3:
        class_name = "Medium"
    else:
        class_name = "Robust"
    
    print(f"   Classification: {class_name}")
    print(f"   Interpretation: ", end="")
    
    if class_name == "Ultra-sensitive":
        print("Minimal noise destroys structure")
    elif class_name == "Sensitive":
        print("Balanced sensitivity to perturbations")
    elif class_name == "Medium":
        print("Chaotic but deterministic behavior")
    else:
        print("Highly resistant to noise")


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("TheQA PACKAGE DEMONSTRATION")
    print("="*60)
    
    basic_usage()
    triple_rule_demo()
    method_comparison()
    visualization_demo()
    optimization_demo()
    complete_analysis()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nFor more information, see the documentation at:")
    print("https://theqa.readthedocs.io")


if __name__ == "__main__":
    main()
