"""
Comprehensive analysis of the Collatz conjecture using TheQA.

This script demonstrates:
1. Computing σc for different Collatz variants (qn+1)
2. Analyzing the quantization phenomenon
3. Visualizing phase transitions
4. Finding optimal (F,C) pairs for Collatz
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from theqa import (
    CollatzSystem, TripleRule, compute_sigma_c,
    PeakCounter, EntropyCalculator, SpectralAnalyzer,
    AutoCorrelation, ZeroCrossings,
    VarianceCriterion, IQRCriterion, EntropyCriterion
)
from theqa.visualize import plot_phase_transition


def analyze_collatz_variants():
    """Analyze different qn+1 variants of Collatz."""
    print("="*60)
    print("COLLATZ VARIANTS ANALYSIS")
    print("="*60)
    
    # Test different multipliers
    q_values = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    results = []
    
    for q in q_values:
        print(f"\nAnalyzing {q}n+1 system...")
        
        # Test multiple starting values for robustness
        sigma_c_values = []
        
        for n in [27, 31, 47, 63, 97]:
            system = CollatzSystem(n=n, q=q)
            sequence = system.generate(max_steps=5000)
            
            if len(sequence) > 10:
                sigma_c = compute_sigma_c(sequence, method='adaptive')
                sigma_c_values.append(sigma_c)
        
        if sigma_c_values:
            mean_sigma_c = np.mean(sigma_c_values)
            std_sigma_c = np.std(sigma_c_values)
            
            results.append({
                'q': q,
                'sigma_c': mean_sigma_c,
                'std': std_sigma_c,
                'n_samples': len(sigma_c_values)
            })
            
            print(f"  σc = {mean_sigma_c:.3f} ± {std_sigma_c:.3f}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Analyze quantization
    print("\n" + "-"*40)
    print("QUANTIZATION ANALYSIS")
    print("-"*40)
    
    # Find clusters
    unique_values = []
    tolerance = 0.01
    
    for sigma in df['sigma_c']:
        is_new = True
        for uv in unique_values:
            if abs(sigma - uv) < tolerance:
                is_new = False
                break
        if is_new:
            unique_values.append(sigma)
    
    print(f"\nQuantized σc values found: {len(unique_values)}")
    for i, val in enumerate(sorted(unique_values)):
        print(f"  Level {i+1}: σc ≈ {val:.3f}")
        # Check if it's a rational multiple of π
        ratio = val / (np.pi/2)
        print(f"    Ratio to π/2: {ratio:.3f}")
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    # Plot 1: σc vs q
    plt.subplot(1, 2, 1)
    plt.errorbar(df['q'], df['sigma_c'], yerr=df['std'], 
                fmt='bo-', capsize=5, capthick=2)
    plt.xlabel('Multiplier q in qn+1')
    plt.ylabel('Critical Threshold σc')
    plt.title('Critical Thresholds for Collatz Variants')
    plt.grid(True, alpha=0.3)
    
    # Add quantization levels
    for val in unique_values:
        plt.axhline(y=val, color='r', linestyle='--', alpha=0.5)
    
    # Plot 2: Empirical scaling law
    plt.subplot(1, 2, 2)
    log_q = np.log(df['q']) / np.log(2)
    plt.plot(log_q, df['sigma_c'], 'bo', label='Data')
    
    # Fit scaling law
    coeffs = np.polyfit(log_q, df['sigma_c'], 1)
    fit_line = np.poly1d(coeffs)
    x_fit = np.linspace(min(log_q), max(log_q), 100)
    plt.plot(x_fit, fit_line(x_fit), 'r-', 
            label=f'σc = {coeffs[0]:.3f}·log₂(q) + {coeffs[1]:.3f}')
    
    plt.xlabel('log₂(q)')
    plt.ylabel('Critical Threshold σc')
    plt.title('Scaling Law for qn+1 Systems')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('collatz_variants_analysis.png', dpi=150)
    print(f"\nPlot saved to 'collatz_variants_analysis.png'")
    
    return df


def detailed_collatz_fingerprint():
    """Create detailed fingerprint of standard Collatz (3n+1)."""
    print("\n" + "="*60)
    print("DETAILED COLLATZ FINGERPRINT")
    print("="*60)
    
    system = CollatzSystem(n=27, q=3)
    
    # All available features
    features = {
        'peaks': PeakCounter(transform='log'),
        'peaks_linear': PeakCounter(transform=None),
        'entropy': EntropyCalculator(bins=20),
        'entropy_fine': EntropyCalculator(bins=50),
        'spectral_freq': SpectralAnalyzer(feature_type='dominant_freq'),
        'spectral_entropy': SpectralAnalyzer(feature_type='spectral_entropy'),
        'spectral_peaks': SpectralAnalyzer(feature_type='n_peaks'),
        'autocorr': AutoCorrelation(feature_type='value'),
        'autocorr_decay': AutoCorrelation(feature_type='decay_rate'),
        'zero_cross': ZeroCrossings(detrend=True),
    }
    
    # All criteria
    criteria = {
        'variance': VarianceCriterion(threshold=0.1),
        'iqr': IQRCriterion(threshold=0.2),
        'entropy': EntropyCriterion(threshold=0.1),
    }
    
    results = []
    
    print("\nComputing comprehensive fingerprint...")
    print("-" * 50)
    print(f"{'Feature':<20} {'Criterion':<10} {'σc':<10} {'Time (s)':<10}")
    print("-" * 50)
    
    for f_name, feature in features.items():
        for c_name, criterion in criteria.items():
            tr = TripleRule(
                system=system,
                feature=feature,
                criterion=criterion
            )
            
            result = tr.compute(n_trials=200, method='empirical')
            
            results.append({
                'feature': f_name,
                'criterion': c_name,
                'sigma_c': result.sigma_c,
                'ci_lower': result.ci_lower,
                'ci_upper': result.ci_upper,
                'time': result.time_elapsed
            })
            
            print(f"{f_name:<20} {c_name:<10} {result.sigma_c:<10.3f} "
                  f"{result.time_elapsed:<10.2f}")
    
    # Create fingerprint heatmap
    df = pd.DataFrame(results)
    pivot = df.pivot(index='feature', columns='criterion', values='sigma_c')
    
    plt.figure(figsize=(8, 10))
    plt.imshow(pivot.values, cmap='viridis', aspect='auto')
    plt.colorbar(label='σc')
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            plt.text(j, i, f'{pivot.values[i, j]:.3f}',
                    ha='center', va='center', color='white')
    
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel('Criterion')
    plt.ylabel('Feature')
    plt.title('Complete σc Fingerprint for Collatz System')
    plt.tight_layout()
    plt.savefig('collatz_fingerprint.png', dpi=150)
    print(f"\nFingerprint saved to 'collatz_fingerprint.png'")
    
    return df


def phase_transition_analysis():
    """Detailed analysis of the phase transition."""
    print("\n" + "="*60)
    print("PHASE TRANSITION ANALYSIS")
    print("="*60)
    
    system = CollatzSystem(n=27)
    sequence = system.generate(max_steps=2000)
    log_seq = np.log(sequence + 1)
    
    # Fine-grained sigma values around transition
    sigmas = np.concatenate([
        np.linspace(0.001, 0.05, 20),
        np.linspace(0.05, 0.15, 40),  # Dense around σc
        np.linspace(0.15, 0.5, 20)
    ])
    
    # Multiple features to track
    features_to_track = {
        'variance': [],
        'mean_peaks': [],
        'entropy': [],
        'iqr': []
    }
    
    print("\nAnalyzing phase transition...")
    
    for sigma in sigmas:
        peak_counts = []
        entropies = []
        
        for _ in range(200):
            noise = np.random.normal(0, sigma, len(log_seq))
            noisy = log_seq + noise
            
            # Count peaks
            peaks, _ = signal.find_peaks(noisy, prominence=sigma/2)
            peak_counts.append(len(peaks))
            
            # Calculate entropy
            hist, _ = np.histogram(noisy, bins=20)
            p = hist / np.sum(hist)
            p = p[p > 0]
            entropy = -np.sum(p * np.log(p))
            entropies.append(entropy)
        
        features_to_track['variance'].append(np.var(peak_counts))
        features_to_track['mean_peaks'].append(np.mean(peak_counts))
        features_to_track['entropy'].append(np.mean(entropies))
        features_to_track['iqr'].append(
            np.percentile(peak_counts, 75) - np.percentile(peak_counts, 25)
        )
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, (name, values) in enumerate(features_to_track.items()):
        ax = axes[idx // 2, idx % 2]
        ax.plot(sigmas, values, 'b-', linewidth=2)
        ax.axvline(x=0.117, color='r', linestyle='--', 
                  label='σc = 0.117', linewidth=2)
        ax.set_xlabel('Noise Level σ')
        ax.set_ylabel(name.replace('_', ' ').title())
        ax.set_title(f'{name.replace("_", " ").title()} vs Noise')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xscale('log')
    
    plt.suptitle('Multi-Feature Phase Transition Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('collatz_phase_transition_detailed.png', dpi=150)
    print(f"\nDetailed transition plot saved to 'collatz_phase_transition_detailed.png'")


def optimal_measurement_design():
    """Find optimal (F,C) pairs for specific applications."""
    print("\n" + "="*60)
    print("OPTIMAL MEASUREMENT DESIGN FOR COLLATZ")
    print("="*60)
    
    system = CollatzSystem(n=27)
    
    # Test comprehensive set of features and criteria
    test_features = {
        'peaks_log': PeakCounter(transform='log'),
        'peaks_sqrt': PeakCounter(transform='sqrt'),
        'entropy_20': EntropyCalculator(bins=20),
        'entropy_50': EntropyCalculator(bins=50),
        'spectral': SpectralAnalyzer(),
        'autocorr': AutoCorrelation(),
    }
    
    test_criteria = {
        'var_0.05': VarianceCriterion(threshold=0.05),
        'var_0.1': VarianceCriterion(threshold=0.1),
        'var_0.2': VarianceCriterion(threshold=0.2),
        'iqr': IQRCriterion(threshold=0.2),
        'entropy': EntropyCriterion(threshold=0.1),
    }
    
    # Goal 1: Maximum sensitivity (minimum σc)
    print("\n1. Optimizing for MAXIMUM SENSITIVITY...")
    
    best_sensitivity = float('inf')
    best_sensitive = None
    
    for f_name, feature in test_features.items():
        for c_name, criterion in test_criteria.items():
            tr = TripleRule(system=system, feature=feature, criterion=criterion)
            result = tr.compute(n_trials=50, method='adaptive')
            
            if result.sigma_c < best_sensitivity:
                best_sensitivity = result.sigma_c
                best_sensitive = (f_name, c_name, result.sigma_c)
    
    print(f"   Best for sensitivity: {best_sensitive[0]} + {best_sensitive[1]}")
    print(f"   Achieves σc = {best_sensitive[2]:.4f}")
    
    # Goal 2: Maximum robustness (maximum σc)
    print("\n2. Optimizing for MAXIMUM ROBUSTNESS...")
    
    best_robustness = 0
    best_robust = None
    
    for f_name, feature in test_features.items():
        for c_name, criterion in test_criteria.items():
            tr = TripleRule(system=system, feature=feature, criterion=criterion)
            result = tr.compute(n_trials=50, method='adaptive')
            
            if result.sigma_c > best_robustness:
                best_robustness = result.sigma_c
                best_robust = (f_name, c_name, result.sigma_c)
    
    print(f"   Best for robustness: {best_robust[0]} + {best_robust[1]}")
    print(f"   Achieves σc = {best_robust[2]:.4f}")
    
    # Goal 3: Distinguish Collatz from other systems
    print("\n3. Optimizing for SYSTEM DISCRIMINATION...")
    
    from theqa import FibonacciSystem, LogisticMap
    
    systems = {
        'collatz': CollatzSystem(n=27),
        'fibonacci': FibonacciSystem(n=100),
        'logistic': LogisticMap(r=3.9, length=500)
    }
    
    best_discrimination = 0
    best_discriminator = None
    
    for f_name, feature in test_features.items():
        for c_name, criterion in test_criteria.items():
            sigma_c_values = []
            
            for sys_name, sys in systems.items():
                tr = TripleRule(system=sys, feature=feature, criterion=criterion)
                result = tr.compute(n_trials=30, method='adaptive')
                sigma_c_values.append(result.sigma_c)
            
            # Discrimination score = spread of σc values
            spread = np.std(sigma_c_values) / (np.mean(sigma_c_values) + 1e-10)
            
            if spread > best_discrimination:
                best_discrimination = spread
                best_discriminator = (f_name, c_name, spread, sigma_c_values)
    
    print(f"   Best discriminator: {best_discriminator[0]} + {best_discriminator[1]}")
    print(f"   Discrimination score: {best_discriminator[2]:.3f}")
    print(f"   σc values: {[f'{x:.3f}' for x in best_discriminator[3]]}")


def main():
    """Run complete Collatz analysis."""
    print("\n" + "="*60)
    print("COMPLETE COLLATZ ANALYSIS WITH TheQA")
    print("="*60)
    
    # 1. Analyze variants
    df_variants = analyze_collatz_variants()
    
    # 2. Create detailed fingerprint
    df_fingerprint = detailed_collatz_fingerprint()
    
    # 3. Phase transition analysis
    phase_transition_analysis()
    
    # 4. Optimal measurement design
    optimal_measurement_design()
    
    # Save results
    df_variants.to_csv('collatz_variants_results.csv', index=False)
    df_fingerprint.to_csv('collatz_fingerprint_results.csv', index=False)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nResults saved to:")
    print("  - collatz_variants_results.csv")
    print("  - collatz_fingerprint_results.csv")
    print("  - collatz_variants_analysis.png")
    print("  - collatz_fingerprint.png")
    print("  - collatz_phase_transition_detailed.png")


if __name__ == "__main__":
    main()
