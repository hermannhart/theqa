"""
Simple Example: Understanding Variance Between Trials
For Prof. Vaienti - Demonstrating how variance is measured across multiple trials
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def simple_variance_example():
    """
    Small data example showing how variance between trials works
    """
    
    # Small deterministic sequence (e.g., first 10 values of Collatz(7))
    # 7 → 22 → 11 → 34 → 17 → 52 → 26 → 13 → 40 → 20
    sequence = np.array([7, 22, 11, 34, 17, 52, 26, 13, 40, 20])
    
    print("SIMPLE EXAMPLE: VARIANCE BETWEEN TRIALS")
    print("="*50)
    print(f"Original sequence: {sequence}")
    print(f"Length: {len(sequence)}")
    
    # Transform (standardize to make peak detection easier)
    transformed = (sequence - np.mean(sequence)) / np.std(sequence)
    print(f"\nStandardized: {[f'{x:.2f}' for x in transformed]}")
    
    # Find peaks in original (deterministic) sequence
    prominence = 0.5  # Fixed threshold
    peaks_original, _ = signal.find_peaks(transformed, prominence=prominence)
    print(f"\nOriginal peaks at positions: {peaks_original}")
    print(f"Number of peaks: {len(peaks_original)}")
    
    # Now test with different noise levels
    print("\n" + "-"*50)
    print("TESTING DIFFERENT NOISE LEVELS")
    print("-"*50)
    
    noise_levels = [0.0, 0.1, 0.3, 0.5]
    n_trials = 10  # Small number for clarity
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    
    for idx, sigma in enumerate(noise_levels):
        print(f"\nNoise level σ = {sigma}:")
        
        # Run multiple trials
        peak_counts = []
        
        # Show first 3 trials in detail
        for trial in range(n_trials):
            # Add noise
            noise = np.random.normal(0, sigma, len(transformed))
            noisy_sequence = transformed + noise
            
            # Find peaks
            peaks, _ = signal.find_peaks(noisy_sequence, prominence=prominence)
            peak_counts.append(len(peaks))
            
            # Print details for first 3 trials
            if trial < 3:
                print(f"  Trial {trial+1}: noise = {[f'{n:.2f}' for n in noise[:5]]}... → {len(peaks)} peaks")
        
        # Calculate variance
        variance = np.var(peak_counts)
        mean_peaks = np.mean(peak_counts)
        
        print(f"\nPeak counts across {n_trials} trials: {peak_counts}")
        print(f"Mean: {mean_peaks:.1f}, Variance: {variance:.3f}")
        
        # Visualize
        ax = axes[idx]
        
        # Plot histogram of peak counts
        if variance > 0:
            bins = np.arange(min(peak_counts)-0.5, max(peak_counts)+1.5, 1)
            ax.hist(peak_counts, bins=bins, alpha=0.7, edgecolor='black')
        else:
            # All values are the same
            ax.bar([mean_peaks], [n_trials], width=0.8, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Number of peaks')
        ax.set_ylabel('Frequency')
        ax.set_title(f'σ = {sigma}, Variance = {variance:.3f}')
        ax.grid(True, alpha=0.3)
        
        # Set integer x-ticks
        if variance > 0:
            ax.set_xticks(range(int(min(peak_counts)), int(max(peak_counts))+1))
    
    plt.suptitle('Distribution of Peak Counts Across Trials', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Show the critical transition
    print("\n" + "="*50)
    print("FINDING CRITICAL THRESHOLD σc")
    print("="*50)
    
    sigmas = np.logspace(-3, 0, 20)
    variances = []
    
    for sigma in sigmas:
        peak_counts = []
        for _ in range(50):  # More trials for better statistics
            noise = np.random.normal(0, sigma, len(transformed))
            noisy_sequence = transformed + noise
            peaks, _ = signal.find_peaks(noisy_sequence, prominence=prominence)
            peak_counts.append(len(peaks))
        
        variance = np.var(peak_counts)
        variances.append(variance)
    
    # Find transition
    threshold = 0.1
    transition_idx = np.where(np.array(variances) > threshold)[0]
    
    if len(transition_idx) > 0:
        sigma_c = sigmas[transition_idx[0]]
        print(f"Critical threshold σc ≈ {sigma_c:.3f}")
        print(f"(where variance exceeds {threshold})")
    
    # Plot variance vs sigma
    plt.figure(figsize=(8, 6))
    plt.semilogx(sigmas, variances, 'b-', linewidth=2)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
    if len(transition_idx) > 0:
        plt.axvline(x=sigma_c, color='g', linestyle='--', label=f'σc = {sigma_c:.3f}')
    plt.xlabel('Noise level σ')
    plt.ylabel('Variance of peak counts')
    plt.title('Variance vs Noise Level: Finding σc')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    # Explain probability measure
    print("\n" + "="*50)
    print("PROBABILITY MEASURE EXPLANATION")
    print("="*50)
    print("The probability measure we use:")
    print("1. We perform M independent trials (realizations)")
    print("2. Each trial: same deterministic sequence + different random noise")
    print("3. We measure a feature F (e.g., number of peaks) for each trial")
    print("4. Variance = E[(F - E[F])²] across trials")
    print("\nThis is the EMPIRICAL measure over the noise distribution")
    print("Alternative measures could weight trials differently")

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    simple_variance_example()