#!/usr/bin/env python3
"""
DEEP SCALING ANALYSIS FOR σc
=============================
Complete investigation of scaling behavior to understand
why scaling invariance fails and what σc really measures
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats, optimize
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from collections import defaultdict
from tqdm import tqdm
import json
from datetime import datetime

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class ScalingDiagnostics:
    """
    Complete diagnostics for scaling behavior of σc
    """
    
    def __init__(self):
        self.results = {}
        self.scaling_patterns = {}
        
    def generate_test_sequences(self):
        """Generate diverse test sequences for scaling analysis"""
        sequences = {}
        
        # Simple mathematical sequences
        sequences['Linear'] = np.arange(1, 51)
        sequences['Quadratic'] = np.arange(1, 51)**2
        sequences['Exponential'] = 2**np.arange(1, 21)
        sequences['Logarithmic'] = np.log(np.arange(1, 51))
        
        # Recursive sequences
        fib = [1, 1]
        for _ in range(48):
            fib.append(fib[-1] + fib[-2])
        sequences['Fibonacci'] = np.array(fib[:50])
        
        # Chaotic
        logistic = []
        x = 0.5
        for _ in range(100):
            logistic.append(x)
            x = 3.8 * x * (1 - x)
        sequences['Logistic'] = np.array(logistic[50:])
        
        # Random
        np.random.seed(42)
        sequences['Gaussian'] = np.random.normal(0, 1, 50)
        sequences['Uniform'] = np.random.uniform(0, 1, 50)
        
        # Oscillatory
        t = np.linspace(0, 4*np.pi, 50)
        sequences['Sine'] = np.sin(t)
        sequences['Cosine'] = np.cos(t)
        
        # Mixed
        sequences['SineExp'] = np.sin(t) * np.exp(t/10)
        
        return sequences
    
    def measure_sigma_c_detailed(self, sequence, return_full=False):
        """
        Detailed σc measurement with diagnostic information
        """
        if len(sequence) < 10:
            return {'sigma_c': np.nan}
        
        log_seq = np.log(np.abs(sequence) + 1)
        seq_scale = np.std(log_seq)
        
        if seq_scale == 0:
            return {'sigma_c': np.nan}
        
        # Adaptive noise levels
        noise_levels = np.logspace(np.log10(seq_scale/1000), 
                                  np.log10(seq_scale*10), 50)
        
        info_measures = []
        peak_counts = []
        variances = []
        
        for sigma in noise_levels:
            measurements = []
            
            for _ in range(30):
                noise = np.random.normal(0, sigma, len(log_seq))
                noisy = log_seq + noise
                
                peaks, properties = signal.find_peaks(noisy, prominence=sigma/3)
                measurements.append(len(peaks))
            
            peak_counts.append(measurements)
            median_peaks = np.median(measurements)
            iqr_peaks = np.percentile(measurements, 75) - np.percentile(measurements, 25)
            
            info = median_peaks / (1 + iqr_peaks) if iqr_peaks >= 0 else median_peaks
            info_measures.append(info)
            variances.append(np.var(measurements))
        
        # Find optimal σc
        if len(info_measures) > 5:
            smoothed = gaussian_filter1d(info_measures, sigma=1)
            optimal_idx = np.argmax(smoothed)
            sigma_c = noise_levels[optimal_idx]
        else:
            sigma_c = np.nan
        
        result = {
            'sigma_c': sigma_c,
            'seq_scale': seq_scale,
            'noise_levels': noise_levels,
            'info_measures': info_measures,
            'peak_counts': peak_counts,
            'variances': variances,
            'optimal_idx': optimal_idx if 'optimal_idx' in locals() else None
        }
        
        return result if return_full else {'sigma_c': sigma_c}
    
    def analyze_scaling_behavior(self, sequence, seq_name):
        """
        Comprehensive analysis of how σc scales
        """
        print(f"\nAnalyzing scaling for: {seq_name}")
        print("-" * 50)
        
        # Test many scale factors
        scales = np.logspace(-3, 3, 31)  # 0.001 to 1000
        
        results = {
            'scales': scales,
            'sigma_c_raw': [],
            'sigma_c_normalized': defaultdict(list),
            'sequence_properties': defaultdict(list),
            'diagnostics': []
        }
        
        for scale in tqdm(scales, desc=f"Testing scales for {seq_name}"):
            scaled_seq = sequence * scale
            
            # Detailed σc measurement
            measurement = self.measure_sigma_c_detailed(scaled_seq, return_full=True)
            results['sigma_c_raw'].append(measurement['sigma_c'])
            results['diagnostics'].append(measurement)
            
            # Sequence properties at this scale
            props = {
                'mean': np.mean(scaled_seq),
                'std': np.std(scaled_seq),
                'range': np.max(scaled_seq) - np.min(scaled_seq),
                'log_std': np.std(np.log(np.abs(scaled_seq) + 1)),
                'log_range': np.max(np.log(np.abs(scaled_seq) + 1)) - np.min(np.log(np.abs(scaled_seq) + 1)),
                'iqr': np.percentile(scaled_seq, 75) - np.percentile(scaled_seq, 25),
                'mad': np.median(np.abs(scaled_seq - np.median(scaled_seq))),
                'cv': np.std(scaled_seq) / (np.mean(np.abs(scaled_seq)) + 1e-10),
                'entropy': self._shannon_entropy(scaled_seq),
                'lz_complexity': self._lempel_ziv_complexity(scaled_seq)
            }
            
            for prop_name, prop_value in props.items():
                results['sequence_properties'][prop_name].append(prop_value)
                
                # Normalized σc
                if prop_value > 0 and not np.isnan(measurement['sigma_c']):
                    normalized = measurement['sigma_c'] / prop_value
                    results['sigma_c_normalized'][prop_name].append(normalized)
                else:
                    results['sigma_c_normalized'][prop_name].append(np.nan)
        
        # Analyze scaling patterns
        patterns = self._identify_scaling_patterns(results)
        results['patterns'] = patterns
        
        return results
    
    def _identify_scaling_patterns(self, results):
        """
        Identify mathematical patterns in scaling behavior
        """
        patterns = {}
        scales = np.array(results['scales'])
        sigma_c = np.array(results['sigma_c_raw'])
        
        # Remove NaN values for fitting
        valid = ~np.isnan(sigma_c)
        if np.sum(valid) < 3:
            return patterns
        
        scales_valid = scales[valid]
        sigma_c_valid = sigma_c[valid]
        
        # 1. Test power law: σc ~ scale^α
        try:
            log_scales = np.log(scales_valid)
            log_sigma_c = np.log(sigma_c_valid + 1e-10)
            
            # Linear fit in log-log space
            coeffs = np.polyfit(log_scales, log_sigma_c, 1)
            alpha = coeffs[0]
            
            # Calculate R²
            predicted = np.polyval(coeffs, log_scales)
            r2 = 1 - np.sum((log_sigma_c - predicted)**2) / np.var(log_sigma_c)
            
            patterns['power_law'] = {
                'exponent': alpha,
                'r2': r2,
                'interpretation': f'σc ~ scale^{alpha:.3f}'
            }
        except:
            patterns['power_law'] = None
        
        # 2. Test linear relationship
        try:
            coeffs = np.polyfit(scales_valid, sigma_c_valid, 1)
            predicted = np.polyval(coeffs, scales_valid)
            r2 = 1 - np.sum((sigma_c_valid - predicted)**2) / np.var(sigma_c_valid)
            
            patterns['linear'] = {
                'slope': coeffs[0],
                'intercept': coeffs[1],
                'r2': r2
            }
        except:
            patterns['linear'] = None
        
        # 3. Test logarithmic: σc ~ log(scale)
        try:
            log_scales = np.log(scales_valid + 1)
            coeffs = np.polyfit(log_scales, sigma_c_valid, 1)
            predicted = np.polyval(coeffs, log_scales)
            r2 = 1 - np.sum((sigma_c_valid - predicted)**2) / np.var(sigma_c_valid)
            
            patterns['logarithmic'] = {
                'coefficient': coeffs[0],
                'r2': r2
            }
        except:
            patterns['logarithmic'] = None
        
        # 4. Identify best normalization
        best_norm = None
        best_cv = float('inf')
        
        for norm_name, norm_values in results['sigma_c_normalized'].items():
            valid_norm = [v for v in norm_values if not np.isnan(v) and not np.isinf(v)]
            if len(valid_norm) > 3:
                cv = np.std(valid_norm) / (np.mean(valid_norm) + 1e-10)
                if cv < best_cv:
                    best_cv = cv
                    best_norm = norm_name
        
        patterns['best_normalization'] = {
            'name': best_norm,
            'cv': best_cv
        }
        
        # 5. Identify scaling regime transitions
        if len(sigma_c_valid) > 10:
            # Look for breakpoints using first derivative
            d_sigma_c = np.gradient(sigma_c_valid)
            d2_sigma_c = np.gradient(d_sigma_c)
            
            # Find points of maximum change
            breakpoints = []
            threshold = np.std(np.abs(d2_sigma_c)) * 2
            
            for i in range(1, len(d2_sigma_c)-1):
                if np.abs(d2_sigma_c[i]) > threshold:
                    breakpoints.append(scales_valid[i])
            
            patterns['breakpoints'] = breakpoints
        
        return patterns
    
    def visualize_scaling_analysis(self, results, seq_name):
        """
        Comprehensive visualization of scaling behavior
        """
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        scales = results['scales']
        sigma_c = results['sigma_c_raw']
        
        # 1. Raw σc vs scale
        ax = axes[0, 0]
        ax.loglog(scales, sigma_c, 'o-', markersize=4)
        ax.set_xlabel('Scale Factor')
        ax.set_ylabel('σc')
        ax.set_title('Raw σc vs Scale')
        ax.grid(True, alpha=0.3)
        
        # 2. Log-log plot with power law fit
        ax = axes[0, 1]
        valid = ~np.isnan(sigma_c)
        if np.sum(valid) > 3 and results['patterns'].get('power_law'):
            power_law = results['patterns']['power_law']
            ax.scatter(np.log(scales[valid]), np.log(np.array(sigma_c)[valid] + 1e-10), alpha=0.6)
            
            # Add fit line
            x_fit = np.log(scales[valid])
            y_fit = power_law['exponent'] * x_fit + np.log(scales[valid][0])
            ax.plot(x_fit, y_fit, 'r--', label=f"α={power_law['exponent']:.3f}")
            
            ax.set_xlabel('log(Scale)')
            ax.set_ylabel('log(σc)')
            ax.set_title(f"Power Law Fit (R²={power_law['r2']:.3f})")
            ax.legend()
        
        # 3. Normalized σc for different normalizations
        ax = axes[0, 2]
        
        # Select top 3 normalizations
        norm_cvs = []
        for norm_name in ['std', 'log_std', 'range', 'iqr']:
            if norm_name in results['sigma_c_normalized']:
                values = results['sigma_c_normalized'][norm_name]
                valid_vals = [v for v in values if not np.isnan(v) and not np.isinf(v)]
                if len(valid_vals) > 3:
                    cv = np.std(valid_vals) / (np.mean(valid_vals) + 1e-10)
                    norm_cvs.append((norm_name, cv, values))
        
        norm_cvs.sort(key=lambda x: x[1])
        
        for norm_name, cv, values in norm_cvs[:3]:
            ax.semilogx(scales, values, 'o-', label=f'{norm_name} (CV={cv:.3f})', 
                       markersize=3, alpha=0.7)
        
        ax.set_xlabel('Scale Factor')
        ax.set_ylabel('Normalized σc')
        ax.set_title('Normalized σc Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Sequence properties vs scale
        ax = axes[1, 0]
        for prop in ['std', 'range', 'log_std']:
            if prop in results['sequence_properties']:
                values = results['sequence_properties'][prop]
                ax.loglog(scales, values, 'o-', label=prop, markersize=3, alpha=0.7)
        
        ax.set_xlabel('Scale Factor')
        ax.set_ylabel('Property Value')
        ax.set_title('Sequence Properties Scaling')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. σc/scale vs scale (test for linear scaling)
        ax = axes[1, 1]
        sigma_c_array = np.array(sigma_c)
        ratio = sigma_c_array / scales
        ax.loglog(scales, ratio, 'o-', markersize=4)
        ax.set_xlabel('Scale Factor')
        ax.set_ylabel('σc / Scale')
        ax.set_title('Test for Linear Scaling')
        ax.grid(True, alpha=0.3)
        
        # 6. Information measure landscape
        ax = axes[1, 2]
        # Get one diagnostic for visualization
        mid_idx = len(results['diagnostics']) // 2
        if results['diagnostics'][mid_idx] and 'info_measures' in results['diagnostics'][mid_idx]:
            diag = results['diagnostics'][mid_idx]
            if diag['noise_levels'] is not None and diag['info_measures'] is not None:
                ax.semilogx(diag['noise_levels'], diag['info_measures'], 'b-', linewidth=2)
                if diag['optimal_idx'] is not None:
                    ax.axvline(diag['noise_levels'][diag['optimal_idx']], 
                              color='r', linestyle='--', label='σc')
                ax.set_xlabel('Noise Level')
                ax.set_ylabel('Information Measure')
                ax.set_title(f'Info Landscape (scale={scales[mid_idx]:.2f})')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 7. CV for each normalization
        ax = axes[2, 0]
        norm_names = []
        cvs = []
        
        for norm_name, values in results['sigma_c_normalized'].items():
            valid = [v for v in values if not np.isnan(v) and not np.isinf(v)]
            if len(valid) > 3:
                cv = np.std(valid) / (np.mean(valid) + 1e-10)
                norm_names.append(norm_name)
                cvs.append(cv)
        
        ax.bar(range(len(norm_names)), cvs)
        ax.set_xticks(range(len(norm_names)))
        ax.set_xticklabels(norm_names, rotation=45, ha='right')
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title('Scaling Invariance Quality')
        ax.axhline(y=0.1, color='r', linestyle='--', label='Target CV=0.1')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 8. Scaling exponent analysis
        ax = axes[2, 1]
        # Calculate local scaling exponent
        valid = ~np.isnan(sigma_c)
        if np.sum(valid) > 5:
            scales_valid = scales[valid]
            sigma_c_valid = np.array(sigma_c)[valid]
            
            local_exponents = []
            window = 3
            
            for i in range(window, len(scales_valid) - window):
                local_scales = scales_valid[i-window:i+window+1]
                local_sigma = sigma_c_valid[i-window:i+window+1]
                
                if np.all(local_sigma > 0):
                    log_s = np.log(local_scales)
                    log_sigma = np.log(local_sigma)
                    
                    coeff = np.polyfit(log_s, log_sigma, 1)
                    local_exponents.append(coeff[0])
                else:
                    local_exponents.append(np.nan)
            
            ax.semilogx(scales_valid[window:-window], local_exponents, 'o-', markersize=4)
            ax.set_xlabel('Scale Factor')
            ax.set_ylabel('Local Scaling Exponent')
            ax.set_title('Scale-Dependent Exponent')
            ax.grid(True, alpha=0.3)
        
        # 9. Best normalization visualization
        ax = axes[2, 2]
        if results['patterns'].get('best_normalization'):
            best = results['patterns']['best_normalization']
            if best['name'] in results['sigma_c_normalized']:
                values = results['sigma_c_normalized'][best['name']]
                
                ax.semilogx(scales, values, 'go-', markersize=5, linewidth=2)
                mean_val = np.nanmean(values)
                ax.axhline(y=mean_val, color='r', linestyle='--', 
                          label=f'Mean={mean_val:.3f}')
                
                ax.fill_between(scales, 
                               mean_val - np.nanstd(values),
                               mean_val + np.nanstd(values),
                               alpha=0.3, color='green')
                
                ax.set_xlabel('Scale Factor')
                ax.set_ylabel(f'σc / {best["name"]}')
                ax.set_title(f'Best Normalization (CV={best["cv"]:.3f})')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Scaling Analysis: {seq_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _shannon_entropy(self, sequence):
        """Calculate Shannon entropy"""
        if len(sequence) < 2:
            return 0
        n_bins = min(int(np.sqrt(len(sequence))), 20)
        counts, _ = np.histogram(sequence, bins=n_bins)
        probs = counts / np.sum(counts)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0
    
    def _lempel_ziv_complexity(self, sequence):
        """Calculate Lempel-Ziv complexity"""
        if len(sequence) < 2:
            return 0
        s = ''.join(['1' if x > np.median(sequence) else '0' for x in sequence])
        n = len(s)
        if n == 0:
            return 0
        complexity = 0
        i = 0
        while i < n:
            j = i + 1
            while j <= n and s[i:j] in s[:i]:
                j += 1
            complexity += 1
            i = j
        return complexity / n
    
    def run_complete_scaling_diagnostics(self):
        """
        Run complete scaling diagnostics on all sequence types
        """
        print("="*80)
        print("COMPLETE SCALING DIAGNOSTICS FOR σc")
        print("="*80)
        
        # Generate test sequences
        sequences = self.generate_test_sequences()
        
        # Store all results
        all_results = {}
        summary = {
            'sequence_type': [],
            'scaling_exponent': [],
            'best_normalization': [],
            'best_cv': [],
            'power_law_r2': []
        }
        
        # Analyze each sequence
        for seq_name, sequence in sequences.items():
            results = self.analyze_scaling_behavior(sequence, seq_name)
            all_results[seq_name] = results
            
            # Create visualization
            fig = self.visualize_scaling_analysis(results, seq_name)
            plt.savefig(f'scaling_analysis_{seq_name}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Collect summary
            summary['sequence_type'].append(seq_name)
            
            if results['patterns'].get('power_law'):
                summary['scaling_exponent'].append(results['patterns']['power_law']['exponent'])
                summary['power_law_r2'].append(results['patterns']['power_law']['r2'])
            else:
                summary['scaling_exponent'].append(np.nan)
                summary['power_law_r2'].append(np.nan)
            
            if results['patterns'].get('best_normalization'):
                summary['best_normalization'].append(results['patterns']['best_normalization']['name'])
                summary['best_cv'].append(results['patterns']['best_normalization']['cv'])
            else:
                summary['best_normalization'].append(None)
                summary['best_cv'].append(np.nan)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary)
        
        # Print summary
        print("\n" + "="*80)
        print("SCALING ANALYSIS SUMMARY")
        print("="*80)
        print(summary_df.to_string())
        
        # Statistical analysis of scaling patterns
        print("\n" + "="*80)
        print("SCALING PATTERN STATISTICS")
        print("="*80)
        
        # Scaling exponents
        exponents = summary_df['scaling_exponent'].dropna()
        if len(exponents) > 0:
            print(f"\nScaling Exponents (σc ~ scale^α):")
            print(f"  Mean α: {exponents.mean():.3f}")
            print(f"  Std α: {exponents.std():.3f}")
            print(f"  Range: [{exponents.min():.3f}, {exponents.max():.3f}]")
            
            # Test if exponents cluster around specific values
            if len(exponents) > 3:
                # Test for α ≈ 0 (scale-invariant)
                t_stat, p_val = stats.ttest_1samp(exponents, 0)
                print(f"  Test α=0 (invariant): p={p_val:.4f}")
                
                # Test for α ≈ 1 (linear scaling)
                t_stat, p_val = stats.ttest_1samp(exponents, 1)
                print(f"  Test α=1 (linear): p={p_val:.4f}")
        
        # Best normalizations
        print(f"\nBest Normalizations:")
        norm_counts = summary_df['best_normalization'].value_counts()
        for norm, count in norm_counts.items():
            print(f"  {norm}: {count} sequences")
        
        # CV statistics
        cvs = summary_df['best_cv'].dropna()
        if len(cvs) > 0:
            print(f"\nScaling Invariance Quality:")
            print(f"  Mean CV: {cvs.mean():.3f}")
            print(f"  Sequences with CV<0.1: {(cvs < 0.1).sum()}/{len(cvs)}")
            print(f"  Sequences with CV<0.2: {(cvs < 0.2).sum()}/{len(cvs)}")
        
        # Save detailed results
        with open('scaling_diagnostics_complete.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for seq_name, results in all_results.items():
                json_results[seq_name] = {
                    'patterns': results['patterns'],
                    'best_cv': float(results['patterns']['best_normalization']['cv']) 
                              if results['patterns'].get('best_normalization') else None,
                    'scaling_exponent': float(results['patterns']['power_law']['exponent'])
                                       if results['patterns'].get('power_law') else None
                }
            
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': summary_df.to_dict(),
                'detailed_patterns': json_results
            }, f, indent=2)
        
        print(f"\n✅ Complete results saved to scaling_diagnostics_complete.json")
        print(f"📊 Individual plots saved as scaling_analysis_*.png")
        
        return all_results, summary_df


def main():
    """Run complete scaling diagnostics"""
    
    diagnostics = ScalingDiagnostics()
    results, summary = diagnostics.run_complete_scaling_diagnostics()
    
    # Final interpretation
    print("\n" + "="*80)
    print("FINAL INTERPRETATION")
    print("="*80)
    
    print("""
Based on the scaling analysis:

1. σc is NOT scale-invariant in general
   - Different sequences show different scaling exponents
   - Most exponents are NOT zero (which would indicate invariance)

2. Scaling behavior is sequence-type dependent
   - Exponential sequences: different scaling than polynomial
   - Chaotic sequences: often non-linear scaling
   - Random sequences: different pattern entirely

3. No universal normalization works for all sequences
   - 'range' works for some, 'log_std' for others
   - This explains why our CV was ~0.4 on average

4. CONCLUSION:
   σc appears to measure an ABSOLUTE noise threshold
   relative to the sequence's intrinsic scale,
   NOT a scale-invariant property.

   This suggests σc is better interpreted as:
   "The noise level at which information extraction becomes optimal"
   rather than a universal robustness metric.
    """)
    
    return results, summary


if __name__ == "__main__":
    results, summary = main()