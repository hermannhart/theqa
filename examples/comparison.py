"""
Compare different dynamical systems using TheQA.

This script demonstrates:
1. Side-by-side comparison of multiple systems
2. Classification into universality classes
3. Visualization of the σc landscape
4. Statistical analysis of system properties
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from theqa import (
    compute_sigma_c, TripleRule,
    # Systems
    CollatzSystem, FibonacciSystem, LogisticMap, TentMap,
    JosephusSystem, PrimeGapSystem, CellularAutomaton,
    # Features and criteria
    PeakCounter, EntropyCalculator, SpectralAnalyzer,
    VarianceCriterion
)


def comprehensive_system_comparison():
    """Compare all available systems."""
    print("="*60)
    print("COMPREHENSIVE SYSTEM COMPARISON")
    print("="*60)
    
    # Define systems to analyze
    systems = {
        # Ultra-sensitive (σc < 0.01)
        'Fibonacci': FibonacciSystem(n=100),
        'Lucas': FibonacciSystem(n=100, a=2, b=1),
        'Prime Gaps': PrimeGapSystem(n_primes=100),
        
        # Sensitive (0.01 < σc < 0.1)
        'Josephus(41,3)': JosephusSystem(n=41, k=3),
        'Josephus(100,7)': JosephusSystem(n=100, k=7),
        
        # Medium (0.1 < σc < 0.3)
        'Collatz(3n+1)': CollatzSystem(n=27, q=3),
        'Collatz(5n+1)': CollatzSystem(n=27, q=5),
        'Logistic(3.6)': LogisticMap(r=3.6, length=500),
        'Logistic(3.9)': LogisticMap(r=3.9, length=500),
        'Tent(1.5)': TentMap(r=1.5, length=500),
        'CA Rule 30': CellularAutomaton(rule=30, width=101, steps=100),
        'CA Rule 90': CellularAutomaton(rule=90, width=101, steps=100),
        'CA Rule 110': CellularAutomaton(rule=110, width=101, steps=100),
        
        # Robust (σc > 0.3)
        'Logistic(4.0)': LogisticMap(r=4.0, length=500),
        'Tent(2.0)': TentMap(r=2.0, length=500),
    }
    
    results = []
    
    print("\nAnalyzing systems...")
    print("-" * 60)
    print(f"{'System':<20} {'σc':<10} {'Class':<15} {'Time (s)':<10}")
    print("-" * 60)
    
    for name, system in systems.items():
        try:
            # Generate sequence
            if hasattr(system, 'generate'):
                sequence = system.generate()
            else:
                sequence = system
            
            # Skip if too short
            if len(sequence) < 10:
                print(f"{name:<20} {'SKIP':<10} {'Too short':<15}")
                continue
            
            # Compute σc
            sigma_c = compute_sigma_c(sequence, method='adaptive')
            
            # Classify
            if sigma_c < 0.01:
                class_name = "Ultra-sensitive"
            elif sigma_c < 0.1:
                class_name = "Sensitive"
            elif sigma_c < 0.3:
                class_name = "Medium"
            else:
                class_name = "Robust"
            
            results.append({
                'system': name,
                'sigma_c': sigma_c,
                'class': class_name,
                'log_sigma_c': np.log10(sigma_c) if sigma_c > 0 else -5
            })
            
            print(f"{name:<20} {sigma_c:<10.4f} {class_name:<15}")
            
        except Exception as e:
            print(f"{name:<20} {'ERROR':<10} {str(e)[:40]}")
    
    return pd.DataFrame(results)


def visualize_system_landscape(df):
    """Create visualization of the σc landscape."""
    print("\n" + "="*60)
    print("VISUALIZING SYSTEM LANDSCAPE")
    print("="*60)
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define colors for each class
    class_colors = {
        'Ultra-sensitive': '#2E86AB',
        'Sensitive': '#A23B72',
        'Medium': '#F18F01',
        'Robust': '#C73E1D'
    }
    
    # Plot 1: σc values with class coloring
    df_sorted = df.sort_values('sigma_c')
    colors = [class_colors[c] for c in df_sorted['class']]
    
    bars = ax1.bar(range(len(df_sorted)), df_sorted['sigma_c'], color=colors, alpha=0.7)
    ax1.set_xlabel('System')
    ax1.set_ylabel('Critical Threshold (σc)')
    ax1.set_title('Critical Thresholds Across Systems')
    ax1.set_xticks(range(len(df_sorted)))
    ax1.set_xticklabels(df_sorted['system'], rotation=45, ha='right')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Add class boundaries
    ax1.axhline(y=0.01, color='black', linestyle='--', alpha=0.5)
    ax1.axhline(y=0.1, color='black', linestyle='--', alpha=0.5)
    ax1.axhline(y=0.3, color='black', linestyle='--', alpha=0.5)
    ax1.axhline(y=np.pi/2, color='red', linestyle='--', alpha=0.5, label='π/2 bound')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=class_name, alpha=0.7)
                      for class_name, color in class_colors.items()]
    ax1.legend(handles=legend_elements, loc='upper left')
    
    # Plot 2: Distribution by class
    class_counts = df['class'].value_counts()
    ax2.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
            colors=[class_colors[c] for c in class_counts.index])
    ax2.set_title('Distribution of Systems by Class')
    
    plt.tight_layout()
    plt.savefig('system_landscape.png', dpi=150)
    print("\nLandscape visualization saved to 'system_landscape.png'")


def statistical_analysis(df):
    """Perform statistical analysis of system properties."""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    # Basic statistics by class
    print("\nStatistics by universality class:")
    print("-" * 50)
    
    grouped = df.groupby('class')['sigma_c'].agg(['mean', 'std', 'min', 'max', 'count'])
    print(grouped)
    
    # Test for significant differences between classes
    print("\n\nANOVA test for class differences:")
    classes = df['class'].unique()
    groups = [df[df['class'] == c]['sigma_c'].values for c in classes]
    
    # Filter out empty groups
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.4e}")
        
        if p_value < 0.05:
            print("✓ Significant differences between classes (p < 0.05)")
        else:
            print("✗ No significant differences between classes")


def fingerprint_comparison():
    """Compare system fingerprints across multiple features."""
    print("\n" + "="*60)
    print("SYSTEM FINGERPRINT COMPARISON")
    print("="*60)
    
    # Select representative systems
    systems = {
        'Fibonacci': FibonacciSystem(n=100),
        'Collatz': CollatzSystem(n=27),
        'Logistic': LogisticMap(r=3.9),
        'Josephus': JosephusSystem(n=41, k=3),
        'CA Rule 30': CellularAutomaton(rule=30)
    }
    
    # Multiple features
    features = {
        'peaks': PeakCounter(transform='log'),
        'entropy': EntropyCalculator(),
        'spectral': SpectralAnalyzer(),
    }
    
    # Compute fingerprints
    fingerprints = []
    
    for sys_name, system in systems.items():
        for feat_name, feature in features.items():
            tr = TripleRule(
                system=system,
                feature=feature,
                criterion=VarianceCriterion()
            )
            result = tr.compute(method='adaptive')
            
            fingerprints.append({
                'system': sys_name,
                'feature': feat_name,
                'sigma_c': result.sigma_c
            })
    
    # Create fingerprint matrix
    df_fp = pd.DataFrame(fingerprints)
    pivot = df_fp.pivot(index='system', columns='feature', values='sigma_c')
    
    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='coolwarm',
                cbar_kws={'label': 'σc'})
    plt.title('System Fingerprints: σc for Different Features')
    plt.tight_layout()
    plt.savefig('system_fingerprints.png', dpi=150)
    print("\nFingerprint comparison saved to 'system_fingerprints.png'")
    
    return pivot


def clustering_analysis(df):
    """Perform clustering analysis to find system groups."""
    print("\n" + "="*60)
    print("CLUSTERING ANALYSIS")
    print("="*60)
    
    # Prepare data for clustering
    # We'll use multiple system properties
    extended_data = []
    
    for _, row in df.iterrows():
        system_name = row['system']
        
        # Get the actual system
        if 'Fibonacci' in system_name:
            seq = FibonacciSystem(n=100).generate()
        elif 'Collatz' in system_name:
            seq = CollatzSystem(n=27).generate()
        elif 'Logistic' in system_name:
            seq = LogisticMap(r=3.9).generate()
        else:
            continue  # Skip for now
        
        # Compute various properties
        properties = {
            'sigma_c': row['sigma_c'],
            'log_sigma_c': row['log_sigma_c'],
            'mean': np.mean(seq),
            'std': np.std(seq),
            'range': np.max(seq) - np.min(seq),
            'trend': np.polyfit(range(len(seq)), seq, 1)[0] if len(seq) > 1 else 0
        }
        
        extended_data.append(properties)
    
    if len(extended_data) < 3:
        print("Not enough data for clustering analysis")
        return
    
    # Create DataFrame
    df_extended = pd.DataFrame(extended_data)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_extended)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1],
                         c=df_extended['sigma_c'], cmap='viridis',
                         s=100, alpha=0.7)
    plt.colorbar(scatter, label='σc')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('System Clustering in Feature Space')
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    for i, name in enumerate(df.iloc[:len(features_pca)]['system']):
        plt.annotate(name, (features_pca[i, 0], features_pca[i, 1]),
                    fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('system_clustering.png', dpi=150)
    print("\nClustering visualization saved to 'system_clustering.png'")


def quantum_comparison():
    """Compare classical vs quantum bounds."""
    print("\n" + "="*60)
    print("CLASSICAL VS QUANTUM COMPARISON")
    print("="*60)
    
    try:
        from theqa.quantum import classical_to_quantum_bound
        
        # Systems to compare
        classical_systems = {
            'Random Walk': 0.45,
            'Collatz': 0.117,
            'Logistic Chaos': 0.21,
            'Fibonacci': 0.001,
        }
        
        # Compute quantum bounds
        comparison = []
        for name, classical_sc in classical_systems.items():
            quantum_sc = classical_to_quantum_bound(classical_sc)
            enhancement = quantum_sc / classical_sc
            
            comparison.append({
                'system': name,
                'classical': classical_sc,
                'quantum': quantum_sc,
                'enhancement': enhancement
            })
        
        df_quantum = pd.DataFrame(comparison)
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar plot comparison
        x = np.arange(len(df_quantum))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, df_quantum['classical'], width,
                        label='Classical', color='blue', alpha=0.7)
        bars2 = ax1.bar(x + width/2, df_quantum['quantum'], width,
                        label='Quantum', color='red', alpha=0.7)
        
        ax1.set_xlabel('System')
        ax1.set_ylabel('Critical Threshold')
        ax1.set_title('Classical vs Quantum Critical Thresholds')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_quantum['system'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add bounds
        ax1.axhline(y=np.pi/2, color='blue', linestyle='--', alpha=0.5,
                   label='Classical bound (π/2)')
        ax1.axhline(y=np.pi, color='red', linestyle='--', alpha=0.5,
                   label='Quantum bound (π)')
        
        # Enhancement factors
        ax2.bar(df_quantum['system'], df_quantum['enhancement'],
               color='green', alpha=0.7)
        ax2.set_xlabel('System')
        ax2.set_ylabel('Quantum Enhancement Factor')
        ax2.set_title('Quantum Advantage in Critical Thresholds')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('quantum_comparison.png', dpi=150)
        print("\nQuantum comparison saved to 'quantum_comparison.png'")
        
        # Print results
        print("\nQuantum enhancement factors:")
        print("-" * 40)
        for _, row in df_quantum.iterrows():
            print(f"{row['system']:<15} {row['enhancement']:.2f}x")
            
    except ImportError:
        print("Quantum module not available. Skipping quantum comparison.")


def main():
    """Run complete system comparison analysis."""
    print("\n" + "="*60)
    print("TheQA SYSTEM COMPARISON ANALYSIS")
    print("="*60)
    
    # 1. Comprehensive comparison
    df = comprehensive_system_comparison()
    
    # 2. Visualize landscape
    visualize_system_landscape(df)
    
    # 3. Statistical analysis
    statistical_analysis(df)
    
    # 4. Fingerprint comparison
    fingerprints = fingerprint_comparison()
    
    # 5. Clustering analysis
    clustering_analysis(df)
    
    # 6. Quantum comparison
    quantum_comparison()
    
    # Save results
    df.to_csv('system_comparison_results.csv', index=False)
    fingerprints.to_csv('system_fingerprints.csv')
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nResults saved to:")
    print("  - system_comparison_results.csv")
    print("  - system_fingerprints.csv")
    print("  - system_landscape.png")
    print("  - system_fingerprints.png")
    print("  - system_clustering.png")
    print("  - quantum_comparison.png")


if __name__ == "__main__":
    main()
