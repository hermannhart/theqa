#!/usr/bin/env python3
"""
Critical Test: Is σc a Universal Information Phenomenon or Quantum-Specific?
=============================================================================
Testing if mathematical sequences and quantum systems share the same
underlying transition mechanism or if it's coincidental similarity.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats, optimize
from scipy.linalg import expm, logm
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TransitionCharacteristics:
    """Complete characterization of the transition"""
    sigma_c: float
    order_parameter: np.ndarray
    susceptibility: np.ndarray
    correlation_length: np.ndarray
    scaling_exponent: float
    universality_class: str
    fisher_information: float
    entanglement_analog: float

class UniversalityTest:
    """
    Test whether σc represents a universal information transition
    or is specific to quantum decoherence
    """
    
    def __init__(self):
        self.results = {}
        
    def generate_test_battery(self) -> Dict[str, np.ndarray]:
        """
        Generate three categories of systems to test universality
        """
        systems = {}
        
        # CATEGORY 1: Pure Mathematical (no physical meaning)
        # These should NOT show DQCP if it's quantum-specific
        
        # Arithmetic sequences
        systems['Arithmetic'] = np.arange(1, 101).astype(float)
        systems['Geometric'] = (2.0**np.arange(20))
        
        # Number theoretic
        def primes(n):
            sieve = [True] * n
            for i in range(2, int(n**0.5) + 1):
                if sieve[i]:
                    sieve[i*i::i] = [False] * len(sieve[i*i::i])
            prime_list = [i for i in range(2, n) if sieve[i]][:100]
            return np.array(prime_list, dtype=float)
        
        systems['Primes'] = primes(1000)
        
        # Combinatorial
        def fibonacci(n):
            fib = [1.0, 1.0]
            for _ in range(n-2):
                fib.append(fib[-1] + fib[-2])
            fib_array = np.array(fib)
            return fib_array / np.max(fib_array)  # Normalize to prevent overflow
        
        systems['Fibonacci'] = fibonacci(100)
        
        # CATEGORY 2: Classical Physical (deterministic dynamics)
        # These might show transitions but NOT quantum universality
        
        # Logistic map at different parameters
        def logistic_map(r, n=100):
            x = 0.5
            trajectory = []
            for _ in range(500):  # Skip transient
                x = r * x * (1 - x)
            for _ in range(n):
                x = r * x * (1 - x)
                trajectory.append(x)
            return np.array(trajectory)
        
        systems['Logistic_Periodic'] = logistic_map(3.2)
        systems['Logistic_Chaos'] = logistic_map(3.9)
        
        # Classical harmonic oscillator
        t = np.linspace(0, 10*np.pi, 100)
        systems['Harmonic'] = np.sin(t) * np.exp(-t/20)  # Damped
        
        # CATEGORY 3: Quantum-Inspired (should show DQCP-like behavior)
        # These SHOULD show DQCP characteristics if theory is correct
        
        # Simulated quantum walk
        def quantum_walk(steps=100):
            # Simplified 1D quantum walk
            pos = np.zeros(2*steps + 1)
            pos[steps] = 1.0  # Start in center
            
            for _ in range(steps):
                # Hadamard evolution (simplified)
                new_pos = np.zeros_like(pos)
                for i in range(1, len(pos)-1):
                    new_pos[i-1] += pos[i] / np.sqrt(2)
                    new_pos[i+1] += pos[i] / np.sqrt(2)
                pos = new_pos
                
            return pos[steps-50:steps+50] if steps >= 50 else pos  # Center 100 points
        
        systems['Quantum_Walk'] = quantum_walk()
        
        # Simulated entangled state
        def entangled_state(n=100):
            # Create pseudo-entangled correlations
            base = np.random.random(n//2)
            entangled = np.zeros(n)
            entangled[::2] = base
            entangled[1::2] = 1 - base  # Anti-correlated pairs
            return entangled
        
        systems['Entangled'] = entangled_state()
        
        # CATEGORY 4: Stochastic (random processes)
        # Control group - should show different behavior
        
        np.random.seed(42)
        systems['Gaussian_Noise'] = np.random.normal(0, 1, 100)
        systems['Brownian'] = np.cumsum(np.random.normal(0, 0.1, 100))
        
        return systems
    
    def measure_order_parameter(self, system: np.ndarray, sigma: float) -> float:
        """
        Universal order parameter: information coherence
        Not quantum-specific
        """
        if len(system) < 10:
            return 0.0
            
        # Ensure float arrays
        system = np.asarray(system, dtype=float)
        noisy = system + np.random.normal(0, sigma, len(system))
        
        # Coherence measure: correlation between original and noisy
        if np.std(system) > 0 and np.std(noisy) > 0:
            # Ensure both are 1D arrays
            system_flat = system.flatten()
            noisy_flat = noisy.flatten()
            
            # Check they have same length
            min_len = min(len(system_flat), len(noisy_flat))
            if min_len > 1:
                correlation = np.corrcoef(system_flat[:min_len], 
                                        noisy_flat[:min_len])[0, 1]
            else:
                correlation = 0.0
        else:
            correlation = 0.0
            
        # Handle NaN
        if np.isnan(correlation):
            correlation = 0.0
            
        # Information overlap
        def entropy(x):
            hist, _ = np.histogram(x, bins=10, density=True)
            hist = hist[hist > 0]
            if len(hist) > 0:
                return -np.sum(hist * np.log(hist + 1e-10))
            return 0.0
        
        entropy_loss = abs(entropy(system) - entropy(noisy))
        
        # Combined order parameter
        order = correlation * np.exp(-entropy_loss)
        
        return float(order)
    
    def measure_susceptibility(self, system: np.ndarray, sigma: float) -> float:
        """
        Response to perturbation - universal quantity
        """
        epsilon = sigma * 0.01 if sigma > 0 else 0.001
        
        order_plus = self.measure_order_parameter(system, sigma + epsilon)
        order_minus = self.measure_order_parameter(system, max(sigma - epsilon, 0))
        
        susceptibility = abs(order_plus - order_minus) / (2 * epsilon)
        
        return float(susceptibility)
    
    def measure_correlation_length(self, system: np.ndarray, sigma: float) -> float:
        """
        Spatial/temporal correlation scale
        """
        system = np.asarray(system, dtype=float)
        noisy = system + np.random.normal(0, sigma, len(system))
        
        # Autocorrelation function
        correlations = []
        for lag in range(1, min(len(noisy)//2, 20)):
            if lag < len(noisy):
                try:
                    corr = np.corrcoef(noisy[:-lag], noisy[lag:])[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                    correlations.append(abs(corr))
                    
                    if abs(corr) < 1/np.e:
                        return float(lag)
                except:
                    correlations.append(0.0)
        
        return float(len(correlations)) if correlations else 1.0
    
    def measure_fisher_information(self, system: np.ndarray, sigma: float) -> float:
        """
        Fisher information about the noise parameter
        Fundamental quantity in information theory
        """
        if sigma <= 0:
            return 0.0
            
        system = np.asarray(system, dtype=float)
        n_samples = 50
        log_likelihoods = []
        
        for _ in range(n_samples):
            noisy = system + np.random.normal(0, sigma, len(system))
            # Simplified: use variance as sufficient statistic
            var_estimate = np.var(noisy - system)
            
            # Log likelihood derivative
            dll = (var_estimate - sigma**2) / sigma**3
            log_likelihoods.append(dll)
        
        # Fisher information is expected value of (dll)^2
        fisher = np.mean(np.array(log_likelihoods)**2) if log_likelihoods else 0.0
        
        return float(fisher)
    
    def measure_entanglement_analog(self, system: np.ndarray, sigma: float) -> float:
        """
        Classical analog of entanglement: mutual information between halves
        """
        if len(system) < 4:
            return 0.0
            
        system = np.asarray(system, dtype=float)
        noisy = system + np.random.normal(0, sigma, len(system))
        
        # Split system in half
        mid = len(noisy) // 2
        first_half = noisy[:mid]
        second_half = noisy[mid:2*mid] if len(noisy) >= 2*mid else noisy[mid:]
        
        # Mutual information (simplified)
        if len(first_half) == len(second_half) and len(first_half) > 1:
            try:
                correlation = np.corrcoef(first_half, second_half)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                    
                if abs(correlation) < 1:
                    mi = -0.5 * np.log(1 - correlation**2)
                else:
                    mi = 0.0
            except:
                mi = 0.0
        else:
            mi = 0.0
            
        return float(mi)
    
    def find_transition(self, system: np.ndarray) -> TransitionCharacteristics:
        """
        Complete characterization of the transition
        """
        system = np.asarray(system, dtype=float)
        
        # Determine noise range
        system_std = np.std(system)
        if system_std == 0:
            system_std = np.mean(np.abs(system)) + 1
            
        noise_levels = np.logspace(np.log10(system_std/1000), 
                                  np.log10(system_std*10), 50)
        
        # Measure all quantities
        order_params = []
        susceptibilities = []
        correlations = []
        fisher_infos = []
        entanglements = []
        
        for sigma in noise_levels:
            order_params.append(self.measure_order_parameter(system, sigma))
            susceptibilities.append(self.measure_susceptibility(system, sigma))
            correlations.append(self.measure_correlation_length(system, sigma))
            fisher_infos.append(self.measure_fisher_information(system, sigma))
            entanglements.append(self.measure_entanglement_analog(system, sigma))
        
        # Find critical point (maximum susceptibility)
        max_idx = np.argmax(susceptibilities)
        sigma_c = noise_levels[max_idx]
        
        # Determine scaling exponent
        scaling = self.measure_scaling_exponent(system, sigma_c)
        
        # Classify universality
        if scaling < 0.1:
            universality = "Non-scaling (deterministic)"
        elif 0.4 < scaling < 0.6:
            universality = "Mean-field (ν=1/2)"
        elif 0.6 < scaling < 0.7:
            universality = "3D Ising-like"
        elif scaling > 0.9:
            universality = "2D Ising-like"
        else:
            universality = f"Unknown (ν={scaling:.2f})"
        
        return TransitionCharacteristics(
            sigma_c=float(sigma_c),
            order_parameter=np.array(order_params),
            susceptibility=np.array(susceptibilities),
            correlation_length=np.array(correlations),
            scaling_exponent=float(scaling),
            universality_class=universality,
            fisher_information=float(fisher_infos[max_idx]),
            entanglement_analog=float(entanglements[max_idx])
        )
    
    def measure_scaling_exponent(self, system: np.ndarray, sigma_c: float) -> float:
        """
        Measure critical exponent ν
        """
        system = np.asarray(system, dtype=float)
        
        # Test different system sizes (subsampling)
        sizes = [25, 50, 75, len(system)]
        sigma_cs = []
        
        for size in sizes:
            if size <= len(system) and size >= 10:
                subsystem = system[:size]
                # Quick estimate of σc for this size
                mini_transition = self.find_transition_quick(subsystem)
                sigma_cs.append(mini_transition)
        
        # Fit power law: σc ~ L^ν
        if len(sigma_cs) > 2:
            sigma_cs = np.array(sigma_cs)
            valid = ~np.isnan(sigma_cs) & (sigma_cs > 0)
            if np.sum(valid) > 2:
                log_sizes = np.log(np.array(sizes)[valid])
                log_sigmas = np.log(sigma_cs[valid])
                nu, _ = np.polyfit(log_sizes, log_sigmas, 1)
                return abs(float(nu))
        
        return 0.5  # Default
    
    def find_transition_quick(self, system: np.ndarray) -> float:
        """Quick σc estimate for scaling analysis"""
        system = np.asarray(system, dtype=float)
        system_std = np.std(system)
        if system_std == 0:
            system_std = 1
            
        # Just test 10 points
        noise_levels = np.logspace(np.log10(system_std/100), 
                                  np.log10(system_std*10), 10)
        
        susceptibilities = []
        for sigma in noise_levels:
            susc = self.measure_susceptibility(system, sigma)
            susceptibilities.append(susc)
        
        max_idx = np.argmax(susceptibilities)
        return float(noise_levels[max_idx])
    
    def analyze_universality(self) -> pd.DataFrame:
        """
        Main analysis: test all systems and compare
        """
        systems = self.generate_test_battery()
        results = []
        
        print("Testing Universality Hypothesis")
        print("="*60)
        
        for name, system in tqdm(systems.items(), desc="Analyzing"):
            transition = self.find_transition(system)
            
            # Determine category
            if name in ['Arithmetic', 'Geometric', 'Primes', 'Fibonacci']:
                category = 'Mathematical'
            elif name in ['Logistic_Periodic', 'Logistic_Chaos', 'Harmonic']:
                category = 'Classical'
            elif name in ['Quantum_Walk', 'Entangled']:
                category = 'Quantum'
            else:
                category = 'Stochastic'
            
            results.append({
                'System': name,
                'Category': category,
                'σc': transition.sigma_c,
                'Scaling_ν': transition.scaling_exponent,
                'Universality': transition.universality_class,
                'Max_Susceptibility': float(np.max(transition.susceptibility)),
                'Fisher_Info': transition.fisher_information,
                'Entanglement_Analog': transition.entanglement_analog,
                'Correlation_Length': float(np.max(transition.correlation_length))
            })
        
        return pd.DataFrame(results)
    
    def plot_phase_diagrams(self, results_df: pd.DataFrame):
        """
        Create phase diagrams to visualize universality
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. σc by category
        ax = axes[0, 0]
        categories = results_df['Category'].unique()
        colors = {'Mathematical': 'blue', 'Classical': 'green', 
                 'Quantum': 'red', 'Stochastic': 'orange'}
        
        for cat in categories:
            data = results_df[results_df['Category'] == cat]
            x = np.random.normal(list(categories).index(cat), 0.1, len(data))
            ax.scatter(x, data['σc'], label=cat, alpha=0.6, s=100, 
                      color=colors.get(cat, 'gray'))
        
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories)
        ax.set_ylabel('σc')
        ax.set_title('Critical Noise by System Category')
        ax.legend()
        ax.set_yscale('log')
        
        # 2. Scaling exponents
        ax = axes[0, 1]
        for cat in categories:
            data = results_df[results_df['Category'] == cat]
            ax.scatter(data['Scaling_ν'], data['Max_Susceptibility'], 
                      label=cat, alpha=0.6, s=100, color=colors.get(cat, 'gray'))
        
        ax.set_xlabel('Scaling Exponent ν')
        ax.set_ylabel('Max Susceptibility')
        ax.set_title('Universality Classes')
        
        # Add theoretical values
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Mean-field')
        ax.axvline(0.63, color='blue', linestyle='--', alpha=0.5, label='3D Ising')
        ax.legend()
        
        # 3. Fisher Information vs Entanglement
        ax = axes[1, 0]
        for cat in categories:
            data = results_df[results_df['Category'] == cat]
            ax.scatter(data['Fisher_Info'], data['Entanglement_Analog'], 
                      label=cat, alpha=0.6, s=100, color=colors.get(cat, 'gray'))
        
        ax.set_xlabel('Fisher Information')
        ax.set_ylabel('Entanglement Analog')
        ax.set_title('Information Theoretic Measures')
        ax.legend()
        
        # 4. Phase diagram
        ax = axes[1, 1]
        scatter = ax.scatter(results_df['σc'], results_df['Correlation_Length'],
                           c=results_df['Scaling_ν'], cmap='viridis', s=100)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Scaling Exponent ν')
        
        ax.set_xlabel('σc')
        ax.set_ylabel('Max Correlation Length')
        ax.set_title('Phase Diagram')
        ax.set_xscale('log')
        
        plt.suptitle('Universality Analysis: Mathematical vs Physical Systems', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def statistical_tests(self, results_df: pd.DataFrame):
        """
        Statistical tests for universality
        """
        print("\n" + "="*60)
        print("STATISTICAL TESTS FOR UNIVERSALITY")
        print("="*60)
        
        # Test 1: Do different categories have different σc distributions?
        categories = results_df['Category'].unique()
        
        print("\n1. ANOVA for σc across categories:")
        groups = [results_df[results_df['Category'] == cat]['σc'].values 
                 for cat in categories]
        
        # Filter out empty groups
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) > 1:
            f_stat, p_value = stats.f_oneway(*groups)
            print(f"   F-statistic: {f_stat:.3f}, p-value: {p_value:.4f}")
        else:
            p_value = 1.0
            print("   Not enough groups for ANOVA")
        
        # Test 2: Do mathematical and quantum systems share universality classes?
        print("\n2. Universality class distribution:")
        contingency = pd.crosstab(results_df['Category'], 
                                  results_df['Universality'])
        print(contingency)
        
        if contingency.shape[0] > 1 and contingency.shape[1] > 1:
            chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
            print(f"   Chi-square test: χ² = {chi2:.3f}, p = {p_chi:.4f}")
        else:
            p_chi = None
            print("   Not enough data for chi-square test")
        
        # Test 3: Correlation between measures
        print("\n3. Correlation between information measures:")
        corr_matrix = results_df[['Fisher_Info', 'Entanglement_Analog', 
                                  'Max_Susceptibility']].corr()
        print(corr_matrix)
        
        # Test 4: Do mathematical sequences show quantum-like scaling?
        math_systems = results_df[results_df['Category'] == 'Mathematical']
        quantum_systems = results_df[results_df['Category'] == 'Quantum']
        
        p_val = None
        if len(quantum_systems) > 0 and len(math_systems) > 0:
            print("\n4. Comparing Mathematical vs Quantum scaling:")
            math_scaling = math_systems['Scaling_ν'].dropna()
            quantum_scaling = quantum_systems['Scaling_ν'].dropna()
            
            if len(math_scaling) > 0 and len(quantum_scaling) > 0:
                u_stat, p_val = stats.mannwhitneyu(math_scaling, quantum_scaling)
                print(f"   Mann-Whitney U: {u_stat:.3f}, p = {p_val:.4f}")
        
        return {
            'anova_p': p_value,
            'chi2_p': p_chi,
            'math_quantum_p': p_val
        }

def main():
    """
    Main experimental test
    """
    print("="*60)
    print("TESTING: Is σc Universal or Quantum-Specific?")
    print("="*60)
    print()
    
    tester = UniversalityTest()
    
    # Run analysis
    results_df = tester.analyze_universality()
    
    # Display results
    print("\nRESULTS:")
    print(results_df.to_string())
    
    # Statistical tests
    test_results = tester.statistical_tests(results_df)
    
    # Visualizations
    fig = tester.plot_phase_diagrams(results_df)
    plt.savefig('universality_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save results
    results_df.to_csv('universality_results.csv', index=False)
    
    # INTERPRETATION
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    if test_results['anova_p'] and test_results['anova_p'] < 0.05:
        print("✗ Different categories have significantly different σc")
        print("  → Suggests category-specific phenomena")
    else:
        print("✓ No significant difference in σc across categories")
        print("  → Suggests universal phenomenon")
    
    if test_results['chi2_p'] and test_results['chi2_p'] < 0.05:
        print("✗ Different universality class distributions")
        print("  → Categories follow different critical behavior")
    else:
        print("✓ Similar universality class distributions")
        print("  → Universal critical behavior")
    
    if test_results['math_quantum_p'] and test_results['math_quantum_p'] > 0.05:
        print("✓ Mathematical and Quantum systems show similar scaling")
        print("  → SUPPORTS universal information transition!")
    else:
        print("✗ Different scaling for Math vs Quantum")
        print("  → Suggests different mechanisms")
    
    # Final verdict
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    # Count evidence
    evidence_universal = sum([
        test_results['anova_p'] > 0.05 if test_results['anova_p'] else False,
        test_results['chi2_p'] > 0.05 if test_results['chi2_p'] else False,
        test_results['math_quantum_p'] > 0.05 if test_results['math_quantum_p'] else False
    ])
    
    if evidence_universal >= 2:
        print("""
        EVIDENCE SUPPORTS UNIVERSAL PHENOMENON:
        
        σc appears to mark a universal information transition that occurs
        whenever coherent structure meets incoherent noise, regardless of
        whether the system is quantum, classical, or purely mathematical.
        
        This suggests a fundamental principle of information processing
        under noise that transcends specific physical implementations.
        """)
    else:
        print("""
        EVIDENCE SUGGESTS CATEGORY-SPECIFIC PHENOMENA:
        
        While all systems show transitions, the mechanisms appear different
        for mathematical vs physical systems. σc might be measuring different
        things in different contexts.
        
        More investigation needed to determine if there's an underlying
        universal principle or just superficial similarity.
        """)
    
    return results_df

if __name__ == "__main__":
    results = main()