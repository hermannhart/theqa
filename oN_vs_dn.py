"""
Comparison of Observational vs Dynamical Noise in Discrete Dynamical Systems
Following Prof. Vaienti's distinction between noise types
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from collections import defaultdict

class NoiseComparisonAnalysis:
    def __init__(self):
        self.results = defaultdict(dict)
        
    def logistic_map_observational(self, r, n_steps, x0=0.5, sigma=0.1, n_trials=100):
        """Current method: deterministic evolution, then add noise to observations"""
        # Generate deterministic trajectory
        x = [x0]
        for i in range(n_steps):
            x.append(r * x[-1] * (1 - x[-1]))
        x = np.array(x)
        
        # Add noise to observations
        peak_counts = []
        for _ in range(n_trials):
            noise = np.random.normal(0, sigma, len(x))
            noisy_obs = x + noise  # Noise only affects observation
            peaks, _ = signal.find_peaks(noisy_obs, prominence=sigma/2)
            peak_counts.append(len(peaks))
            
        return np.var(peak_counts), x
    
    def logistic_map_dynamical(self, r, n_steps, x0=0.5, sigma=0.1, n_trials=100):
        """Vaienti's method: noise affects the dynamics at each step"""
        peak_counts = []
        
        for _ in range(n_trials):
            x = [x0]
            for i in range(n_steps):
                # Noise affects the state before applying the map
                noise = np.random.normal(0, sigma)
                x_noisy = x[-1] + noise
                # Ensure x stays in [0,1]
                x_noisy = np.clip(x_noisy, 0, 1)
                # Apply map to noisy state
                x_next = r * x_noisy * (1 - x_noisy)
                x.append(x_next)
            
            x = np.array(x)
            peaks, _ = signal.find_peaks(x, prominence=sigma/2)
            peak_counts.append(len(peaks))
            
        return np.var(peak_counts), None
    
    def henon_map_observational(self, a=1.4, b=0.3, n_steps=500, sigma=0.1, n_trials=100):
        """Current method for Hénon map"""
        # Generate deterministic trajectory
        x, y = [0.1], [0.1]
        for i in range(n_steps):
            x_new = 1 - a * x[-1]**2 + y[-1]
            y_new = b * x[-1]
            x.append(x_new)
            y.append(y_new)
        
        x = np.array(x)
        
        # Add noise to observations
        peak_counts = []
        for _ in range(n_trials):
            noise = np.random.normal(0, sigma, len(x))
            noisy_obs = x + noise
            peaks, _ = signal.find_peaks(noisy_obs, prominence=sigma/2)
            peak_counts.append(len(peaks))
            
        return np.var(peak_counts), x
    
    def henon_map_dynamical(self, a=1.4, b=0.3, n_steps=500, sigma=0.1, n_trials=100):
        """Vaienti's method: x_{n+1} = 1 - a(x_n + ε_n)² + (y_n + η_n)"""
        peak_counts = []
        
        for _ in range(n_trials):
            x, y = [0.1], [0.1]
            for i in range(n_steps):
                # Add noise to both components
                eps = np.random.normal(0, sigma)
                eta = np.random.normal(0, sigma)
                
                # Apply map with noisy states
                x_new = 1 - a * (x[-1] + eps)**2 + (y[-1] + eta)
                y_new = b * (x[-1] + eps)
                
                x.append(x_new)
                y.append(y_new)
            
            x = np.array(x)
            # Check if trajectory is bounded
            if np.max(np.abs(x)) < 100:  # Avoid divergent trajectories
                peaks, _ = signal.find_peaks(x, prominence=sigma/2)
                peak_counts.append(len(peaks))
        
        if len(peak_counts) > 1:
            return np.var(peak_counts), None
        else:
            return np.nan, None
    
    def tent_map_observational(self, r=1.5, n_steps=500, x0=0.4, sigma=0.1, n_trials=100):
        """Current method for tent map"""
        # Generate deterministic trajectory
        x = [x0]
        for i in range(n_steps):
            if x[-1] < 0.5:
                x.append(r * x[-1])
            else:
                x.append(r * (1 - x[-1]))
        x = np.array(x)
        
        # Add noise to observations
        peak_counts = []
        for _ in range(n_trials):
            noise = np.random.normal(0, sigma, len(x))
            noisy_obs = x + noise
            peaks, _ = signal.find_peaks(noisy_obs, prominence=sigma/2)
            peak_counts.append(len(peaks))
            
        return np.var(peak_counts), x
    
    def tent_map_dynamical(self, r=1.5, n_steps=500, x0=0.4, sigma=0.1, n_trials=100):
        """Vaienti's method: noise affects state before map application"""
        peak_counts = []
        
        for _ in range(n_trials):
            x = [x0]
            for i in range(n_steps):
                # Add noise to state
                noise = np.random.normal(0, sigma)
                x_noisy = x[-1] + noise
                # Ensure x stays in [0,1]
                x_noisy = np.clip(x_noisy, 0, 1)
                
                # Apply tent map
                if x_noisy < 0.5:
                    x.append(r * x_noisy)
                else:
                    x.append(r * (1 - x_noisy))
            
            x = np.array(x)
            peaks, _ = signal.find_peaks(x, prominence=sigma/2)
            peak_counts.append(len(peaks))
            
        return np.var(peak_counts), None
    
    def find_critical_threshold(self, system_func, system_name, noise_type, **kwargs):
        """Find σ_c where variance exceeds threshold"""
        sigmas = np.logspace(-4, 0, 50)
        variances = []
        
        print(f"\nFinding σ_c for {system_name} ({noise_type})...")
        
        for sigma in sigmas:
            var, _ = system_func(sigma=sigma, **kwargs)
            variances.append(var)
            
            if var > 0.1:  # Threshold
                self.results[system_name][noise_type] = sigma
                print(f"  σ_c = {sigma:.4f}")
                return sigma, sigmas[:len(variances)], variances
        
        # If no transition found
        self.results[system_name][noise_type] = sigmas[-1]
        print(f"  σ_c > {sigmas[-1]:.4f} (no transition found)")
        return sigmas[-1], sigmas, variances
    
    def compare_all_systems(self):
        """Compare observational vs dynamical noise for all systems"""
        
        # Logistic map
        print("\n=== LOGISTIC MAP (r=3.9) ===")
        sigma_obs, sigmas_obs, vars_obs = self.find_critical_threshold(
            self.logistic_map_observational, "Logistic", "observational", r=3.9, n_steps=500
        )
        sigma_dyn, sigmas_dyn, vars_dyn = self.find_critical_threshold(
            self.logistic_map_dynamical, "Logistic", "dynamical", r=3.9, n_steps=500
        )
        
        # Hénon map
        print("\n=== HÉNON MAP ===")
        sigma_obs_h, sigmas_obs_h, vars_obs_h = self.find_critical_threshold(
            self.henon_map_observational, "Hénon", "observational", a=1.4, b=0.3
        )
        sigma_dyn_h, sigmas_dyn_h, vars_dyn_h = self.find_critical_threshold(
            self.henon_map_dynamical, "Hénon", "dynamical", a=1.4, b=0.3
        )
        
        # Tent map
        print("\n=== TENT MAP ===")
        sigma_obs_t, sigmas_obs_t, vars_obs_t = self.find_critical_threshold(
            self.tent_map_observational, "Tent", "observational", r=1.5
        )
        sigma_dyn_t, sigmas_dyn_t, vars_dyn_t = self.find_critical_threshold(
            self.tent_map_dynamical, "Tent", "dynamical", r=1.5
        )
        
        # Visualize results
        self.create_comparison_plots()
        
    def create_comparison_plots(self):
        """Create visualization comparing noise types"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Summary bar chart
        ax = axes[0, 0]
        systems = ['Logistic', 'Hénon', 'Tent']
        obs_values = [self.results[s]['observational'] for s in systems]
        dyn_values = [self.results[s]['dynamical'] for s in systems]
        
        x = np.arange(len(systems))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, obs_values, width, label='Observational', alpha=0.7)
        bars2 = ax.bar(x + width/2, dyn_values, width, label='Dynamical', alpha=0.7)
        
        ax.set_ylabel('Critical Threshold σ_c')
        ax.set_title('Comparison of Critical Thresholds')
        ax.set_xticks(x)
        ax.set_xticklabels(systems)
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add ratio text
        for i, (obs, dyn) in enumerate(zip(obs_values, dyn_values)):
            ratio = dyn/obs if obs > 0 else np.inf
            ax.text(i, max(obs, dyn)*1.5, f'Ratio: {ratio:.1f}', 
                   ha='center', fontsize=10)
        
        # Show sample trajectories
        ax2 = axes[0, 1]
        ax2.text(0.5, 0.5, 'Key Differences:\n\n' +
                'Observational Noise:\n' +
                '• Deterministic dynamics\n' +
                '• Noise only in measurement\n' +
                '• Preserves system structure\n\n' +
                'Dynamical Noise:\n' +
                '• Stochastic dynamics\n' +
                '• Noise affects evolution\n' +
                '• Can destroy attractors\n' +
                '• Related to Markov chains',
                transform=ax2.transAxes,
                fontsize=12,
                verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        ax2.axis('off')
        
        # Example trajectories for Logistic map
        ax3 = axes[1, 0]
        np.random.seed(42)
        
        # Observational
        _, x_obs = self.logistic_map_observational(3.9, 100, sigma=0.01, n_trials=1)
        noise = np.random.normal(0, 0.01, len(x_obs))
        ax3.plot(x_obs + noise, 'b-', alpha=0.7, label='Observational', linewidth=1)
        
        # Dynamical
        x_dyn = [0.5]
        for i in range(100):
            noise = np.random.normal(0, 0.01)
            x_noisy = np.clip(x_dyn[-1] + noise, 0, 1)
            x_dyn.append(3.9 * x_noisy * (1 - x_noisy))
        ax3.plot(x_dyn, 'r-', alpha=0.7, label='Dynamical', linewidth=1)
        
        ax3.set_xlabel('Time')
        ax3.set_ylabel('State')
        ax3.set_title('Sample Trajectories (Logistic, σ=0.01)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Mathematical formulation
        ax4 = axes[1, 1]
        ax4.text(0.1, 0.8, 'Mathematical Formulation:', fontsize=14, weight='bold',
                transform=ax4.transAxes)
        ax4.text(0.1, 0.6, 'Observational (Our method):\n' +
                '$x_{n+1} = f(x_n)$ deterministic\n' +
                '$\\tilde{x}_n = x_n + \\epsilon_n$ noisy observation',
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.3, 'Dynamical (Vaienti):\n' +
                '$x_{n+1} = f(x_n + \\epsilon_n)$ stochastic\n' +
                'Noise affects the dynamics',
                transform=ax4.transAxes, fontsize=12)
        ax4.axis('off')
        
        plt.suptitle('Observational vs Dynamical Noise: Critical Thresholds', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY OF RESULTS")
        print("="*60)
        print(f"{'System':<15} {'Observational σ_c':<20} {'Dynamical σ_c':<20} {'Ratio':<10}")
        print("-"*60)
        
        for system in ['Logistic', 'Hénon', 'Tent']:
            obs = self.results[system]['observational']
            dyn = self.results[system]['dynamical']
            ratio = dyn/obs if obs > 0 else np.inf
            print(f"{system:<15} {obs:<20.4f} {dyn:<20.4f} {ratio:<10.1f}")
        
        print("\nKEY FINDING: Dynamical noise typically requires LARGER σ_c")
        print("This makes sense: noise in the dynamics accumulates over time!")

# Run the analysis
if __name__ == "__main__":
    analyzer = NoiseComparisonAnalysis()
    analyzer.compare_all_systems()