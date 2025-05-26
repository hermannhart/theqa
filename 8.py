"""
Systematischer Test: Welche Art von Rauschen funktioniert am besten mit Collatz?
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.optimize import curve_fit
import time

class NoiseTypeComparison:
    """Teste verschiedene fundamentale Rauschtypen auf Collatz-Sequenzen"""
    
    def __init__(self):
        self.results = {}
        self.kb = 1.380649e-23
        self.h = 6.62607015e-34
        
    def collatz_sequence(self, n):
        """Generiere Collatz-Sequenz"""
        seq = []
        while n != 1:
            seq.append(n)
            n = n // 2 if n % 2 == 0 else 3 * n + 1
        seq.append(1)
        return np.array(seq, dtype=float)
    
    # === VERSCHIEDENE RAUSCHGENERATOREN ===
    
    def white_noise(self, size, amplitude=1.0):
        """Weißes Rauschen (Gaußverteilt)"""
        return np.random.normal(0, amplitude, size)
    
    def pink_noise(self, size, amplitude=1.0):
        """1/f Rauschen (Rosa Rauschen)"""
        # Generiere im Frequenzbereich
        freqs = np.fft.fftfreq(size, 1.0)
        
        # 1/f Spektrum
        spectrum = np.zeros(size, dtype=complex)
        for i in range(1, size//2):
            f = abs(freqs[i])
            if f > 0:
                # 1/f Amplitude mit zufälliger Phase
                amp = amplitude / np.sqrt(f)
                phase = np.random.uniform(0, 2*np.pi)
                spectrum[i] = amp * np.exp(1j * phase)
                spectrum[-i] = np.conj(spectrum[i])
        
        # Inverse FFT
        noise = np.real(np.fft.ifft(spectrum))
        return noise / np.std(noise) * amplitude
    
    def brown_noise(self, size, amplitude=1.0):
        """1/f² Rauschen (Braunes Rauschen)"""
        # Integriertes weißes Rauschen
        white = self.white_noise(size)
        brown = np.cumsum(white)
        return brown / np.std(brown) * amplitude
    
    def quantum_inspired_noise(self, size, amplitude=1.0, temperature=300):
        """Quanten-inspiriertes Rauschen mit Nullpunktfluktuationen"""
        # Basis: Weißes Rauschen
        base_noise = self.white_noise(size)
        
        # Füge "Quantencharakter" hinzu
        # Simuliere diskrete Energieniveaus
        n_levels = 10
        level_spacing = amplitude / n_levels
        
        # Quantisiere das Rauschen
        quantized = np.round(base_noise / level_spacing) * level_spacing
        
        # Füge thermische Fluktuationen hinzu
        thermal = self.white_noise(size) * np.sqrt(self.kb * temperature) * 1e10
        
        return quantized + thermal * amplitude
    
    def telegraph_noise(self, size, amplitude=1.0, switch_prob=0.01):
        """Random Telegraph Noise (RTN)"""
        noise = np.ones(size)
        state = 1
        
        for i in range(size):
            if np.random.random() < switch_prob:
                state *= -1
            noise[i] = state * amplitude
            
        return noise
    
    def levy_noise(self, size, amplitude=1.0, alpha=1.5):
        """Lévy-Flug Rauschen (heavy-tailed)"""
        # Lévy-stabile Verteilung
        # Verwendet Chambers-Mallows-Weil Methode
        u = np.random.uniform(-np.pi/2, np.pi/2, size)
        w = np.random.exponential(1.0, size)
        
        if alpha == 1:
            noise = np.tan(u)
        else:
            const = np.sin(alpha * u) / (np.cos(u) ** (1/alpha))
            noise = const * (np.cos((1-alpha) * u) / w) ** ((1-alpha)/alpha)
        
        # Clip extreme Werte
        noise = np.clip(noise, -10, 10)
        return noise * amplitude
    
    # === ANALYSE-METHODEN ===
    
    def analyze_with_noise(self, sequence, noise_func, noise_levels):
        """Analysiere Sequenz mit spezifischem Rauschtyp"""
        mi_values = []
        peak_counts_mean = []
        peak_counts_std = []
        
        for noise_level in noise_levels:
            measurements = []
            
            # Mehrere Messungen für Statistik
            for _ in range(30):
                # Generiere Rauschen
                noise = noise_func(len(sequence), noise_level)
                
                # Log-space Analyse (wie in deinem Paper)
                log_seq = np.log(sequence + 1)  # +1 um log(0) zu vermeiden
                noisy_seq = log_seq + noise
                
                # Peak Detection
                peaks, _ = signal.find_peaks(noisy_seq, prominence=noise_level/2)
                measurements.append(len(peaks))
            
            # Berechne MI-Approximation
            mean_m = np.mean(measurements)
            var_m = np.var(measurements)
            
            if var_m > 0:
                mi = mean_m / (1 + var_m)
            else:
                mi = mean_m
                
            mi_values.append(mi)
            peak_counts_mean.append(mean_m)
            peak_counts_std.append(np.std(measurements))
        
        return mi_values, peak_counts_mean, peak_counts_std
    
    def find_optimal_parameters(self, sequence, noise_func, param_name, param_range):
        """Finde optimale Parameter für spezifischen Rauschtyp"""
        best_mi = 0
        best_param = None
        
        noise_level = 0.1  # Fester Noise Level für Parametersuche
        
        for param in param_range:
            measurements = []
            
            for _ in range(20):
                # Generiere Rauschen mit Parameter
                if param_name == 'alpha':
                    noise = self.levy_noise(len(sequence), noise_level, alpha=param)
                elif param_name == 'switch_prob':
                    noise = self.telegraph_noise(len(sequence), noise_level, switch_prob=param)
                else:
                    noise = noise_func(len(sequence), noise_level)
                
                log_seq = np.log(sequence + 1)
                noisy_seq = log_seq + noise
                peaks, _ = signal.find_peaks(noisy_seq)
                measurements.append(len(peaks))
            
            mean_m = np.mean(measurements)
            var_m = np.var(measurements)
            mi = mean_m / (1 + var_m) if var_m > 0 else 0
            
            if mi > best_mi:
                best_mi = mi
                best_param = param
        
        return best_param, best_mi
    
    def run_comprehensive_test(self):
        """Führe umfassenden Test aller Rauschtypen durch"""
        print("UMFASSENDER TEST: RAUSCHTYPEN AUF COLLATZ-SEQUENZEN")
        print("="*70)
        
        # Test-Sequenz
        test_seq = self.collatz_sequence(27)
        print(f"Test-Sequenz: Collatz(27), Länge: {len(test_seq)}")
        
        # Noise Level Range (wie in deinem Paper)
        noise_levels = np.logspace(-4, 0, 50)
        
        # Definiere Rauschtypen
        noise_types = {
            'White (Gaussian)': self.white_noise,
            'Pink (1/f)': self.pink_noise,
            'Brown (1/f²)': self.brown_noise,
            'Quantum-inspired': self.quantum_inspired_noise,
            'Telegraph (RTN)': self.telegraph_noise,
            'Lévy (α=1.5)': lambda size, amp: self.levy_noise(size, amp, 1.5)
        }
        
        # Teste jeden Rauschtyp
        all_results = {}
        
        for name, noise_func in noise_types.items():
            print(f"\nTeste {name}...")
            start_time = time.time()
            
            mi_values, means, stds = self.analyze_with_noise(
                test_seq, noise_func, noise_levels
            )
            
            # Finde optimales Rauschen
            optimal_idx = np.argmax(mi_values)
            optimal_noise = noise_levels[optimal_idx]
            max_mi = mi_values[optimal_idx]
            
            elapsed = time.time() - start_time
            
            print(f"  Optimales σ: {optimal_noise:.4f}")
            print(f"  Max MI: {max_mi:.2f}")
            print(f"  Peak Count bei σ_opt: {means[optimal_idx]:.1f} ± {stds[optimal_idx]:.1f}")
            print(f"  Zeit: {elapsed:.1f}s")
            
            all_results[name] = {
                'noise_levels': noise_levels,
                'mi_values': mi_values,
                'optimal_noise': optimal_noise,
                'max_mi': max_mi,
                'peak_means': means,
                'peak_stds': stds
            }
        
        self.results = all_results
        
        # Ranking
        print("\n" + "="*70)
        print("RANKING DER RAUSCHTYPEN (nach max MI):")
        sorted_results = sorted(all_results.items(), 
                              key=lambda x: x[1]['max_mi'], 
                              reverse=True)
        
        for i, (name, data) in enumerate(sorted_results, 1):
            print(f"{i}. {name}: MI={data['max_mi']:.2f}, σ_opt={data['optimal_noise']:.4f}")
        
        return all_results
    
    def test_parameter_sensitivity(self):
        """Teste Sensitivität verschiedener Parameter"""
        print("\n\nPARAMETER-SENSITIVITÄTS-ANALYSE")
        print("="*70)
        
        test_seq = self.collatz_sequence(27)
        
        # Teste Lévy α Parameter
        print("\nLévy-Rauschen: Teste verschiedene α-Werte")
        alpha_range = [0.5, 1.0, 1.5, 1.8, 2.0]
        
        for alpha in alpha_range:
            noise_func = lambda size, amp: self.levy_noise(size, amp, alpha)
            mi_values, _, _ = self.analyze_with_noise(
                test_seq, noise_func, np.logspace(-4, 0, 20)
            )
            max_mi = np.max(mi_values)
            print(f"  α={alpha}: max MI = {max_mi:.2f}")
        
        # Teste RTN Switch Probability
        print("\nTelegraph Noise: Teste verschiedene Switch-Wahrscheinlichkeiten")
        switch_probs = [0.001, 0.005, 0.01, 0.05, 0.1]
        
        for prob in switch_probs:
            noise_func = lambda size, amp: self.telegraph_noise(size, amp, prob)
            mi_values, _, _ = self.analyze_with_noise(
                test_seq, noise_func, np.logspace(-4, 0, 20)
            )
            max_mi = np.max(mi_values)
            print(f"  p={prob}: max MI = {max_mi:.2f}")
    
    def analyze_noise_characteristics(self):
        """Analysiere Charakteristiken der verschiedenen Rauschtypen"""
        print("\n\nRAUSCH-CHARAKTERISTIKEN")
        print("="*70)
        
        n_samples = 10000
        amplitude = 1.0
        
        # Generiere Samples
        noise_samples = {
            'White': self.white_noise(n_samples, amplitude),
            'Pink': self.pink_noise(n_samples, amplitude),
            'Brown': self.brown_noise(n_samples, amplitude),
            'Quantum': self.quantum_inspired_noise(n_samples, amplitude),
            'Telegraph': self.telegraph_noise(n_samples, amplitude),
            'Lévy': self.levy_noise(n_samples, amplitude, 1.5)
        }
        
        print("\nStatistische Eigenschaften:")
        print(f"{'Typ':<12} {'Mean':>8} {'Std':>8} {'Skew':>8} {'Kurt':>8} {'Min':>8} {'Max':>8}")
        print("-" * 70)
        
        for name, samples in noise_samples.items():
            mean = np.mean(samples)
            std = np.std(samples)
            skew = stats.skew(samples)
            kurt = stats.kurtosis(samples)
            min_val = np.min(samples)
            max_val = np.max(samples)
            
            print(f"{name:<12} {mean:>8.3f} {std:>8.3f} {skew:>8.3f} "
                  f"{kurt:>8.3f} {min_val:>8.3f} {max_val:>8.3f}")
        
        return noise_samples
    
    def create_visualizations(self):
        """Erstelle umfassende Visualisierungen"""
        fig = plt.figure(figsize=(16, 12))
        
        # Layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. SR Kurven für alle Rauschtypen
        ax1 = fig.add_subplot(gs[0, :])
        
        for name, data in self.results.items():
            ax1.semilogx(data['noise_levels'], data['mi_values'], 
                        linewidth=2, label=name, alpha=0.8)
        
        ax1.set_xlabel('Noise Level σ')
        ax1.set_ylabel('MI Approximation')
        ax1.set_title('Stochastic Resonance Curves for Different Noise Types')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2-7. Individuelle Rauschtyp-Beispiele
        noise_samples = self.analyze_noise_characteristics()
        
        positions = [(1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
        
        for (name, samples), (row, col) in zip(noise_samples.items(), positions):
            ax = fig.add_subplot(gs[row, col])
            
            # Zeige erste 500 Samples
            ax.plot(samples[:500], 'b-', linewidth=0.5, alpha=0.7)
            ax.set_title(f'{name} Noise')
            ax.set_xlabel('Sample')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
            
            # Füge Statistik als Text hinzu
            std = np.std(samples)
            skew = stats.skew(samples)
            ax.text(0.95, 0.95, f'σ={std:.2f}\nskew={skew:.2f}', 
                   transform=ax.transAxes, 
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=8)
        
        plt.suptitle('Comprehensive Noise Type Analysis for Collatz Sequences', 
                    fontsize=16)
        plt.tight_layout()
        plt.savefig('noise_types_collatz_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def theoretical_analysis(self):
        """Theoretische Analyse der Ergebnisse"""
        print("\n\nTHEORETISCHE ANALYSE")
        print("="*70)
        
        # Finde besten Rauschtyp
        best_type = max(self.results.items(), key=lambda x: x[1]['max_mi'])[0]
        best_data = self.results[best_type]
        
        print(f"\nBester Rauschtyp: {best_type}")
        print(f"Optimales σ: {best_data['optimal_noise']:.4f}")
        print(f"Max MI: {best_data['max_mi']:.2f}")
        
        # Vergleiche mit deinem Paper-Ergebnis
        print("\nVergleich mit Original-Paper:")
        print(f"Paper σ_opt: 0.001")
        print(f"Bester Test σ_opt: {best_data['optimal_noise']:.4f}")
        print(f"Verhältnis: {best_data['optimal_noise']/0.001:.1f}x")
        
        # Interpretiere Ergebnisse
        print("\nINTERPRETATION:")
        
        if 'Pink' in best_type or '1/f' in best_type:
            print("✓ 1/f Rauschen funktioniert am besten")
            print("  → Deutet auf Skaleninvarianz in Collatz hin")
            print("  → Verbindung zu Selbstähnlichkeit")
        
        if 'Quantum' in best_type:
            print("✓ Quanten-inspiriertes Rauschen effektiv")
            print("  → Diskrete Natur könnte wichtig sein")
            print("  → Nullpunktfluktuationen-Analogie")
        
        if abs(best_data['optimal_noise'] - 0.001) < 0.0005:
            print("✓ Ergebnis bestätigt Original-Paper!")
            print("  → Robustheit der Methode")

# Hauptprogramm
def main():
    print("WELCHES RAUSCHEN PASST AM BESTEN ZU COLLATZ?")
    print("="*70)
    
    tester = NoiseTypeComparison()
    
    # Führe Haupttest durch
    results = tester.run_comprehensive_test()
    
    # Parameter-Sensitivität
    tester.test_parameter_sensitivity()
    
    # Visualisierungen
    tester.create_visualizations()
    
    # Theoretische Analyse
    tester.theoretical_analysis()
    
    print("\n\nFAZIT:")
    print("="*70)
    print("Die Analyse zeigt, welcher fundamentale Rauschtyp")
    print("am besten mit der Collatz-Struktur harmoniert.")
    print("\nDies gibt Hinweise auf die tiefere Natur")
    print("der Collatz-Vermutung und warum Stochastic")
    print("Resonance überhaupt funktioniert!")
    
    return results

if __name__ == "__main__":
    results = main()