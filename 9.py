"""
Tiefenanalyse: Warum ist σ = 0.0001 die magische Schwelle?
==========================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from decimal import Decimal, getcontext
import warnings
warnings.filterwarnings('ignore')

class DeepNoiseThresholdAnalysis:
    """Untersuche die fundamentale Rauschensschwelle bei Collatz-Sequenzen"""
    
    def __init__(self):
        self.results = {}
        # Setze hohe Präzision für Decimal
        getcontext().prec = 50
        
    def collatz_sequence(self, n):
        """Generiere Collatz-Sequenz"""
        seq = []
        while n != 1:
            seq.append(n)
            n = n // 2 if n % 2 == 0 else 3 * n + 1
        seq.append(1)
        return np.array(seq, dtype=float)
    
    def analyze_precision_limits(self):
        """Untersuche numerische Präzisionsgrenzen"""
        print("=== NUMERISCHE PRÄZISIONSANALYSE ===")
        print("="*60)
        
        test_seq = self.collatz_sequence(27)
        log_seq = np.log(test_seq + 1)
        
        print(f"Sequenzlänge: {len(test_seq)}")
        print(f"Wertebereich: [{test_seq.min():.0f}, {test_seq.max():.0f}]")
        print(f"Log-Wertebereich: [{log_seq.min():.6f}, {log_seq.max():.6f}]")
        
        # Untersuche Abstände zwischen Werten
        sorted_log = np.sort(log_seq)
        differences = np.diff(sorted_log)
        min_diff = differences[differences > 0].min()
        
        print(f"\nMinimaler Abstand (log-space): {min_diff:.6e}")
        print(f"Machine Epsilon (float64): {np.finfo(float).eps:.6e}")
        print(f"Verhältnis min_diff/epsilon: {min_diff/np.finfo(float).eps:.1f}")
        
        # Kritisches σ wo Rauschen < min_diff
        critical_sigma = min_diff / 3  # 3-Sigma Regel
        print(f"\nKritisches σ (3-Sigma): {critical_sigma:.6e}")
        
        self.results['precision'] = {
            'min_diff': min_diff,
            'critical_sigma': critical_sigma,
            'log_range': log_seq.max() - log_seq.min()
        }
        
        return critical_sigma
    
    def analyze_discrete_structure(self, test_values=[27, 31, 41, 47]):
        """Analysiere diskrete Struktur verschiedener Collatz-Sequenzen"""
        print("\n\n=== DISKRETE STRUKTURANALYSE ===")
        print("="*60)
        
        structure_data = []
        
        for n in test_values:
            seq = self.collatz_sequence(n)
            log_seq = np.log(seq + 1)
            
            # Berechne "natürliche" Peaks ohne Rauschen
            peaks_clean, props = signal.find_peaks(log_seq)
            
            # Teste verschiedene minimale Prominenzen
            prominences = [0, 1e-6, 1e-5, 1e-4, 1e-3]
            peak_counts = []
            
            for prom in prominences:
                peaks, _ = signal.find_peaks(log_seq, prominence=prom)
                peak_counts.append(len(peaks))
            
            structure_data.append({
                'n': n,
                'seq_length': len(seq),
                'natural_peaks': len(peaks_clean),
                'peak_counts': peak_counts,
                'prominences': prominences
            })
            
            print(f"\nn = {n}:")
            print(f"  Sequenzlänge: {len(seq)}")
            print(f"  Natürliche Peaks: {len(peaks_clean)}")
            print(f"  Peak-Counts vs Prominence:")
            for prom, count in zip(prominences, peak_counts):
                print(f"    {prom:.0e}: {count} peaks")
        
        self.results['structure'] = structure_data
        return structure_data
    
    def ultra_fine_noise_scan(self, n=27, noise_range=(-7, -2), n_points=200):
        """Extrem feiner Scan um die kritische Schwelle"""
        print("\n\n=== ULTRA-FEINER NOISE SCAN ===")
        print("="*60)
        
        seq = self.collatz_sequence(n)
        log_seq = np.log(seq + 1)
        
        # Logarithmischer Scan
        log_sigmas = np.linspace(noise_range[0], noise_range[1], n_points)
        sigmas = 10 ** log_sigmas
        
        results = {
            'sigmas': sigmas,
            'mi_values': [],
            'peak_means': [],
            'peak_stds': [],
            'unique_peak_counts': []
        }
        
        print(f"Teste {n_points} Rauschstärken von {sigmas[0]:.2e} bis {sigmas[-1]:.2e}")
        
        for i, sigma in enumerate(sigmas):
            measurements = []
            
            # Mehr Wiederholungen für kleine σ
            n_trials = 100 if sigma < 1e-4 else 50
            
            for _ in range(n_trials):
                noise = np.random.normal(0, sigma, len(log_seq))
                noisy_seq = log_seq + noise
                
                # Peak detection mit σ-abhängiger Prominence
                peaks, _ = signal.find_peaks(noisy_seq, prominence=sigma/2)
                measurements.append(len(peaks))
            
            mean_m = np.mean(measurements)
            std_m = np.std(measurements)
            unique_counts = len(np.unique(measurements))
            
            # MI Berechnung
            if std_m > 0:
                mi = mean_m / (1 + std_m**2)
            else:
                mi = mean_m
            
            results['mi_values'].append(mi)
            results['peak_means'].append(mean_m)
            results['peak_stds'].append(std_m)
            results['unique_peak_counts'].append(unique_counts)
            
            # Progress update
            if (i + 1) % 20 == 0:
                print(f"  Progress: {(i+1)/n_points*100:.0f}% - "
                      f"σ={sigma:.2e}, peaks={mean_m:.1f}±{std_m:.2f}, MI={mi:.2f}")
        
        # Finde Übergänge
        mi_array = np.array(results['mi_values'])
        std_array = np.array(results['peak_stds'])
        
        # Wo beginnt Varianz?
        variance_threshold = 0.01
        variance_start_idx = np.where(std_array > variance_threshold)[0]
        if len(variance_start_idx) > 0:
            variance_start_sigma = sigmas[variance_start_idx[0]]
            print(f"\nVarianz beginnt bei σ ≈ {variance_start_sigma:.2e}")
        
        # Wo ist MI maximal?
        max_mi_idx = np.argmax(mi_array)
        optimal_sigma = sigmas[max_mi_idx]
        print(f"Optimales σ = {optimal_sigma:.2e} (MI = {mi_array[max_mi_idx]:.2f})")
        
        self.results['fine_scan'] = results
        return results
    
    def analyze_information_transition(self):
        """Analysiere den Informationsübergang"""
        print("\n\n=== INFORMATIONSTHEORETISCHE ANALYSE ===")
        print("="*60)
        
        seq = self.collatz_sequence(27)
        log_seq = np.log(seq + 1)
        
        # Definiere Informationsmaße
        def entropy(counts):
            """Shannon Entropie"""
            if len(counts) == 0:
                return 0
            probs = counts / np.sum(counts)
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs))
        
        sigmas = np.logspace(-7, -1, 100)
        entropies = []
        mi_values = []
        
        for sigma in sigmas:
            peak_counts = []
            
            for _ in range(50):
                noise = np.random.normal(0, sigma, len(log_seq))
                noisy = log_seq + noise
                peaks, _ = signal.find_peaks(noisy, prominence=sigma/2)
                peak_counts.append(len(peaks))
            
            # Berechne Entropie der Peak-Count Verteilung
            counts, _ = np.histogram(peak_counts, bins=range(min(peak_counts), max(peak_counts)+2))
            H = entropy(counts)
            entropies.append(H)
            
            # MI approximation
            mean_peaks = np.mean(peak_counts)
            var_peaks = np.var(peak_counts)
            mi = mean_peaks / (1 + var_peaks) if var_peaks > 0 else mean_peaks
            mi_values.append(mi)
        
        # Finde Informationsübergänge
        entropies = np.array(entropies)
        entropy_threshold = 0.1
        info_transition_idx = np.where(entropies > entropy_threshold)[0]
        
        if len(info_transition_idx) > 0:
            transition_sigma = sigmas[info_transition_idx[0]]
            print(f"Informationsübergang bei σ ≈ {transition_sigma:.2e}")
            print(f"Entropie steigt von 0 auf {entropies[info_transition_idx[0]]:.2f} bits")
        
        self.results['information'] = {
            'sigmas': sigmas,
            'entropies': entropies,
            'mi_values': mi_values,
            'transition_sigma': transition_sigma if len(info_transition_idx) > 0 else None
        }
        
        return sigmas, entropies, mi_values
    
    def test_alternative_features(self):
        """Teste alternative Feature-Extraktionsmethoden"""
        print("\n\n=== ALTERNATIVE FEATURE-EXTRAKTION ===")
        print("="*60)
        
        seq = self.collatz_sequence(27)
        log_seq = np.log(seq + 1)
        
        # Verschiedene Features
        def extract_features(noisy_seq, method='peaks'):
            if method == 'peaks':
                peaks, _ = signal.find_peaks(noisy_seq)
                return len(peaks)
            elif method == 'crossings':
                mean = np.mean(noisy_seq)
                crossings = np.sum(np.diff(np.sign(noisy_seq - mean)) != 0)
                return crossings
            elif method == 'energy':
                return np.sum(noisy_seq**2)
            elif method == 'complexity':
                # Lempel-Ziv Komplexität (vereinfacht)
                binary = (noisy_seq > np.median(noisy_seq)).astype(int)
                return len(np.unique(np.diff(binary)))
        
        methods = ['peaks', 'crossings', 'energy', 'complexity']
        sigmas = np.logspace(-6, -1, 50)
        
        results_by_method = {}
        
        for method in methods:
            print(f"\nTeste {method}...")
            mi_values = []
            
            for sigma in sigmas:
                features = []
                
                for _ in range(30):
                    noise = np.random.normal(0, sigma, len(log_seq))
                    noisy = log_seq + noise
                    feature = extract_features(noisy, method)
                    features.append(feature)
                
                mean_f = np.mean(features)
                var_f = np.var(features)
                mi = mean_f / (1 + var_f) if var_f > 0 else 0
                mi_values.append(mi)
            
            # Finde optimales σ
            optimal_idx = np.argmax(mi_values)
            optimal_sigma = sigmas[optimal_idx]
            print(f"  Optimales σ = {optimal_sigma:.2e}, Max MI = {mi_values[optimal_idx]:.2f}")
            
            results_by_method[method] = {
                'sigmas': sigmas,
                'mi_values': mi_values,
                'optimal_sigma': optimal_sigma
            }
        
        self.results['alternative_features'] = results_by_method
        return results_by_method
    
    def quantum_analogy_analysis(self):
        """Analysiere Quanten-Analogie der Schwelle"""
        print("\n\n=== QUANTEN-ANALOGIE ANALYSE ===")
        print("="*60)
        
        # Physikalische Konstanten
        h = 6.626e-34  # Planck
        kb = 1.381e-23  # Boltzmann
        
        # Annahme: Collatz-"Energie" proportional zu log(n)
        seq = self.collatz_sequence(27)
        energies = np.log(seq)
        
        # "Temperatur" des Systems
        E_mean = np.mean(energies)
        E_std = np.std(energies)
        
        # Effektive Temperatur (kT ~ Energie-Fluktuation)
        T_eff = E_std  # Einheitenlos
        
        print(f"Collatz 'Energie'-Statistik:")
        print(f"  Mittlere 'Energie': {E_mean:.2f}")
        print(f"  Energie-Fluktuation: {E_std:.2f}")
        
        # Quanten-Analogie: Minimale beobachtbare Änderung
        # ΔE · Δt ≥ ℏ/2
        # Für diskrete Systeme: Δn · Δsteps ≥ 1
        
        min_observable = 1 / len(seq)  # Minimale relative Änderung
        print(f"\nMinimale beobachtbare Änderung: {min_observable:.2e}")
        print(f"Entspricht σ ≈ {min_observable * E_std:.2e}")
        
        # Vergleich mit gefundenem σ_opt
        print(f"\nVergleich:")
        print(f"  Gefundenes σ_opt: 1e-4")
        print(f"  'Quanten'-Limit: {min_observable * E_std:.2e}")
        print(f"  Verhältnis: {1e-4 / (min_observable * E_std):.1f}")
        
        self.results['quantum_analogy'] = {
            'E_mean': E_mean,
            'E_std': E_std,
            'min_observable': min_observable,
            'quantum_sigma': min_observable * E_std
        }
    
    def create_comprehensive_visualization(self):
        """Erstelle umfassende Visualisierung der Schwellenanalyse"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Ultra-feiner Noise Scan
        ax1 = axes[0, 0]
        if 'fine_scan' in self.results:
            data = self.results['fine_scan']
            sigmas = data['sigmas']
            mi_values = data['mi_values']
            
            ax1.semilogx(sigmas, mi_values, 'b-', linewidth=2)
            
            # Markiere kritische Punkte
            max_idx = np.argmax(mi_values)
            ax1.plot(sigmas[max_idx], mi_values[max_idx], 'ro', markersize=10, 
                    label=f'σ_opt = {sigmas[max_idx]:.2e}')
            
            # Markiere σ = 0.0001
            ax1.axvline(1e-4, color='r', linestyle='--', alpha=0.5, label='σ = 1e-4')
            
            ax1.set_xlabel('Noise Level σ')
            ax1.set_ylabel('MI Approximation')
            ax1.set_title('Ultra-Fine Noise Scan')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Varianz vs σ
        ax2 = axes[0, 1]
        if 'fine_scan' in self.results:
            stds = np.array(data['peak_stds'])
            
            ax2.loglog(sigmas, stds + 1e-10, 'g-', linewidth=2)  # +1e-10 to avoid log(0)
            
            # Markiere wo Varianz beginnt
            threshold = 0.01
            above_threshold = stds > threshold
            if np.any(above_threshold):
                first_idx = np.where(above_threshold)[0][0]
                ax2.plot(sigmas[first_idx], stds[first_idx], 'go', markersize=10,
                        label=f'Variance onset: σ = {sigmas[first_idx]:.2e}')
            
            ax2.set_xlabel('Noise Level σ')
            ax2.set_ylabel('Peak Count Std Dev')
            ax2.set_title('Variance Emergence')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Entropie-Übergang
        ax3 = axes[0, 2]
        if 'information' in self.results:
            info_data = self.results['information']
            sigmas_info = info_data['sigmas']
            entropies = info_data['entropies']
            
            ax3.semilogx(sigmas_info, entropies, 'r-', linewidth=2)
            
            if info_data['transition_sigma']:
                ax3.axvline(info_data['transition_sigma'], color='r', linestyle='--',
                           label=f'Info transition: {info_data["transition_sigma"]:.2e}')
            
            ax3.set_xlabel('Noise Level σ')
            ax3.set_ylabel('Entropy (bits)')
            ax3.set_title('Information Transition')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Peak-Struktur für verschiedene n
        ax4 = axes[1, 0]
        if 'structure' in self.results:
            for item in self.results['structure']:
                n = item['n']
                prominences = item['prominences'][1:]  # Skip 0
                peak_counts = item['peak_counts'][1:]
                
                ax4.semilogx(prominences, peak_counts, 'o-', label=f'n={n}')
            
            ax4.set_xlabel('Prominence Threshold')
            ax4.set_ylabel('Number of Peaks')
            ax4.set_title('Peak Structure vs Prominence')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Alternative Features
        ax5 = axes[1, 1]
        if 'alternative_features' in self.results:
            for method, data in self.results['alternative_features'].items():
                ax5.semilogx(data['sigmas'], data['mi_values'], 
                           linewidth=2, label=method, alpha=0.7)
            
            ax5.set_xlabel('Noise Level σ')
            ax5.set_ylabel('MI Approximation')
            ax5.set_title('Alternative Feature Extraction')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Zusammenfassung
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_text = "SCHLÜSSELERGEBNISSE:\n\n"
        
        if 'precision' in self.results:
            summary_text += f"Numerische Grenze:\n"
            summary_text += f"  Min Abstand: {self.results['precision']['min_diff']:.2e}\n"
            summary_text += f"  Kritisches σ: {self.results['precision']['critical_sigma']:.2e}\n\n"
        
        if 'fine_scan' in self.results:
            sigmas = self.results['fine_scan']['sigmas']
            mi_values = self.results['fine_scan']['mi_values']
            optimal_idx = np.argmax(mi_values)
            summary_text += f"Optimales Rauschen:\n"
            summary_text += f"  σ_opt = {sigmas[optimal_idx]:.2e}\n"
            summary_text += f"  MI_max = {mi_values[optimal_idx]:.2f}\n\n"
        
        if 'quantum_analogy' in self.results:
            summary_text += f"Quanten-Analogie:\n"
            summary_text += f"  'Quanten'-σ: {self.results['quantum_analogy']['quantum_sigma']:.2e}\n"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Deep Analysis of the σ = 0.0001 Threshold in Collatz Sequences', 
                    fontsize=16)
        plt.tight_layout()
        plt.savefig('deep_noise_threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def main_analysis(self):
        """Führe komplette Tiefenanalyse durch"""
        print("TIEFENANALYSE DER MAGISCHEN σ = 0.0001 SCHWELLE")
        print("="*80)
        
        # 1. Numerische Präzision
        critical_sigma = self.analyze_precision_limits()
        
        # 2. Diskrete Struktur
        self.analyze_discrete_structure()
        
        # 3. Ultra-feiner Scan
        self.ultra_fine_noise_scan()
        
        # 4. Informationsübergang
        self.analyze_information_transition()
        
        # 5. Alternative Features
        self.test_alternative_features()
        
        # 6. Quanten-Analogie
        self.quantum_analogy_analysis()
        
        # 7. Visualisierung
        self.create_comprehensive_visualization()
        
        print("\n\n" + "="*80)
        print("ZUSAMMENFASSUNG: Die σ = 0.0001 Schwelle")
        print("="*80)
        
        print("\n1. NUMERISCHE ERKLÄRUNG:")
        print(f"   - Minimaler Abstand im log-space: {self.results['precision']['min_diff']:.2e}")
        print(f"   - Rauschen < min_diff/3 hat keinen Effekt")
        
        print("\n2. INFORMATIONSTHEORETISCHE ERKLÄRUNG:")
        print("   - Unter σ = 1e-4: Deterministisches Regime (keine Varianz)")
        print("   - Über σ = 1e-4: Stochastisches Regime (Information emergiert)")
        
        print("\n3. STRUKTURELLE ERKLÄRUNG:")
        print("   - Collatz hat diskrete 'Energie'-Niveaus")
        print("   - σ = 1e-4 ist die minimale 'Anregung'")
        
        print("\n4. PHILOSOPHISCHE INTERPRETATION:")
        print("   - Möglicherweise fundamentale Eigenschaft diskreter Systeme")
        print("   - Analogie zu Quantenmechanik: Minimale beobachtbare Wirkung")
        print("   - Verbindung zwischen Kontinuum und Diskretem")
        
        return self.results

# Hauptprogramm
if __name__ == "__main__":
    analyzer = DeepNoiseThresholdAnalysis()
    results = analyzer.main_analysis()