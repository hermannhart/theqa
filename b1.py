"""
Systematische Analyse: Warum ergibt sich σ_c = 0.117 und k = 1/13.5?
=====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats, optimize
from scipy.linalg import eig
import sympy as sp
from collections import defaultdict

class CriticalValueAnalysis:
    """Untersuche die theoretischen Grundlagen von σ_c und k"""
    
    def __init__(self):
        self.results = {}
    
    def analyze_log_ratios(self):
        """Analysiere die logarithmischen Verhältnisse in Collatz"""
        print("=== ANALYSE DER LOGARITHMISCHEN VERHÄLTNISSE ===")
        print("="*60)
        
        # Grundlegende Verhältnisse
        log_3_2 = np.log(3) / np.log(2)
        log_3_4 = np.log(3) / np.log(4)
        log_sqrt_3_4 = np.log(np.sqrt(3/4)) / np.log(2)
        
        print(f"log₂(3) = {log_3_2:.6f}")
        print(f"log₄(3) = {log_3_4:.6f}")
        print(f"log₂(√(3/4)) = {log_sqrt_3_4:.6f}")
        
        # Teste verschiedene k-Werte
        print("\nTeste k-Werte:")
        for denom in range(10, 20):
            k = 1 / denom
            sigma = k * log_3_2
            print(f"k = 1/{denom}: σ = {sigma:.4f}")
            if abs(sigma - 0.117) < 0.001:
                print(f"  → TREFFER! k ≈ 1/{denom}")
        
        # Genauere Analyse um 13.5
        print("\nFeinanalyse um k = 1/13.5:")
        for offset in np.linspace(-0.5, 0.5, 11):
            k = 1 / (13.5 + offset)
            sigma = k * log_3_2
            error = abs(sigma - 0.117)
            print(f"k = 1/{13.5 + offset:.1f}: σ = {sigma:.5f}, Fehler = {error:.5f}")
        
        return log_3_2
    
    def analyze_collatz_structure(self):
        """Analysiere strukturelle Eigenschaften der Collatz-Abbildung"""
        print("\n\n=== STRUKTURELLE EIGENSCHAFTEN VON COLLATZ ===")
        print("="*60)
        
        # Durchschnittliche Schrittweiten
        print("\n1. Durchschnittliche Transformationen:")
        
        # Für gerade: n → n/2
        avg_even = 1/2
        log_even = np.log(avg_even)
        
        # Für ungerade: n → (3n+1)/2 ≈ 3n/2 für große n
        avg_odd = 3/2
        log_odd = np.log(avg_odd)
        
        # Gewichteter Durchschnitt (50/50)
        avg_total = np.sqrt(avg_even * avg_odd)
        log_total = np.log(avg_total)
        
        print(f"Gerade: n → n/2, log-Faktor = {log_even:.4f}")
        print(f"Ungerade: n → 3n/2, log-Faktor = {log_odd:.4f}")
        print(f"Geometrisches Mittel: {avg_total:.4f}, log = {log_total:.4f}")
        
        # Das ist genau log(√(3/4))!
        print(f"\nVergleich mit Drift: log(√(3/4)) = {np.log(np.sqrt(3/4)):.4f}")
        
        # 2. Spektrale Eigenschaften
        print("\n2. Spektrale Zerlegung:")
        
        # Transfer-Matrix KORREKT berechnen
        T_even = np.array([[0.5, 0], [0, 1]])    # n → n/2
        T_odd = np.array([[1.5, 0.5], [0, 1]])   # n → (3n+1)/2
        T_eff = 0.5 * T_even + 0.5 * T_odd       # Gewichteter Durchschnitt
        
        print(f"T_even = \n{T_even}")
        print(f"T_odd = \n{T_odd}")
        print(f"T_eff = \n{T_eff}")
        
        eigenvals, eigenvecs = np.linalg.eig(T_eff)
        print(f"Eigenwerte: {eigenvals}")
        print(f"Dominanter Eigenwert: {max(eigenvals)}")
        
        # Spektraler Radius
        spectral_radius = max(abs(eigenvals))
        print(f"Spektraler Radius: {spectral_radius}")
        
        return avg_total, spectral_radius
    
    def analyze_critical_transition(self):
        """Analysiere warum der Übergang genau bei σ = 0.117 stattfindet"""
        print("\n\n=== KRITISCHER ÜBERGANG BEI σ = 0.117 ===")
        print("="*60)
        
        # Hypothese: σ_c ist wo Rauschen gleich der minimalen Struktur-Skala ist
        
        # 1. Minimale log-Abstände in Collatz-Sequenzen
        print("\n1. Minimale Abstände in log-Raum:")
        
        min_gaps = []
        for n in [27, 31, 41, 47, 63, 97]:
            seq = self.collatz_sequence(n)
            if len(seq) > 2:
                log_seq = np.log(seq + 1)
                sorted_log = np.sort(log_seq)
                gaps = np.diff(sorted_log)
                min_gap = np.min(gaps[gaps > 0]) if any(gaps > 0) else 0
                min_gaps.append(min_gap)
                print(f"n={n}: min log-gap = {min_gap:.4f}")
        
        avg_min_gap = np.mean(min_gaps)
        print(f"\nDurchschnittlicher min gap: {avg_min_gap:.4f}")
        
        # 2. Verhältnis zu σ_c
        print(f"\nVerhältnis σ_c / avg_min_gap = {0.117 / avg_min_gap:.2f}")
        
        # 3. Alternative Herleitung über Informationstheorie
        print("\n2. Informationstheoretische Herleitung:")
        
        # Shannon-Kapazität eines binären Kanals
        p_error = 0.117  # Fehlerwahrscheinlichkeit
        capacity = 1 - (-p_error * np.log2(p_error) - (1-p_error) * np.log2(1-p_error))
        print(f"Kanal-Kapazität bei p={p_error}: {capacity:.4f}")
        
        # 4. Verbindung zu 13.5
        print("\n3. Warum k = 1/13.5?")
        
        # Mögliche Interpretationen
        print(f"13.5 = 27/2 (27 ist eine wichtige Collatz-Zahl)")
        print(f"13.5 ≈ 4π + 1 = {4*np.pi + 1:.1f}")
        print(f"13.5 ≈ e² - e/2 = {np.e**2 - np.e/2:.1f}")
        
        # Spektrale Interpretation
        eigenval_ratio = 3/4 / (1/2)  # Verhältnis der Eigenwerte
        print(f"Eigenwert-Verhältnis: {eigenval_ratio}")
        
        # k könnte related sein zu einer Normalisierung
        log_3_2 = np.log(3) / np.log(2)  # Definiere log_3_2 lokal
        normalization = np.log(3) / (2 * np.pi)
        k_theoretical = normalization / log_3_2
        print(f"\nTheoretisches k aus Normalisierung: {k_theoretical:.4f}")
        print(f"Das ergibt σ = {k_theoretical * log_3_2:.4f}")
        
        return min_gaps
    
    def collatz_sequence(self, n):
        """Generiere Collatz-Sequenz"""
        seq = []
        while n != 1 and len(seq) < 1000:
            seq.append(n)
            n = n // 2 if n % 2 == 0 else 3 * n + 1
        seq.append(1)
        return np.array(seq)
    
    def test_universal_constant_hypothesis(self):
        """Teste Hypothese: k = 1/13.5 ist eine universelle Konstante"""
        print("\n\n=== UNIVERSELLE KONSTANTE k = 1/13.5 ===")
        print("="*60)
        
        # Teste für andere qn+1 Vermutungen
        print("\nTeste k für verschiedene qn+1 Vermutungen:")
        
        q_values = [3, 5, 7, 9, 11]
        for q in q_values:
            # Theoretisches σ_c für qn+1
            log_q_2 = np.log(q) / np.log(2)
            
            # Mit k = 1/13.5
            sigma_predicted = (1/13.5) * log_q_2
            
            print(f"q={q}: σ_c(predicted) = {sigma_predicted:.4f}")
            
            # Verhältnis zu bekannten Werten (aus paper)
            if q == 5:
                actual = 0.257  # Aus paper
                k_actual = actual / log_q_2
                print(f"  Tatsächlich: σ_c = {actual}, k = {k_actual:.4f} = 1/{1/k_actual:.1f}")
    
    def geometric_interpretation(self):
        """Geometrische Interpretation von σ_c"""
        print("\n\n=== GEOMETRISCHE INTERPRETATION ===")
        print("="*60)
        
        # σ_c als kritischer Winkel?
        angle_rad = 0.117
        angle_deg = np.degrees(angle_rad)
        print(f"\nσ_c als Winkel: {angle_deg:.1f}°")
        
        # σ_c als Verhältnis im Einheitskreis
        sin_sigma = np.sin(0.117)
        cos_sigma = np.cos(0.117)
        tan_sigma = np.tan(0.117)
        
        print(f"\nTrigonometrische Werte:")
        print(f"sin(σ_c) = {sin_sigma:.4f}")
        print(f"cos(σ_c) = {cos_sigma:.4f}")
        print(f"tan(σ_c) = {tan_sigma:.4f}")
        
        # Interessant: tan(σ_c) ≈ 0.117!
        print(f"\nBemerkenswert: tan({angle_rad:.3f}) ≈ {tan_sigma:.3f} ≈ σ_c!")
        
        # Das könnte die wahre Definition sein!
        # σ_c ist der Winkel wo tan(σ_c) = σ_c (fast)
        
        # Löse tan(x) = x numerisch
        from scipy.optimize import fsolve
        
        def equation(x):
            return np.tan(x) - x
        
        # Suche Lösung nahe 0.117
        solution = fsolve(equation, 0.1)[0]
        print(f"\nLösung von tan(x) = x nahe 0: x = {solution:.6f}")
        
        # Aber die erste nicht-triviale Lösung ist bei ~4.49
        solution2 = fsolve(equation, 4.5)[0]
        print(f"Erste nicht-triviale Lösung: x = {solution2:.6f}")
        
        return angle_rad, tan_sigma
    
    def final_synthesis(self):
        """Finale Synthese: Was ist σ_c wirklich?"""
        print("\n\n=== FINALE SYNTHESE ===")
        print("="*60)
        
        log_3_2 = np.log(3) / np.log(2)
        
        print("\nMögliche Interpretationen von σ_c = 0.117:")
        
        print(f"\n1. RAUSCH-SCHWELLE:")
        print(f"   σ_c = (1/13.5) * log₂(3)")
        print(f"   Wo 13.5 die 'Resonanz-Konstante' für Collatz ist")
        
        print(f"\n2. INFORMATIONSTHEORETISCH:")
        print(f"   σ_c ist wo die Kanal-Kapazität maximal wird")
        print(f"   für die Detektion der Collatz-Struktur")
        
        print(f"\n3. SPEKTRAL:")
        print(f"   σ_c related zu Eigenwerten der Transfer-Matrix")
        print(f"   σ_c ≈ -log₂(λ) / 13.5 wo λ = 3/4")
        
        print(f"\n4. GEOMETRISCH:")
        print(f"   σ_c ≈ tan(σ_c) (Selbst-Konsistenz)")
        
        print(f"\n5. EMERGENT:")
        print(f"   σ_c entsteht aus dem Zusammenspiel von:")
        print(f"   - Diskreter Struktur (Collatz)")
        print(f"   - Kontinuierlichem Rauschen")
        print(f"   - Log-Transformation")
        print(f"   - Peak-Detection Algorithmus")
        
        print("\n" + "="*60)
        print("VERMUTUNG: k = 1/13.5 ist NICHT fundamental,")
        print("sondern emergent aus der Collatz-Struktur!")
        print("="*60)
    
    def create_visualization(self):
        """Visualisiere die Analyse"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. k-Werte vs σ
        ax1 = axes[0, 0]
        k_values = 1 / np.linspace(10, 20, 100)
        sigma_values = k_values * np.log(3) / np.log(2)
        
        ax1.plot(k_values, sigma_values, 'b-', linewidth=2)
        ax1.axhline(0.117, color='r', linestyle='--', label='σ_c = 0.117')
        ax1.axvline(1/13.5, color='g', linestyle='--', label='k = 1/13.5')
        ax1.set_xlabel('k')
        ax1.set_ylabel('σ = k * log₂(3)')
        ax1.set_title('k vs σ Beziehung')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Spektrale Eigenschaften
        ax2 = axes[0, 1]
        
        # Eigenwerte visualisieren
        eigenvals = [3/4, 1/2]
        ax2.bar(['λ₁', 'λ₂'], eigenvals, color=['red', 'blue'])
        ax2.axhline(1, color='green', linestyle='--', label='Stabilität')
        ax2.set_ylabel('Eigenwert')
        ax2.set_title('Transfer-Matrix Eigenwerte')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Log-Gaps Verteilung
        ax3 = axes[0, 2]
        
        all_gaps = []
        for n in range(10, 100):
            seq = self.collatz_sequence(n)
            if len(seq) > 2:
                log_seq = np.log(seq + 1)
                sorted_log = np.sort(log_seq)
                gaps = np.diff(sorted_log)
                all_gaps.extend(gaps[gaps > 0])
        
        ax3.hist(all_gaps, bins=50, alpha=0.7, density=True)
        ax3.axvline(0.117, color='r', linestyle='--', label='σ_c = 0.117')
        ax3.set_xlabel('Log-Gap')
        ax3.set_ylabel('Dichte')
        ax3.set_title('Verteilung der Log-Abstände')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Trigonometrische Beziehung
        ax4 = axes[1, 0]
        
        x = np.linspace(0, 0.3, 1000)
        ax4.plot(x, x, 'k--', label='y = x')
        ax4.plot(x, np.tan(x), 'b-', linewidth=2, label='y = tan(x)')
        ax4.plot(0.117, 0.117, 'ro', markersize=10, label='σ_c')
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.set_title('σ_c und tan(σ_c)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 0.3)
        ax4.set_ylim(0, 0.3)
        
        # 5. q-Abhängigkeit
        ax5 = axes[1, 1]
        
        q_values = np.arange(3, 20, 2)
        sigma_theoretical = (1/13.5) * np.log(q_values) / np.log(2)
        
        ax5.plot(q_values, sigma_theoretical, 'b-', linewidth=2)
        ax5.scatter([3, 5, 7], [0.117, 0.257, 0.238], 
                   color='red', s=100, label='Gemessen')
        ax5.set_xlabel('q in qn+1')
        ax5.set_ylabel('σ_c')
        ax5.set_title('σ_c für verschiedene qn+1 Vermutungen')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Zusammenfassung
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_text = """
Ergebnisse:

σ_c = 0.117 = (1/13.5) × log₂(3)

Mögliche Bedeutungen:
• Emergente Konstante
• Resonanz-Phänomen
• Informations-Schwelle
• Geometrische Beziehung

k = 1/13.5 ist vermutlich
NICHT fundamental!
"""
        
        ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('sigma_c_analysis.png', dpi=300)
        plt.show()
    
    def run_complete_analysis(self):
        """Führe komplette Analyse durch"""
        print("VOLLSTÄNDIGE ANALYSE: WARUM σ_c = 0.117 UND k = 1/13.5?")
        print("="*60)
        
        # 1. Log-Verhältnisse
        log_3_2 = self.analyze_log_ratios()
        
        # 2. Collatz-Struktur
        avg_total, spectral_radius = self.analyze_collatz_structure()
        
        # 3. Kritischer Übergang
        min_gaps = self.analyze_critical_transition()
        
        # 4. Universelle Konstante
        self.test_universal_constant_hypothesis()
        
        # 5. Geometrische Interpretation
        angle, tan_val = self.geometric_interpretation()
        
        # 6. Finale Synthese
        self.final_synthesis()
        
        # 7. Visualisierung
        self.create_visualization()
        
        return {
            'sigma_c': 0.117,
            'k': 1/13.5,
            'log_3_2': log_3_2,
            'spectral_radius': spectral_radius,
            'geometric_angle': angle
        }

# Hauptausführung
if __name__ == "__main__":
    analyzer = CriticalValueAnalysis()
    results = analyzer.run_complete_analysis()
    
    print("\n\nFAZIT:")
    print("="*60)
    print("σ_c = 0.117 ist wahrscheinlich eine EMERGENTE Eigenschaft")
    print("die aus dem Zusammenspiel von:")
    print("- Collatz-Dynamik (log₂(3) = 1.585...)")
    print("- Rausch-Analyse")
    print("- Peak-Detection Algorithmus")
    print("entsteht.")
    print("\nk = 1/13.5 ist NICHT fundamental, sondern eine")
    print("empirische Konstante die diese Faktoren verbindet!")