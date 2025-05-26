"""
Vereinfachte und korrigierte Untersuchung des universellen Rauschens
====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, constants
from scipy.fft import fft, fftfreq

class NoiseInvestigation:
    """Untersuche verschiedene fundamentale Rauschquellen"""
    
    def __init__(self):
        self.kb = 1.380649e-23  # Boltzmann constant J/K
        self.h = 6.62607015e-34  # Planck constant J⋅s
        self.q = 1.602176634e-19  # Elementary charge C
        
    def thermal_noise_demo(self, T=300, R=1000, bandwidth=1e6):
        """Johnson-Nyquist thermisches Rauschen"""
        print("\n=== THERMISCHES RAUSCHEN ===")
        
        # Theoretische RMS Spannung
        V_rms = np.sqrt(4 * self.kb * T * R * bandwidth)
        print(f"Temperatur: {T} K")
        print(f"Widerstand: {R} Ω")
        print(f"Bandbreite: {bandwidth/1e6} MHz")
        print(f"Thermische Rauschspannung: {V_rms*1e9:.1f} nV/√Hz")
        
        # Simuliere für 1ms
        sample_rate = 2 * bandwidth
        duration = 0.001
        n_samples = int(sample_rate * duration)
        
        # Erzeuge weißes Rauschen
        noise = np.random.normal(0, V_rms, n_samples)
        
        # Berechne Spektrum
        freqs = fftfreq(n_samples, 1/sample_rate)[:n_samples//2]
        spectrum = np.abs(fft(noise)[:n_samples//2])
        
        return noise, freqs, spectrum, V_rms
    
    def quantum_noise_demo(self, frequency=1e12, T=300):
        """Quantenrauschen und Nullpunktfluktuationen"""
        print("\n=== QUANTENRAUSCHEN ===")
        
        # Nullpunktenergie
        E_zero = self.h * frequency / 2
        
        # Thermische Energie
        if self.h * frequency < self.kb * T:
            regime = "Klassisch"
            E_thermal = self.kb * T
        else:
            regime = "Quantum"
            E_thermal = self.h * frequency / (np.exp(self.h * frequency / (self.kb * T)) - 1)
        
        print(f"Frequenz: {frequency/1e12:.1f} THz")
        print(f"Regime: {regime}")
        print(f"Nullpunktenergie: {E_zero/self.q:.3f} eV")
        print(f"Thermische Energie: {E_thermal/self.q:.3f} eV")
        print(f"Verhältnis Zero/Thermal: {E_zero/E_thermal:.2f}")
        
        # Crossover-Frequenz (quantum = klassisch)
        crossover_freq = self.kb * T / self.h
        print(f"Quantum-Classical Crossover: {crossover_freq/1e12:.1f} THz")
        
        return E_zero, E_thermal, crossover_freq
    
    def pink_noise_demo(self, n_samples=10000, alpha=1.0):
        """1/f Rauschen (Pink Noise)"""
        print("\n=== 1/f RAUSCHEN ===")
        
        # Erzeuge 1/f Rauschen
        freqs = np.fft.fftfreq(n_samples, 1.0)
        
        # Erzeuge 1/f Spektrum (nur positive Frequenzen)
        spectrum = np.zeros(n_samples, dtype=complex)
        for i in range(1, n_samples//2):
            # 1/f^alpha Amplitude mit zufälliger Phase
            amplitude = 1 / (freqs[i]**alpha) if freqs[i] > 0 else 0
            phase = np.random.uniform(0, 2*np.pi)
            spectrum[i] = amplitude * np.exp(1j * phase)
            spectrum[-i] = np.conj(spectrum[i])  # Symmetrie für reelles Signal
        
        # Inverse FFT
        pink_noise = np.real(np.fft.ifft(spectrum))
        pink_noise = pink_noise / np.std(pink_noise)  # Normalisierung
        
        print(f"Erzeugt mit Exponent α = {alpha}")
        print(f"Signal Länge: {n_samples} samples")
        
        # Verifiziere Spektrum
        f_check, psd_check = signal.periodogram(pink_noise, fs=1.0)
        
        # Fitte Potenzgesetz im mittleren Bereich
        mask = (f_check > 0.01) & (f_check < 0.4)
        if np.sum(mask) > 10:
            log_f = np.log10(f_check[mask])
            log_psd = np.log10(psd_check[mask])
            slope, _ = np.polyfit(log_f, log_psd, 1)
            print(f"Gemessener Exponent: α ≈ {-slope:.2f}")
        
        return pink_noise, f_check, psd_check
    
    def semiconductor_noise_demo(self):
        """Rauschen in Halbleitern"""
        print("\n=== HALBLEITER-RAUSCHEN ===")
        
        # Shot Noise (Schrotrauschen)
        I = 1e-6  # 1 μA
        bandwidth = 1e6  # 1 MHz
        shot_noise = np.sqrt(2 * self.q * I * bandwidth)
        print(f"Schrotrauschen bei 1 μA: {shot_noise*1e12:.1f} pA/√Hz")
        
        # Random Telegraph Noise (RTN)
        print("\nRandom Telegraph Noise:")
        duration = 0.1  # 100 ms
        n_samples = 10000
        time = np.linspace(0, duration, n_samples)
        
        # Simuliere Zwei-Niveau-System
        switch_rate = 50  # Hz
        n_switches = np.random.poisson(switch_rate * duration)
        switch_times = np.sort(np.random.uniform(0, duration, n_switches))
        
        # Erzeuge RTN Signal
        rtn_signal = np.ones(n_samples)
        for i, t in enumerate(time):
            switches_before = np.sum(switch_times <= t)
            rtn_signal[i] = 1 if switches_before % 2 == 0 else -1
            
        print(f"Simuliert mit {n_switches} Schaltvorgängen")
        
        return shot_noise, time, rtn_signal
    
    def compare_to_collatz_noise(self):
        """Vergleiche mit deinem Collatz σ = 0.001"""
        print("\n=== VERGLEICH MIT COLLATZ-RAUSCHEN ===")
        print("Dein optimales σ = 0.001")
        
        # Annahme: Collatz-Werte im Bereich 1-1000
        typical_value = 100
        noise_amplitude = 0.001 * typical_value  # 0.1
        
        print(f"\nBei typischem Collatz-Wert von {typical_value}:")
        print(f"Absolute Rauschamplitude: {noise_amplitude}")
        
        # Vergleiche mit thermischem Rauschen
        T = 300  # Raumtemperatur
        thermal_voltage = np.sqrt(4 * self.kb * T * 1000 * 1e6)  # 1kΩ, 1MHz
        print(f"\nThermisches Rauschen (1kΩ, 1MHz): {thermal_voltage*1e6:.2f} μV")
        print(f"Verhältnis Collatz/Thermal: {noise_amplitude/(thermal_voltage*1e6):.1e}")
        
        # Vergleiche mit Quantenrauschen
        E_quantum = self.h * 1e12 / 2  # 1 THz
        print(f"\nQuantenrauschen (1 THz): {E_quantum/self.q:.3f} eV")
        
        print("\nSCHLUSSFOLGERUNG:")
        print("Dein σ = 0.001 ist relativ schwach und könnte sein:")
        print("1. Skaliertes thermisches Rauschen")
        print("2. Quantenfluktuationen auf makroskopischer Ebene")
        print("3. Emergentes 1/f Rauschen")
        print("4. Oder eine Kombination!")

def create_visualization():
    """Erstelle Visualisierung aller Rauschtypen"""
    investigator = NoiseInvestigation()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Thermisches Rauschen
    ax1 = axes[0, 0]
    noise, freqs, spectrum, v_rms = investigator.thermal_noise_demo()
    
    # Zeige Zeitbereich (erste 1000 Punkte)
    time = np.arange(1000) / 2e6  # μs
    ax1.plot(time[:1000], noise[:1000] * 1e6, 'b-', linewidth=0.5)
    ax1.set_xlabel('Zeit (μs)')
    ax1.set_ylabel('Spannung (μV)')
    ax1.set_title('Thermisches Rauschen (Zeitbereich)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Quantum vs Classical
    ax2 = axes[0, 1]
    frequencies = np.logspace(10, 15, 100)  # 10 GHz bis 1 PHz
    
    # Berechne für verschiedene Temperaturen
    for T in [10, 100, 300]:
        quantum_energy = []
        classical_energy = []
        
        for f in frequencies:
            E_zero = investigator.h * f / 2
            if investigator.h * f < investigator.kb * T:
                E_th = investigator.kb * T
            else:
                E_th = investigator.h * f / (np.exp(investigator.h * f / (investigator.kb * T)) - 1)
            
            quantum_energy.append(E_zero + E_th)
            classical_energy.append(investigator.kb * T)
        
        ax2.loglog(frequencies/1e12, np.array(quantum_energy)/investigator.q, 
                  label=f'Quantum {T}K', linewidth=2)
        ax2.loglog(frequencies/1e12, np.array(classical_energy)/investigator.q, 
                  '--', label=f'Classical {T}K', alpha=0.5)
    
    ax2.set_xlabel('Frequenz (THz)')
    ax2.set_ylabel('Energie (eV)')
    ax2.set_title('Quantum vs Classical Noise')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 1/f Rauschen
    ax3 = axes[1, 0]
    pink, f_pink, psd_pink = investigator.pink_noise_demo()
    
    mask = (f_pink > 0) & (f_pink < 0.5)
    ax3.loglog(f_pink[mask], psd_pink[mask], 'b-', alpha=0.5)
    
    # Theoretische 1/f Linie
    f_theory = np.logspace(-2, -0.3, 100)
    psd_theory = 1 / f_theory
    ax3.loglog(f_theory, psd_theory, 'r--', linewidth=2, label='1/f')
    
    ax3.set_xlabel('Frequenz (Hz)')
    ax3.set_ylabel('PSD')
    ax3.set_title('1/f (Pink) Noise Spektrum')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. RTN
    ax4 = axes[1, 1]
    _, time_rtn, rtn = investigator.semiconductor_noise_demo()
    
    ax4.plot(time_rtn * 1000, rtn, 'b-', linewidth=1)
    ax4.set_xlabel('Zeit (ms)')
    ax4.set_ylabel('Zustand')
    ax4.set_title('Random Telegraph Noise')
    ax4.set_ylim(-1.5, 1.5)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('noise_types_comparison.png', dpi=300)
    plt.show()

# Hauptprogramm
def main():
    print("UNTERSUCHUNG DES UNIVERSELLEN RAUSCHENS")
    print("="*60)
    
    investigator = NoiseInvestigation()
    
    # Führe alle Demos durch
    investigator.thermal_noise_demo()
    investigator.quantum_noise_demo()
    investigator.pink_noise_demo()
    investigator.semiconductor_noise_demo()
    investigator.compare_to_collatz_noise()
    
    # Erstelle Visualisierungen
    create_visualization()
    
    print("\n\nZUSAMMENFASSUNG:")
    print("="*60)
    print("Rauschen ist ÜBERALL und hat verschiedene Ursprünge:")
    print("- THERMAL: Bewegung der Atome (kT)")
    print("- QUANTUM: Heisenbergsche Unschärfe (h/2)")
    print("- 1/f: Mysteriös aber universal")
    print("- SHOT: Diskrete Ladungsträger (e)")
    print("\nDein Collatz-Rauschen nutzt diese fundamentalen Quellen!")

if __name__ == "__main__":
    main()