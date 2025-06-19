from braket.aws import AwsDevice
from braket.circuits import Circuit
import numpy as np
import matplotlib.pyplot as plt

# Konfiguration
SV1_DEVICE = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
NUM_QUBITS = 2
NUM_SHOTS = 1000

class QuantumTripleRule:
    def __init__(self):
        self.device = AwsDevice(SV1_DEVICE)
    
    def prepare_quantum_state(self, circuit):
        """Zustandspräparation (ρ)"""
        # Erzeuge verschränkten Bell-Zustand
        circuit.h(0)
        circuit.cnot(0, 1)
        return circuit
    
    def apply_decoherence(self, circuit, noise_level):
        """Dekoherenzkanal (D)"""
        # Amplitudendämpfung
        circuit.rx(0, noise_level)
        circuit.rx(1, noise_level)
        # Dephasierung
        circuit.rz(0, noise_level/2)
        circuit.rz(1, noise_level/2)
        return circuit
    
    def measure_observable(self, circuit):
        """Messung (M)"""
        # Messung in verschiedenen Basen
        circuit.h(0)  # Basiswechsel
        circuit.h(1)
        # Explizite Messung für jedes Qubit
        circuit.measure([0, 1])  # Korrekte Syntax für Braket
        return circuit
    
    def compute_von_neumann_entropy(self, counts):
        """Berechne Von-Neumann-Entropie"""
        probabilities = [count/NUM_SHOTS for count in counts.values()]
        entropy = -np.sum([p * np.log2(p) if p > 0 else 0 for p in probabilities])
        return entropy
    
    def analyze_quantum_threshold(self):
        """Hauptanalyse für Quantensysteme"""
        # Teste bis π (Quantengrenze)
        noise_levels = np.linspace(0, np.pi, 30)
        entropies = []
        
        for noise in noise_levels:
            circuit = Circuit()
            
            # Wende Triple Rule an
            circuit = self.prepare_quantum_state(circuit)
            circuit = self.apply_decoherence(circuit, noise)
            circuit = self.measure_observable(circuit)
            
            # Führe Circuit aus
            result = self.device.run(circuit, shots=NUM_SHOTS).result()
            
            # Berechne Entropie
            entropy = self.compute_von_neumann_entropy(result.measurement_counts)
            entropies.append(entropy)
        
        return noise_levels, entropies
    
    def find_critical_threshold(self, noise_levels, entropies):
        """Finde kritischen Schwellwert"""
        gradient = np.gradient(entropies)
        critical_idx = np.argmax(gradient)
        return noise_levels[critical_idx]
    
    def plot_results(self, noise_levels, entropies):
        """Visualisiere Ergebnisse"""
        plt.figure(figsize=(10, 6))
        plt.plot(noise_levels, entropies, 'b-', label='Quantenentropie')
        plt.axvline(x=np.pi/2, color='r', linestyle='--', 
                   label='Klassische Grenze π/2')
        plt.axvline(x=np.pi, color='g', linestyle='--', 
                   label='Quantengrenze π')
        plt.xlabel('Noise Level σ')
        plt.ylabel('Von-Neumann-Entropie')
        plt.title('Quantum Triple Rule: Phase Transition')
        plt.legend()
        plt.grid(True)
        plt.savefig('quantum_triple_rule.png')
        plt.close()

def main():
    print("Starting Quantum Triple Rule Analysis...")
    
    analyzer = QuantumTripleRule()
    noise_levels, entropies = analyzer.analyze_quantum_threshold()
    
    # Finde kritischen Schwellwert
    sigma_c = analyzer.find_critical_threshold(noise_levels, entropies)
    
    print(f"Quantenkritischer Schwellwert σc^(Q) = {sigma_c:.3f}")
    print(f"Verhältnis zu π: {sigma_c/np.pi:.3f}")
    
    # Visualisiere Ergebnisse
    analyzer.plot_results(noise_levels, entropies)
    print("Analysis completed. Results saved to 'quantum_triple_rule.png'")

if __name__ == "__main__":
    main()
