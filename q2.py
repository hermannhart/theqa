from braket.aws import AwsDevice
from braket.circuits import Circuit
import numpy as np
import matplotlib.pyplot as plt

# Konfiguration für IonQ Forte
DEVICE_ARN = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1"
NUM_QUBITS = 2
NUM_SHOTS = 1000

class QuantumTripleRule:
    def __init__(self):
        self.device = AwsDevice(DEVICE_ARN)
    
    def prepare_quantum_state(self, circuit):
        """Zustandspräparation (ρ)"""
        # Optimiert für Ionenfallen-QC
        circuit.h(0)
        circuit.cnot(0, 1)
        return circuit
    
    def apply_decoherence(self, circuit, noise_level):
        """Dekoherenzkanal (D) - angepasst für IonQ"""
        # IonQ-spezifische Rauschmodellierung
        circuit.rx(0, noise_level)
        circuit.rz(0, noise_level/2)
        circuit.rx(1, noise_level)
        circuit.rz(1, noise_level/2)
        return circuit
    
    def measure_observable(self, circuit):
        """Messung (M)"""
        circuit.h(0)
        circuit.h(1)
        circuit.measure([0, 1])
        return circuit
    
    def analyze_quantum_threshold(self):
        """Hauptanalyse"""
        noise_levels = np.linspace(0, np.pi, 20)  # Weniger Punkte für QPU
        entropies = []
        
        for noise in noise_levels:
            circuit = Circuit()
            circuit = self.prepare_quantum_state(circuit)
            circuit = self.apply_decoherence(circuit, noise)
            circuit = self.measure_observable(circuit)
            
            result = self.device.run(circuit, shots=NUM_SHOTS).result()
            counts = result.measurement_counts
            
            # Von-Neumann-Entropie
            probabilities = [count/NUM_SHOTS for count in counts.values()]
            entropy = -np.sum([p * np.log2(p) if p > 0 else 0 for p in probabilities])
            entropies.append(entropy)
        
        return noise_levels, entropies

def main():
    print("Starting Quantum Triple Rule Analysis on IonQ Forte...")
    
    analyzer = QuantumTripleRule()
    noise_levels, entropies = analyzer.analyze_quantum_threshold()
    
    # Kritischen Schwellwert finden
    gradient = np.gradient(entropies)
    critical_idx = np.argmax(gradient)
    sigma_c = noise_levels[critical_idx]
    
    print(f"Quantenkritischer Schwellwert σc^(Q) = {sigma_c:.3f}")
    print(f"Verhältnis zu π: {sigma_c/np.pi:.3f}")
    
    # Visualisierung
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, entropies, 'b-', label='Quantenentropie')
    plt.axvline(x=np.pi/2, color='r', linestyle='--', label='Klassische Grenze π/2')
    plt.axvline(x=np.pi, color='g', linestyle='--', label='Quantengrenze π')
    plt.xlabel('Noise Level σ')
    plt.ylabel('Von-Neumann-Entropie')
    plt.title('Quantum Triple Rule: IonQ Forte Phase Transition')
    plt.legend()
    plt.grid(True)
    plt.savefig('quantum_triple_rule_ionq.png')
    plt.close()

if __name__ == "__main__":
    main()