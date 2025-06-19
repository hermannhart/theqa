from braket.aws import AwsDevice
from braket.circuits import Circuit
import numpy as np

DEVICE = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1"
NUM_SHOTS = 100

def minimal_quantum_triple_rule_test():
    device = AwsDevice(DEVICE)
    noise_levels = [np.pi/4, np.pi/2, 3*np.pi/4]
    results = []
    
    for noise in noise_levels:
        circuit = Circuit()
        
        # Verbesserte Quantenzustandspräparation
        circuit.h(0)  # Superposition
        circuit.cnot(0, 1)  # Verschränkung
        
        # Rauschsimulation
        circuit.rx(0, noise)
        circuit.rx(1, noise)
        
        # Messung
        circuit.measure([0, 1])
        
        # Ausführung
        result = device.run(circuit, shots=NUM_SHOTS).result()
        counts = result.measurement_counts
        
        # Verbesserte Entropieberechnung
        probabilities = [count/NUM_SHOTS for count in counts.values()]
        entropy = -np.sum([p * np.log2(p) if p > 0 else 0 for p in probabilities])
        results.append(entropy)
    
    return noise_levels, results

def main():
    print("Starting minimal Quantum Triple Rule test...")
    noise_levels, results = minimal_quantum_triple_rule_test()
    
    for noise, entropy in zip(noise_levels, results):
        ratio = noise/np.pi
        print(f"Noise σ = {ratio:.3f}π: Entropy = {entropy:.3f}")

if __name__ == "__main__":
    main()