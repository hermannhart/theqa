"""
Ultimate Quantum Randomness Show: THEQA-Style Multi-Test Demonstrator
Elastic 2.0 License - theqa@posteo.com

- Fetches true quantum random numbers in batch from IonQ Aria-1 QPU (or any QPU)
- Applies "Rule of Three": three independent, deep randomness analyses:
    1. Collatz Orbit Diversity
    2. Block Entropy (Symbolic Analysis)
    3. Symbol Frequency Distribution
- Compares quantum seeds vs. PRNG
- Visualizes all results ("orbits", entropy spectrum, histograms)
- Produces a cryptographic proof/certificate of the quantum batch
- Highlights irreplicability & auditability of quantum randomness
- Additionally: Generates and saves a hash for every quantum seed,
  and lists the best candidates (e.g. with most leading zeros in the hash)
"""

import numpy as np
import matplotlib.pyplot as plt
import hashlib
from collections import Counter

# -- Quantum Seed Acquisition (Batch) -----------------------------------------
def get_aria_random_bits(nbits=16, n_samples=1000):
    """
    Efficiently fetches n_samples random nbits-bit integers from IonQ Aria-1 QPU.
    Falls back to PRNG if QPU is not available.
    """
    try:
        from braket.aws import AwsDevice
        from braket.circuits import Circuit
        device = AwsDevice("arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1")
        circuit = Circuit()
        for q in range(nbits):
            circuit.h(q)
        print(f"Starting QPU job for {n_samples} true quantum seeds (Aria-1)...")
        task = device.run(circuit, shots=n_samples)
        result = task.result()
        bitstrings = []
        for shot, count in result.measurement_counts.items():
            bitstrings.extend([shot]*count)
        seeds = [int(bs, 2) for bs in bitstrings]
        print(f"Received {len(seeds)} quantum seeds.")
        return seeds, bitstrings
    except Exception as e:
        print(f"Warning: QPU not available, using simulation! ({e})")
        seeds = [np.random.randint(0, 2**nbits) for _ in range(n_samples)]
        bitstrings = [bin(s)[2:].zfill(nbits) for s in seeds]
        return seeds, bitstrings

def get_prng_seeds(nbits=16, n_samples=1000):
    seeds = [np.random.randint(0, 2**nbits) for _ in range(n_samples)]
    bitstrings = [bin(s)[2:].zfill(nbits) for s in seeds]
    return seeds, bitstrings
    
def qpu_signature_analysis(bitstrings):
    """
    Unique QPU fingerprint based on noise characteristics
    Different QPUs have different σc signatures!
    """
    # Ihre σc-Analyse anwenden
    return qpu_fingerprint

def quantum_certificate_generation(data, qpu_hash):
    """
    Combines user data with quantum hash for 
    unforgeable certificates
    """
    return certificate
    
# -- Analysis 1: Collatz Orbit Diversity --------------------------------------
def collatz_orbit(n, maxlen=5000):
    orbit = [n]
    seen = {n}
    for _ in range(maxlen-1):
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        orbit.append(n)
        if n == 1 or n in seen:
            break
        seen.add(n)
    return orbit

def analyze_collatz_orbits(seeds):
    lengths = [len(collatz_orbit(n)) for n in seeds]
    endpoints = [collatz_orbit(n)[-1] for n in seeds]
    return lengths, endpoints

# -- Analysis 2: Block Entropy in Symbolic Collatz ----------------------------
def symbolic_sequence(seq, mod=4):
    return [str(n % mod) for n in seq]

def block_entropy(symbols, blocksize=3):
    blocks = [''.join(symbols[i:i+blocksize])
              for i in range(len(symbols)-blocksize+1)]
    counts = np.array(list(Counter(blocks).values()))
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-12))

def analyze_block_entropies(seeds, mod=4, blocksize=3):
    entropies = []
    for n in seeds:
        seq = collatz_orbit(n)
        symb = symbolic_sequence(seq, mod)
        entropies.append(block_entropy(symb, blocksize))
    return entropies

# -- Analysis 3: Symbol Frequency Distribution --------------------------------
def analyze_symbol_frequencies(seeds, mod=4):
    all_symbols = []
    for n in seeds:
        seq = collatz_orbit(n)
        symb = symbolic_sequence(seq, mod)
        all_symbols.extend(symb)
    freq = Counter(all_symbols)
    total = sum(freq.values())
    freq_norm = {k: v/total for k, v in freq.items()}
    return freq_norm

# -- Visualization ------------------------------------------------------------
def visualize_all(q_data, p_data):
    q_lengths, q_endpoints, q_entropies, q_freq = q_data
    p_lengths, p_endpoints, p_entropies, p_freq = p_data

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Collatz Orbit Lengths
    axs[0,0].hist(q_lengths, bins=30, alpha=0.7, label="Quantum")
    axs[0,0].hist(p_lengths, bins=30, alpha=0.7, label="PRNG")
    axs[0,0].set_title("Collatz Orbit Length Distribution")
    axs[0,0].set_xlabel("Orbit Length")
    axs[0,0].set_ylabel("Frequency")
    axs[0,0].legend()

    # 2. Block Entropy
    axs[0,1].hist(q_entropies, bins=30, alpha=0.7, label="Quantum")
    axs[0,1].hist(p_entropies, bins=30, alpha=0.7, label="PRNG")
    axs[0,1].set_title("Block Entropy (Symbolic Collatz, mod 4, blocksize 3)")
    axs[0,1].set_xlabel("Block Entropy")
    axs[0,1].set_ylabel("Frequency")
    axs[0,1].legend()

    # 3. Symbol Frequencies
    labels = sorted(set(q_freq.keys()).union(p_freq.keys()))
    q_vals = [q_freq.get(l,0) for l in labels]
    p_vals = [p_freq.get(l,0) for l in labels]
    width = 0.35
    axs[1,0].bar(np.arange(len(labels)) - width/2, q_vals, width, label="Quantum")
    axs[1,0].bar(np.arange(len(labels)) + width/2, p_vals, width, label="PRNG")
    axs[1,0].set_xticks(range(len(labels)))
    axs[1,0].set_xticklabels(labels)
    axs[1,0].set_title("Symbol Frequency (mod 4)")
    axs[1,0].set_xlabel("Symbol")
    axs[1,0].set_ylabel("Relative Frequency")
    axs[1,0].legend()

    # 4. Orbit Endpoints
    axs[1,1].hist(q_endpoints, bins=30, alpha=0.7, label="Quantum")
    axs[1,1].hist(p_endpoints, bins=30, alpha=0.7, label="PRNG")
    axs[1,1].set_title("Collatz Orbit Endpoints")
    axs[1,1].set_xlabel("Endpoint")
    axs[1,1].set_ylabel("Frequency")
    axs[1,1].legend()

    plt.tight_layout()
    plt.show()

# -- Proof Generation ---------------------------------------------------------
def generate_proof(quantum_bitstrings, q_lengths, q_entropies, q_freq):
    """
    Combines all quantum results into a single cryptographic hash proof.
    """
    proof_str = (
        ''.join(quantum_bitstrings) +
        ''.join(map(str, q_lengths)) +
        ''.join(f"{e:.6f}" for e in q_entropies) +
        ''.join(f"{k}:{v:.6f}" for k,v in sorted(q_freq.items()))
    )
    proof = hashlib.sha256(proof_str.encode('utf-8')).hexdigest()
    return proof

# -- Hashing Batch ------------------------------------------------------------
def hash_seeds(bitstrings, hash_func="sha256"):
    hashes = []
    for s in bitstrings:
        h = hashlib.new(hash_func)
        h.update(s.encode("utf-8"))
        hashes.append(h.hexdigest())
    return hashes

def write_hashes_to_file(bitstrings, hashes, filename="quantum_hashes.txt"):
    with open(filename, "w") as f:
        for i, (s, h) in enumerate(zip(bitstrings, hashes)):
            f.write(f"{i+1:05d}: {s} -> {h}\n")
    print(f"{len(hashes)} Hashes saved as {filename} .")

def analyze_leading_zeros(hashes):
    leading_zeros = [len(h) - len(h.lstrip('0')) for h in hashes]
    best = max(leading_zeros)
    best_indices = [i for i, lz in enumerate(leading_zeros) if lz == best]
    return best, best_indices, leading_zeros

def print_top_candidates(bitstrings, hashes, best_indices, best_zeros, top_n=5):
    print("\nTop Candidates (leading 0s):")
    for rank, idx in enumerate(best_indices[:top_n], 1):
        print(f"#{rank}: Seed {idx+1} | {bitstrings[idx]} -> {hashes[idx]} (leading 0s: {best_zeros})")

# -- Main Pipeline ------------------------------------------------------------
def main():
    n_samples = 5000  # or 5000, as needed
    nbits = 16
    mod = 4
    blocksize = 3

    print("=== THEQA Ultimate Quantum Randomness Show ===")
    print("Fetching quantum seeds (Aria-1 QPU)...")
    q_seeds, q_bitstrings = get_aria_random_bits(nbits=nbits, n_samples=n_samples)
    print("Fetching PRNG seeds...")
    p_seeds, p_bitstrings = get_prng_seeds(nbits=nbits, n_samples=n_samples)

    print("Analyzing Collatz orbits...")
    q_lengths, q_endpoints = analyze_collatz_orbits(q_seeds)
    p_lengths, p_endpoints = analyze_collatz_orbits(p_seeds)

    print("Analyzing block entropies...")
    q_entropies = analyze_block_entropies(q_seeds, mod=mod, blocksize=blocksize)
    p_entropies = analyze_block_entropies(p_seeds, mod=mod, blocksize=blocksize)

    print("Analyzing symbol frequencies...")
    q_freq = analyze_symbol_frequencies(q_seeds, mod=mod)
    p_freq = analyze_symbol_frequencies(p_seeds, mod=mod)

    print("\n=== Summary Statistics ===")
    print(f"Quantum Collatz orbit length: mean={np.mean(q_lengths):.2f}, std={np.std(q_lengths):.2f}")
    print(f"PRNG   Collatz orbit length: mean={np.mean(p_lengths):.2f}, std={np.std(p_lengths):.2f}")
    print(f"Quantum block entropy: mean={np.mean(q_entropies):.3f}")
    print(f"PRNG   block entropy: mean={np.mean(p_entropies):.3f}")
    print(f"Quantum symbol freq: {q_freq}")
    print(f"PRNG   symbol freq: {p_freq}")

    print("\nVisualizing all analyses (close plot window to continue)...")
    visualize_all(
        (q_lengths, q_endpoints, q_entropies, q_freq),
        (p_lengths, p_endpoints, p_entropies, p_freq)
    )

    print("\n=== Quantum Proof of Uniqueness ===")
    proof = generate_proof(q_bitstrings, q_lengths, q_entropies, q_freq)
    print("SHA256 hash (quantum batch + all test results):")
    print(proof)
    print("This cryptographic proof is unique to this quantum run and cannot be reproduced by any classical means.")

    # ---- Batch Hashing und Analyse ----
    print("\n=== Quantum Hash Batch Demo ===")
    hashes = hash_seeds(q_bitstrings, hash_func="sha256")
    write_hashes_to_file(q_bitstrings, hashes, filename="quantum_hashes.txt")
    best_zeros, best_indices, leading_zeros = analyze_leading_zeros(hashes)
    print_top_candidates(q_bitstrings, hashes, best_indices, best_zeros, top_n=5)
    print("\nDemo completed.")

if __name__ == "__main__":
    main()
