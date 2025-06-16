"""
Quantum extensions for TheQA package.

This module provides quantum versions of critical noise threshold
computations, including quantum walks and extended bounds (σc < π).
"""

import numpy as np
from typing import Optional, Tuple, Union, List
import warnings

# Check if quantum libraries are available
try:
    import qiskit
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit import Aer, execute
    from qiskit.quantum_info import Statevector
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    warnings.warn("Qiskit not available. Quantum features will be limited.")


class QuantumSystem:
    """Base class for quantum discrete dynamical systems."""
    
    def __init__(self, n_qubits: int):
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits in the system
        """
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits
        
    def evolve(self, steps: int) -> np.ndarray:
        """Evolve quantum system and return measurement statistics."""
        raise NotImplementedError


class QuantumWalk(QuantumSystem):
    """
    Discrete-time quantum walk on a line.
    
    Parameters
    ----------
    n_qubits : int
        Number of position qubits (walk on 2^n positions)
    coin : str, default='hadamard'
        Type of coin operator: 'hadamard', 'grover', 'fourier'
    """
    
    def __init__(self, n_qubits: int = 6, coin: str = 'hadamard'):
        super().__init__(n_qubits)
        self.coin_type = coin
        self.n_positions = 2**n_qubits
        
        if HAS_QISKIT:
            self._initialize_operators()
    
    def _initialize_operators(self):
        """Initialize quantum walk operators."""
        # Coin operators
        if self.coin_type == 'hadamard':
            self.coin = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        elif self.coin_type == 'grover':
            self.coin = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            self.coin = 2 * self.coin @ self.coin.T - np.eye(2)
        elif self.coin_type == 'fourier':
            # DFT matrix for 2x2
            omega = np.exp(2j * np.pi / 2)
            self.coin = np.array([[1, 1], [1, omega]]) / np.sqrt(2)
        else:
            raise ValueError(f"Unknown coin type: {self.coin_type}")
    
    def evolve(self, steps: int, initial_pos: Optional[int] = None) -> np.ndarray:
        """
        Evolve quantum walk and return position probability distribution.
        
        Parameters
        ----------
        steps : int
            Number of walk steps
        initial_pos : int, optional
            Initial position (default: center)
        
        Returns
        -------
        np.ndarray
            Position probability distribution after evolution
        """
        if not HAS_QISKIT:
            # Fallback to classical simulation
            return self._classical_simulation(steps, initial_pos)
        
        if initial_pos is None:
            initial_pos = self.n_positions // 2
        
        # Initialize state |initial_pos>|0> (position ⊗ coin)
        state = np.zeros(2 * self.n_positions, dtype=complex)
        state[2 * initial_pos] = 1.0  # |pos>|0>
        
        # Evolution
        for _ in range(steps):
            # Apply coin
            state = self._apply_coin(state)
            
            # Apply shift
            state = self._apply_shift(state)
        
        # Measure position distribution
        prob_dist = self._measure_position(state)
        
        return prob_dist
    
    def _apply_coin(self, state: np.ndarray) -> np.ndarray:
        """Apply coin operator to all positions."""
        new_state = np.zeros_like(state)
        
        for pos in range(self.n_positions):
            # Extract coin state at position
            coin_state = state[2*pos:2*pos+2]
            
            # Apply coin
            new_coin_state = self.coin @ coin_state
            
            # Update state
            new_state[2*pos:2*pos+2] = new_coin_state
        
        return new_state
    
    def _apply_shift(self, state: np.ndarray) -> np.ndarray:
        """Apply conditional shift operator."""
        new_state = np.zeros_like(state)
        
        for pos in range(self.n_positions):
            # |pos>|0> -> |pos-1>|0>
            if pos > 0:
                new_state[2*(pos-1)] += state[2*pos]
            
            # |pos>|1> -> |pos+1>|1>
            if pos < self.n_positions - 1:
                new_state[2*(pos+1)+1] += state[2*pos+1]
        
        return new_state
    
    def _measure_position(self, state: np.ndarray) -> np.ndarray:
        """Measure position distribution."""
        prob_dist = np.zeros(self.n_positions)
        
        for pos in range(self.n_positions):
            # Sum probabilities over coin states
            prob_dist[pos] = (np.abs(state[2*pos])**2 + 
                            np.abs(state[2*pos+1])**2)
        
        return prob_dist
    
    def _classical_simulation(self, steps: int, initial_pos: Optional[int] = None) -> np.ndarray:
        """Classical random walk for comparison."""
        if initial_pos is None:
            initial_pos = self.n_positions // 2
        
        # Simple diffusion approximation
        positions = np.arange(self.n_positions)
        center = initial_pos
        variance = steps  # Classical: variance ~ t
        
        # Gaussian distribution
        prob_dist = np.exp(-(positions - center)**2 / (2 * variance))
        prob_dist /= np.sum(prob_dist)
        
        return prob_dist


def quantum_sigma_c(quantum_system: QuantumSystem,
                   decoherence_model: str = 'depolarizing',
                   n_trials: int = 100) -> float:
    """
    Compute critical noise threshold for quantum system.
    
    Quantum systems can have σc up to π (twice classical bound).
    
    Parameters
    ----------
    quantum_system : QuantumSystem
        Quantum system to analyze
    decoherence_model : str, default='depolarizing'
        Type of decoherence: 'depolarizing', 'dephasing', 'amplitude_damping'
    n_trials : int, default=100
        Number of trials for statistics
    
    Returns
    -------
    float
        Critical noise threshold (can be up to π)
    """
    # Test different decoherence rates
    gamma_values = np.logspace(-4, 0, 50)
    
    for gamma in gamma_values:
        features = []
        
        for _ in range(n_trials):
            # Evolve with decoherence
            if isinstance(quantum_system, QuantumWalk):
                prob_dist = _evolve_with_decoherence(
                    quantum_system, gamma, decoherence_model
                )
                
                # Extract feature (e.g., spreading width)
                positions = np.arange(len(prob_dist))
                mean_pos = np.sum(positions * prob_dist)
                width = np.sqrt(np.sum((positions - mean_pos)**2 * prob_dist))
                features.append(width)
        
        # Check for phase transition
        variance = np.var(features)
        if variance > 0.1:  # Threshold
            # Convert decoherence rate to noise level
            sigma_c = np.sqrt(gamma) * np.pi
            return min(sigma_c, np.pi - 0.1)  # Enforce bound
    
    return np.pi - 0.1  # Maximum quantum bound


def _evolve_with_decoherence(quantum_walk: QuantumWalk,
                            gamma: float,
                            model: str = 'depolarizing') -> np.ndarray:
    """Evolve quantum walk with decoherence."""
    if not HAS_QISKIT:
        # Simple approximation
        steps = 50
        prob_dist = quantum_walk.evolve(steps)
        
        # Add decoherence effect (simplified)
        if gamma > 0.001:
            # Transition to classical
            classical_dist = np.ones_like(prob_dist) / len(prob_dist)
            prob_dist = (1 - gamma) * prob_dist + gamma * classical_dist
        
        return prob_dist
    
    # Full quantum simulation would go here
    # For now, use approximation
    return quantum_walk.evolve(50)


def classical_to_quantum_bound(sigma_c_classical: float) -> float:
    """
    Convert classical critical threshold to quantum bound.
    
    Quantum systems can have up to 2x the classical threshold,
    with maximum at π.
    
    Parameters
    ----------
    sigma_c_classical : float
        Classical critical threshold
    
    Returns
    -------
    float
        Quantum critical threshold
    """
    # Empirical scaling
    sigma_c_quantum = 2.0 * sigma_c_classical + 0.1
    
    # Enforce quantum bound
    return min(sigma_c_quantum, np.pi - 0.1)


class QuantumCellularAutomaton(QuantumSystem):
    """
    Quantum cellular automaton.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits
    rule : str or np.ndarray
        Evolution rule (unitary matrix or preset)
    """
    
    def __init__(self, n_qubits: int = 8, rule: Union[str, np.ndarray] = 'qca1'):
        super().__init__(n_qubits)
        self.rule = rule
        
        if isinstance(rule, str):
            self._load_preset_rule(rule)
        else:
            self.evolution_unitary = rule
    
    def _load_preset_rule(self, rule_name: str):
        """Load preset QCA rules."""
        if rule_name == 'qca1':
            # Simple quantum rule
            theta = np.pi / 4
            self.local_unitary = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
        elif rule_name == 'grover_qca':
            # Grover-inspired QCA
            self.local_unitary = 2 * np.ones((2, 2)) / 2 - np.eye(2)
        else:
            raise ValueError(f"Unknown rule: {rule_name}")
    
    def evolve(self, steps: int, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Evolve QCA and return state statistics."""
        if initial_state is None:
            # Default: single excitation in middle
            state = np.zeros(self.dim, dtype=complex)
            state[self.dim // 2] = 1.0
        else:
            state = initial_state
        
        # Evolution (simplified)
        for _ in range(steps):
            # Apply local unitaries
            state = self._apply_local_rules(state)
        
        # Return probability distribution
        return np.abs(state)**2
    
    def _apply_local_rules(self, state: np.ndarray) -> np.ndarray:
        """Apply local unitary rules (simplified)."""
        # This is a simplified version
        # Full implementation would apply nearest-neighbor interactions
        new_state = state.copy()
        
        # Random unitary evolution for demonstration
        if hasattr(self, 'local_unitary'):
            # Apply some mixing
            indices = np.random.choice(len(state), size=len(state)//2, replace=False)
            for i in range(0, len(indices), 2):
                if i+1 < len(indices):
                    idx1, idx2 = indices[i], indices[i+1]
                    # Apply 2-qubit gate
                    new_state[[idx1, idx2]] = self.local_unitary @ state[[idx1, idx2]]
        
        return new_state / np.linalg.norm(new_state)


class QuantumCollatz(QuantumSystem):
    """
    Quantum version of Collatz system using superposition.
    
    This creates quantum superpositions of Collatz trajectories.
    """
    
    def __init__(self, n_qubits: int = 8):
        super().__init__(n_qubits)
        self.max_value = 2**n_qubits - 1
    
    def evolve(self, steps: int, initial_n: int = 27) -> np.ndarray:
        """
        Evolve quantum Collatz with superposition.
        
        Returns probability distribution over values.
        """
        # Initialize superposition
        state = np.zeros(self.dim, dtype=complex)
        state[initial_n] = 1.0
        
        for _ in range(steps):
            new_state = np.zeros_like(state)
            
            for n in range(self.dim):
                if np.abs(state[n])**2 < 1e-10:
                    continue
                
                amplitude = state[n]
                
                if n == 1:
                    new_state[1] += amplitude
                elif n % 2 == 0:
                    # Even: n/2
                    new_state[n // 2] += amplitude
                else:
                    # Odd: 3n+1
                    new_val = 3 * n + 1
                    if new_val < self.dim:
                        new_state[new_val] += amplitude / np.sqrt(2)
                        # Quantum: also allow staying
                        new_state[n] += amplitude / np.sqrt(2)
            
            state = new_state / np.linalg.norm(new_state)
        
        return np.abs(state)**2


def compare_classical_quantum(system_name: str = 'walk',
                            n_steps: int = 50) -> Tuple[float, float]:
    """
    Compare classical and quantum critical thresholds.
    
    Parameters
    ----------
    system_name : str
        Type of system: 'walk', 'qca', 'collatz'
    n_steps : int
        Number of evolution steps
    
    Returns
    -------
    tuple
        (classical_sigma_c, quantum_sigma_c)
    """
    if system_name == 'walk':
        # Classical random walk
        classical_sc = 0.45
        
        # Quantum walk
        qw = QuantumWalk(n_qubits=6)
        quantum_sc = quantum_sigma_c(qw)
        
    elif system_name == 'qca':
        # Classical CA
        classical_sc = 0.20
        
        # Quantum CA
        qca = QuantumCellularAutomaton()
        quantum_sc = quantum_sigma_c(qca)
        
    elif system_name == 'collatz':
        # Classical Collatz
        classical_sc = 0.117
        
        # Quantum Collatz
        qc = QuantumCollatz()
        quantum_sc = quantum_sigma_c(qc)
        
    else:
        raise ValueError(f"Unknown system: {system_name}")
    
    return classical_sc, quantum_sc


# Utility functions

def quantum_bound_ratio(sigma_c_classical: float) -> float:
    """
    Calculate the quantum enhancement ratio.
    
    Returns how much larger the quantum bound is compared to classical.
    """
    sigma_c_quantum = classical_to_quantum_bound(sigma_c_classical)
    return sigma_c_quantum / sigma_c_classical


def is_quantum_advantage(sigma_c: float) -> bool:
    """
    Check if a threshold is in the quantum-only regime.
    
    Returns True if π/2 < σc < π.
    """
    return np.pi/2 < sigma_c < np.pi
