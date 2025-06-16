"""
Dynamical systems for TheQA package.

This module provides various discrete dynamical systems for analysis.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, List


class BaseSystem(ABC):
    """Abstract base class for dynamical systems."""
    
    @abstractmethod
    def generate(self, **kwargs) -> np.ndarray:
        """Generate sequence from the system."""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self._get_params()})"
    
    def _get_params(self):
        """Get string representation of parameters."""
        return ""


class CollatzSystem(BaseSystem):
    """
    Collatz conjecture sequences (generalized qn+1).
    
    Parameters
    ----------
    n : int
        Starting value
    q : int, default=3
        Multiplier for odd values (3 for standard Collatz)
    """
    
    def __init__(self, n: int = 27, q: int = 3):
        self.n = n
        self.q = q
    
    def generate(self, max_steps: int = 10000) -> np.ndarray:
        """Generate Collatz sequence until reaching 1 or max_steps."""
        sequence = []
        current = self.n
        steps = 0
        
        while current != 1 and steps < max_steps:
            sequence.append(current)
            if current % 2 == 0:
                current = current // 2
            else:
                current = self.q * current + 1
            steps += 1
        
        if current == 1:
            sequence.append(1)
        
        return np.array(sequence, dtype=float)
    
    def _get_params(self):
        return f"n={self.n}, q={self.q}"


class FibonacciSystem(BaseSystem):
    """
    Fibonacci and generalized Fibonacci sequences.
    
    Parameters
    ----------
    n : int
        Length of sequence to generate
    a : float, default=0
        First initial value
    b : float, default=1
        Second initial value
    """
    
    def __init__(self, n: int = 100, a: float = 0, b: float = 1):
        self.n = n
        self.a = a
        self.b = b
    
    def generate(self) -> np.ndarray:
        """Generate Fibonacci sequence."""
        if self.n <= 0:
            return np.array([])
        elif self.n == 1:
            return np.array([self.a])
        
        sequence = [self.a, self.b]
        for i in range(2, self.n):
            sequence.append(sequence[-1] + sequence[-2])
        
        return np.array(sequence, dtype=float)
    
    def _get_params(self):
        return f"n={self.n}"


class LogisticMap(BaseSystem):
    """
    Discrete logistic map: x_{n+1} = r * x_n * (1 - x_n).
    
    Parameters
    ----------
    r : float
        Growth rate parameter
    x0 : float, default=0.5
        Initial value
    length : int, default=1000
        Length of sequence
    """
    
    def __init__(self, r: float = 3.9, x0: float = 0.5, length: int = 1000):
        self.r = r
        self.x0 = x0
        self.length = length
    
    def generate(self) -> np.ndarray:
        """Generate logistic map sequence."""
        sequence = [self.x0]
        
        for i in range(1, self.length):
            x = sequence[-1]
            sequence.append(self.r * x * (1 - x))
        
        return np.array(sequence)
    
    def _get_params(self):
        return f"r={self.r}, x0={self.x0}"


class TentMap(BaseSystem):
    """
    Tent map: piecewise linear chaotic map.
    
    Parameters
    ----------
    r : float, default=1.5
        Parameter controlling the height
    x0 : float, default=0.4
        Initial value
    length : int, default=1000
        Length of sequence
    """
    
    def __init__(self, r: float = 1.5, x0: float = 0.4, length: int = 1000):
        self.r = r
        self.x0 = x0
        self.length = length
    
    def generate(self) -> np.ndarray:
        """Generate tent map sequence."""
        sequence = [self.x0]
        
        for i in range(1, self.length):
            x = sequence[-1]
            if x < 0.5:
                sequence.append(self.r * x)
            else:
                sequence.append(self.r * (1 - x))
        
        return np.array(sequence)
    
    def _get_params(self):
        return f"r={self.r}"


class JosephusSystem(BaseSystem):
    """
    Josephus problem sequence.
    
    Parameters
    ----------
    n : int
        Number of people in circle
    k : int, default=2
        Count for elimination
    """
    
    def __init__(self, n: int = 41, k: int = 2):
        self.n = n
        self.k = k
    
    def generate(self) -> np.ndarray:
        """Generate Josephus elimination sequence."""
        people = list(range(1, self.n + 1))
        sequence = []
        idx = 0
        
        while len(people) > 0:
            idx = (idx + self.k - 1) % len(people)
            sequence.append(people.pop(idx))
        
        return np.array(sequence, dtype=float)
    
    def _get_params(self):
        return f"n={self.n}, k={self.k}"


class PrimeGapSystem(BaseSystem):
    """
    Sequence of gaps between consecutive primes.
    
    Parameters
    ----------
    n_primes : int
        Number of prime gaps to generate
    """
    
    def __init__(self, n_primes: int = 100):
        self.n_primes = n_primes
    
    def generate(self) -> np.ndarray:
        """Generate prime gap sequence."""
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(np.sqrt(n)) + 1):
                if n % i == 0:
                    return False
            return True
        
        primes = []
        num = 2
        
        while len(primes) < self.n_primes + 1:
            if is_prime(num):
                primes.append(num)
            num += 1
        
        gaps = np.diff(primes)
        return gaps.astype(float)
    
    def _get_params(self):
        return f"n_primes={self.n_primes}"


class CustomSystem(BaseSystem):
    """
    Custom system from user-provided sequence or function.
    
    Parameters
    ----------
    sequence : array-like or callable
        If array-like, used directly as sequence
        If callable, should return sequence when called
    """
    
    def __init__(self, sequence: Union[List, np.ndarray, callable]):
        self.sequence = sequence
    
    def generate(self, **kwargs) -> np.ndarray:
        """Generate or return the custom sequence."""
        if callable(self.sequence):
            return np.array(self.sequence(**kwargs), dtype=float)
        else:
            return np.array(self.sequence, dtype=float)
    
    def _get_params(self):
        if callable(self.sequence):
            return "custom_function"
        else:
            return f"length={len(self.sequence)}"


class CellularAutomaton(BaseSystem):
    """
    Elementary cellular automaton.
    
    Parameters
    ----------
    rule : int
        Rule number (0-255)
    width : int
        Width of the automaton
    steps : int
        Number of time steps
    init_state : str or array-like
        Initial state: 'random', 'single', or custom array
    """
    
    def __init__(self, rule: int = 30, width: int = 101, 
                 steps: int = 100, init_state: str = 'single'):
        self.rule = rule
        self.width = width
        self.steps = steps
        self.init_state = init_state
        
        # Convert rule to binary
        self.rule_binary = format(rule, '08b')
    
    def generate(self) -> np.ndarray:
        """Generate cellular automaton evolution."""
        # Initialize
        if self.init_state == 'random':
            state = np.random.randint(0, 2, self.width)
        elif self.init_state == 'single':
            state = np.zeros(self.width, dtype=int)
            state[self.width // 2] = 1
        else:
            state = np.array(self.init_state, dtype=int)
        
        # Store evolution
        evolution = [state.copy()]
        
        # Evolve
        for _ in range(self.steps - 1):
            new_state = np.zeros_like(state)
            
            for i in range(self.width):
                # Get neighborhood (with periodic boundary)
                left = state[(i - 1) % self.width]
                center = state[i]
                right = state[(i + 1) % self.width]
                
                # Apply rule
                pattern = 4 * left + 2 * center + right
                new_state[i] = int(self.rule_binary[7 - pattern])
            
            state = new_state
            evolution.append(state.copy())
        
        # Convert to 1D sequence (sum of each row)
        sequence = [np.sum(row) for row in evolution]
        return np.array(sequence, dtype=float)
    
    def _get_params(self):
        return f"rule={self.rule}"


class HenonMap(BaseSystem):
    """
    Hénon map: 2D chaotic map projected to 1D.
    
    Parameters
    ----------
    a : float
        First parameter
    b : float
        Second parameter
    x0 : float
        Initial x value
    y0 : float
        Initial y value
    length : int
        Length of sequence
    """
    
    def __init__(self, a: float = 1.4, b: float = 0.3,
                 x0: float = 0.1, y0: float = 0.1, length: int = 1000):
        self.a = a
        self.b = b
        self.x0 = x0
        self.y0 = y0
        self.length = length
    
    def generate(self) -> np.ndarray:
        """Generate Hénon map sequence (x-coordinate)."""
        x, y = self.x0, self.y0
        sequence = [x]
        
        for _ in range(1, self.length):
            x_new = 1 - self.a * x**2 + y
            y_new = self.b * x
            x, y = x_new, y_new
            sequence.append(x)
        
        return np.array(sequence)
    
    def _get_params(self):
        return f"a={self.a}, b={self.b}"


# Convenience functions for quick system generation
def collatz(n=27, q=3, **kwargs):
    """Generate Collatz sequence."""
    return CollatzSystem(n, q).generate(**kwargs)


def fibonacci(n=100, **kwargs):
    """Generate Fibonacci sequence."""
    return FibonacciSystem(n).generate(**kwargs)


def logistic(r=3.9, length=1000, **kwargs):
    """Generate logistic map sequence."""
    return LogisticMap(r, length=length).generate(**kwargs)


def tent(r=1.5, length=1000, **kwargs):
    """Generate tent map sequence."""
    return TentMap(r, length=length).generate(**kwargs)
