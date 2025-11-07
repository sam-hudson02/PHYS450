from enum import Enum
import numpy as np
from scipy.constants import hbar as hbar_SI, e as eC

class DisorderType(Enum):
    NONE = 0
    ONSITE = 1
    HOPPING = 2
    BOTH = 3

class Hamiltonian:
    def __init__(self, q: np.ndarray, n: int, hop: np.ndarray,
                 mag: np.ndarray, onsite: float = 0.0,
                 d: float = 0.346e-9,
                 disorder_type: DisorderType = DisorderType.NONE,
                 disorder_strength: float = 0.0):
        self.q = q
        self.n = n
        self.d = d
        self.hop = hop
        self.mag = mag
        self.onsite_energy = onsite
        self.disorder_type = disorder_type
        self.disorder_strength = disorder_strength
        self.hbar_ev = hbar_SI / eC
        self.onsite_disorder_array = None
        self.hopping_disorder_array = None
        self.a = 0.246e-9  # lattice constant in meters
        self.v = (np.sqrt(3) * self.hop[0] * self.a) / (2 * self.hbar_ev)
        self.qc = self.hop[1] / (self.v * self.hbar_ev)

    def _get_pi(self, q: np.ndarray, mag: np.ndarray, i: int, dag: bool) -> np.ndarray:
        """Compute pi term for Hamiltonian matrix.

        Args:
            q (np.ndarray): A 2D momentum vector.
            i (int): layer number.
            dag (bool): If True, compute the conjugate transpose.
        Returns:
            float: The pi term for layer n.
        """
        bx, by = mag
        qx, qy = q
        zn = (i - (self.n - 1) / 2) * self.d
        shift = eC * zn / hbar_SI
        if dag:
            result = (qx - shift * by) - \
                1j * (qy + shift * bx)
        else:
            result = (qx + shift * by) + \
                1j * (qy - shift * bx)
        return result

    def _construct_matrix(self) -> np.ndarray:
        """
        Compute the n layer Hamiltonian matrix for a given momentum vector
        ,coupling parameters and magnetic field.

        Args:
            q (np.ndarray): A 2D momentum vector.

        Returns:
            np.ndarray: The Hamiltonian matrix.
        """
        _, gamma_1 = self.hop
        ham = np.zeros((2 * self.n, 2 * self.n), dtype=complex)
        for i in range(self.n):
            # intralayer hopping
            pi = self._get_pi(self.q, self.mag, i, dag=False)
            pi_dagger = self._get_pi(self.q, self.mag, i, dag=True)
            ham[2 * i, 2 * i + 1] = self.hbar_ev * self.v * pi
            ham[2 * i + 1, 2 * i] = self.hbar_ev * self.v * pi_dagger

            # onsite energy
            ham[2 * i, 2 * i] += self.onsite_energy
            ham[2 * i + 1, 2 * i + 1] -= self.onsite_energy

            # interlayer hopping
            if i < self.n - 1:
                ham[2 * i + 1, 2 * (i + 1)] = gamma_1
                ham[2 * (i + 1), 2 * i + 1] = gamma_1

        # onsite disorder
        if self.disorder_type == DisorderType.ONSITE \
        or self.disorder_type == DisorderType.BOTH:
            if self.onsite_disorder_array is None:
                self.onsite_disorder_array = np.random.uniform(
                    -self.disorder_strength,
                    self.disorder_strength,
                    size=self.n
                )
            for i in range(self.n):
                onsite_energy = self.onsite_disorder_array[i]
                ham[2 * i, 2 * i] += onsite_energy
                ham[2 * i + 1, 2 * i + 1] += onsite_energy

        # hopping disorder
        if self.disorder_type == DisorderType.HOPPING \
        or self.disorder_type == DisorderType.BOTH:
            if self.hopping_disorder_array is None:
                self.hopping_disorder_array = np.random.uniform(
                    -self.disorder_strength,
                    self.disorder_strength,
                    size=self.n - 1
                )
            for i in range(self.n - 1):
                hopping_variation = self.hopping_disorder_array[i]
                ham[2 * i + 1, 2 * (i + 1)] += hopping_variation
                ham[2 * (i + 1), 2 * i + 1] += hopping_variation
        return ham

    def matrix(self) -> np.ndarray:
        return self._construct_matrix()
    
    def eigh(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the hamiltonian.
        Args:
            q (np.ndarray): A 2D momentum vector.
            mag (np.ndarray): A 2D magnetic field vector.
        Returns:
            tuple[np.ndarray, np.ndarray]: Eigenvalues, Eigenvectors
        """
        ham = self.matrix()
        evals, evecs = np.linalg.eigh(ham)
        return evals, evecs

    def update_q(self, q: np.ndarray):
        self.q = q

    def evals(self) -> np.ndarray:
        return np.linalg.eigvalsh(self.matrix())

    def evecs(self) -> np.ndarray:
        v = np.linalg.eigh(self.matrix())[1]
        return v

    def zero_energy_states(self) -> tuple[list[np.ndarray], list[complex]]:
        evals, evecs = self.eigh()
        zero_states_vec = []
        zero_states_e = []
        for i, val in enumerate(evals):
            print(f"Eigenvalue {i}: {val}")
            if np.isclose(val, 0, atol=1e-6):
                zero_state_vec = evecs[:, i]
                zero_states_vec.append(zero_state_vec)
                zero_states_e.append(val)
        return zero_states_vec, zero_states_e

    def egap(self) -> float:
        evals = self.evals()
        ind = None
        for i, eval in enumerate(evals):
            if eval >= 0:
                ind = i
                break
        if ind is None:
            raise ValueError("No positive eigenvalues found to compute energy gap.")
        if ind == 0:
            raise ValueError("No negative eigenvalues found to compute energy gap.")
        above = np.abs(evals[ind+1] - evals[ind])
        below = np.abs(evals[ind] - evals[ind - 1])
        gap = min(above, below)
        return gap


