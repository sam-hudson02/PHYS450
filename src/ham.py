from enum import Enum
import numpy as np
from scipy.constants import hbar as hbar_SI, e as eC

class DisorderType(Enum):
    NONE = 0
    ONSITE = 1

class Hamiltonian:
    def __init__(self, q: np.ndarray, n: int, hop: np.ndarray,
                 mag: np.ndarray, d: float = 0.346e-9,
                 disorder_type: DisorderType = DisorderType.NONE,
                 disorder_strength: float = 0.0):
        self.q = q
        self.n = n
        self.d = d
        self.hop = hop
        self.mag = mag
        self.disorder_type = disorder_type
        self.disorder_strength = disorder_strength
        self.hbar_ev = hbar_SI / eC
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
            pi = self._get_pi(self.q, self.mag, i, dag=False)
            pi_dagger = self._get_pi(self.q, self.mag, i, dag=True)
            ham[2 * i, 2 * i + 1] = self.hbar_ev * self.v * pi
            ham[2 * i + 1, 2 * i] = self.hbar_ev * self.v * pi_dagger
            if i < self.n - 1:
                ham[2 * i + 1, 2 * (i + 1)] = gamma_1
                ham[2 * (i + 1), 2 * i + 1] = gamma_1
        if self.disorder_type == DisorderType.ONSITE:
            for i in range(self.n):
                onsite_energy = np.random.uniform(-self.disorder_strength, self.disorder_strength)
                ham[i, i] = onsite_energy
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

    @property
    def egap(self) -> float:
        zero_states = self.zero_energy_states()[1]
        if len(zero_states) < 2:
            return 0.0
        if len(zero_states) == 2:
            evals, _ = self.eigh()
            sorted_evals = np.sort(np.abs(evals))
            return sorted_evals[1] - sorted_evals[0]
        raise ValueError("More than two zero energy states found.")

