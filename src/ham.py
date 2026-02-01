from enum import Enum
import numpy as np
from scipy.constants import h, hbar as hbar_SI, e as eC

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
                 disorder_strength: float = 0.0,
                 bernal_fault: bool = False,
                 bernal_layer: int = 2):
        print(f"q: {q}")
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
        self.bernal_fault = bernal_fault
        self.bernal_layer = bernal_layer
        self._matrix = self._construct_matrix()
        threshold = self.soliton_threshold()
        zero_thresh = self.zero_energy_threshold()
        print(f"Hamiltonian matrix constructed for n={n} layers.")
        print(f"Soliton formation threshold magnetic field: {threshold:.2f} T")
        print(f"Zero-energy states threshold magnetic field: {zero_thresh:.2f} T")

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

        if self.bernal_fault:
            ham = self.construct_bf(ham)
        self._matrix = ham
        return ham

    def construct_bf(self, ham: np.ndarray) -> np.ndarray:
        """
        Constructs a Hamiltonian matrix with a Bernal fault between layers i and i+1.
        Args:
            i (int): The layer index where the fault is introduced.
        Returns:
            np.ndarray: The Hamiltonian matrix with the Bernal fault.
        """
        i = self.bernal_layer
        if i < 0 or i >= self.n - 1:
            raise ValueError(f"Layer index out of bounds for Bernal fault:\
                             \n{i} not in [0, {self.n - 2}]")

        # Remove interlayer hopping between specified layers
        ham[2 * i + 1, 2 * (i + 1)] = 0.0
        ham[2 * (i + 1), 2 * i + 1] = 0.0

        # Add hopping between the other sublattices of the two layers
        ham[2 * i, 2 * (i + 1) + 1] = self.hop[1]
        ham[2 * (i + 1) + 1, 2 * i] = self.hop[1]

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
                ham[2 * i, 2 * (i + 1) + 1] += hopping_variation
                ham[2 * (i + 1) + 1, 2 * i] += hopping_variation

        return ham

    def matrix(self) -> np.ndarray:
        return self._matrix
    
    def eigh(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the hamiltonian.
        Args:
            q (np.ndarray): A 2D momentum vector.
            mag (np.ndarray): A 2D magnetic field vector.
        Returns:
            tuple[np.ndarray, np.ndarray]: Eigenvalues, Eigenvectors
        """
        ham = self._matrix
        evals, evecs = np.linalg.eigh(ham)
        return evals, evecs

    def update_q(self, q: np.ndarray) -> np.ndarray:
        """
        Update the momentum vector and reconstruct the Hamiltonian matrix.
        Args:
            q (np.ndarray): A 2D momentum vector.
        Returns:
            np.ndarray: The updated Hamiltonian matrix.
        """
        self.q = q
        return self._construct_matrix()

    def evals(self) -> np.ndarray:
        return np.linalg.eigvalsh(self._matrix)

    def evecs(self) -> np.ndarray:
        v = np.linalg.eigh(self.matrix())[1]
        return v

    def zero_energy_states(self) -> tuple[list[np.ndarray], list[complex]]:
        evals, evecs = self.eigh()
        zero_states_vec = []
        zero_states_e = []
        """
        for i, val in enumerate(evals):
            print(f"Eigenvalue {i}: {val}")
            if np.isclose(val, 0, atol=1e-2):
                zero_state_vec = evecs[:, i]
                zero_states_vec.append(zero_state_vec)
                zero_states_e.append(val)
        """
        # two lowest positive and two highest negative eigenvalues
        pos_indices = [i for i, val in enumerate(evals) if val >= 0]
        neg_indices = [i for i, val in enumerate(evals) if val < 0]
        pos_indices.sort(key=lambda i: evals[i])
        neg_indices.sort(key=lambda i: evals[i], reverse=True)
        selected_indices = neg_indices[:2] + pos_indices[:2]
        for i in selected_indices:
            zero_state_vec = evecs[:, i]
            zero_states_vec.append(zero_state_vec)
            zero_states_e.append(evals[i])
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

    def flux_all(self) -> float:
        """
        Calculate the magnetic flux through the multilayer system.
        Returns:
            float: The magnetic flux in Weber.
        """
        bx, _ = self.mag
        flux = self.a * self.d * (self.n - 1) * bx
        return flux

    def soliton_threshold(self) -> float:
        """
        Calculate the soliton formation threshold magnetic field.
        Returns:
            float: The threshold magnetic field in Tesla.
        """
        flux_0 = h / eC
        gamma_1 = self.hop[1]
        gamma_0 = self.hop[0]
        frac = (flux_0 * 2 * gamma_1) / (np.sqrt(3) * np.pi * gamma_0)
        bx = frac / (self.a * self.d * (self.n - 1))
        return bx
    
    def zero_energy_threshold(self):
        """
        Calculate the magnetic field at which zero-energy states disappear.
        Returns:
            float: The threshold magnetic field in Tesla.
        """
        flux_0 = h / eC
        gamma_1 = self.hop[1]
        gamma_0 = self.hop[0]
        frac = (2 * gamma_1 * flux_0 * self.n) / (np.sqrt(3) * np.pi * gamma_0)
        bx = frac / (self.a * self.d * (self.n - 1))
        return bx

    def zero_energy_threshold_bernal(self):
        """
        Calculate the magnetic field at which zero-energy states disappear.
        Returns:
            float: The threshold magnetic field in Tesla.
        """
        n_1 = self.n - (self.bernal_layer + 1)
        n_2 = self.bernal_layer + 1
        flux_0 = h / eC
        gamma_1 = self.hop[1]
        gamma_0 = self.hop[0]
        frac_1 = (2 * gamma_1 * flux_0 * n_1) / (np.sqrt(3) * np.pi * gamma_0)
        frac_2 = (2 * gamma_1 * flux_0 * n_2) / (np.sqrt(3) * np.pi * gamma_0)
        bx_1 = frac_1 / (self.a * self.d * (n_1 - 1))
        bx_2 = frac_2 / (self.a * self.d * (n_2 - 1))
        return bx_1, bx_2

