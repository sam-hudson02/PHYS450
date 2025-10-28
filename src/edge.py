import numpy as np
from scipy.constants import hbar as hbar_SI, e as eC
import matplotlib.pyplot as plt
from alive_progress import alive_bar

class EdgeStates:
    def __init__(self, n: int, hop: np.ndarray,
                 d: float = 0.346e-9):
        self.n = n
        self.d = d
        self.hop = hop
        self.hbar_ev = hbar_SI / eC
        self.a = 0.246e-9
        self.v = (np.sqrt(3) * self.hop[0] * self.a) / (2 * self.hbar_ev)

    def get_pi(self, q: np.ndarray, mag: np.ndarray, i: int, dag: bool) -> np.ndarray:
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

    def hamiltonian(self, q: np.ndarray, mag: np.ndarray) -> np.ndarray:
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
            pi = self.get_pi(q, mag, i, dag=False)
            pi_dagger = self.get_pi(q, mag, i, dag=True)
            ham[2 * i, 2 * i + 1] = self.hbar_ev * self.v * pi
            ham[2 * i + 1, 2 * i] = self.hbar_ev * self.v * pi_dagger
            if i < self.n - 1:
                ham[2 * i + 1, 2 * (i + 1)] = gamma_1
                ham[2 * (i + 1), 2 * i + 1] = gamma_1
        return ham

    def solve_hamiltonian(self, q: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """
        Solve the hamiltonian.
        Args:
            q (np.ndarray): A 2D momentum vector.
            mag (np.ndarray): A 2D magnetic field vector.
        Returns:
            np.ndarray: |psi|^2 for each layer.
        """
        ham = self.hamiltonian(q, mag)
        _, evecs = np.linalg.eigh(ham)
        psi_sq = np.abs(evecs) ** 2
        print(psi_sq.shape)
        print(psi_sq)
        return psi_sq
