import numpy as np
from scipy.constants import hbar as hbar_SI, e as eC
import matplotlib.pyplot as plt

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
            np.ndarray: |psi|^2 for the zero energy state.
        """
        ham = self.hamiltonian(q, mag)
        evals, evecs = np.linalg.eigh(ham)
        zero_state = None
        for i, val in enumerate(evals):
            if np.isclose(val, 0, atol=1e-9):
                zero_state = evecs[:, i]
                break
        if zero_state is None:
            raise ValueError("No zero energy state found.")
        psi_sq = np.abs(zero_state)**2
        return psi_sq

    def plot_edge_state(self, psi_sq: np.ndarray, mag: np.ndarray):
        """
        Plot the edge state probability density as bar graph.
        Args:
            psi_sq (np.ndarray): The probability density |psi|^2.
        """ 
        j_max = 2 * self.n + 1
        j = np.arange(1, j_max) # atomic site
        plt.figure(figsize=(8, 6))
        plt.xlabel('j')
        plt.ylabel(r'$|\psi|^2$')
        plt.title(f'Edge State Probability Density\nMagnetic Field: Bx={mag[0]} T, By={mag[1]} T')
        plt.bar(j, psi_sq, width=0.8, color='blue', alpha=0.7, label=r'$|\psi|^2$')
        plt.xticks(ticks=np.arange(0, j_max, 20))
        plt.legend()
        plt.savefig(f'./plots/edge_state_Bx{mag[0]}_N{self.n}.png')

    def run(self, q: np.ndarray, mag: np.ndarray):
        """
        Run the edge state calculation and plot the result.
        Args:
            q (np.ndarray): A 2D momentum vector.
            mag (np.ndarray): A 2D magnetic field vector.
        """
        psi_sq = self.solve_hamiltonian(q, mag)
        self.plot_edge_state(psi_sq, mag)

