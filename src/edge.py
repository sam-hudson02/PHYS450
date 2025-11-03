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
        self.qc = self.hop[1] / (self.v * self.hbar_ev)

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

    def solve_hamiltonian(self, q: np.ndarray, mag: np.ndarray) -> list[np.ndarray]:
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
        zero_states = []
        for i, val in enumerate(evals):
            print(f"Eigenvalue {i}: {val}")
            if np.isclose(val, 0, atol=1e-2):
                zero_state = evecs[:, i]
                zero_states.append(zero_state)
        return zero_states

    def plot_psi(self, psi: np.ndarray, mag: np.ndarray, i: int):
        """
        Plot the edge state wavefunction as bar graph with a and b sites next to each other both labeled as site j.
        Args:
            psi (np.ndarray): The wavefunction psi.
        """ 

        # Example data
        m = np.arange(1, self.n + 1)  # cell index
        values1 = psi[0::2]  # a sites
        values2 = psi[1::2]  # b sites

        # Bar width
        width = 0.35

        # Create the plot
        _, ax = plt.subplots(figsize=(10, 4))

        # Two bar sets (one slightly shifted)
        ax.bar(m - width/2, values1, width=width, color='gray', edgecolor='black')
        ax.bar(m + width/2, values2, width=width, color='white', edgecolor='black')

        # Axis labels
        ax.set_xlim(0.5, len(m) + 0.5)
        ax.set_xlabel(r'cell index $m$', fontsize=14)
        ax.set_ylim(-0.8, 0.8)
        ax.set_yticks([-0.8, 0, 0.8])
        ax.set_xticks(m)
        ax.set_xticklabels(m)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)

        ax.tick_params(direction='in', top=True, right=True)

        plt.legend([r'a sites', r'b sites'])

        # save plot
        plt.title(f'Edge State Wavefunction\nMagnetic Field: Bx={mag[0]} T, By={mag[1]} T')
        plt.savefig(f'./plots/edge_wavefunction_Bx{mag[0]}_N{self.n}_{i}.png', dpi=300)

    def plot_prob_dist(self, psi: np.ndarray, mag: np.ndarray, i: int):
        """
        Plot the edge state probability density as bar graph.
        Args:
            psi_sq (np.ndarray): The probability density |psi|^2.
        """ 
        psi_sq = np.abs(psi)**2
        j_max = 2 * self.n + 1
        j = np.arange(1, j_max) # atomic site
        plt.figure(figsize=(8, 6))
        plt.xlabel('j')
        plt.ylabel(r'$|\psi|^2$')
        plt.title(f'Edge State Probability Density\nMagnetic Field: Bx={mag[0]} T, By={mag[1]} T')
        plt.bar(j, psi_sq, width=0.8, color='blue', alpha=0.7, label=r'$|\psi|^2$')
        plt.xticks(ticks=np.arange(0, j_max, 20))
        plt.legend()
        plt.savefig(f'./plots/edge_state_Bx{mag[0]}_N{self.n}_{i}.png')

    def run(self, q: np.ndarray, mag: np.ndarray):
        """
        Run the edge state calculation and plot the result.
        Args:
            q (np.ndarray): A 2D momentum vector.
            mag (np.ndarray): A 2D magnetic field vector.
        """
        states = self.solve_hamiltonian(q, mag)
        for i, psi in enumerate(states):
            self.plot_prob_dist(psi, mag, i)
            self.plot_psi(psi, mag, i)

