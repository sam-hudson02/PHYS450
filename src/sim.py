import numpy as np
from scipy.constants import e, hbar
import matplotlib.pyplot as plt

HBAR_EV = hbar / e


class Simulation:
    def __init__(self, n: int, hop: np.ndarray, mag: np.ndarray,
                 d: int, max_qx_qc: float = 1.5, samples: int = 400):
        self.n = n
        self.d = d
        self.hop = hop
        self.mag = mag
        self.samples = samples
        self.hbar_ev = hbar / e
        self.a = 0.246e-9
        self.v = (np.sqrt(3) * self.hop[0] * self.a) / (2 * self.hbar_ev)
        self.qc = self.hop[1] / (self.v * self.hbar_ev)
        self.max_qx_qc = max_qx_qc
        max_qx = max_qx_qc * self.qc
        self.qxs = np.linspace(0, max_qx, self.samples)

    def get_pi(self, q: np.ndarray, dag: bool) -> np.ndarray:
        """Compute pi term for Hamiltonian matrix.

        Args:
            dag (bool): If True, compute the conjugate transpose.
        Returns:
            float: The pi term for layer n.
        """
        bx, by = self.mag
        qx, qy = q
        if dag:
            result = (qx - e * self.n * self.d * by / HBAR_EV) - \
                1j * (qy + e * self.n * self.d * bx / HBAR_EV)
        else:
            result = (qx + e * self.n * self.d * by / HBAR_EV) + \
                1j * (qy - e * self.n * self.d * bx / HBAR_EV)
        return result

    def hamiltonian(self, q: np.ndarray) -> np.ndarray:
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
            pi = self.get_pi(q, dag=False)
            pi_dagger = self.get_pi(q, dag=True)
            ham[2 * i, 2 * i + 1] = self.hbar_ev * self.v * pi
            ham[2 * i + 1, 2 * i] = self.hbar_ev * self.v * pi_dagger
            if i < self.n - 1:
                ham[2 * i + 1, 2 * (i + 1)] = gamma_1
                ham[2 * (i + 1), 2 * i + 1] = gamma_1
        return ham

    def plot_graph(self, energies, qx_vals):
        fig, ax = plt.subplots(figsize=(7, 5))

        hitrate = 10
        to_plot = energies.shape[1] // (2 * hitrate)
        mid = energies.shape[1] // 2
        for n in range(to_plot):
            band = n * hitrate
            upper_band = mid + band
            lower_band = mid - band - 1
            ax.plot(qx_vals, energies[:, upper_band],
                    color='black', lw=0.9, alpha=0.9)
            ax.plot(qx_vals, energies[:, lower_band],
                    color='black', lw=0.9, alpha=0.9)

        # --- Aesthetic touches ---
        ax.set_xlabel(r"$q_x / q_c$", fontsize=12)
        ax.set_ylabel(r"$E / \gamma_1$", fontsize=12)
        ax.set_title(
            "Band Structure of Rhombohedral Multilayer Graphene", fontsize=13)
        ax.axhline(0, color='gray', lw=0.6, ls='--')  # zero-energy line
        ax.set_xlim(0, 1.5)
        ax.set_ylim(-2, 2)
        ax.grid(False)
        plt.tight_layout()
        plt.show()

    def run(self):
        energies = np.zeros((self.samples, 2 * self.n))
        for i, qx in enumerate(self.qxs):
            q = np.array([qx, 0])
            ham = self.hamiltonian(q)
            evals = np.linalg.eigvalsh(ham)
            energies[i, :] = np.sort(evals) / self.hop[1]

        self.plot_graph(energies, self.qxs / self.qc)
