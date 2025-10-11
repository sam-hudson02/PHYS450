import numpy as np
from scipy.constants import hbar as hbar_SI, e as eC
import matplotlib.pyplot as plt
from alive_progress import alive_bar


class Simulation:
    def __init__(self, n: int, hop: np.ndarray, mag: np.ndarray,
                 d: float = 0.346e-9, max_qx_qc: float = 1.5,
                 samples: int = 400, hitrate: int = 10):
        self.n = n
        self.d = d
        self.hop = hop
        self.mag = mag
        self.samples = samples
        self.hitrate = hitrate
        self.hbar_ev = hbar_SI / eC
        self.a = 0.246e-9
        self.v = (np.sqrt(3) * self.hop[0] * self.a) / (2 * self.hbar_ev)
        self.qc = self.hop[1] / (self.v * self.hbar_ev)
        self.max_qx_qc = max_qx_qc
        max_qx = max_qx_qc * self.qc
        self.qxs = np.linspace(0, max_qx, self.samples)
        print(f"v = {self.v:.3e} m/s")
        print(f"q_c = {self.qc:.3e} 1/m")
        z_typ = (self.n / 2) * self.d
        B_est = (hbar_SI * self.qc) / (eC * z_typ)
        print(f"Expected bifurcation field scale: {B_est:.2f} T")
        print(f"Actual magnetic field: {mag[0]:.2f} T")

    def get_pi(self, q: np.ndarray, i: int, dag: bool) -> np.ndarray:
        """Compute pi term for Hamiltonian matrix.

        Args:
            q (np.ndarray): A 2D momentum vector.
            i (int): layer number.
            dag (bool): If True, compute the conjugate transpose.
        Returns:
            float: The pi term for layer n.
        """
        bx, by = self.mag
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
            pi = self.get_pi(q, i, dag=False)
            pi_dagger = self.get_pi(q, i, dag=True)
            ham[2 * i, 2 * i + 1] = self.hbar_ev * self.v * pi
            ham[2 * i + 1, 2 * i] = self.hbar_ev * self.v * pi_dagger
            if i < self.n - 1:
                ham[2 * i + 1, 2 * (i + 1)] = gamma_1
                ham[2 * (i + 1), 2 * i + 1] = gamma_1
        return ham

    def plot_graph(self, energies, qx_vals, save):
        fig, ax = plt.subplots(figsize=(7, 5))

        to_plot = energies.shape[1] // (2 * self.hitrate)
        mid = energies.shape[1] // 2
        for n in range(to_plot):
            band = n * self.hitrate
            upper_band = mid + band
            lower_band = mid - band - 1
            ax.plot(qx_vals, energies[:, upper_band],
                    color='black', lw=0.9, alpha=0.9)
            ax.plot(qx_vals, energies[:, lower_band],
                    color='black', lw=0.9, alpha=0.9)

        ax.set_xlabel(r"$q_x / q_c$", fontsize=12)
        ax.set_ylabel(r"$\epsilon / \gamma_1$", fontsize=12)
        ax.set_title(f"$B_x = {self.mag[0]}$T", fontsize=13)
        ax.axhline(0, color='gray', lw=0.6, ls='--')  # zero-energy line
        ax.set_xlim(0, 1.5)
        ax.set_ylim(-2, 2)
        ax.grid(False)
        plt.tight_layout()
        if save:
            title = f"./plots/RMG_{self.mag[0]}_{self.n}.png"
            print(f"Saving figure as {title}")
            plt.savefig(title, dpi=300)
        plt.show()

    def save(self, energies, qx_vals):
        np.savez(f"./data/RMG_{self.mag[0]}_{self.n}.npz",
                 energies=energies, qx_vals=qx_vals)
    
    def open(self, filename):
        data = np.load(filename)
        energies = data['energies']
        qx_vals = data['qx_vals']
        return energies, qx_vals

    def run(self, save: bool = False):
        energies = np.zeros((self.samples, 2 * self.n))
        with alive_bar(self.samples, title="Computing bands") as bar:
            for i, qx in enumerate(self.qxs):
                q = np.array([qx, 0])
                ham = self.hamiltonian(q)
                evals = np.linalg.eigvalsh(ham)
                energies[i, :] = np.sort(evals) / self.hop[1]
                bar()

        self.plot_graph(energies, self.qxs / self.qc, save)
