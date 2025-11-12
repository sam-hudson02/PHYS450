from numpy.random import f
from ham import DisorderType
import numpy as np
from alive_progress import alive_bar
from scipy.constants import hbar as hbar_SI, e as eC
from ham import Hamiltonian
import matplotlib.pyplot as plt
import os

class Simulation:
    def __init__(self, hop: np.ndarray, mag: np.ndarray, n: int, d: float = 0.346e-9,
                 disorder_type: DisorderType = DisorderType.NONE, disorder_strength: float = 10):
        self.hop = hop
        self.mag = mag
        self.d = d
        self.n = n
        self.a = 0.246e-9
        self.disorder_type = disorder_type
        self.disorder_strength = disorder_strength
        self.L = n * d
        self.lorentzian_width = 0.005 * hop[1]
        hbar_ev = hbar_SI / eC
        self.v = (np.sqrt(3) * self.hop[0] * self.a) / (2 * hbar_ev)
        self.qc = self.hop[1] / (self.v * hbar_ev)
        print(f"v = {self.v:.3e} m/s")
        print(f"q_c = {self.qc:.3e} 1/m")
        z_typ = (self.n / 2) * self.d
        B_est = (hbar_SI * self.qc) / (eC * z_typ)
        print(f"Expected bifurcation field scale: {B_est:.2f} T")
        print(f"Actual magnetic field: {mag[0]:.2f} T")

    def _plot_band_structure(self, energies, qx_vals, hitrate: int = 10, 
                   bernal_fault: bool = False):
        _, ax = plt.subplots(figsize=(7, 5))

        to_plot = energies.shape[1] // (2 * hitrate)
        mid = energies.shape[1] // 2
        for n in range(to_plot):
            band = n * hitrate
            upper_band = mid + band
            lower_band = mid - band - 1
            if bernal_fault and (n == 0 or n == 1):
                ax.plot(qx_vals, energies[:, upper_band],
                        color='blue', lw=1.1, alpha=1.0)
                ax.plot(qx_vals, energies[:, lower_band],
                        color='blue', lw=1.1, alpha=1.0)
            else:
                ax.plot(qx_vals, energies[:, upper_band],
                        color='black', lw=1.0, alpha=1.0)
                ax.plot(qx_vals, energies[:, lower_band],
                        color='black', lw=1.0, alpha=1.0)

        ax.set_xlabel(r"$q_x / q_c$", fontsize=12)
        ax.set_ylabel(r"$\epsilon / \gamma_1$", fontsize=12)
        ax.set_title(f"$B_x = {self.mag[0]}$T", fontsize=13)
        ax.axhline(0, color='gray', lw=0.6, ls='--')  # zero-energy line
        ax.set_xlim(0, 1.5)
        ax.set_ylim(-2, 2)
        ax.grid(False)
        plt.tight_layout()
        title = f"./plots/band_structure/RMG_{self.mag[0]}_{self.n}.png"
        print(f"Saving figure as {title}")
        plt.savefig(title, dpi=300)
        plt.show()

    def _plot_eg(self, eg_onsite_list: list[tuple[float, float]],
                 eg_hopping_list, max_disorder_strength: float):
        _, ax = plt.subplots(figsize=(7, 5))

        # convert to meV
        disorder_strengths = np.linspace(0, max_disorder_strength, len(eg_onsite_list)) * 1e3
        means_onsite = np.array([eg[0] for eg in eg_onsite_list]) * 1e3
        err_onsite = np.array([eg[1] for eg in eg_onsite_list]) * 1e3
        means_hopping = np.array([eg[0] for eg in eg_hopping_list]) * 1e3
        err_hopping = np.array([eg[1] for eg in eg_hopping_list]) * 1e3

        # plot each onsite as empty triangle and hopping as empty cirle
        ax.errorbar(disorder_strengths, means_onsite, yerr=err_onsite,
                    fmt='^', color='black', ecolor='black', elinewidth=1,
                    capsize=4, label='Onsite Disorder')
        ax.errorbar(disorder_strengths, means_hopping, yerr=err_hopping,
                    fmt='o', color='black', ecolor='black', elinewidth=1,
                    capsize=4, label='Hopping Disorder')
        ax.set_xlabel(r"$\delta$ (meV)", fontsize=12)
        ax.set_ylabel(r"$E_g$ (meV)", fontsize=12)
        ax.set_title(f"Energy Gap vs Disorder Strength\n$B_x = {self.mag[0]}$T, N={self.n}", fontsize=13)
        ax.grid(False)
        plt.tight_layout()
        if not os.path.exists("./plots/eg_disorder"):
            os.makedirs("./plots/eg_disorder")
        file = f"./plots/eg_disorder/EG_RMG_{self.mag[0]}_{self.n}.png"
        print(f"Saving figure as {file}")
        plt.savefig(file, dpi=300)

    def _plot_psi(self, psi: np.ndarray, i: int):
        """
        Plot the edge state wavefunction as bar graph with a and b sites next to each other both labeled as site j.
        Args:
            psi (np.ndarray): The wavefunction psi.
        """ 

        m = np.arange(1, self.n + 1)  # cell index
        values1 = psi[0::2]  # a sites
        values2 = psi[1::2]  # b sites

        width = 0.35

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

        if not os.path.exists("./plots/psi_edge"):
            os.makedirs("./plots/psi_edge")

        plt.title(f'Edge State Wavefunction\nMagnetic Field: Bx={self.mag[0]} T, By={self.mag[1]} T')
        plt.savefig(f'./plots/psi_edge/edge_state_Bx{self.mag[0]}_N{self.n}_{i}.png')

    def _plot_prob_dist(self, psi: np.ndarray, i: int):
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
        plt.title(f'Edge State Probability Density\nMagnetic Field: Bx={self.mag[0]} T, By={self.mag[1]} T')
        plt.bar(j, psi_sq, width=0.8, color='blue', alpha=0.7, label=r'$|\psi|^2$')
        plt.xticks(ticks=np.arange(0, j_max, 20))
        plt.legend()
        plt.savefig(f'./plots/edge_state_Bx{self.mag[0]}_N{self.n}_{i}.png')


    def _plot_dos(self, energies: np.ndarray, dos: np.ndarray, max_e: float):
        plt.figure(figsize=(8, 6))
        plt.plot(energies, dos, color='blue', lw=1.5)
        plt.xlabel('Energy (eV)')
        plt.ylabel('Density of States (1/eV)')
        plt.title(f'Density of States\nMagnetic Field: Bx={self.mag[0]} T, By={self.mag[1]} T')
        plt.xlim(-max_e, max_e)
        plt.grid(False)
        if not os.path.exists("./plots/dos"):
            os.makedirs("./plots/dos")
        plt.savefig(f'./plots/dos/dos_Bx{self.mag[0]}_N{self.n}.png')

    def _plot_evals_comparison(self, collected_evals: list[np.ndarray], disorder_strengths: list[float]):
        plt.figure(figsize=(8, 6))
        # disorder sterngth on x axis and eig values on y axis
        for i, evals in enumerate(collected_evals):
            ds = disorder_strengths[i]
            # make each sample a horizontal red line with some width at y = ds, x = eval
            # so multiple samples together look like a bar chart
            plt.scatter([ds] * len(evals), evals / self.hop[1], color='red', s=1, alpha=0.5)
        plt.xlabel('Disorder Strength (meV)')
        plt.ylabel('Eigenvalues (eV)')
        plt.title(f'Eigenvalues vs Disorder Strength\nMagnetic Field: Bx={self.mag[0]} T, By={self.mag[1]} T')
        plt.grid(False)
        if not os.path.exists("./plots/evals_comparison"):
            os.makedirs("./plots/evals_comparison")
        plt.savefig(f'./plots/evals_comparison/evals_Bx{self.mag[0]}_N{self.n}.png')

    def band_structure(self, samples: int, hitrate: int, max_qx_qc = 1.5,
                       onsite_e: float = 0.0, bernal_fault: bool = False,
                       bernal_layer: int = 2):
        energies = np.zeros((samples, 2 * self.n))
        max_qx = max_qx_qc * self.qc
        qxs = np.linspace(0, max_qx, samples)
        ham = Hamiltonian(qxs[0], self.n, self.hop, self.mag, onsite_e, self.d,
                          self.disorder_type, self.disorder_strength)
        with alive_bar(samples, title="Computing bands") as bar:
            for i, qx in enumerate(qxs):
                q = np.array([qx, 0])
                ham.update_q(q)
                if bernal_fault:
                    matrix = ham.bernal_fault(bernal_layer)
                else:
                    matrix = ham.matrix()
                evals = np.linalg.eigvalsh(matrix)
                energies[i, :] = np.sort(evals) / self.hop[1]
                bar()
        self._plot_band_structure(energies, qxs / self.qc, hitrate, bernal_fault)
        return ham

    def eg_disorder(self, max_disorder_strength: float = 10, samples: int=20):
        eg_onsite_list: list[tuple[float, float]] = []
        eg_hopping_list: list[tuple[float, float]] = []
        q = np.array([0.0, 0.0])
        onsite_e = self.hop[1] * 0.1
        for i in range(11):
            disorder_strength = (max_disorder_strength / 10) * i
            egs_onsite: list[float] = []
            egs_hopping: list[float] = []
            for _ in range(samples):
                ham_onsite = Hamiltonian(q, self.n, self.hop, self.mag, onsite_e, self.d,
                              DisorderType.ONSITE, disorder_strength)
                egs_onsite.append(ham_onsite.egap())
                ham_hopping = Hamiltonian(q, self.n, self.hop, self.mag, onsite_e, self.d,
                              DisorderType.HOPPING, disorder_strength)
                egs_hopping.append(ham_hopping.egap())

            mean_onsite = float(np.mean(egs_onsite))
            err_onsite = float(np.std(egs_onsite)) / np.sqrt(samples)
            print(f"Onsite Disorder Strength: {disorder_strength} eV, Mean EG: {mean_onsite} eV")
            eg_onsite_list.append((mean_onsite, err_onsite))

            mean_hopping = float(np.mean(egs_hopping))
            err_hopping = float(np.std(egs_hopping)) / np.sqrt(samples)
            print(f"Hopping Disorder Strength: {disorder_strength} eV, Mean EG: {mean_hopping} eV")
            eg_hopping_list.append((mean_hopping, err_hopping))
        self._plot_eg(eg_onsite_list, eg_hopping_list, max_disorder_strength)

    def psi_edge(self, q: np.ndarray):
        ham = Hamiltonian(q, self.n, self.hop, self.mag, self.d)
        zero_states = ham.zero_energy_states()[0]
        for i, psi in enumerate(zero_states):
            self._plot_psi(psi, i)

    def dos(self, q: np.ndarray, max_e_gamma: float = 2, energy_points: int = 400):
        max_e = max_e_gamma * self.hop[1]
        ham = Hamiltonian(q, self.n, self.hop, self.mag, self.d)
        evals = ham.evals()
        energies = np.linspace(-max_e, max_e, energy_points)
        energies_over_gamma = energies / self.hop[1]
        dos = np.zeros(energy_points)
        for i, energy_over_gamma in enumerate(energies):
            e = energy_over_gamma
            for ev in evals:
                dos[i] += (1 / (np.pi)) * (self.lorentzian_width / ((e - ev)**2 + self.lorentzian_width**2))
        self._plot_dos(energies_over_gamma, dos, max_e_gamma)

    def compare_evals(self, q: np.ndarray, max_disorder_strength: float = 50, 
                      samples:int = 200, passes: int = 400):
        collected_evals = []
        disorder_strengths = []
        with alive_bar(samples*passes, title="Collecting eigenvalues") as bar:
            for i in range(samples):
                group = []
                disorder_strength = (max_disorder_strength / samples) * i
                for _ in range(passes):
                    ham = Hamiltonian(q, self.n, self.hop, self.mag, 0.0, self.d,
                                    self.disorder_type, disorder_strength)
                    evals = ham.evals()
                    group.extend(evals)
                    bar()
                collected_evals.append(group)
                disorder_strengths.append(disorder_strength)
        self._plot_evals_comparison(collected_evals, disorder_strengths)


    def prob_edge(self, q: np.ndarray):
        ham = Hamiltonian(q, self.n, self.hop, self.mag, self.d)
        zero_states = ham.zero_energy_states()[0]
        for i, psi in enumerate(zero_states):
            self._plot_prob_dist(psi, i)

