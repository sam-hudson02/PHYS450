from logging import raiseExceptions
from ham import DisorderType
import numpy as np
from alive_progress import alive_bar
from scipy.constants import hbar as hbar_SI, e as eC
from ham import Hamiltonian
import matplotlib.pyplot as plt
import os

class Simulation:
    def __init__(self, ham: Hamiltonian):
        self.ham = ham

    def _plot_band_structure(self, energies, qx_vals, hitrate: int = 10):
        _, ax = plt.subplots(figsize=(7, 5))

        to_plot = energies.shape[1] // (2 * hitrate)
        mid = energies.shape[1] // 2
        for n in range(to_plot):
            band = n * hitrate
            upper_band = mid + band
            lower_band = mid - band - 1
            if self.ham.bernal_fault and (n == 0 or n == 1):
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
        ax.set_title(f"$B_x = {self.ham.mag[0]}$T", fontsize=13)
        ax.axhline(0, color='gray', lw=0.6, ls='--')  # zero-energy line
        ax.set_xlim(0, 1.5)
        ax.set_ylim(-2, 2)
        ax.grid(False)
        plt.tight_layout()
        title = f"./plots/band_structure/RMG_{self.ham.mag[0]}_{self.ham.n}.png"
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
        ax.set_title(f"Energy Gap vs Disorder Strength\n$B_x = {self.ham.mag[0]}$T, N={self.ham.n}", fontsize=13)
        ax.grid(False)
        plt.tight_layout()
        if not os.path.exists("./plots/eg_disorder"):
            os.makedirs("./plots/eg_disorder")
        file = f"./plots/eg_disorder/EG_RMG_{self.ham.mag[0]}_{self.ham.n}.png"
        print(f"Saving figure as {file}")
        plt.savefig(file, dpi=300)

    def _plot_psi(self, psi: np.ndarray, i: int):
        """
        Plot the edge state wavefunction as bar graph with a and b sites next to each other both labeled as site j.
        Args:
            psi (np.ndarray): The wavefunction psi.
        """ 

        m = np.arange(1, self.ham.n + 1)  # cell index
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
        ax.set_yticks([-1, 0, 1])
        ax.set_xticks(m)
        ax.set_xticklabels(m)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)

        ax.tick_params(direction='in', top=True, right=True)

        plt.legend([r'a sites', r'b sites'])

        if not os.path.exists("./plots/psi_edge"):
            os.makedirs("./plots/psi_edge")

        plt.title(f'Edge State Wavefunction\nMagnetic Field: Bx={self.ham.mag[0]} T, By={self.ham.mag[1]} T')
        plt.savefig(f'./plots/psi_edge/edge_state_Bx{self.ham.mag[0]}_N{self.ham.n}_{i}.png')

    def _plot_prob_dist(self, psi: np.ndarray, i: int):
        """
        Plot the edge state probability density as bar graph.
        Args:
            psi_sq (np.ndarray): The probability density |psi|^2.
        """ 
        psi_sq = np.abs(psi)**2
        j_max = 2 * self.ham.n + 1
        j = np.arange(1, j_max) # atomic site
        plt.figure(figsize=(8, 6))
        plt.xlabel('j')
        plt.ylabel(r'$|\psi|^2$')
        plt.title(f'Edge State Probability Density\nMagnetic Field: Bx={self.ham.mag[0]} T, By={self.ham.mag[1]} T')
        plt.bar(j, psi_sq, width=0.8, color='blue', alpha=0.7, label=r'$|\psi|^2$')
        plt.xticks(ticks=np.arange(0, j_max, 20))
        plt.legend()
        plt.savefig(f'./plots/edge_state_Bx{self.ham.mag[0]}_N{self.ham.n}_{i}.png')


    def _plot_dos(self, energies: np.ndarray, dos: np.ndarray, max_e: float):
        plt.figure(figsize=(8, 6))
        plt.plot(energies, dos, color='blue', lw=1.5)
        plt.xlabel('Energy (eV)')
        plt.ylabel('Density of States (1/eV)')
        plt.title(f'Density of States\nMagnetic Field: Bx={self.ham.mag[0]} T, By={self.ham.mag[1]} T')
        plt.xlim(-max_e, max_e)
        plt.grid(False)
        if not os.path.exists("./plots/dos"):
            os.makedirs("./plots/dos")
        plt.savefig(f'./plots/dos/dos_Bx{self.ham.mag[0]}_N{self.ham.n}.png')

    def _plot_evals_comparison(self, collected_evals: list[np.ndarray], disorder_strengths: list[float]):
        plt.figure(figsize=(8, 6))
        # disorder sterngth on x axis and eig values on y axis
        for i, evals in enumerate(collected_evals):
            ds = disorder_strengths[i]
            # make each sample a horizontal red line with some width at y = ds, x = eval
            # so multiple samples together look like a bar chart
            plt.scatter([ds] * len(evals), evals / self.ham.hop[1], color='red', s=1, alpha=0.5)
        plt.xlabel('Disorder Strength (meV)')
        plt.ylabel('Eigenvalues (eV)')
        plt.title(f'Eigenvalues vs Disorder Strength\nMagnetic Field: Bx={self.ham.mag[0]} T, By={self.ham.mag[1]} T')
        plt.grid(False)
        if not os.path.exists("./plots/evals_comparison"):
            os.makedirs("./plots/evals_comparison")
        plt.savefig(f'./plots/evals_comparison/evals_Bx{self.ham.mag[0]}_N{self.ham.n}.png')

    def band_structure(self, samples: int, hitrate: int, max_qx_qc = 1.5):
        energies = np.zeros((samples, 2 * self.ham.n))
        max_qx = max_qx_qc * self.ham.qc
        qxs = np.linspace(0, max_qx, samples)
        with alive_bar(samples, title="Computing bands") as bar:
            for i, qx in enumerate(qxs):
                q = np.array([qx, 0])
                matrix = self.ham.update_q(q)
                evals = np.linalg.eigvalsh(matrix)
                energies[i, :] = np.sort(evals) / self.ham.hop[1]
                bar()
        self._plot_band_structure(energies, qxs / self.ham.qc, hitrate)
        return self.ham

    def eg_disorder(self, max_disorder_strength: float = 10, samples: int=20):
        eg_onsite_list: list[tuple[float, float]] = []
        eg_hopping_list: list[tuple[float, float]] = []
        for i in range(11):
            disorder_strength = (max_disorder_strength / 10) * i
            egs_onsite: list[float] = []
            egs_hopping: list[float] = []
            for _ in range(samples):
                self.ham.update_disorder(disorder_type=DisorderType.ONSITE, disorder_strength=disorder_strength)
                egs_onsite.append(self.ham.egap())
                self.ham.update_disorder(disorder_type=DisorderType.HOPPING, disorder_strength=disorder_strength)
                egs_hopping.append(self.ham.egap())

            mean_onsite = float(np.mean(egs_onsite))
            err_onsite = float(np.std(egs_onsite)) / np.sqrt(samples)
            print(f"Onsite Disorder Strength: {disorder_strength} eV, Mean EG: {mean_onsite} eV")
            eg_onsite_list.append((mean_onsite, err_onsite))

            mean_hopping = float(np.mean(egs_hopping))
            err_hopping = float(np.std(egs_hopping)) / np.sqrt(samples)
            print(f"Hopping Disorder Strength: {disorder_strength} eV, Mean EG: {mean_hopping} eV")
            eg_hopping_list.append((mean_hopping, err_hopping))
        self._plot_eg(eg_onsite_list, eg_hopping_list, max_disorder_strength)

    def psi_edge(self):
        zero_states = self.ham.zero_energy_states()[0]
        for i, psi in enumerate(zero_states):
            self._plot_psi(psi, i)

    def compare_evals(self, max_disorder_strength: float = 50,
                      samples:int = 200, passes: int = 400):
        collected_evals = []
        disorder_strengths = []
        with alive_bar(samples*passes, title="Collecting eigenvalues") as bar:
            for i in range(samples):
                group = []
                disorder_strength = (max_disorder_strength / samples) * i
                for _ in range(passes):
                    self.ham.update_disorder(disorder_strength=disorder_strength)
                    evals = self.ham.evals()
                    group.extend(evals)
                    bar()
                collected_evals.append(group)
                disorder_strengths.append(disorder_strength)
        self._plot_evals_comparison(collected_evals, disorder_strengths)

    def plot_multi_prob_dist(self, states_list: tuple[list[np.ndarray], list[complex]],
                             sub_folder: str = ""):
        """
        Plot multiple edge state probability densities with different colours.
        Args:
            q (np.ndarray): The momentum vector q.
            psi_list (list[np.ndarray]): List of wavefunctions psi.
        """
        psi_list, e_list = states_list
        plt.figure(figsize=(8, 6))
        j_max = 2 * self.ham.n + 1
        # atomic sites
        j = np.arange(1, j_max)
        bottom = np.zeros(j_max - 1)
        for i, psi in enumerate(psi_list):
            psi_sq = np.abs(psi)**2
            e = e_list[i] / self.ham.hop[1]
            e_sci_notation = f"{e:.4e}"
            cmap = plt.get_cmap('Set1')
            color = cmap(i % cmap.N)
            plt.bar(j, psi_sq, width=0.8, bottom=bottom, color=color, alpha=0.7,
                    label=f'$\epsilon / \gamma_1$ = {e_sci_notation}')
            bottom += psi_sq
        plt.xlabel('j')
        plt.ylabel(r'$|\psi|^2$')
        plt.title(f'Edge State Probability Densities\nMagnetic Field: Bx={self.ham.mag[0]} T, By={self.ham.mag[1]} T')
        plt.xticks(ticks=np.arange(0, j_max, 20))
        plt.legend()
        dir_path = f'./plots{sub_folder}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.savefig(f'./plots{sub_folder}edge_state_multi_Bx{self.ham.mag[0]}_N{self.ham.n}.png')


    def dos_at_e(self, e: float, evals: np.ndarray)-> float:
        fac = 0.005
        loz = fac * self.ham.hop[1]
        total = 0
        for eval in evals:
            total += loz / ((e - eval)**2 + loz**2)
        return total / (np.pi * self.ham.n * 2)


    def dos(self, energy_range: float = 1.0, steps: int = 1000, r: int = 1):
        e_range = energy_range * self.ham.hop[1]
        es = np.linspace(-e_range, e_range, steps)
        dos_values = np.zeros(steps)
        for _ in range(r):
            self.ham.update_disorder()
            evals = self.ham.evals()
            for i, e in enumerate(es):
                dos_values[i] += self.dos_at_e(e, evals)
        dos_values /= r
        self._plot_dos(es, dos_values, e_range)



    def prob_edge(self, ham: Hamiltonian, sub_folder: str = ""):
        ham.update_q(np.array([0, 0]))
        zero_states = ham.zero_energy_states()
        self.plot_multi_prob_dist(zero_states, sub_folder)

