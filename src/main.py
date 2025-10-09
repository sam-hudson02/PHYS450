import numpy as np
from scipy.constants import e, hbar
import matplotlib.pyplot as plt

HBAR_EV = hbar / e


def get_pi(mag: np.ndarray, q: np.ndarray, n: int, dagger: bool) -> complex:
    """Compute pi term for Hamiltonian matrix.

    Args:
        mag (np.ndarray): A 2D magnetic field vector.
        q (np.ndarray): A 2D momentum vector.
        n (int): layer number.
        dagger (bool): If True, compute the conjugate transpose.
    Returns:
        float: The pi term for layer n.
    """
    bx, by = mag
    qx, qy = q
    d = 0.335e-9  # interlayer distance in meters
    if dagger:
        result = (qx - e * n * by / HBAR_EV) - 1j * (qy + e * n * bx / HBAR_EV)
        return result
    else:
        return (qx + e * n * by / HBAR_EV) + 1j * (qy - e * n * bx / HBAR_EV)


def get_v(gamma_0: float | int) -> float:
    """Compute the Fermi velocity.

    Args:
        gamma_0 (float | int): Intralayer coupling parameter.

    Returns:
        float: The Fermi velocity.
    """
    a = 0.246e-9  # lattice constant in meters
    return (np.sqrt(3) * gamma_0 * a) / (2 * HBAR_EV)


def get_qc(hop: np.ndarray) -> float:
    gamma_0, gamma_1 = hop
    v = get_v(gamma_0)
    return gamma_1 / (HBAR_EV * v)


def hamiltonian(mag: np.ndarray, q: np.ndarray, n: int,
                hop: np.ndarray) -> np.ndarray:
    """
    Compute the n layer Hamiltonian matrix for a given momentum vector
    ,coupling parameters and magnetic field.

    Args:
        mag (np.ndarray): A 2D magnetic field vector.
        q (np.ndarray): A 2D momentum vector.
        hop (np.ndarray): Intralayer, Interlayer coupling parameter.
        n (int): Number of layers.

    Returns:
        np.ndarray: The Hamiltonian matrix.
    """
    gamma_0, gamma_1 = hop
    ham = np.zeros((2 * n, 2 * n), dtype=complex)
    for i in range(n):
        v = get_v(gamma_0)
        pi = get_pi(mag, q, i, dagger=False)
        pi_dagger = get_pi(mag, q, i, dagger=True)
        ham[2 * i, 2 * i + 1] = HBAR_EV * v * pi
        ham[2 * i + 1, 2 * i] = HBAR_EV * v * pi_dagger
        if i < n - 1:
            ham[2 * i + 1, 2 * (i + 1)] = gamma_1
            ham[2 * (i + 1), 2 * i + 1] = gamma_1
    return ham


def plot_graph(energies, qx_vals):
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

    ax.set_xlabel(r"$q_x / q_c$", fontsize=12)
    ax.set_ylabel(r"$E / \gamma_1$", fontsize=12)
    ax.set_title(
        "Band Structure of Rhombohedral Multilayer Graphene", fontsize=13)
    ax.axhline(0, color='gray', lw=0.6, ls='--')  # zero-energy line
    ax.set_xlim(0, 1.5)
    ax.set_ylim(-2, 2)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig("./plots/rmg_bandstructure.png", dpi=300)
    plt.show()


def main():
    samples = 400
    n = 300  # Number of layers
    hop = np.array([3.16, 0.381])  # Coupling parameters in eV
    qc = get_qc(hop)
    mag = np.array([5000000000, 0])  # Magnetic field in Tesla
    max_qx = 1.5 * qc
    qxs = np.linspace(0, max_qx, samples)
    qx_over_qc = np.linspace(0, max_qx / qc, samples)
    print(f"{qxs}")
    results = np.zeros((samples, 2*n))
    for qx in qxs:
        q = np.array([qx, 0.0])
        ham = hamiltonian(mag, q, n, hop)
        eigvals = np.linalg.eigvalsh(ham)
        energy_over_hop = eigvals / hop[1]
        results[np.where(qxs == qx)[0][0], :] = energy_over_hop
    plot_graph(results, qx_over_qc)


if __name__ == "__main__":
    main()
