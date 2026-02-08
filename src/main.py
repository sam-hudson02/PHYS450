import numpy as np
from sim import Simulation
from ham import DisorderType, Hamiltonian


def edge_states():
    hop = np.array([3.16, 0.381])  # Coupling parameters in eV
    mag = np.array([0, 0])  # Magnetic field in Tesla
    sim1 = Simulation(hop, mag, n=10)
    sim1.band_structure(samples=400, hitrate=1)

def egap():
    hop = np.array([3.16, 0.381])  # Coupling parameters in eV
    mag = np.array([0, 0])  # Magnetic field in Tesla
    sim = Simulation(hop, mag, n=12)
    sim.eg_disorder(max_disorder_strength=0.01, samples=1000)

def compare_evals():
    disorder_strength = 0.1
    disorder_type = DisorderType.BOTH
    hop = np.array([3.16, 0.381])  # Coupling parameters in eV
    mag = np.array([0, 0])  # Magnetic field in Tesla
    sim = Simulation(hop, mag, n=20, disorder_type=disorder_type)
    qx = 0.8 * sim.ham.qc
    q = np.array([qx, 0.0])
    sim.compare_evals(q, max_disorder_strength=disorder_strength)

def band_structure_psi():
    disorder_strength = 0.000
    disorder_type = DisorderType.NONE
    bernal = True
    bernal_layer = 3 
    hop = np.array([3.16, 0.381])  # Coupling parameters in eV
    mag = np.array([0, 0])  # Magnetic field in Tesla
    onsite_energy = 0
    sim = Simulation(hop, mag, n=20, disorder_type=disorder_type,
                     disorder_strength=disorder_strength)
    sim.band_structure(samples=400, hitrate=1, onsite_e=onsite_energy)
    q = np.array([0, 0])
    sim.psi_edge(q)

def band_structure_prob():
    mags = [0, 50, 100, 150, 200, 250, 300]
    sub_folder = "/l3_fault/"
    disorder_strength = 0.1
    disorder_type = DisorderType.ONSITE
    bernal = True
    bernal_layer = 7
    n = 20
    mag = np.array([0, 0])  # Magnetic field in Tesla
    ham = Hamiltonian(n=n, disorder_type=disorder_type, disorder_strength=disorder_strength,
                        bernal_fault=bernal, bernal_layer=bernal_layer)
    sim = Simulation(ham)
    for bx in mags:
        mag = np.array([bx, 0])  # Magnetic field in Tesla
        ham.update_mag(mag)
        sim.band_structure(samples=400, hitrate=1)
        sim.prob_edge(ham, sub_folder=sub_folder)

    zero_energy_bx_1, zero_energy_bx_2 = ham.zero_energy_threshold_bernal()
    soliton_bx = ham.soliton_threshold()

    # look at soliton threshold field
    q = np.array([0, 0])
    ham.update_q(q)

    mag = np.array([soliton_bx+1, 0])
    ham.update_mag(mag)

    sim.band_structure(samples=400, hitrate=1)
    sim.prob_edge(ham, sub_folder=sub_folder)

    # look at zero energy threshold field
    mag = np.array([zero_energy_bx_1, 0])
    ham.update_mag(mag)

    sim.band_structure(samples=400, hitrate=1)
    sim.prob_edge(ham, sub_folder=sub_folder)

    mag = np.array([zero_energy_bx_2, 0])
    ham.update_mag(mag)

    sim.band_structure(samples=400, hitrate=1)
    sim.prob_edge(ham, sub_folder=sub_folder)

def main():
    band_structure_prob()


if __name__ == "__main__":
    main()
