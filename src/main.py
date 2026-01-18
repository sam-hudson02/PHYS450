import numpy as np
from sim import Simulation
from ham import DisorderType


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

def dos():
    disorder_strength = 50
    disorder_type = DisorderType.ONSITE
    hop = np.array([3.16, 0.381])  # Coupling parameters in eV
    mag = np.array([50, 0])  # Magnetic field in Tesla
    sim = Simulation(hop, mag, n=20, disorder_type=disorder_type, disorder_strength=disorder_strength)
    qx = 0.8 * sim.qc
    q = np.array([qx, 0.0])
    sim.dos(q, max_e_gamma=2, energy_points=400)
    sim.band_structure(samples=400, hitrate=1)

def compare_evals():
    disorder_strength = 0.1
    disorder_type = DisorderType.BOTH
    hop = np.array([3.16, 0.381])  # Coupling parameters in eV
    mag = np.array([0, 0])  # Magnetic field in Tesla
    sim = Simulation(hop, mag, n=20, disorder_type=disorder_type)
    qx = 0.8 * sim.qc
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
    sim.band_structure(samples=400, hitrate=1, onsite_e=onsite_energy, bernal_fault=bernal,
                     bernal_layer=bernal_layer)
    q = np.array([0, 0])
    sim.psi_edge(q, bernal, bernal_layer)

def band_structure_prob():
    disorder_strength = 0.000
    disorder_type = DisorderType.NONE
    bernal = True
    bernal_layer = 3 
    hop = np.array([3.16, 0.381])  # Coupling parameters in eV
    mag = np.array([0, 0])  # Magnetic field in Tesla
    onsite_energy = 0
    sim = Simulation(hop, mag, n=20, disorder_type=disorder_type,
                     disorder_strength=disorder_strength)
    sim.band_structure(samples=400, hitrate=1, onsite_e=onsite_energy, bernal_fault=bernal,
                     bernal_layer=bernal_layer)
    q = np.array([0, 0])
    sim.psi_edge(q, bernal, bernal_layer)

def main():
    band_structure()


if __name__ == "__main__":
    main()
