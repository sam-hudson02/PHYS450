import numpy as np
from edge import EdgeStates
from sim import Simulation
from ham import DisorderType
from minimal import Simulation as MinimalSimulation


def minimal():
    hop = np.array([3.16, 0.381])  # Coupling parameters in eV
    mag = np.array([5, 0])  # Magnetic field in Tesla
    sim = MinimalSimulation(n=300, hop=hop, mag=mag)
    sim.run()

def edge_states():
    hop = np.array([3.16, 0.381])  # Coupling parameters in eV
    mag = np.array([0, 0])  # Magnetic field in Tesla
    sim1 = Simulation(hop, mag, n=10)
    sim1.band_structure(samples=400, hitrate=1)

def egap():
    hop = np.array([3.16, 0.381])  # Coupling parameters in eV
    mag = np.array([10, 0])  # Magnetic field in Tesla
    dt = DisorderType.ONSITE
    sim = Simulation(hop, mag, n=12, disorder_type=dt, disorder_strength=10.0)
    sim.eg_disorder()

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

def main():
    compare_evals()


if __name__ == "__main__":
    main()
