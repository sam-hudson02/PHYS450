import numpy as np
from minimal import Simulation
from edge import EdgeStates


def minimal():
    hop = np.array([3.16, 0.381])  # Coupling parameters in eV
    mag = np.array([1, 0])  # Magnetic field in Tesla
    sim = Simulation(n=1000, hop=hop, mag=mag, hitrate=25)
    sim.run(save=True)

def edge_states(): 
    hop = np.array([3.16, 0.381])  # Coupling parameters in eV
    edge = EdgeStates(n=10, hop=hop)
    mag = np.array([0, 0])  # Magnetic field in Tesla
    q = np.array([0, 0])  # Momentum vector
    edge.run(q, mag)

def main():
    edge_states()


if __name__ == "__main__":
    main()
