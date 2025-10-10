import numpy as np
from scipy.constants import e, hbar
import matplotlib.pyplot as plt
from sim import Simulation


def main():
    hop = np.array([3.16, 0.381])  # Coupling parameters in eV
    mag = np.array([1, 0])  # Magnetic field in Tesla
    sim = Simulation(n=1000, hop=hop, mag=mag, hitrate=25)
    sim.run(save=True)


if __name__ == "__main__":
    main()
