import numpy as np
from sim import Simulation
from ham import DisorderType, Hamiltonian


def egap():
    bernal = True
    bernal_layer = 9
    mags = [0, 50, 100, 150]
    onsite_energy = 0.02425
    ham = Hamiltonian(n=20, bernal_fault=bernal, bernal_layer=bernal_layer, onsite=onsite_energy)
    sim = Simulation(ham)
    for bx in mags:
        mag = np.array([bx, 0])  # Magnetic field in Tesla
        ham.update_mag(mag)
        sim.eg_disorder(max_disorder_strength=0.01, samples=1000)

def band_structure_prob():
    mags = [0, 50, 100, 150, 200, 250, 300]
    disorder_strength = 0.0
    disorder_type = DisorderType.ONSITE
    bernal = True
    bernal_layer = 7
    n = 20
    mag = np.array([0, 0])  # Magnetic field in Tesla
    onsite_energy = 0.00
    ham = Hamiltonian(n=n, disorder_type=disorder_type, disorder_strength=disorder_strength,
                        bernal_fault=bernal, bernal_layer=bernal_layer, onsite=onsite_energy)
    sim = Simulation(ham)
    for bx in mags:
        mag = np.array([bx, 0])  # Magnetic field in Tesla
        ham.update_mag(mag)
        sim.band_structure(samples=400, hitrate=1)
        sim.prob_edge(ham)

    zero_energy_bx_1, zero_energy_bx_2 = ham.zero_energy_threshold_bernal()
    soliton_bx = ham.soliton_threshold()

    # look at soliton threshold field
    q = np.array([0, 0])
    ham.update_q(q)

    mag = np.array([soliton_bx+1, 0])
    ham.update_mag(mag)

    sim.band_structure(samples=400, hitrate=1)
    sim.prob_edge(ham)

    # look at zero energy threshold field
    mag = np.array([zero_energy_bx_1, 0])
    ham.update_mag(mag)

    sim.band_structure(samples=400, hitrate=1)
    sim.prob_edge(ham)

    mag = np.array([zero_energy_bx_2, 0])
    ham.update_mag(mag)

    sim.band_structure(samples=400, hitrate=1)
    sim.prob_edge(ham)

def dos():
    disorder_strength = 0.1
    disorder_type = DisorderType.NONE
    q = np.array([50, 0])
    r = 10
    mag = np.array([0, 0]) # Magnetic field in Tesla
    bernal = True
    bernal_layer = 7
    ham = Hamiltonian(n=20, disorder_type=disorder_type, disorder_strength=disorder_strength,
                      mag=mag, q=q, bernal_fault=bernal, bernal_layer=bernal_layer)
    sim = Simulation(ham)
    sim.dos(energy_range=2, r=r)

def main():
    #egap()
    dos()
    #band_structure_prob()

if __name__ == "__main__":
    main()
