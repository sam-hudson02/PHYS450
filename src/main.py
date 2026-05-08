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
    mags = [0.1, 25, 50, 75, 100, 125, 150, 200, 250, 300, 10000]
    disorder_strength = 0.0
    disorder_type = DisorderType.ONSITE
    bernal = True
    bernal_layer = 15
    n = 50
    start_pair = 0
    end_pair = 0
    # highlight second band pair blue
    band_index = [0]
    mag = np.array([0, 0])  # Magnetic field in Tesla
    onsite_energy = 0.00
    ham = Hamiltonian(n=n, disorder_type=disorder_type, disorder_strength=disorder_strength,
                        bernal_fault=bernal, bernal_layer=bernal_layer, onsite=onsite_energy)
    sim = Simulation(ham)
    zero_energy_bx = ham.zero_energy_threshold()
    soliton_bx = ham.soliton_threshold()
    soliton_bx_colision = ham.soliton_collision_threshold()
    print(f"soliton threshold: {soliton_bx:.2f} T")
    print(f"soliton collision threshold: {soliton_bx_colision:.2f} T")
    mags.append(soliton_bx)
    mags.append(soliton_bx_colision)
    mags.append(zero_energy_bx)

    for bx in mags:
        mag = np.array([bx, 0])  # Magnetic field in Tesla
        ham.update_mag(mag)
        sim.band_structure(samples=400, hitrate=1, band_index=band_index)
        sim.prob_edge(ham, start_pair, end_pair)

def band_structure():
    q = np.array([0, 0])
    mag = np.array([25, 0]) # Magnetic field in Tesla
    ham = Hamiltonian(n=20, mag=mag, q=q)
    sim = Simulation(ham)
    sim.band_structure(samples=400, hitrate=1)


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

def check_ham():
    n = 20
    extra_hop = False
    mag_dep = True
    bx = 0
    bernal = True
    bernal_layer = 7
    mag = np.array([bx, 0]) # Magnetic field in Tesla
    ham = Hamiltonian(n=n, mag=mag, extra_hop=extra_hop, mag_dep=mag_dep,
                      bernal_fault=bernal, bernal_layer=bernal_layer)
    sim = Simulation(ham)
    print("Hamiltonian check:")
    print(ham._matrix)
    sim.band_structure(samples=400, hitrate=1)

def trig_warp():
    bxs = [0, 25]
    for bx in bxs:
        n = 20
        extra_hop = True
        mag_dep = True
        mag = np.array([bx, 0]) # Magnetic field in Tesla
        ham = Hamiltonian(n=n, mag=mag, extra_hop=extra_hop, mag_dep=mag_dep)
        sim = Simulation(ham)
        sim.band_structure(samples=400, hitrate=1)
        sim.trig_warp(ham)

def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=400)
    #band_structure()
    #egap()
    #dos()
    band_structure_prob()
    #check_ham()
    #trig_warp()

if __name__ == "__main__":
    main()
