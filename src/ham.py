from enum import Enum
import numpy as np
from scipy.constants import h, hbar as hbar_SI, e as eC

class DisorderType(Enum):
    NONE = 0
    ONSITE = 1
    HOPPING = 2
    BOTH = 3

class Hamiltonian:
    def __init__(self, n: int,
                 q: np.ndarray = np.array([0.0, 0.0]),
                 hop: np.ndarray = np.array([3.16, 0.381]),
                 mag: np.ndarray = np.array([0.0, 0.0]),
                 onsite: float = 0.0,
                 d: float = 0.346e-9,
                 disorder_type: DisorderType = DisorderType.NONE,
                 disorder_strength: float = 0.0,
                 bernal_fault: bool = False,
                 bernal_layer: int = 2,
                 extra_hop: bool = False,
                 mag_dep: bool = True):
        print(f"q: {q}")
        self.q = q
        self.n = n
        self.d = d
        self.hop = hop
        self.gamma_0 = hop[0]
        self.gamma_1 = hop[1]
        self.mag = mag
        self.extra_hop = extra_hop
        self.onsite_energy = onsite
        self.disorder_type = disorder_type
        self.disorder_strength = disorder_strength
        self.hbar_ev = hbar_SI / eC
        self.onsite_disorder_array = None
        self.hopping_disorder_array = None
        self.a = 0.246e-9  # lattice constant in meters
        self.v = (np.sqrt(3) * self.hop[0] * self.a) / (2 * self.hbar_ev)
        self.qc = self.hop[1] / (self.v * self.hbar_ev)
        self.bernal_fault = bernal_fault
        self.bernal_layer = bernal_layer
        self.extra_hop_terms = np.array([-0.017, 0.38, 0.14])
        self.mag_dep = mag_dep
        self._matrix = self._construct_matrix()
        threshold = self.soliton_threshold()
        zero_thresh = self.zero_energy_threshold()
        print(f"Hamiltonian matrix constructed for n={n} layers.")
        print(f"Soliton formation threshold magnetic field: {threshold:.2f} T")
        print(f"Zero-energy states threshold magnetic field: {zero_thresh:.2f} T")

    def fermi_v(self, hop: float) -> float:
        """Calculate the Fermi velocity for a given intralayer hopping parameter.
        Args:
            hop (float): The intralayer hopping parameter in eV.
        Returns:
            float: The Fermi velocity in m/s.
        """
        return (np.sqrt(3) * hop * self.a) / (2 * self.hbar_ev)

    @property
    def file_path(self) -> str:
        bernal_str = f"bernal_{self.bernal_layer}" if self.bernal_fault else "no_bernal"
        return f"n_{self.n}_{bernal_str}"

    @property 
    def disorder_text(self) -> str:
        if self.disorder_type == DisorderType.ONSITE:
            return f"onsite_disorder_{self.disorder_strength:.2f}"
        elif self.disorder_type == DisorderType.HOPPING:
            return f"hopping_disorder_{self.disorder_strength:.2f}"
        elif self.disorder_type == DisorderType.BOTH:
            return f"both_disorder_{self.disorder_strength:.2f}"
        else:
            return "no_disorder"


    def _get_pi(self, q: np.ndarray, mag: np.ndarray, i: float | int, dag: bool, 
                mag_dep: bool = True) -> np.ndarray:
        """Compute pi term for Hamiltonian matrix.

        Args:
            q (np.ndarray): A 2D momentum vector.
            i (int): layer number.
            dag (bool): If True, compute the conjugate transpose.
        Returns:
            float: The pi term for layer n.
        """
        bx, by = mag
        qx, qy = q
        zn = (i - (self.n - 1) / 2) * self.d
        if mag_dep:
            shift = eC * zn / hbar_SI
        else:
            shift = 0
        real_part = qx + shift * by
        imag_part = qy - shift * bx
        if dag:
            result = (real_part) - 1j * imag_part
        else:
            result = (real_part) + 1j * imag_part
        return result

    def _construct_matrix(self) -> np.ndarray:
        """
        Compute the n layer Hamiltonian matrix for a given momentum vector
        ,coupling parameters and magnetic field.

        Args:
            q (np.ndarray): A 2D momentum vector.

        Returns:
            np.ndarray: The Hamiltonian matrix.
        """

        _, gamma_1 = self.hop
        gamma_2, gamma_3, gamma_4 = self.extra_hop_terms
        v_3, v_4 = self.fermi_v(gamma_3), self.fermi_v(gamma_4)

        ham = np.zeros((2 * self.n, 2 * self.n), dtype=complex)
        for i in range(self.n):
            # intralayer hopping
            pi = self._get_pi(self.q, self.mag, i, dag=False)
            pi_dagger = self._get_pi(self.q, self.mag, i, dag=True)
            ham[2 * i, 2 * i + 1] = self.hbar_ev * self.v * pi_dagger
            ham[2 * i + 1, 2 * i] = self.hbar_ev * self.v * pi

            # onsite energy
            ham[2 * i, 2 * i] += self.onsite_energy
            ham[2 * i + 1, 2 * i + 1] -= self.onsite_energy

            if self.extra_hop:
                try:
                    pi_half = self._get_pi(self.q, self.mag, i + 0.5, dag=False,
                                           mag_dep=self.mag_dep)
                    pi_half_dagger = self._get_pi(self.q, self.mag, i + 0.5, dag=True,
                                                  mag_dep=self.mag_dep)
                    ham[2 * i, 2 * i + 2] += -self.hbar_ev * v_4 * pi_half_dagger
                    ham[2 * i + 2, 2 * i] += -self.hbar_ev * v_4 * pi_half
                    ham[2 * i , 2 * i + 3] += self.hbar_ev * v_3 * pi_half
                    ham[2 * i + 3, 2 * i] += self.hbar_ev * v_3 * pi_half_dagger
                except IndexError:
                    pass

            # interlayer hopping
            if i < self.n - 1:
                ham[2 * i + 1, 2 * (i + 1)] = gamma_1
                ham[2 * (i + 1), 2 * i + 1] = gamma_1
                if self.extra_hop:
                    pi_half = self._get_pi(self.q, self.mag, i + 0.5, dag=False,
                                           mag_dep=self.mag_dep)
                    pi_half_dagger = self._get_pi(self.q, self.mag, i + 0.5, dag=True,
                                                  mag_dep=self.mag_dep)
                    ham[2 * i + 1, 2 * (i + 1) + 1] += -self.hbar_ev * v_4 * pi_half_dagger
                    ham[2 * (i + 1) + 1, 2 * i + 1] += -self.hbar_ev * v_4 * pi_half

        # onsite disorder
        if self.disorder_type == DisorderType.ONSITE \
        or self.disorder_type == DisorderType.BOTH:
            if self.onsite_disorder_array is None:
                self.onsite_disorder_array = np.random.uniform(
                    -self.disorder_strength,
                    self.disorder_strength,
                    size=self.n
                )
            for i in range(self.n):
                onsite_energy = self.onsite_disorder_array[i]
                ham[2 * i, 2 * i] += onsite_energy
                ham[2 * i + 1, 2 * i + 1] += onsite_energy

        # hopping disorder
        if self.disorder_type == DisorderType.HOPPING \
        or self.disorder_type == DisorderType.BOTH:
            if self.hopping_disorder_array is None:
                self.hopping_disorder_array = np.random.uniform(
                    -self.disorder_strength,
                    self.disorder_strength,
                    size=self.n - 1
                )
            for i in range(self.n - 1):
                hopping_variation = self.hopping_disorder_array[i]
                ham[2 * i + 1, 2 * (i + 1)] += hopping_variation
                ham[2 * (i + 1), 2 * i + 1] += hopping_variation

        if self.bernal_fault:
            ham = self.construct_bf(ham)
        self._matrix = ham
        return ham

    def construct_bf(self, ham: np.ndarray) -> np.ndarray:
        """
        Constructs a Hamiltonian matrix with a Bernal fault between layers i and i+1.
        Args:
            i (int): The layer index where the fault is introduced.
        Returns:
            np.ndarray: The Hamiltonian matrix with the Bernal fault.
        """
        i = self.bernal_layer
        if i < 0 or i >= self.n - 1:
            raise ValueError(f"Layer index out of bounds for Bernal fault:\
                             \n{i} not in [0, {self.n - 2}]")


        # get existing interlayer hopping between layers i and i+1
        # this accounts for any disorder that may have been added to the hopping terms
        hop1 = ham[2 * i + 1, 2 * (i + 1)] 
        hop2 = ham[2 * (i + 1), 2 * i + 1]

        # Remove interlayer hopping between specified layers
        ham[2 * i + 1, 2 * (i + 1)] = 0
        ham[2 * (i + 1), 2 * i + 1] = 0

        # Add hopping between the other sublattices of the two layers
        ham[2 * i, 2 * (i + 1) + 1] = hop1
        ham[2 * (i + 1) + 1, 2 * i] = hop2

        return ham

    def matrix(self) -> np.ndarray:
        return self._matrix
    
    def eigh(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the hamiltonian.
        Args:
            q (np.ndarray): A 2D momentum vector.
            mag (np.ndarray): A 2D magnetic field vector.
        Returns:
            tuple[np.ndarray, np.ndarray]: Eigenvalues, Eigenvectors
        """
        ham = self._matrix
        evals, evecs = np.linalg.eigh(ham)
        return evals, evecs

    def update_q(self, q: np.ndarray) -> np.ndarray:
        """
        Update the momentum vector and reconstruct the Hamiltonian matrix.
        Args:
            q (np.ndarray): A 2D momentum vector.
        Returns:
            np.ndarray: The updated Hamiltonian matrix.
        """
        self.q = q
        return self._construct_matrix()

    def  update_mag(self, mag: np.ndarray) -> np.ndarray:
        """
        Update the magnetic field vector and reconstruct the Hamiltonian matrix.
        Args:
            mag (np.ndarray): A 2D magnetic field vector.
        Returns:
            np.ndarray: The updated Hamiltonian matrix.
        """
        self.mag = mag
        return self._construct_matrix()

    def update_disorder(self, disorder_type: DisorderType | None = None,
                        disorder_strength: float | None = None) -> np.ndarray:
        """
        Randomly draw new disorder values for the onsite and hopping disorder arrays and reconstruct the Hamiltonian matrix.
        Returns:
            np.ndarray: The updated Hamiltonian matrix with shuffled disorder.
        """
        if disorder_type is not None:
            self.disorder_type = disorder_type
        if disorder_strength is not None:
            self.disorder_strength = disorder_strength

        self.onsite_disorder_array = np.random.uniform(
            -self.disorder_strength,
            self.disorder_strength,
            size=self.n
        )
        self.hopping_disorder_array = np.random.uniform(
            -self.disorder_strength,
            self.disorder_strength,
            size=self.n - 1
        )
        return self._construct_matrix()

    def evals(self) -> np.ndarray:
        return np.linalg.eigvalsh(self._matrix)

    def evecs(self) -> np.ndarray:
        v = np.linalg.eigh(self.matrix())[1]
        return v

    def zero_energy_states(self, start: int = 0, end: int = 0) -> tuple[list[np.ndarray], list[complex]]:
        evals, evecs = self.eigh()
        print(evals)
        zero_states_vec = []
        zero_states_e = []
        """
        for i, val in enumerate(evals):
            print(f"Eigenvalue {i}: {val}")
            if np.isclose(val, 0, atol=1e-2):
                zero_state_vec = evecs[:, i]
                zero_states_vec.append(zero_state_vec)
                zero_states_e.append(val)
        # two lowest positive and two highest negative eigenvalues

        pos_indices = [i for i, val in enumerate(evals) if val >= 0]
        neg_indices = [i for i, val in enumerate(evals) if val < 0]
        pos_indices.sort(key=lambda i: evals[i])
        neg_indices.sort(key=lambda i: evals[i], reverse=True)
        selected_indices = neg_indices[:num] + pos_indices[:num]
        for i in selected_indices:
            zero_state_vec = evecs[:, i]
            zero_states_vec.append(zero_state_vec)
            zero_states_e.append(evals[i])
        """
        middle_pos = self.n
        middle_neg = self.n - 1

        for i in range(start, end+1):
            print(i)
            print(f"Selected eigenvalues for zero-energy states {i}: {evals[middle_neg - i]}, {evals[middle_pos + i]}")
            zero_state_vec_pos = evecs[:, middle_pos + i]
            zero_state_vec_neg = evecs[:, middle_neg - i]
            zero_states_vec.append(zero_state_vec_pos)
            zero_states_vec.append(zero_state_vec_neg)
            zero_states_e.append(evals[middle_pos + i])
            zero_states_e.append(evals[middle_neg - i])


        return zero_states_vec, zero_states_e

    def egap(self) -> float:
        evals = self.evals()
        # return the middle values
        mid = len(evals) // 2
        gap = evals[mid] - evals[mid - 1]
        return gap

    def flux_all(self) -> float:
        """
        Calculate the magnetic flux through the multilayer system.
        Returns:
            float: The magnetic flux in Weber.
        """
        bx, _ = self.mag
        flux = self.a * self.d * (self.n - 1) * bx
        return flux

    def soliton_threshold(self) -> float:
        """
        Calculate the soliton formation threshold magnetic field.
        Returns:
            float: The threshold magnetic field in Tesla.
        """
        flux_0 = h / eC
        gamma_1 = self.hop[1]
        gamma_0 = self.hop[0]
        frac = (flux_0 * 2 * gamma_1) / (np.sqrt(3) * np.pi * gamma_0)
        bx = frac / (self.a * self.d * (self.n - 1))
        return bx

    def soliton_collision_threshold(self) -> float:
        """
        Calculate the magnetic field at which soltion collides with zero-energy states 
        at the stracking fault.
        Returns:
            float: The threshold magnetic field in Tesla.
        """
        print(f"Calculating soliton collision threshold for Bernal layer {self.bernal_layer} and total layers {self.n}")
        layer_offset = self.bernal_layer - ((self.n - 1) / 2)
        layer_position = layer_offset * self.d
        print(f"Layer offset from center: {layer_offset}")
        flux_0 = h / eC
        num = flux_0 * self.qc
        denom = np.abs(layer_position * 2 * np.pi)
        print(f"Numerator: {num:.2e}, Denominator: {denom:.2e}")
        bx = num / denom
        return bx
    
    def zero_energy_threshold(self):
        """
        Calculate the magnetic field at which zero-energy states disappear.
        Returns:
            float: The threshold magnetic field in Tesla.
        """
        flux_0 = h / eC
        gamma_1 = self.hop[1]
        gamma_0 = self.hop[0]
        frac = (2 * gamma_1 * flux_0 * self.n) / (np.sqrt(3) * np.pi * gamma_0)
        bx = frac / (self.a * self.d * (self.n - 1))
        return bx

    def zero_energy_threshold_bernal(self):
        """
        Calculate the magnetic field at which zero-energy states disappear.
        Returns:
            float: The threshold magnetic field in Tesla.
        """
        n_1 = self.n - (self.bernal_layer + 1)
        n_2 = self.bernal_layer + 1
        flux_0 = h / eC
        gamma_1 = self.hop[1]
        gamma_0 = self.hop[0]
        frac_1 = (2 * gamma_1 * flux_0 * n_1) / (np.sqrt(3) * np.pi * gamma_0)
        frac_2 = (2 * gamma_1 * flux_0 * n_2) / (np.sqrt(3) * np.pi * gamma_0)
        bx_1 = frac_1 / (self.a * self.d * (n_1 - 1))
        bx_2 = frac_2 / (self.a * self.d * (n_2 - 1))
        return bx_1, bx_2

