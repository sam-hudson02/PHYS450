import numpy as np
import matplotlib.pyplot as plt

def plot_graphene_band_heatmap(a, t, r):
    kx = np.linspace(-r, r, 400)
    ky = np.linspace(-r, r, 400)
    KX, KY = np.meshgrid(kx, ky)
    f_abs = get_abs_f(KX, KY, a)
    E_plus = t * f_abs
    params = {"ytick.color" : "w",
            "xtick.color" : "w",
            "axes.labelcolor" : "w",
            "axes.edgecolor" : "w"}
    plt.rcParams.update(params)
    plt.figure(figsize=(8, 6))
    plt.contourf(KX, KY, E_plus, levels=100, cmap='plasma')
    plt.colorbar(label='Energy (eV)')
    plt.title('Graphene Band Structure Heatmap')
    plt.xlabel(r'$k_x$ $(1/\AA)$')
    plt.ylabel(r'$k_y$ $(1/\AA)$')
    plt.title('Graphene Band Structure Heatmap', color='white')
    plt.savefig('plots/graphene/graphene_band_heatmap_transparent.png', dpi=300, 
                transparent=True)

def plot_graphene_2d_band(a, t, r):
    r += 0.4
    kx = np.linspace(-r, r, 1000)
    ky = np.linspace(0, 0, 1000)
    f_abs = get_abs_f(kx, ky, a)
    E_plus = t * f_abs
    E_minus = -E_plus
    plt.figure(figsize=(8, 6))
    plt.plot(kx, E_plus, label='Conduction Band', color='orange')
    plt.plot(kx, E_minus, label='Valence Band', color='blue')
    plt.title('Graphene Band Structure along kx')
    plt.xlabel('kx (1/Angstrom)')
    plt.ylabel('Energy (eV)')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
    plt.legend()
    plt.savefig('plots/graphene/graphene_band_2d.png', 
                dpi=300)

def plot_graphene_band_3d(a, t, r):
    kx = np.linspace(-r, r, 1001)
    ky = np.linspace(-r, r, 1001)
    KX, KY = np.meshgrid(kx, ky)
    f_abs = get_abs_f(KX, KY, a)
    E_plus = t * f_abs
    E_minus = -E_plus
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(KX, KY, E_plus, rstride=5,cstride=5, cmap='plasma', alpha=1.0)
    ax.plot_surface(KX, KY, E_minus, rstride=5,cstride=5, cmap='viridis_r', alpha=1.0)
    ax.set_title('Graphene Band Structure')
    ax.set_xlabel(r'$k_x$ $(1/\AA)$')
    ax.set_ylabel(r'$k_y$ $(1/\AA)$')
    ax.set_zlabel('Energy (eV)')
    # change view angle
    ax.view_init(elev=10, azim=45)
    plt.savefig('plots/graphene/graphene_band_structure.png', 
                dpi=300)

def plot_png_transparent_band(a, t, r):
    kx = np.linspace(-r, r, 1001)
    ky = np.linspace(-r, r, 1001)
    KX, KY = np.meshgrid(kx, ky)
    f_abs = get_abs_f(KX, KY, a)
    E_plus = t * f_abs
    E_minus = -E_plus
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(KX, KY, E_plus, rstride=5,cstride=5, cmap='plasma', alpha=1.0)
    ax.plot_surface(KX, KY, E_minus, rstride=5,cstride=5, cmap='viridis_r', alpha=1.0)
    # remove axes and background
    ax.set_axis_off()
    plt.grid(False)
    # change view angle
    ax.view_init(elev=10, azim=45)
    plt.savefig('plots/graphene/graphene_band_structure_transparent.png', 
                dpi=300, bbox_inches='tight', transparent=True)



def get_abs_f(kx, ky, a):
    part_1 = np.exp(1j * ky * a / np.sqrt(3))
    part_2 = 2 * np.exp(-1j * ky * a / (2 * np.sqrt(3))) * np.cos(kx * a / 2)
    f = part_1 + part_2
    return np.abs(f)

def main():
    a = 1.42  # Carbon-carbon distance in Angstroms
    r = (4 * np.pi / (3 * a)) + 0.2 # Radius of the first Brillouin zone
    t = 3.033 # Hopping parameter in eV
    plot_png_transparent_band(a, t, r)
    plot_graphene_2d_band(a, t, r)
    plot_graphene_band_3d(a, t, r)
    plot_graphene_band_heatmap(a, t, r)

if __name__ == "__main__":
    main()
