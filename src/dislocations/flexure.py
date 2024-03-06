import gflex
import numpy as np

def flexure_from_displacement_profile(displacement_profile: np.ndarray, te: float = 1.5e4,
                                      spacing: float = 100., half_width_km = 250,
                                      crust_rho: float = 2800, mantle_rho: float = 3300):
    flex = gflex.F1D()

    flex.Quiet = True

    flex.Method = 'FD' # Solution method: * FD (finite difference)
                       #                  * SAS (superposition of analytical solutions)
                       #                  * SAS_NG (ungridded SAS)

    flex.Solver = 'direct' # direct or iterative
    # convergence = 1E-3 # convergence between iterations, if an iterative solution
                         # method is chosen

    flex.g = 9.8 # acceleration due to gravity
    flex.E = 65E9 # Young's Modulus
    flex.nu = 0.25 # Poisson's Ratio
    flex.rho_m = 3300. # MantleDensity
    flex.rho_fill = 1000. # InfiillMaterialDensity

    x_indices = np.array(displacement_profile[:, 0] / spacing + half_width_km * 1.e3 / spacing, dtype=int)

    flex.Te = te
    flex.qs = np.zeros(int(2 * half_width_km * 1.e3 / spacing))
    flex.qs[x_indices] += crust_rho * flex.g * displacement_profile[:, 1]
    flex.dx = spacing
    flex.BC_W = '0Displacement0Slope' # west boundary condition
    flex.BC_E = '0Displacement0Slope' # east boundary condition

    flex.sigma_xx = 100. # Normal stress on the edge of the plate

    flex.initialize()
    flex.run()
    flex.finalize()

    deflection = flex.w

    return deflection[x_indices]