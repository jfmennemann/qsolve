import numpy as np


def eval_data(solver):

    # ---------------------------------------------------------------------------------------------
    index_center_x = solver.index_center_x
    index_center_y = solver.index_center_y
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    N = solver.compute_n_atoms('psi')
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    V = solver.V

    V_x = V[:, index_center_y].squeeze()
    V_y = V[index_center_x, :].squeeze()
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    psi = solver.get('psi')

    psi_x = psi[:, index_center_y].squeeze()
    psi_y = psi[index_center_x, :].squeeze()

    density = np.abs(psi) ** 2

    density_x = np.abs(psi_x) ** 2
    density_y = np.abs(psi_y) ** 2

    real_part_x = np.real(psi_x)
    real_part_y = np.real(psi_y)

    imag_part_x = np.imag(psi_x)
    imag_part_y = np.imag(psi_y)
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    data = type('', (), {})()

    data.N = N

    data.psi = psi

    data.psi_x = psi_x
    data.psi_y = psi_y

    data.density = density

    data.density_x = density_x
    data.density_y = density_y

    data.real_part_x = real_part_x
    data.real_part_y = real_part_y

    data.imag_part_x = imag_part_x
    data.imag_part_y = imag_part_y

    data.V_x = V_x
    data.V_y = V_y
    # ---------------------------------------------------------------------------------------------

    return data
