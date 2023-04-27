import numpy as np


def eval_data(solver):

    dx = solver.get('dx')

    index_center_x = solver.get('index_center_x')
    index_center_y = solver.get('index_center_y')

    data = type('', (), {})()

    # ---------------------------------------------------------------------------------------------
    V = solver.get('V')

    V_x = np.squeeze(V[:, index_center_y])
    V_y = np.squeeze(V[index_center_x, :])

    data.V = V

    data.V_x = V_x
    data.V_y = V_y
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    psi = solver.get('psi')

    psi_x = np.squeeze(psi[:, index_center_y])
    psi_y = np.squeeze(psi[index_center_x, :])

    data.psi = psi

    data.psi_x = psi_x
    data.psi_y = psi_y
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    phase = np.angle(psi)

    data.phase_y = np.squeeze(phase[index_center_x, :])

    tmp = dx * np.sum(psi, 0, keepdims=False)

    data.phase_y_eff = np.angle(tmp)
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    density = np.real(psi * np.conj(psi))

    density_max = np.max(density)

    density_y = np.abs(psi_y) ** 2

    density_xy = density

    if density_max > 0:

        density_xy = density_xy / density_max

    density_y_eff = dx * np.sum(density, (1, ), keepdims=False)

    data.density = density

    data.density_y = density_y

    data.density_xy = density_xy

    data.density_y_eff = density_y_eff
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    real_part = np.real(psi)

    real_part_x = np.real(psi_x)
    real_part_y = np.real(psi_y)

    imag_part_x = np.imag(psi_x)
    imag_part_y = np.imag(psi_y)

    # real_part_xz = np.real(psi_xz)
    # real_part_xy = np.real(psi_xy)

    data.real_part = real_part

    data.real_part_x = real_part_x
    data.real_part_y = real_part_y

    data.imag_part_x = imag_part_x
    data.imag_part_y = imag_part_y

    # data.real_part_xz = real_part_xz
    # data.real_part_xy = real_part_xy
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    data.N = solver.compute_n_atoms('psi')
    # ---------------------------------------------------------------------------------------------

    return data
