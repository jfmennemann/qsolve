import matplotlib.pyplot as plt

from .fig_excitation_u_xz import FigExcitationUXZ
from .fig_excitation_v_xz import FigExcitationVXZ

class FigureEigenstatesBDG3D(object):

    def __init__(self, *, excitations_u, excitations_v, V, psi_0, x, y, z, x_ticks, y_ticks, z_ticks):

        label_x = r'$x \;\, \mathrm{in} \;\, \mu \mathrm{m}$'
        label_y = r'$y \;\, \mathrm{in} \;\, \mu \mathrm{m}$'
        label_z = r'$z \;\, \mathrm{in} \;\, \mu \mathrm{m}$'

        plt.rcParams.update({'font.size': 10})

        # -----------------------------------------------------------------------------------------
        self.fig_name = "figure_eigenstates_bdg_3d"

        self.fig = plt.figure(self.fig_name, figsize=(12, 10), facecolor="white")

        self.fig.suptitle('quasi-excitations (rescaled)', fontsize=14)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        self.gridspec = self.fig.add_gridspec(nrows=5, ncols=4,
                                              left=0.1, right=0.95,
                                              bottom=0.08, top=0.9,
                                              wspace=0.35,
                                              hspace=0.7,
                                              width_ratios=[1, 1, 1, 1],
                                              height_ratios=[1, 1, 1, 1, 1])

        ax_00 = self.fig.add_subplot(self.gridspec[0, 0])
        ax_10 = self.fig.add_subplot(self.gridspec[1, 0])
        ax_20 = self.fig.add_subplot(self.gridspec[2, 0])
        ax_30 = self.fig.add_subplot(self.gridspec[3, 0])
        ax_40 = self.fig.add_subplot(self.gridspec[4, 0])

        ax_01 = self.fig.add_subplot(self.gridspec[0, 1])
        ax_11 = self.fig.add_subplot(self.gridspec[1, 1])
        ax_21 = self.fig.add_subplot(self.gridspec[2, 1])
        ax_31 = self.fig.add_subplot(self.gridspec[3, 1])
        ax_41 = self.fig.add_subplot(self.gridspec[4, 1])

        ax_02 = self.fig.add_subplot(self.gridspec[0, 2])
        ax_12 = self.fig.add_subplot(self.gridspec[1, 2])
        ax_22 = self.fig.add_subplot(self.gridspec[2, 2])
        ax_32 = self.fig.add_subplot(self.gridspec[3, 2])
        ax_42 = self.fig.add_subplot(self.gridspec[4, 2])

        ax_03 = self.fig.add_subplot(self.gridspec[0, 3])
        ax_13 = self.fig.add_subplot(self.gridspec[1, 3])
        ax_23 = self.fig.add_subplot(self.gridspec[2, 3])
        ax_33 = self.fig.add_subplot(self.gridspec[3, 3])
        ax_43 = self.fig.add_subplot(self.gridspec[4, 3])
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        # FigGroundStateGPE2D(ax_00, V, psi_0, x, y, label_x, label_y, x_ticks, y_ticks, title=r'$\psi_0$ (scaled)')

        # FigPotential2D(ax_01, V, x, y, label_x, label_y, x_ticks, y_ticks, r'$V$ (scaled)')

        # n = excitations_u.shape[0]

        import numpy as np

        for nr in np.arange(excitations_u.shape[0]):

            print(nr)
            print(np.max(np.abs(excitations_u[nr, :, :, :])))
            print(np.max(np.abs(excitations_v[nr, :, :, :])))
            print()
            print()

        # input()

        FigExcitationUXZ(ax_00, V, excitations_u, 0, x, y, z, label_x, label_z, x_ticks, z_ticks)
        FigExcitationUXZ(ax_10, V, excitations_u, 1, x, y, z, label_x, label_z, x_ticks, z_ticks)
        FigExcitationUXZ(ax_20, V, excitations_u, 2, x, y, z, label_x, label_z, x_ticks, z_ticks)
        FigExcitationUXZ(ax_30, V, excitations_u, 3, x, y, z, label_x, label_z, x_ticks, z_ticks)
        FigExcitationUXZ(ax_40, V, excitations_u, 4, x, y, z, label_x, label_z, x_ticks, z_ticks)

        FigExcitationVXZ(ax_02, V, excitations_v, 0, x, y, z, label_x, label_z, x_ticks, z_ticks)
        FigExcitationVXZ(ax_12, V, excitations_v, 1, x, y, z, label_x, label_z, x_ticks, z_ticks)
        FigExcitationVXZ(ax_22, V, excitations_v, 2, x, y, z, label_x, label_z, x_ticks, z_ticks)
        FigExcitationVXZ(ax_32, V, excitations_v, 3, x, y, z, label_x, label_z, x_ticks, z_ticks)
        FigExcitationVXZ(ax_42, V, excitations_v, 4, x, y, z, label_x, label_z, x_ticks, z_ticks)

        # FigEigenstateBDGXY(ax_12, V, excitations_u, n-4, x, y, z, label_x, label_y, x_ticks, z_ticks, r'$u_{-4}$ (scaled)')
        # FigEigenstateBDGXY(ax_22, V, excitations_u, n-3, x, y, z, label_x, label_y, x_ticks, z_ticks, r'$u_{-3}$ (scaled)')
        # FigEigenstateBDGXY(ax_32, V, excitations_u, n-2, x, y, z, label_x, label_y, x_ticks, z_ticks, r'$u_{-2}$ (scaled)')
        # FigEigenstateBDGXY(ax_42, V, excitations_u, n-1, x, y, z, label_x, label_y, x_ticks, z_ticks, r'$u_{-1}$ (scaled)')

        # FigEigenstateBDGXY(ax_13, V, excitations_v, n-4, x, y, z, label_x, label_y, x_ticks, z_ticks, r'$v_{-4}$ (scaled)')
        # FigEigenstateBDGXY(ax_23, V, excitations_v, n-3, x, y, z, label_x, label_y, x_ticks, z_ticks, r'$v_{-3}$ (scaled)')
        # FigEigenstateBDGXY(ax_33, V, excitations_v, n-2, x, y, z, label_x, label_y, x_ticks, z_ticks, r'$v_{-2}$ (scaled)')
        # FigEigenstateBDGXY(ax_43, V, excitations_v, n-1, x, y, z, label_x, label_y, x_ticks, z_ticks, r'$v_{-1}$ (scaled)')
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        plt.ion()
        
        plt.draw()
        plt.pause(0.001)
        # -----------------------------------------------------------------------------------------

    def export(self, filepath):

        plt.figure(self.fig_name)

        plt.draw()

        self.fig.canvas.start_event_loop(0.001)

        plt.savefig(filepath,
                    dpi=None,
                    facecolor='w',
                    edgecolor='w',
                    format='png',
                    transparent=False,
                    bbox_inches=None,
                    pad_inches=0,
                    metadata=None)
