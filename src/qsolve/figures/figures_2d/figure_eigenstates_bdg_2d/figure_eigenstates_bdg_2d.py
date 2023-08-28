import matplotlib.pyplot as plt


from .fig_ground_state_gpe_2d import FigGroundStateGPE2D
from .fig_potential_2d import FigPotential2D
from .fig_eigenstate_bdg_2d import FigEigenstateBDG2D


class FigureEigenstatesBDG2D(object):

    def __init__(self, *, eigenvectors_u, eigenvectors_v, V, psi_0, x, y, x_ticks, y_ticks):

        label_x = r'$x \;\, \mathrm{in} \;\, \mu \mathrm{m}$'
        label_y = r'$y \;\, \mathrm{in} \;\, \mu \mathrm{m}$'

        plt.rcParams.update({'font.size': 10})

        # -----------------------------------------------------------------------------------------
        self.fig_name = "figure_eigenstates_bdg_2d"

        self.fig = plt.figure(self.fig_name, figsize=(12, 10), facecolor="white")
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        self.gridspec = self.fig.add_gridspec(nrows=5, ncols=4,
                                              left=0.1, right=0.95,
                                              bottom=0.08, top=0.95,
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

        # ax_02 = self.fig.add_subplot(self.gridspec[0, 2])
        ax_12 = self.fig.add_subplot(self.gridspec[1, 2])
        ax_22 = self.fig.add_subplot(self.gridspec[2, 2])
        ax_32 = self.fig.add_subplot(self.gridspec[3, 2])
        ax_42 = self.fig.add_subplot(self.gridspec[4, 2])

        # ax_03 = self.fig.add_subplot(self.gridspec[0, 3])
        ax_13 = self.fig.add_subplot(self.gridspec[1, 3])
        ax_23 = self.fig.add_subplot(self.gridspec[2, 3])
        ax_33 = self.fig.add_subplot(self.gridspec[3, 3])
        ax_43 = self.fig.add_subplot(self.gridspec[4, 3])
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        FigGroundStateGPE2D(ax_00, V, psi_0, x, y, label_x, label_y, x_ticks, y_ticks, title=r'$\psi_0$ (scaled)')

        FigPotential2D(ax_01, V, x, y, label_x, label_y, x_ticks, y_ticks, r'$V$ (scaled)')

        n = eigenvectors_u.shape[0]

        FigEigenstateBDG2D(ax_10, V, eigenvectors_u, 0, x, y, label_x, label_y, x_ticks, y_ticks, r'$u_0$ (scaled)')
        FigEigenstateBDG2D(ax_20, V, eigenvectors_u, 1, x, y, label_x, label_y, x_ticks, y_ticks, r'$u_1$ (scaled)')
        FigEigenstateBDG2D(ax_30, V, eigenvectors_u, 2, x, y, label_x, label_y, x_ticks, y_ticks, r'$u_2$ (scaled)')
        FigEigenstateBDG2D(ax_40, V, eigenvectors_u, 3, x, y, label_x, label_y, x_ticks, y_ticks, r'$u_3$ (scaled)')

        FigEigenstateBDG2D(ax_11, V, eigenvectors_v, 0, x, y, label_x, label_y, x_ticks, y_ticks, r'$v_0$ (scaled)')
        FigEigenstateBDG2D(ax_21, V, eigenvectors_v, 1, x, y, label_x, label_y, x_ticks, y_ticks, r'$v_1$ (scaled)')
        FigEigenstateBDG2D(ax_31, V, eigenvectors_v, 2, x, y, label_x, label_y, x_ticks, y_ticks, r'$v_2$ (scaled)')
        FigEigenstateBDG2D(ax_41, V, eigenvectors_v, 3, x, y, label_x, label_y, x_ticks, y_ticks, r'$v_3$ (scaled)')

        FigEigenstateBDG2D(ax_12, V, eigenvectors_u, n-4, x, y, label_x, label_y, x_ticks, y_ticks, r'$u_{-4}$ (scaled)')
        FigEigenstateBDG2D(ax_22, V, eigenvectors_u, n-3, x, y, label_x, label_y, x_ticks, y_ticks, r'$u_{-3}$ (scaled)')
        FigEigenstateBDG2D(ax_32, V, eigenvectors_u, n-2, x, y, label_x, label_y, x_ticks, y_ticks, r'$u_{-2}$ (scaled)')
        FigEigenstateBDG2D(ax_42, V, eigenvectors_u, n-1, x, y, label_x, label_y, x_ticks, y_ticks, r'$u_{-1}$ (scaled)')

        FigEigenstateBDG2D(ax_13, V, eigenvectors_v, n-4, x, y, label_x, label_y, x_ticks, y_ticks, r'$v_{-4}$ (scaled)')
        FigEigenstateBDG2D(ax_23, V, eigenvectors_v, n-3, x, y, label_x, label_y, x_ticks, y_ticks, r'$v_{-3}$ (scaled)')
        FigEigenstateBDG2D(ax_33, V, eigenvectors_v, n-2, x, y, label_x, label_y, x_ticks, y_ticks, r'$v_{-2}$ (scaled)')
        FigEigenstateBDG2D(ax_43, V, eigenvectors_v, n-1, x, y, label_x, label_y, x_ticks, y_ticks, r'$v_{-1}$ (scaled)')
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
