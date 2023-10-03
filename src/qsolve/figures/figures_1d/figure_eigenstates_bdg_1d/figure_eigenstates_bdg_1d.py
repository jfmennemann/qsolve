import matplotlib.pyplot as plt

from scipy import constants

from .fig_u_v_re_im_1d import fig_u_v_re_im_1d

from qsolve.figures.style import colors


class FigureEigenstatesBDG1D(object):

    def __init__(self,
                 excitations_u,
                 excitations_v,
                 V,
                 x,
                 u_v_re_im_min,
                 u_v_re_im_max,
                 V_min,
                 V_max,
                 x_ticks,
                 name="figure_eigenstates_bdg_1d"):

        # x_ticks = params["x_ticks"]

        x = x / 1e-6

        Jx = x.shape[0]

        dx = x[1] - x[0]

        x_min = x[0]
        x_max = x_min + Jx * dx

        Jx = x.size

        # -----------------------------------------------------------------------------------------
        settings = type('', (), {})()

        settings.hbar = constants.hbar

        settings.u_v_re_im_min = u_v_re_im_min
        settings.u_v_re_im_max = u_v_re_im_max

        settings.V_min = V_min
        settings.V_max = V_max

        settings.x = x

        settings.Jx = Jx

        settings.x_ticks = x_ticks

        settings.x_min = x_min
        settings.x_max = x_max

        settings.linecolor_V = colors.alizarin
        settings.linewidth_V = 1.1

        settings.label_x = r'$x \;\, \mathrm{in} \;\, \mu \mathrm{m}$'
        settings.label_t = r'$t \;\, \mathrm{in} \;\, \mathrm{ms}$'

        settings.cmap_density = colors.cmap_density

        settings.cmap_real_part = colors.cmap_real_part

        settings.color_gridlines_major = colors.color_gridlines_major
        settings.color_gridlines_minor = colors.color_gridlines_minor

        settings.fontsize_titles = 10
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        plt.rcParams.update({'font.size': 10})
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        self.fig_name = name

        self.fig = plt.figure(self.fig_name, figsize=(8, 8), facecolor="white")
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        self.gridspec = self.fig.add_gridspec(nrows=4, ncols=2,
                                              left=0.125, right=0.9,
                                              bottom=0.08, top=0.95,
                                              wspace=0.6,
                                              hspace=0.7
                                              # width_ratios=[1, 1],
                                              # height_ratios=[1, 1]
                                              )

        ax_00 = self.fig.add_subplot(self.gridspec[0, 0])
        ax_10 = self.fig.add_subplot(self.gridspec[1, 0])
        ax_20 = self.fig.add_subplot(self.gridspec[2, 0])
        ax_30 = self.fig.add_subplot(self.gridspec[3, 0])

        ax_01 = self.fig.add_subplot(self.gridspec[0, 1])
        ax_11 = self.fig.add_subplot(self.gridspec[1, 1])
        ax_21 = self.fig.add_subplot(self.gridspec[2, 1])
        ax_31 = self.fig.add_subplot(self.gridspec[3, 1])
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        self.fig_u_v_re_im_1d_00 = fig_u_v_re_im_1d(ax_00, settings)
        self.fig_u_v_re_im_1d_10 = fig_u_v_re_im_1d(ax_10, settings, legend=False)
        self.fig_u_v_re_im_1d_20 = fig_u_v_re_im_1d(ax_20, settings, legend=False)
        self.fig_u_v_re_im_1d_30 = fig_u_v_re_im_1d(ax_30, settings, legend=False)

        self.fig_u_v_re_im_1d_01 = fig_u_v_re_im_1d(ax_01, settings)
        self.fig_u_v_re_im_1d_11 = fig_u_v_re_im_1d(ax_11, settings, legend=False)
        self.fig_u_v_re_im_1d_21 = fig_u_v_re_im_1d(ax_21, settings, legend=False)
        self.fig_u_v_re_im_1d_31 = fig_u_v_re_im_1d(ax_31, settings, legend=False)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        self.fig_u_v_re_im_1d_00.update(excitations_u[0, :], excitations_v[0, :], V)
        self.fig_u_v_re_im_1d_10.update(excitations_u[1, :], excitations_v[1, :], V)
        self.fig_u_v_re_im_1d_20.update(excitations_u[2, :], excitations_v[2, :], V)
        self.fig_u_v_re_im_1d_30.update(excitations_u[3, :], excitations_v[3, :], V)

        self.fig_u_v_re_im_1d_01.update(excitations_u[4, :], excitations_v[4, :], V)
        self.fig_u_v_re_im_1d_11.update(excitations_u[5, :], excitations_v[5, :], V)
        self.fig_u_v_re_im_1d_21.update(excitations_u[6, :], excitations_v[6, :], V)
        self.fig_u_v_re_im_1d_31.update(excitations_u[-1, :], excitations_v[-1, :], V)
        # -----------------------------------------------------------------------------------------

        plt.ion()
        
        plt.draw()
        plt.pause(0.001)
        # -----------------------------------------------------------------------------------------

    # def update_data(self, psi, V):
    #
    #     # self.fig_psi_abs_squared_1d.update(psi, V)
    #     self.fig_psi_re_im_1d.update(psi, V)

    # def redraw(self):
    #
    #     # plt.figure(self.fig_name)
    #     #
    #     # plt.draw()
    #     #
    #     # self.fig.canvas.start_event_loop(0.001)
    #
    #     # -----------------------------------------------------------------------------------------
    #     # drawing updated values
    #     self.fig.canvas.draw()
    #
    #     # This will run the GUI event
    #     # loop until all UI events
    #     # currently waiting have been processed
    #     self.fig.canvas.flush_events()
    #
    #     # time.sleep(0.1)
    #     # -----------------------------------------------------------------------------------------

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
