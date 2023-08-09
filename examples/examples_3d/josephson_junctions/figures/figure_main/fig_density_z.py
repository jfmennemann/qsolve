import numpy as np

from qsolve.figures.style import colors


class fig_density_z(object):

    def __init__(self, ax, settings):

        self.hbar = settings.hbar

        self.m_atom = settings.m_atom

        self.line_density_z_x1, = ax.plot(settings.z, np.zeros_like(settings.z), linewidth=1, linestyle='-',
                                          color=colors.peter_river, label=r'$|\psi(x_1, 0, z)|^2$')

        self.line_density_z_x2, = ax.plot(settings.z, np.zeros_like(settings.z), linewidth=1, linestyle='-',
                                          color=colors.wet_asphalt, label=r'$|\psi(x_2, 0, z)|^2$')

        ax.set_xlim(settings.z_min, settings.z_max)

        density_min = settings.density_min
        density_max = settings.density_max

        ax.set_ylim(density_min - 0.1 * (density_max - density_min), density_max + 0.1 * (density_max - density_min))

        ax.set_yticks(np.linspace(density_min, density_max, num=3))

        ax.set_xlabel(settings.label_z)
        
        ax.set_xticks(settings.z_ticks)
        
        ax.grid(visible=True, which='major', color=settings.color_gridlines_major, linestyle='-', linewidth=0.5)
        
        ax.set_ylabel(settings.label_density)

        # -----------------------------------------------------------------------------------------
        ax2 = ax.twinx()

        self.line_V_z_x1, = ax2.plot(settings.z, np.zeros_like(settings.z), linewidth=1, linestyle='-',
                                           color=colors.alizarin, label=r'$V(x_1, 0, z)$')

        ax2.set_xlim(settings.z_min, settings.z_max)

        V_min = settings.V_min
        V_max = settings.V_max

        ax2.set_ylim(V_min - 0.1 * (V_max - V_min), V_max + 0.1 * (V_max - V_min))

        ax2.set_yticks(np.linspace(V_min, V_max, num=3))

        ax2.set_ylabel(settings.label_V)

        ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fancybox=False, framealpha=1.0, ncol=1)
        # -----------------------------------------------------------------------------------------

    def update(self, density_z_x1, density_z_x2, V_z_x1):

        scaling_V = self.hbar * 2 * np.pi * 1000

        V_z_x1 = V_z_x1 / scaling_V

        self.line_density_z_x1.set_ydata(density_z_x1)
        self.line_density_z_x2.set_ydata(density_z_x2)

        self.line_V_z_x1.set_ydata(V_z_x1)
