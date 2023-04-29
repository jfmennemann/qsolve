from numpy import zeros_like

from numpy import pi

from .. style import colors


class fig_phase_y(object):

    def __init__(self, ax, settings):

        self.hbar = settings.hbar

        self.m_atom = settings.m_atom

        # -----------------------------------------------------------------------------------------
        self.line_phase_y, = ax.plot(settings.y, zeros_like(settings.y), linewidth=1, linestyle='-', color=colors.wet_asphalt, label=r'$\Re\, \psi$')

        ax.set_xlim(settings.y_min, settings.y_max)

        ax.set_ylim(settings.real_part_min, settings.real_part_max)

        ax.set_xlabel(settings.label_y)
        
        ax.set_xticks(settings.y_ticks)
        
        ax.grid(visible=True, which='major', color=settings.color_gridlines_major, linestyle='-', linewidth=0.5)

        ax.set_ylabel(r'arbitrary units')

        ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), fancybox=settings.fancybox, framealpha=1.0, ncol=1)

        ax.set_title('$x=0$', fontsize=settings.fontsize_titles)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        ax_V_y = ax.twinx()
    
        self.line_V_y, = ax_V_y.plot(settings.y, zeros_like(settings.y), linewidth=1, linestyle='-', color=colors.sun_flower, label=r'$V$')

        ax_V_y.set_xlim(settings.y_min, settings.y_max)
        ax_V_y.set_ylim(settings.V_min, settings.V_max)
        
        ax_V_y.set_ylabel(settings.label_V)

        ax_V_y.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fancybox=settings.fancybox, framealpha=1.0, ncol=1)
        # -----------------------------------------------------------------------------------------

    def update(self, real_part_y, imag_part_y, V_y):
        
        scaling_V = self.hbar * 2 * pi * 1000
        
        V_y = V_y / scaling_V
        
        self.line_phase_y.set_ydata(real_part_y)

        self.line_V_y.set_ydata(V_y)
