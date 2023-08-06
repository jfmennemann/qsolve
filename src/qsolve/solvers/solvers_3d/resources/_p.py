# self._p = {'hbar': self._hbar,
        #            'mu_B': self._mu_B,
        #            'k_B': self._k_B,
        #            'm_atom': self._m_atom,
        #            'Lx': self._Lx,
        #            'Ly': self._Ly,
        #            'Lz': self._Lz}


# def init_external_potential(self, compute_external_potential, parameters_potential):
#
#     self._compute_external_potential = compute_external_potential
#
#     for key, p in parameters_potential.items():
#
#         if type(p) is not tuple:
#
#             _value = p
#
#         else:
#
#             value = p[0]
#             unit = p[1]
#
#             if unit == 'm':
#                 _value = value / self._units.unit_length
#             elif unit == 's':
#                 _value = value / self._units.unit_time
#             elif unit == 'Hz':
#                 _value = value / self._units.unit_frequency
#             elif unit == 'J':
#                 _value = value / self._units.unit_energy
#             elif unit == 'J/m':
#                 _value = value * self._units.unit_length / self._units.unit_energy
#             else:
#                 raise Exception('unknown unit')
#
#         self._p[key] = _value