import sympy as sym


x = sym.symbols('x')

x0 = sym.symbols('x0')

Lx = sym.symbols('Lx')

k = 100  # needs to be even

f_sympy = (1 - sym.cos((sym.pi * (x-x0) / Lx)) ** 2) ** 4 * (1 + 0.2 * sym.sin(k * sym.pi * (x-x0) / Lx))
f_x_sympy = sym.diff(f_sympy, x, 1)
f_xx_sympy = sym.diff(f_sympy, x, 2)


def f(x_numpy):

    Jx = x_numpy.shape[0]

    dx = x_numpy[1] - x_numpy[0]

    x0 = x_numpy[0]

    f_sympy_tmp = f_sympy.subs('Lx', Jx * dx)
    f_sympy_tmp = f_sympy_tmp.subs('x0', x0)

    f_sympy_lambdified = sym.lambdify(x, f_sympy_tmp, "numpy")

    return f_sympy_lambdified(x_numpy)


def f_x(x_numpy):

    Jx = x_numpy.shape[0]

    dx = x_numpy[1] - x_numpy[0]

    x0 = x_numpy[0]

    f_x_sympy_tmp = f_x_sympy.subs('Lx', Jx * dx)
    f_x_sympy_tmp = f_x_sympy_tmp.subs('x0', x0)

    f_x_sympy_lambdified = sym.lambdify(x, f_x_sympy_tmp, "numpy")

    return f_x_sympy_lambdified(x_numpy)


def f_xx(x_numpy):

    Jx = x_numpy.shape[0]

    dx = x_numpy[1] - x_numpy[0]

    x0 = x_numpy[0]

    f_xx_sympy_tmp = f_xx_sympy.subs('Lx', Jx * dx)
    f_xx_sympy_tmp = f_xx_sympy_tmp.subs('x0', x0)

    f_xx_sympy_lambdified = sym.lambdify(x, f_xx_sympy_tmp, "numpy")

    return f_xx_sympy_lambdified(x_numpy)
