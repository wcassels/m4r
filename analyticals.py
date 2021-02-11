import numpy as np
from scipy.optimize import fsolve

def sarler_first(x, y, h=750, k=52, trunc=50):
    """
    Evaluates the analytical equilibrium state for the first Sarler test, for
    arrays of positions x, y
    """
    R = -h / k
    f = lambda x: x*np.tan(0.6*x) + R

    # Get betas 
    betas = fsolve(f, 2.34+5*np.arange(trunc), xtol=1e-14)

    # Both papers are slightly wrong!
    point_lambda = lambda x, y: sarler_first_point_eval(x, y, betas, R)
    return np.vectorize(point_lambda)(x, y)


def sarler_first_point_eval(x, y, betas, R):
    """
    Evaluates the analytical equilibrium state for the first Sarler test, at a single point.
    """
    return -2 * 100 * R * np.sum(np.cos(betas*x) * (betas*np.cosh(betas*(1-y)) - \
        R * np.sinh(betas*(1-y))) / (np.cos(betas*0.6)*(betas*np.cosh(betas) - R * \
        np.sinh(betas)) * (0.6*(R**2 + betas**2) - R)))


def sarler_second(x, y, t, diff, trunc=50, dist=1):
    """
    Returns the truncated analytical solution described in the second test of
    the Sarler paper.
    """
    point_lambda = lambda x, y: 16 * sarler_second_sum(x, t, diff, dist, trunc) \
                    * sarler_second_sum(y, t, diff, dist, trunc) / (np.pi**2)

    return np.vectorize(point_lambda)(x, y)


def sarler_second_sum(eta, t, diff, dist, trunc):
    """
    Returns either sum for the analytical solution described in
    the second test of the Sarler paper.

    Arguments:
    - eta: x or y
    - t: time value
    - diff: diffusivity of the problem
    - dist: eta_max - eta_min
    - n: the term in the sum to compute
    """
    n = np.arange(trunc) # Sarler paper has a typo - sum from 0.
    # another typo - changed eta-1 to eta
    return np.sum(((-1) ** n / (2*n + 1)) * np.exp(-(diff*(2*n+1)**2*np.pi**2*t) \
            / (4*dist**2)) * np.cos((2*n+1)*np.pi*(eta)/(2*dist)))
