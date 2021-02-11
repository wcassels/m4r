def sarler_first(x, y, h=750, k=52, trunc=50):
    """
    Returns the truncated analytical equilibrium state described in the first
    test of the Sarler paper.
    """
    R = -h / k
    f = lambda x: x*np.tan(0.6*x) + R
    # hacky way of getting betas - work on something more precise?
    betas = fsolve(f, 2.34+5*np.arange(trunc))

    # Both papers are slightly wrong!
    return -2 * 100 * R * np.sum(np.cos(betas*x) * (betas*np.cosh(betas*(1-y)) - \
        R * np.sinh(betas*(1-y))) / (np.cos(betas*0.6)*(betas*np.cosh(betas) - R * \
        np.sinh(betas)) * (0.6*(R**2 + betas**2) - R)))


def sarler_second(x, y, t, diff, trunc=50, dist=1):
    """
    Returns the truncated analytical solution described in the second test of
    the Sarler paper.
    """
    sol = np.zeros_like(x)
    m, n = x.shape

    # term_idx = np.arange(1, trunc+1)
    term_idx = np.arange(trunc) # Sarler paper has a typo :O

    for i in range(m):
        for j in range(n):
            x_terms = sarler_second_test_nth(x[i,j], t, diff, dist, term_idx)
            y_terms = sarler_second_test_nth(y[i,j], t, diff, dist, term_idx)
            sum_x = 4 * np.sum(x_terms) / np.pi
            sum_y = 4 * np.sum(y_terms) / np.pi

            sol[i,j] = sum_x * sum_y

    return sol


def sarler_second_nth(eta, t, diff, dist, n):
    """
    Returns the nth term of either sum for the analytical solution described in
    the second test of the Sarler paper.

    Arguments:
    - eta: x or y
    - t: time value
    - diff: diffusivity of the problem
    - dist: eta_max - eta_min
    - n: the term in the sum to compute
    """
    # changed eta-1 to eta?
    return ((-1) ** n / (2*n + 1)) * np.exp(-(diff*(2*n+1)**2*np.pi**2*t) / (4*dist**2)) * np.cos((2*n+1)*np.pi*(eta)/(2*dist))
