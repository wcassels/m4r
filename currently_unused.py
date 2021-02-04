"""
Functions that are currently unused in solution implementations but may be useful
for future implementations or error analysis
"""

import numpy as np

def get_deriv_vecs(N, grid_dist, c, include_first_derivs=False):
    """
    Returns the 5 N-length derivative vectors for the RBF functions, ie:
    dphi/dx, dphi/dy, d2phi/dx2, d2phi/dy2, d2phi/dxdy
    """
    if N == 5:
        if include_first_derivs:
            dphidx = np.array([0, 0, 1, -1, 0]) / np.sqrt(1 + 4*c**2)
            dphidy = np.array([0, 1, 0, 0, -1]) / np.sqrt(1 + 4*c**2)

        # again, relocated the /grid_dist to the step function
        d2phidx2 = (np.array([0, 1, 0, 0, 1], dtype=np.float128) + 4*c**2)
        d2phidx2[1:] /= (1 + 4*c**2)**(3/2)
        d2phidx2[0] /= (8*c**3)

        d2phidy2 = (np.array([0, 0, 1, 1, 0], dtype=np.float128) + 4*c**2)
        d2phidy2[1:] /= (1 + 4*c**2)**(3/2)
        d2phidy2[0] /= (8*c**3)

    elif N == 9:

        if include_first_derivs:
            dphidx = np.array([0, 1, 0, -1, 1, -1, 1, 0, -1], dtype=np.float64)
            dphidx[[4, 5]] /= np.sqrt(1+8*c**2)
            dphidx[[1, 3, 6, 8]] /= np.sqrt(2+8*c**2)

            dphidy = np.array([0, 1, 1, 1, 0, 0, -1, -1, -1], dtype=np.float64)
            dphidy[[2, 7]] /= np.sqrt(1+8*c**2)
            dphidy[[1, 3, 6, 8]] /= np.sqrt(2+8*c**2)

        d2phidx2 = (np.array([0, 1, 1, 1, 0, 0, 1, 1, 1]) + 8*c**2) / grid_dist
        d2phidx2[0] /= c**3 * 2**(9/2)
        d2phidx2[[2, 4, 5, 7]] /= (1 + 8*c**2)**(3/2)
        d2phidx2[[1, 3, 6, 8]] /= (2 + 8*c**2)**(3/2)

        d2phidy2 = (np.array([0, 1, 0, 1, 1, 1, 1, 0, 1]) + 8*c**2) / grid_dist
        d2phidy2[0] /= c**3 * 2**(9/2)
        d2phidy2[[2, 4, 5, 7]] /= (1 + 8*c**2)**(3/2)
        d2phidy2[[1, 3, 6, 8]] /= (2 + 8*c**2)**(3/2)
    else:
        raise NotImplementedError("N must be either 5 or 9")

    if include_first_derivs:
        return dphidx, dphidy, d2phidx2, d2phidy2
    else:
        return d2phidx2, d2phidy2


def step_domain_old(T, weighted_phi):
    """
    Uniform grid domain step using more general method that will be used for
    non-uniform configurations
    """
    T += domain_increment(T, weighted_phi)

    return T[1:-1,1:-1]


def plot_indices_excl_corners(x, y):
    """
    Used for plotting when corner values have not been computed.

    This didn't work :D
    """
    x_list, y_list = np.ravel(np.arange(x)), np.ravel(np.arange(y))

    return np.delete(x_list, [0, x-1, x*(y-1), x*y-1]), np.delete(y_list, [0, x-1, x*(y-1), x*y-1])
