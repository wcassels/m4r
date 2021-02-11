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


def get_grid_indices(i, j, N):
    """
    Helper function returning neighbourhood index values for domain nodes in the
    uniform grid configuration

    In the efficient implementation this goes unused
    """
    if N == 5:
        T_idx_y = [i, i-1, i, i, i+1]
        T_idx_x = [j, j, j-1, j+1, j]
    elif N == 9:
        T_idx_y = [i, i-1, i-1, i-1, i, i, i+1, i+1, i+1]
        T_idx_x = [j, j-1, j, j+1, j-1, j+1, j-1, j, j+1]
    else:
        raise NotImplementedError("N must be either 5 or 9")

    return T_idx_y, T_idx_x

# following functions were written for the uniform grid case but replaced by
# more general ones
def get_Phi(N, c):
    """
    Returns the NxN collocation matrix of RBF function values for the uniform
    grid configuration
    """
    if N == 5:
        nodes = [(0,0), (0,1), (-1,0), (1,0), (0,-1)]
    elif N == 9:
        nodes = [(0,0), (-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1)]
    else:
        raise NotImplementedError("N must be either 5 or 9")

    D_sq = np.zeros((N, N))
    for i, (x1, y1) in enumerate(nodes):
        for j, (x2, y2) in enumerate(nodes):
            D_sq[i,j] = (x2-x1) ** 2 + (y2-y1) ** 2

    cr_0_sq = (N-1) * c ** 2

    # the * grid_dist has been relocated to the step function, perhaps improves numerical precision
    return np.sqrt(D_sq + cr_0_sq)


def get_update_weights(Phi, diffusivity, time_step, grid_dist, shape_param):
    """
    Returns the coefficients in the linear combination of neighbourhood node
    of the increment between time steps. This is a constant in the uniform
    grid configuration, in fact there are only two unique values in the vector.
    The grid_dist factor is omitted as it simplifies elsewhere
    """
    c = shape_param

    # the sum of the two second derivative vectors
    sum_2nd_derivs = (np.array([0, 1, 1, 1, 1], dtype=np.float64) + 8*c**2)
    sum_2nd_derivs[1:] /= (1 + 4*c**2)**(3/2)
    sum_2nd_derivs[0] /= (8*c**3)

    # weighted_phi = Phi^-T .dot(combined_deriv_vec)
    # (in this case don't need to transpose but will do in the general case)
    return np.linalg.solve(Phi.T, sum_2nd_derivs * diffusivity * time_step / (grid_dist**2))


def domain_increment(A, weighted_phi):
    """
    For the uniform grid implementation, return the scaled sum of the second
    derivatives wrt x and y of A - this is the t such that T_new = T_old + t,
    inside the domain of the grid (currently unused since step_grid is vastly
    more efficient in the uniform case, but will be the primary method used in
    solving non-uniform problems)
    """

    m, n = A.shape

    N = weighted_phi.size

    combined_2nd_deriv = np.zeros((m, n))

    # Excluding boundaries for now
    for i in range(1, m-1):
        for j in range(1, n-1):
            combined_2nd_deriv[i,j] = A[rect_utils.get_grid_indices(i, j, N)].dot(weighted_phi)

    return combined_2nd_deriv



def get_Phi_Robin(c, grid_dist, condition_value):
    Phi_Robin = np.zeros((5,5), dtype=np.float64)

    # Max inter-neighourhood distance is 4 for this configuration
    cr_0_sq = (c**2) * 16

    # Row enforcing Robin condition
    Phi_Robin[0] = np.arange(5) / np.sqrt(np.arange(5)**2 + cr_0_sq) - \
                   condition_value * np.sqrt(np.arange(5)**2 + cr_0_sq) * grid_dist

    # Other rows
    for i in range(1, 5):
        for j in range(5):
            Phi_Robin[i,j] = np.sqrt((i-j)**2 + cr_0_sq) * grid_dist


    return Phi_Robin
