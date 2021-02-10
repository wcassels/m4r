"""
Helper functions used in solution computations on a uniform rectangular grid
"""

import numpy as np

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


def get_boundary_neighbourhood(edge, pos):
    """
    Return the neighbourhood indices for a boundary node, with neighbourhood
    configuration "....."
    """
    if edge == "North":
        return ([0, 1, 2, 3, 4], [pos, pos, pos, pos, pos])
    elif edge == "South":
        return ([-1, -2, -3, -4, -5], [pos, pos, pos, pos, pos])
    elif edge == "East":
        return ([pos, pos, pos, pos, pos], [-1, -2, -3, -4, -5])
    elif edge == "West":
        return ([pos, pos, pos, pos, pos], [0, 1, 2, 3, 4])
    else:
        raise ValueError("Invalid boundary selected")


def get_boundary_positions(edge):
        if edge == "North":
            return lambda i: (0, i)
        elif edge == "South":
            return lambda i: (-1, i)
        elif edge == "East":
            return lambda i: (i, -1)
        elif edge == "West":
            return lambda i: (i, 0)
        else:
            raise ValueError("Invalid boundary selected")


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


def get_Phi_Neumann(c, grid_dist):
    """
    Return collocation matrix Phi_Neumann for a rectangular boundary. Uses a
    neighbourhood of 5 linear nodes leading away from the boundary - have to be
    careful with other node distributions since they can easily lead to having
    singular matrices

    Handles N = 5 only currently
    """
    Phi_Neumann = np.zeros((5,5), dtype=np.float64)

    # Max inter-neighbourhood distance is 4 for this configuration
    cr_0_sq = (c**2) * 16

    # Neumann boundary condition row
    for j in range(5):
        Phi_Neumann[0,j] = j / np.sqrt(j**2 + cr_0_sq)

    # Other rows
    for i in range(1, 5):
        for j in range(5):
            Phi_Neumann[i,j] = np.sqrt((i-j)**2 + cr_0_sq) * grid_dist

    return Phi_Neumann


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
