"""
Helper functions used in solution computations on a uniform rectangular grid
"""

import numpy as np


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
    # for j in range(5):
    #     Phi_Neumann[0,j] = j / np.sqrt(j**2 + cr_0_sq)

    # Test 
    Phi_Neumann[0] = np.arange(5) / np.sqrt(np.arange(5)**2 + cr_0_sq)

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
