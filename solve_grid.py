"""
Functions for solving the diffusion equation in the uniform grid node configuration
"""

import numpy as np
import matplotlib.pyplot as plt
import time

import rect_helpers


def domain_increment(A, weighted_phi):
    """
    For the uniform grid implementation, return the scaled sum of the second
    derivatives wrt x and y of A - this is the t such that T_new = T_old + t,
    inside the domain of the grid
    """

    m, n = A.shape

    N = weighted_phi.size

    combined_2nd_deriv = np.zeros((m, n))

    # Excluding boundaries for now
    for i in range(1, m-1):
        for j in range(1, n-1):
            combined_2nd_deriv[i,j] = A[rect_helpers.get_grid_indices(i, j, N)].dot(weighted_phi)

    return combined_2nd_deriv


def set_boundary(A, condition, edge, val, c):
    """
    Compute the boundary values of the rectangle using boundary conditions provided
    """
    m, n = A.shape

    if condition == "Dirichlet":
        if edge == "North":
            A[0] = val
        elif edge == "East":
            A[:,-1] = val
        elif edge == "West":
            A[:,0] = val
        elif edge == "South":
            A[-1] = val
        else:
            raise ValueError("Invalid boundary selected")

        return

    elif condition == "Neumann":
        Phi_Neumann = rect_helpers.get_Phi_Neumann(c, grid_dist)

        # phi_vec = np.sqrt(np.array(0, 1, 1, 2, 2) + (c**2) * 16)
        phi_vec = np.sqrt(np.arange(4) + (c**2)*16)

        if edge == "North":
            boundary_idx = lambda i: i, 0
        elif edge == "South":
            boundary_idx = lambda i: i, -1
        elif edge == "East":
            boundary_idx = lambda i: -1, i
        elif edge == "West":
            boundary_idx = lambda i: 0, i
        else:
            raise ValueError("Invalid boundary selected")

        # Exclude corners for now (or permanently? paper seemingly does not do corners.)
        for i in range(1, n-1):
            rhs = A[rect_helpers.get_boundary_indices(edge, i)]
            rhs[0] = val
            alphas = np.linalg.solve(Neumann_Phi, neighbourhood_vals)
            A[boundary_idx(i)] = alphas.dot(phi_vec)

        return

    elif condition == "Robin":
        pass
    else:
        raise ValueError("Invalid boundary condition selected")


def step_domain(T, update_weights):
    """
    Very fast domain step for uniform grid implementation

    Returns smaller grid since boundary values not computed
    """
    # Doing all the dot products is equivalent to just summing shifted layers
    # of the grid
    T[1:-1,1:-1] += update_weights[0] * T[1:-1,1:-1] + update_weights[1] * (T[:-2,1:-1]
                    + T[2:,1:-1] + T[1:-1,:-2] + T[1:-1,2:])

    return T[1:-1,1:-1]


def grid_step(T, weighted_phi, grid_dist, c, boundary_conditions):
    """
    Step forwards using step_domain to compute values at domain nodes then
    applies boundary conditions

    Boundary conditions - a dictionary with entries {"Edge": ("Condition", value)}
    """
    # Update domain nodes
    step_domain(T, weighted_phi)

    # Apply boundary conditions
    for edge, (condition, value) in boundary_conditions.items():
        set_boundary(T, condition, edge, value, c)

    return T
