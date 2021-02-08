"""
Functions for solving the diffusion equation in the uniform grid node configuration
"""

import numpy as np
import matplotlib.pyplot as plt
import time

import unif_utils


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
            combined_2nd_deriv[i,j] = A[unif_utils.get_grid_indices(i, j, N)].dot(weighted_phi)

    return combined_2nd_deriv


def unif_boundary(A, condition, edge, val, c, grid_dist, robin_ref=0):
    """
    Fast boundary implementation for uniform grid solutions
    """
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
    else:
        if condition == "Neumann":
            Phi_boundary = unif_utils.get_Phi_Neumann(c, grid_dist)
            rhs0 = val
        elif condition == "Robin":
            Phi_boundary = unif_utils.get_Phi_Robin(c, grid_dist, val)
            rhs0 = -val * robin_ref
        else:
            raise ValueError("Invalid boundary condition selected")

        phi_vec = np.sqrt(np.arange(5)**2 + (c**2)*16) * grid_dist

        update_weights = np.linalg.solve(Phi_boundary.T, phi_vec)

        if edge == "North":
            infl_matrix = A[:5,1:-1].T
            infl_matrix[:,0] = rhs0
            A[0,1:-1] = infl_matrix.dot(update_weights)
        elif edge == "South":
            infl_matrix = A[-5:,1:-1][::-1].T
            infl_matrix[:,0] = rhs0
            A[-1,1:-1] = infl_matrix.dot(update_weights)
        elif edge == "West":
            infl_matrix = A[1:-1,:5]
            infl_matrix[:,0] = rhs0
            A[1:-1,0] = infl_matrix.dot(update_weights)
        elif edge == "East":
            infl_matrix = A[1:-1,-5:][:,::-1]
            infl_matrix[:,0] = rhs0
            A[1:-1,-1] = infl_matrix.dot(update_weights)
        else:
            raise ValueError("Invalid boundary selected")

        return



def set_boundary(A, condition, edge, val, c, grid_dist, robin_ref=0):
    """
    Compute the boundary values of the rectangle using boundary conditions provided
    """
    m, n = A.shape

    if edge == "North" or edge == "South":
        mn = n
    elif edge == "East" or edge ==  "West":
        mn = m

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
        Phi_Neumann = unif_utils.get_Phi_Neumann(c, grid_dist)

        phi_vec = np.sqrt(np.arange(5)**2 + (c**2)*16) * grid_dist

        boundary_idx = unif_utils.get_boundary_positions(edge)

        Neumann_update_weights = np.linalg.solve(Phi_Neumann.T, phi_vec)

        # Exclude corners for now (or permanently? paper seemingly does not do corners.)
        for i in range(1, mn-1):
            neighbourhood_x, neighbourhood_y = unif_utils.get_boundary_neighbourhood(edge, i)
            rhs = A[neighbourhood_x, neighbourhood_y]
            rhs[0] = val

            # More general method, slower
            # alphas = np.linalg.solve(Phi_Neumann, rhs)
            # A[boundary_idx(i)] = alphas.dot(phi_vec)

            # Update_weights method
            A[boundary_idx(i)] = Neumann_update_weights.dot(rhs)

        return

    elif condition == "Robin":
        Phi_Robin = unif_utils.get_Phi_Robin(c, grid_dist, val)

        phi_vec = np.sqrt(np.arange(5)**2 + (c**2)*16) * grid_dist

        boundary_idx = unif_utils.get_boundary_positions(edge)

        Robin_update_weights = np.linalg.solve(Phi_Robin.T, phi_vec)

        for i in range(1, mn-1):
            rhs = A[unif_utils.get_boundary_neighbourhood(edge, i)]
            rhs[0] = -val * robin_ref

            # More general method, slower
            # alphas = np.linalg.solve(Phi_Robin, rhs)
            # A[boundary_idx(i)] = alphas.dot(phi_vec)

            # Update_weights method
            A[boundary_idx(i)] = Robin_update_weights.dot(rhs)

        return

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


def grid_step(T, update_weights, grid_dist, c, boundary_conditions, boundary_method=set_boundary):
    """
    Step forwards using step_domain to compute values at domain nodes then
    applies boundary conditions

    Boundary conditions - a dictionary with entries {"Edge": ("Condition", value)}

    bounary_method: the function to use to solve the boundary conditions.
    """
    # Update domain nodes
    step_domain(T, update_weights)

    # Apply boundary conditions
    for edge, (condition, value) in boundary_conditions.items():
        if condition == "Robin":
            # Robin boundaries have two parameters
            boundary_method(T, condition, edge, value[0], c, grid_dist, robin_ref=value[1])
        else:
            boundary_method(T, condition, edge, value[0], c, grid_dist)

    return T
