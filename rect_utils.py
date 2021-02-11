"""
Methods for solving on a uniform rectangular grid
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
    """
    Return a lambda function that, given an integer index, points to the correct
    2d array index
    """
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


def get_boundary_Phi(c, grid_dist, type, robin_val=None):
    """
    Return collocation matrix Phi for a rectangular boundary. Uses a
    neighbourhood of 5 linear nodes leading away from the boundary - have to be
    careful with other node distributions since they can easily lead to having
    singular matrices

    Handles N = 5 only

    Arguments:
    - c: shape param
    - grid_dist: grid_dist
    - type: either "Neumann" or "Robin"
    - robin_val: condition value (for Robin boundaries only)
    """
    # Max inter-neighbourhood distance is 4 for this configuration
    cr_0_sq = (4*c) ** 2 # exclude the grid_dist term here since it cancels in Phi_Neumann[0]
    pos = np.arange(5) * grid_dist
    Phi = np.sqrt((pos-pos[:,np.newaxis]) ** 2 + cr_0_sq * (grid_dist ** 2))

    # Overwrite first row with boundary condition
    if type == "Neumann":
        Phi[0] = np.arange(5) / np.sqrt(np.arange(5)**2 + cr_0_sq)
    elif type == "Robin":
        Phi[0] *= -robin_val
        Phi[0] += np.arange(5) / np.sqrt(np.arange(5)**2 + cr_0_sq)
    else:
        raise ValueError("Invalid boundary type")

    return Phi

"""
Functions for solving the diffusion equation in the uniform grid node configuration
"""

import numpy as np

import rect_utils


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
            Phi_boundary = rect_utils.get_boundary_Phi(c, grid_dist, "Neumann")
            rhs0 = val
        elif condition == "Robin":
            Phi_boundary = rect_utils.get_boundary_Phi(c, grid_dist, "Robin", val)
            rhs0 = -val * robin_ref
        else:
            raise ValueError("Invalid boundary condition selected")

        phi_vec = np.sqrt(np.arange(5)**2 + (c**2)*16) * grid_dist

        update_weights = np.linalg.solve(Phi_boundary.T, phi_vec)

        # Quickly perform the dot products by forming an "influence matrix" for each boundary
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
        Phi_Neumann = rect_utils.get_boundary_Phi(c, grid_dist, "Neumann")

        phi_vec = np.sqrt(np.arange(5)**2 + (c**2)*16) * grid_dist

        boundary_idx = rect_utils.get_boundary_positions(edge)

        Neumann_update_weights = np.linalg.solve(Phi_Neumann.T, phi_vec)

        # Exclude corners for now
        for i in range(1, mn-1):
            neighbourhood_x, neighbourhood_y = rect_utils.get_boundary_neighbourhood(edge, i)
            rhs = A[neighbourhood_x, neighbourhood_y]
            rhs[0] = val

            # Update_weights method
            A[boundary_idx(i)] = Neumann_update_weights.dot(rhs)

        return

    elif condition == "Robin":
        Phi_Robin = rect_utils.get_boundary_Phi(c, grid_dist, "Robin", val)

        phi_vec = np.sqrt(np.arange(5)**2 + (c**2)*16) * grid_dist

        boundary_idx = rect_utils.get_boundary_positions(edge)

        Robin_update_weights = np.linalg.solve(Phi_Robin.T, phi_vec)

        for i in range(1, mn-1):
            rhs = A[rect_utils.get_boundary_neighbourhood(edge, i)]
            rhs[0] = -val * robin_ref

            # Update_weights method
            A[boundary_idx(i)] = Robin_update_weights.dot(rhs)

        return

    else:
        raise ValueError("Invalid boundary condition selected")


def domain_step(T, update_weights):
    """
    Very fast domain step for uniform grid implementation

    Returns smaller grid since boundary values not computed
    """
    # Doing all the dot products is equivalent to just summing shifted layers
    # of the grid
    T[1:-1,1:-1] += update_weights[0] * T[1:-1,1:-1] + update_weights[1] * (T[:-2,1:-1]
                    + T[2:,1:-1] + T[1:-1,:-2] + T[1:-1,2:])

    return T[1:-1,1:-1]


def step(T, update_weights, grid_dist, c, boundary_conditions, boundary_method=set_boundary):
    """
    Step forwards using step_domain to compute values at domain nodes then
    applies boundary conditions

    Boundary conditions - a dictionary with entries {"Edge": ("Condition", value)}

    bounary_method: the function to use to solve the boundary conditions.
    """
    # Update domain nodes
    domain_step(T, update_weights)

    # Apply boundary conditions
    for edge, (condition, value) in boundary_conditions.items():
        if condition == "Robin":
            # Robin boundaries have two parameters
            boundary_method(T, condition, edge, value[0], c, grid_dist, robin_ref=value[1])
        else:
            boundary_method(T, condition, edge, value[0], c, grid_dist)

    return T
