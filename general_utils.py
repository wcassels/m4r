"""
Methods for solving on a general domain
"""

import numpy as np

def domain_update_weights(positions, time_step, diffusivity, c):
    """
    Given a vector of node positions in form x+iy (relative to the centre node),
    get the vector of heat equation update weights for that neighbourhood
    """
    dist_mat_sq = np.abs(positions - positions[:,np.newaxis]) ** 2
    cr_0_sq = np.max(dist_mat_sq) * (c ** 2)

    return np.linalg.solve(np.sqrt(dist_mat_sq + cr_0_sq).T, (dist_mat_sq[0] + 2*cr_0_sq) \
            / ((dist_mat_sq[0] + cr_0_sq) ** 1.5)) * time_step * diffusivity


def general_setup(positions, labels, boundary_vals, time_step, diffusivity, c):
    """
    Given an array of node positions stored in complex form x+iy, returns:
    - a 5xn_points matrix with nth column containing the indexes of node n's
    - neighbours,
    - a 5xn_points matrix with nth column containing either:
        (i) The update weights for a domain node
        (ii) The relative neighbourhood node positions for a boundary node
    - a n_points length list with entries
        (i) None - domain node
        (ii) Some function that encodes the derivative in the normal direction to the boundary

    Arguments:
    - positions: array of positions stored in complex form x+iy
    - labels: an equally sized array with boundary labels for corresponding nodes.
              entries are one of
              * None - domain node
              * "D" - Dirichlet boundary node with value d
              * "N" - Neumann boundary node with value n
              * "R" - Robin boundary node with value r and reference ref
    - boundary_vals: an equally sized array with boundary values for corresponding nodes.
              entries are one of
              * None - domain node
              * (k - Dirichlet or Neumann value
              * (k, ref) - Robin value and reference
    """
    neighbourhood_idx = np.zeros((5, positions.size), dtype=int)
    update_info = np.zeros_like(neighbourhood_idx, dtype=np.float64)
    normal_funcs = [None] * positions.size

    N = 5

    dirichlet_possibilities = positions.copy()
    neumann_possibilities = positions.copy()
    robin_possibilities = positions.copy()

    dirichlet_possibilities[labels == "D"] = np.inf
    neumann_possibilities[labels == "N"] = np.inf
    robin_possibilities[labels == "R"] = np.inf

    poss_dict = {"D": dirichlet_possibilities, "N": neumann_possibilities, "R": robin_possibilities}

    for i in range(positions.size):
        if (node_label := labels[i]) is None:
            # Domain node
            rel_pos = positions-positions[i]
            idx = np.abs(rel_pos).argsort()[:N]
            neigh_pos = rel_pos[idx]

            neighbourhood_idx[:,i] = idx

            update_info[:,i] = domain_update_weights(neigh_pos, time_step, diffusivity, c)
        else:
            # Boundary node
            centre_pos = positions[i]

            possible_neighbours = poss_dict[node_label]

            rel_pos = possible_neighbours - centre_pos
            idx = np.abs(rel_pos).argsort()[:N-1] # since the centre node is not contained in this array

            neighbourhood_idx[0,i] = i
            neighbourhood_idx[1:,i] = idx

            # For boundary nodes we unfortunately cannot use the update weights
            # method due to robin boundary conditions depending on T
            # so instead we use this space to store ABSOLUTE neighbourhood node positions
            update_info[0,i] = 0
            update_info[1:,i] = positions[idx]


    return neighbourhood_idx, update_info


def solve_boundary_node(T, positions, labels, vals, time_step, diffusivity, c):
    """
    Compute the new value at a (non-Dirichlet) boundary node
    """
    rel_pos = positions - positions[0]
    dist_mat_sq = np.abs(rel_pos - rel_pos[:,np.newaxis]) ** 2
    cr_0_sq = np.max(dist_mat_sq) * (c ** 2)

    # Collocation matrix, each row of which enforces one condition on weights alpha
    Phi = np.sqrt(dist_mat_sq + cr_0_sq)

    # initialise rhs vector b as T
    b = T

    # now override rows in Phi and entries in b that correspond to boundary nodes
    # current problem: need to encode direction of normal - gonna use a lambda function
    for i in range(5):
        if (label := labels[i]) is not None:
            if label == "D":
                b[i] = vals[i]
            elif label == "N":

                b[i] = vals[i]

            elif label == "R":
                b[i] = - vals[i][0] * vals[i][1]

            else:
                raise ValueError("Invalid boundary label")

    return


def general_step(T_vec, update_info, neighbourhood_idx, labels, boundary_vals, time_step, diffusivity, c):
    """
    Updates all (domain + boundary) nodes to their values at the next time step
    """
    for i in range(T_vec.size):
        if (node_label := labels[i]) == None:
            # domain node! easy
            T_vec[i] += update_info[:,i].dot(T_vec[neighbourhood_idx[:,i]])
        else:
            if node_label == "D":
                # dirichlet boundary, easy
                # consider the need for this line in general since we should really only
                # need to set this once during the first step
                T_vec[i] = boundary_vals[i]
            else:
                # more difficult boundary!
                neighbourhood_labels = labels[neighbourhood_idx[:,i]]
                neighbourhood_vals = boundary_vals[neighbourhood_idx[:,i]]
                neighbourhood_T = T_vec[neighbourhood_idx[:,i]]
                positions = update_info[:,i]

                T_vec[i] = solve_boundary_node(neighbourhood_T, positions, neighbourhood_labels, neighbourhood_vals, time_step, diffusivity, c)
