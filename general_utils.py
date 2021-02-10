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


def boundary_update_weights(neigh_pos, neigh_labels, neigh_vals, time_step, diffusivity, c):
    """
    Solves the general collocation problem for some node on a boundary, and N-1 neighbours
    which may be domain or boundary nodes (if they are boundary nodes, they must be of a different
    type to the centre node)
    """


def setup_neighbours_and_update_weights(positions, labels, boundary_vals, time_step, diffusivity, c):
    """
    Given an array of node positions stored in complex form x+iy, returns:
    - a 5xn_points matrix with nth column containing the indexes of node n's
    - neighbours,
    - a 5xn_points matrix with nth column containing the update weights for
    - node n

    Arguments:
    - positions: array of positions stored in complex form x+iy
    - labels: an equally sized array with boundary labels for corresponding nodes.
              entries are one of
              * None - domain node
              * "D" - Dirichlet boundary node with value d
              * "N" - Neumann boundary node with value n
              * "R" - Robin boundary node with value r and reference ref
    - boundary_vals: an equally sizzed array with boundary values for corresponding nodes.
              entries are one of
              * None - domain node
              * k - Dirichlet or Neumann value
              * (k, ref) - Robin value and reference
    """
    neighbourhood_idx = np.zeros((5, positions.size), dtype=int)
    update_weights = np.zeros_like(neighbourhood_idx, dtype=np.float64)

    N = 5

    # for now, restricting possible neighbourhood nodes for boundaries to
    # domain nodes, or nodes of other boundary types
    # in an attempt to avoid singular boundary collocation matrices
    # possiblities = {"D": positions[labels != "D"],
    #                 "N": positions[labels != "N"],
    #                 "R": positions[labels != "R"]}

    dirichlet_possibilities = positions.copy()
    neumann_possibilities = positions.copy()
    robin_possibilities = positions.copy()

    dirichlet_possibilities[labels == "D"] = np.inf
    neumann_possibilities[labels == "N"] = np.inf
    robin_possibilities[labels == "R"] = np.inf

    poss_dict = {"D": dirichlet_possibilities, "N": neumann_possibilities, "R": robin_possibilities}

    # labels_dict = {"D": labels[labels != "D"],
    #                "N": labels[labels != "N"],
    #                "R": labels[labels != "R"]}
    #
    # boundary_vals_dict = {"D": boundary_vals[labels != "D"],
    #                       "N": boundary_vals[labels != "N"],
    #                       "R": boundary_vals[labels != "R"]}

    for i in range(positions.size):
        if (node_label := labels[i]) is None:
            # Domain node
            rel_pos = positions-positions[i]
            idx = np.abs(rel_pos).argsort()[:N]
            neigh_pos = rel_pos[idx]

            neighbourhood_idx[:,i] = idx

            update_weights[:,i] = domain_update_weights(neigh_pos, time_step, diffusivity, c)
        else:
            # Boundary node
            centre_pos = positions[i]

            possible_neighbours = poss_dict[node_label]

            rel_pos = possible_neighbours - centre_pos
            idx = np.abs(rel_pos).argsort()[:N-1] # since the centre node is not contained in this array
            neigh_pos = rel_pos[idx]

            neighbourhood_labs = labels[idx]
            neighbourhood_vals = boundary_vals[idx]

            neighbourhood_idx[0,i] = i
            neighbourhood_idx[1:,i] = idx

            update_weights[:,i] = boundary_update_weights(np.insert(neigh_pos, 0, 0), \
                                  np.insert(labels[idx], 0, node_label), np.insert(boundary_vals[idx], \
                                  0, boundary_vals[i]), time_step, diffusivity, c)

    return neighbourhood_idx, update_weights
