"""
Methods for solving on a general domain
"""

import numpy as np

def get_update_weights(positions, c):
    """
    Given a vector of complex-valued node positions (relative to the centre node),
    get the vector of heat equation update weights for that neighbourhood
    """
    dist_mat_sq = np.abs(positions - positions[:,np.newaxis]) ** 2
    cr_0_sq = np.max(dist_mat_sq) * (c ** 2)

    return np.linalg.solve(np.sqrt(dist_mat_sq + cr_0_sq).T, (dist_mat_sq[0] + 2*cr_0_sq) \
            / ((dist_mat_sq[0] + cr_0_sq) ** 1.5))


def setup_neighbours_and_update_weights(domain_positions, time_step, diffusivity, c):
    """
    Given an array of node positions stored in complex form x+iy, returns:
    - a 5xn_points matrix with nth column containing the indexes of node n's
    - neighbours,
    - a 5xn_points matrix with nth column containing the update weights for
    - node n
    """
    neighbourhood_idx = np.zeros((5, domain_positions.size), dtype=int)
    update_weights = np.zeros_like(neighbourhood_idx, dtype=np.float64)

    N = 5

    for i in range(domain_positions.size):
        rel_pos = domain_positions-domain_positions[i]
        idx = np.abs(rel_pos).argsort()[:N]
        neigh_pos = rel_pos[idx]

        neighbourhood_idx[:,i] = idx
        update_weights[:,i] = get_update_weights(neigh_pos, c)

    # Remember to scale the update weights by time_step and diffusivity
    return neighbourhood_idx, update_weights * time_step * diffusivity
