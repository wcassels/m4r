"""
Methods for solving on a general (not necessarily uniform or rectangular) domain
"""

import numpy as np

def sum_2nd_derivs(dists, c, r0):
    """
    Returns the vector of the sum of the second derivates, evaluated using the
    distances provided
    """
    cr_0_sq = (c*r0)**2

    # This is how it simplifies
    return (d**2 + 2*cr_0_sq) / ((d**2 + cr_0_sq)**1.5)


def Phi(positions, c, r0):
    """
    Returns the collocation matrix Phi given neighbour positions relative to the central node
    """

    return np.sqrt(np.abs(positions - positions[:,np.newaxis]) ** 2 + (c*r0)**2)


def neighbours_and_update_weights(domain_positions):
    """
    Not sure if will work but:
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
        rel_positions = domain_positions-domain_positions[i]
        idx = np.abs(rel_positions).argsort()[:N]
        neigh_pos = rel_positions[idx]

        r0 = np.max(np.abs(rel_positions))

        neighbourhood_idx[:,i] = idx
        update_weights[:,i] = np.linalg.solve(Phi(neigh_pos, c, r0), sum_2nd_derivs(neighbourhood_distances, c, r0))


    return neighbourhood_idx, update_weights
