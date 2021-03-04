"""
Methods for solving on a general domain
"""

import numpy as np

import normal_derivs as temp

def domain_update_weights(positions, time_step, diffusivity, c):
    """
    Given a vector of node positions in form x+iy (relative to the centre node),
    get the vector of heat equation update weights for that neighbourhood
    """
    dist_mat_sq = np.abs(positions - positions[:,np.newaxis]) ** 2
    cr_0_sq = np.max(dist_mat_sq) * (c ** 2)

    # print(positions)
    # print(dist_mat_sq)
    # input()

    return np.linalg.solve(np.sqrt(dist_mat_sq + cr_0_sq).T, (dist_mat_sq[0] + 2*cr_0_sq) \
            / ((dist_mat_sq[0] + cr_0_sq) ** 1.5)) * time_step * diffusivity


def general_setup(positions, labels, time_step, diffusivity, c, enforce_sum=False):
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
              * "D" - Dirichlet boundary node with constant value d
              * "D-f" - Dirichlet boundary node with value f(x, y, t)
              * "N" - Neumann boundary node with value n
              * "R" - Robin boundary node with value r and reference ref
    - boundary_vals: an equally sized array with boundary values for corresponding nodes.
              entries are one of
              * None - domain node
              * k - Constant Dirichlet, or Neumann value
              * Reference to a function of the form f(x, y, t) - Non-constant Dirichlet node
              * (k, ref) - Robin value and reference
    - enforce_sum: enforces the condition that update weights should sum to 1 (currently testing)
    """
    neighbourhood_idx = np.zeros((5, positions.size), dtype=int)

    # if we're storing position data in boundary columns, this needs to be complex-valued
    update_info = np.zeros_like(neighbourhood_idx, dtype=np.complex128)

    N = 5

    # dirichlet_possibilities = positions.copy()
    neumann_possibilities = positions.copy()
    robin_possibilities = positions.copy()

    # hacky way of excluding same-type boundaries...
    # except don't need to exclude dirichlet since we aren't taking derivatives
    neumann_possibilities[labels == "N"] = np.inf
    robin_possibilities[labels == "R"] = np.inf

    boundary_possibilities = positions.copy()
    # boundary_possibilities[labels == "D"] = np.inf
    # boundary_possibilities[labels == "N"] = np.inf
    # boundary_possibilities[labels == "R"] = np.inf

    poss_dict = {"D": positions, "D-f": positions, "N": neumann_possibilities, "R": robin_possibilities}

    for i in range(positions.size):
        if (node_label := labels[i]) is None:
            # Domain node
            rel_pos = positions-positions[i]
            idx = np.abs(rel_pos).argsort()[:N]
            neigh_pos = rel_pos[idx]

            neighbourhood_idx[:,i] = idx
            # print('-------------')
            # print(positions[idx])
            update_info[:,i] = domain_update_weights(neigh_pos, time_step, diffusivity, c)
            # print(update_info[:,i])
            # print(sum(update_info[:,i]))
            if enforce_sum:
                update_info[0,i] = -sum(update_info[1:,i])
            # update_info[1:,i] *= -update_info[0,i]/sum(update_info[1:,i]) # this way seems worse?
        else:
            # Boundary node
            centre_pos = positions[i]

            possible_neighbours = poss_dict[node_label]
            # possible_neighbours = positions
            # possible_neighbours = boundary_possibilities

            rel_pos = possible_neighbours - centre_pos
            idx = np.abs(rel_pos).argsort()[:N-1] # since the centre node is not contained in this array

            neighbourhood_idx[0,i] = i
            neighbourhood_idx[1:,i] = idx

            # For boundary nodes we unfortunately cannot use the update weights
            # method due to robin boundary conditions depending on T
            # so instead we use this space to store ABSOLUTE neighbourhood node positions
            update_info[0,i] = centre_pos
            update_info[1:,i] = positions[idx]


    return neighbourhood_idx, update_info


def solve_boundary_node(T, positions, labels, vals, normal_derivs, c, t):
    """
    Compute the new value at a (non-Dirichlet) boundary node
    """
    rel_pos = positions - positions[0]
    dist_mat_sq = np.abs(rel_pos - rel_pos[:,np.newaxis]) ** 2
    cr_0_sq = np.max(dist_mat_sq) * (c ** 2)

    # Collocation matrix, each row of which enforces one condition on weights alpha
    Phi = np.sqrt(dist_mat_sq + cr_0_sq)

    # phi_vec
    phi_vec = Phi[0].copy()

    # initialise rhs vector b as T
    b = T
    # print(b)

    # now override rows in Phi and entries in b that correspond to boundary nodes
    for i in range(5):
        if (label := labels[i]) is not None:
            if label == "D":
                b[i] = vals[i]
            elif label == "D-f":
                b[i] = vals[i](positions.real, positions.imag, t)
            elif label == "N":
                Phi[i] = normal_derivs[i](positions[i], positions)
                b[i] = vals[i]

            elif label == "R":
                Phi[i] *= -vals[i][0]
                Phi[i] += normal_derivs[i](positions[i], positions)
                b[i] = -vals[i][0] * vals[i][1]

            else:
                raise ValueError("Invalid boundary label")

    print(b)
    print(Phi)
    print(np.linalg.cond(Phi))
    t = np.linalg.solve(Phi, b)
    print(t.dot(temp.radial(positions[0], positions, c)))
    # input()
    return np.linalg.solve(Phi, b).dot(phi_vec)


def general_step(T_vec, update_info, neighbourhood_idx, labels, boundary_vals, deriv_lambdas, c, t):
    """
    Updates all (domain + boundary) nodes to their values at the next time step
    """
    domain_idx = np.where(labels == None)[0]
    boundary_idx = np.where(labels != None)[0]

    T_old = T_vec.copy()
    # First loop through domain nodes
    for i in domain_idx:
        # cast to real to avoid warnings (we know these entries are real-valued anyway)
        T_vec[i] += update_info[:,i].real.dot(T_old[neighbourhood_idx[:,i]])

    # now solve the boundaries
    for i in boundary_idx:
        if labels[i] == "D":
            # constant dirichlet boundary, easy
            # consider the need for this line in general since we should really only
            # need to set this once during the first step
            T_vec[i] = boundary_vals[i]
        elif labels[i] == "D-f":
            # time-dependent Dirichlet boundary
            pos = update_info[0,i]
            T_vec[i] = boundary_vals[i](pos.real, pos.imag, t)
        else:
            # more difficult boundary!
            neighbourhood_labels = labels[neighbourhood_idx[:,i]]
            neighbourhood_vals = boundary_vals[neighbourhood_idx[:,i]]
            neighbourhood_T = T_vec[neighbourhood_idx[:,i]]
            neighbourhood_normals = deriv_lambdas[neighbourhood_idx[:,i]]
            positions = update_info[:,i]

            T_vec[i] = solve_boundary_node(neighbourhood_T, positions, neighbourhood_labels, neighbourhood_vals, neighbourhood_normals, c, t)
