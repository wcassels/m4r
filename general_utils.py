"""
Methods for solving on a general domain
"""

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt


def setup(positions, labels, boundary_vals, normal_derivs, time_step, diffusivity, c, N, dtype=np.float64, reg=0, method="Sarler", c_boundary=None, N_boundary=None, return_conds=False):
    """
    Setup for the final, fully general, weight based solution procedure.
    Returns idx, weights (including the +1 on central nodes) or if N_boundary is
    specified, returns idx, weights, boundary_flags (alternative setup only)
    plus the condition number distribution if requested

    Handles all boundaries
    """
    if c_boundary is None:
        c_boundary = c

    if method == "Sarler" or method == "Sarler implicit":
        return sarler_setup(positions, labels, boundary_vals, normal_derivs, time_step, diffusivity, c, c_boundary, N, dtype, reg)
    elif method == "Alternative" or method == "Alternative implicit":
        return alternative_setup(positions, labels, boundary_vals, normal_derivs, time_step, diffusivity, c, c_boundary, N, dtype, N_boundary)
    else:
        raise ValueError("Invalid solution method argument")


def sarler_setup(positions, labels, boundary_vals, normal_derivs, time_step, diffusivity, c, c_boundary, N, dtype=np.float64, reg=0):
    """
    Weight based implementation of original scheme presented by Sarler.
    Returns idx, weights (including the +1 on central nodes)
    """
    neighbourhood_idx = np.zeros((N, positions.size), dtype=int)

    # if we're storing position data in boundary columns, this needs to be complex-valued
    if dtype == np.float64:
        weights = np.zeros_like(neighbourhood_idx, dtype=np.float64)
    elif dtype == np.float128:
        weights = np.zeros_like(neighbourhood_idx, dtype=np.float128)
    else:
        raise ValueError("Unsupported dtype")

    neumann_possibilities = positions.copy()
    robin_possibilities = positions.copy()

    # hacky way of excluding same-type boundaries...
    # except don't need to exclude dirichlet since we aren't taking derivatives
    neumann_possibilities[labels == "N"] = np.inf
    neumann_possibilities[labels == "R"] = np.inf
    robin_possibilities[labels == "R"] = np.inf
    robin_possibilities[labels == "N"] = np.inf

    boundary_possibilities = positions.copy()

    poss_dict = {"D": positions, "D-f": positions, "N": neumann_possibilities, "R": robin_possibilities}

    for i in range(positions.size):
        print(f"{i}/{positions.size}")
        if (node_label := labels[i]) is None:
            # Domain node
            rel_pos = positions-positions[i]
            idx = np.abs(rel_pos).argsort()[:N]
            neigh_pos = rel_pos[idx]
            neighbourhood_idx[:,i] = idx

            weights[:,i] = sarler_domain_weights(neigh_pos, time_step, diffusivity, c, dtype=dtype)
            weights[0,i] += 1
        elif node_label == "D":
            # Dirichlet node
            weights[0,i] = 1
        else:
            # Neumann or Robin node
            centre_pos = positions[i]
            possible_neighbours = poss_dict[node_label]
            rel_pos = possible_neighbours - centre_pos

            # generalise this regularisation
            if reg:
                theta = np.arccos(centre_pos.real / np.abs(centre_pos))
                if centre_pos.imag < 0:
                    theta *= -1

                thetas = np.arccos(possible_neighbours.real / np.abs(possible_neighbours))
                thetas[possible_neighbours.imag < 0] *= -1
                idx = (np.abs(rel_pos) + reg * np.abs(theta-thetas)).argsort()[:N-1] # since the centre node is not contained in this array
            else:
                idx = np.abs(rel_pos).argsort()[:N-1]

            neighbourhood_idx[0,i] = i
            neighbourhood_idx[1:,i] = idx

            w_idx = neighbourhood_idx[:,i]

            # weights[:,i] = sarler_boundary_weights(np.insert(positions[idx], 0, positions[i]), np.insert(labels[idx], 0, node_label), np.insert(normal_derivs[idx], 0, normal_derivs[i]), c)
            weights[:,i] = sarler_boundary_weights(positions[w_idx], labels[w_idx], boundary_vals[w_idx], normal_derivs[w_idx], c_boundary, dtype=dtype)

    return neighbourhood_idx, weights


def sarler_domain_weights(positions, time_step, diffusivity, c, dtype=np.float64):
    """
    Given a vector of node positions in form x+iy (relative to the centre node),
    get the vector of heat equation update weights for that neighbourhood
    """
    N = positions.size

    dist_mat_sq = np.zeros((N, N), dtype=dtype)
    rhs = np.zeros(N, dtype=dtype)

    dist_mat_sq[:,:] = np.abs(positions - positions[:,np.newaxis]) ** 2
    cr_0_sq = np.max(dist_mat_sq) * (c ** 2)

    PhiT = np.sqrt(dist_mat_sq + cr_0_sq).T
    rhs[:] = (dist_mat_sq[0] + 2*cr_0_sq) / ((dist_mat_sq[0] + cr_0_sq) ** 1.5)

    return np.linalg.solve(PhiT, rhs) * time_step * diffusivity


def sarler_boundary_weights(positions, labels, boundary_vals, normal_derivs, c, dtype=np.float64):
    """
    Given a vector of node positions in form x+iy (relative to the centre node),
    get the vector of boundary interpolation weights for that neighbourhood
    """
    N = positions.size

    dist_mat_sq = np.zeros((N, N), dtype=dtype)
    rel_pos = positions - positions[0]
    dist_mat_sq[:,:] = np.abs(rel_pos - rel_pos[:,np.newaxis]) ** 2
    cr_0_sq = np.max(dist_mat_sq) * (c ** 2)

    # Collocation matrix
    Phi = np.sqrt(dist_mat_sq + cr_0_sq)

    # phi_vec
    phi_vec = Phi[0].copy()

    # now override rows in Phi that correspond to boundary nodes
    for i in range(N):
        if (label := labels[i]) is not None:
            if label == "D":
                pass
            elif label == "N":
                Phi[i] = normal_derivs[i](positions[i], positions)
            elif label == "R":
                Phi[i] *= -boundary_vals[i][0]
                Phi[i] += normal_derivs[i](positions[i], positions)
            else:
                raise ValueError("Invalid boundary label")

    return np.linalg.solve(Phi.T, phi_vec)



def alternative_setup(positions, labels, boundary_vals, normal_derivs, time_step, diffusivity, c, c_boundary, N, dtype, N_boundary=None):
    """
    Solution procedure with my boundary modifications
    """
    if N_boundary is None:
        m = N
    else:
        m = max(N, N_boundary)

    neighbourhood_idx = np.zeros((m, positions.size), dtype=int)

    # if the boundary domains of influences  are a different size to regular ones,
    # have to store flags for each node for whether their domains contain
    # boundary nodes
    if N_boundary:
        boundary_flags = np.zeros(positions.size, dtype=bool)

    # if we're storing position data in boundary columns, this needs to be complex-valued
    if dtype == np.float64:
        weights = np.zeros_like(neighbourhood_idx, dtype=np.float64)
    elif dtype == np.float128:
        weights = np.zeros_like(neighbourhood_idx, dtype=np.float128)
    else:
        raise ValueError("Unsupported dtype")


    # dirichlet_possibilities = positions.copy()
    neumann_possibilities = positions.copy()
    robin_possibilities = positions.copy()

    # hacky way of excluding same-type boundaries...
    # except don't need to exclude dirichlet since we aren't taking derivatives
    neumann_possibilities[labels == "N"] = np.inf
    robin_possibilities[labels == "R"] = np.inf

    boundary_possibilities = positions.copy()
    boundary_possibilities[labels == "D"] = np.inf
    boundary_possibilities[labels == "N"] = np.inf
    boundary_possibilities[labels == "R"] = np.inf

    # poss_dict = {"D": positions, "D-f": positions, "N": neumann_possibilities, "R": robin_possibilities}
    poss_dict = {"D": boundary_possibilities, "D-f": boundary_possibilities, "N": boundary_possibilities, "R": boundary_possibilities}

    # awful code but convenient way to get average condition numbers
    # global avg_cond
    # avg_cond = 0
    # global num_domain
    # num_domain = 0

    for i in range(positions.size):
        # Boundary node
        node_label = labels[i]
        if node_label == "D":
            weights[0,i] = 1
        else:
            centre_pos = positions[i]

            # need to avoid singular collocation matrix for neighbourhoods
            # bordering the boundary. so can only have one Neumann node in these
            # neighbourhoods
            if node_label is None:
                rel_pos = positions - centre_pos
                global_idx = np.abs(rel_pos).argsort()
                # neighbourhood_idx[0,i] = i
                # fill up the neighbours while ensuring there's <= 1 Neumann neighbour
                N_flag = False
                num_neighbours = 0
                for index in global_idx:
                    if labels[index] == "N":
                        if N_flag is False:
                            if not N_boundary:
                                neighbourhood_idx[num_neighbours,i] = index
                                N_flag = True
                                num_neighbours += 1
                            elif N_boundary and num_neighbours <= N_boundary:
                                print("Added Neumann", index)
                                neighbourhood_idx[num_neighbours,i] = index
                                boundary_flags[i] = True
                                N_flag = True
                                num_neighbours += 1
                            else:
                                # boundary domain size has been reached but new neighbour node is a boundary one!
                                # reject all excess neighbour nodes past the N_boundary threshold and continue
                                # as a purely domain node
                                if N_boundary > N:
                                    num_neighbours = N_boundary
                                    neighbourhood_idx[N:,i] = 0
                                    boundary_flags[i] = False
                                    break
                                else:
                                    continue

                        else:
                            continue
                    else:
                        neighbourhood_idx[num_neighbours,i] = index
                        num_neighbours += 1

                    if N_boundary:
                        if num_neighbours >= N and N_flag is False:
                            break
                        elif num_neighbours >= N_boundary and N_flag is True:
                            break
                    else:
                        if num_neighbours >= N:
                            break

            else:
                if N_boundary:
                    boundary_flags[i] = True
                    m = N_boundary
                else:
                    m = N
                possible_neighbours = poss_dict[node_label]
                rel_pos = possible_neighbours - centre_pos

                idx = (np.abs(rel_pos)).argsort()[:m-1] # since the centre node is not contained in this array

                neighbourhood_idx[0,i] = i
                neighbourhood_idx[1:m,i] = idx

            if node_label is None:
                shape_param = c
            else:
                shape_param = c_boundary

            if N_boundary and boundary_flags[i]:
                m = N_boundary
            else:
                m = N

            w_idx = neighbourhood_idx[:m,i]
            weights[:m,i] = alternative_weights(positions[w_idx], labels[w_idx], boundary_vals[w_idx], normal_derivs[w_idx], time_step, diffusivity, shape_param, dtype=dtype)

    # save condition numbers
    # input(f"N_\omega={N}, avg cond is {avg_cond/num_domain}")
    # np.save(f"data/disk_N_conds/{N}", avg_cond/num_domain)

    if N_boundary:
        return neighbourhood_idx, weights, boundary_flags
    else:
        return neighbourhood_idx, weights


def alternative_weights(positions, labels, boundary_vals, normal_derivs, time_step, diffusivity, c, dtype=np.float64):
    """
    Alternative scheme, where boundary conditions are enforced in all near-boundary
    neighbourhoods, not just the boundary ones. Essentially treats all neighbourhoods
    in the same way as boundary ones are treated in the original scheme
    """
    N = positions.size

    dist_mat_sq = np.zeros((N, N), dtype=dtype)
    rel_pos = positions - positions[0]
    dist_mat_sq[:,:] = np.abs(rel_pos - rel_pos[:,np.newaxis]) ** 2
    cr_0_sq = np.max(dist_mat_sq) * (c ** 2)

    # Collocation matrix, each row of which enforces one condition on weights alpha
    Phi = np.sqrt(dist_mat_sq + cr_0_sq)

    phi_vec = Phi[0].copy()

    # override rows in Phi that correspond to boundary nodes
    for i in range(N):
        if (label := labels[i]) is not None:
            if label == "D":
                pass
            elif label == "N":
                Phi[i] = normal_derivs[i](positions[i], positions)
            elif label == "R":
                Phi[i] *= -boundary_vals[i][0]
                Phi[i] += normal_derivs[i](positions[i], positions)
            else:
                raise ValueError("Invalid boundary label")

    if labels[0] is None:
        # domain node, evaluate the Laplacian
        rhs = np.zeros(N, dtype=dtype)
        rhs[:] = (dist_mat_sq[0] + 2*cr_0_sq) / ((dist_mat_sq[0] + cr_0_sq) ** 1.5)

        # These weights have the +1 for the central node incorporated, for later convenience
        w = np.linalg.solve(Phi.T, rhs) * diffusivity * time_step
        w[0] += 1
        return w
    else:
        # boundary node, interpolate
        return np.linalg.solve(Phi.T, phi_vec) # then just needs to be dotted with T (modified with boundary condition values)


def step(T, weights, neighbourhood_idx, labels, rhs_vals, N, N_boundary=None, boundary_flags=None, method="Sarler"):
    """
    Step for the final, fully general, weight based solution procedure

    Takes filtered rhs_vals argument
    """
    if method == "Sarler":
        return sarler_step(T, weights, neighbourhood_idx, labels, rhs_vals)
        pass
    elif method == "Alternative":
        if N_boundary is None:
            return alternative_step(T, weights, neighbourhood_idx, labels, rhs_vals, N)
        else:
            return alternative_step_flex(T, weights, neighbourhood_idx, labels, rhs_vals, N, N_boundary, boundary_flags)
        pass
    elif method == "Sarler implicit":
        return sarler_implicit_step(T, weights, neighbourhood_idx, labels, rhs_vals)
        pass
    elif method == "Alternative implicit":
        return alternative_implicit_step(T, weights, neighbourhood_idx, labels, rhs_vals, jumps=1)
        pass


def sarler_step(T, weights, neighbourhood_idx, labels, rhs_vals):
    """
    Weight based Sarler step

    Takes filtered rhs_vals argument
    """
    domain_idx = np.where(labels == None)[0]
    boundary_idx = np.where(labels != None)[0]
    T_old = T.copy()

    # update domain nodes
    for i in domain_idx:
        T[i] = T_old[neighbourhood_idx[:,i]].dot(weights[:,i])

    T_mod = T.copy()
    T_mod[labels != None] = rhs_vals[labels != None]

    # update boundary nodes
    for i in boundary_idx:
        if labels[i] == "D":
            # manually fix Dirichlet nodes
            T[i] = rhs_vals[i]
        else:
            T[i] = weights[:,i].dot(T_mod[neighbourhood_idx[:,i]])

    return T


def alternative_step(T, weights, neighbourhood_idx, labels, rhs_vals, N):
    """
    Weight based alternative step

    Takes filtered rhs_vals argument
    """
    domain_idx = np.where(labels == None)[0]
    boundary_idx = np.where(labels != None)[0]

    T_mod = T.copy()
    T_mod[labels != None] = rhs_vals[labels != None]

    # step
    for i in domain_idx:
        T[i] = weights[:N,i].dot(T_mod[neighbourhood_idx[:N,i]])


    T_mod = T.copy()
    T_mod[labels != None] = rhs_vals[labels != None]
    for i in boundary_idx:

        if labels[i] == "D":
            T[i] = rhs_vals[i]
        else:
            T[i] = weights[:N,i].dot(T_mod[neighbourhood_idx[:N,i]])

    return T

def alternative_step_flex(T, weights, neighbourhood_idx, labels, rhs_vals, N, N_boundary, boundary_flags):
    """
    Weight based alternative step, with the modification that boundary domains of influences can
    be of a different size to domain ones

    Takes filtered rhs_vals argument
    """
    domain_idx = np.where(labels == None)[0]
    boundary_idx = np.where(labels != None)[0]

    T_mod = T.copy()
    T_mod[labels != None] = rhs_vals[labels != None]

    # step
    for i in domain_idx:
        if boundary_flags[i]:
            m = N_boundary
        else:
            m = N
        T[i] = weights[:m,i].dot(T_mod[neighbourhood_idx[:m,i]])

    # interpolate boundaries with new domain values
    T_mod = T.copy()
    T_mod[labels != None] = rhs_vals[labels != None]
    for i in boundary_idx:
        if labels[i] == "D":
            T[i] = rhs_vals[i]
        else:
            T[i] = weights[:N_boundary,i].dot(T_mod[neighbourhood_idx[:N_boundary,i]])

    return T

def sarler_implicit_step(T, weights, neighbourhood_idx, labels, rhs_vals):
    """
    Implicit Sarler step, using GMRES
    """
    N = T.size
    M = scipy.sparse.lil_matrix((N, N))

    # set rhs vals values
    T[labels != None] = rhs_vals[labels != None]

    for i in range(N):
        if labels[i] == None:
            M[i,i] = 2
            M[i,neighbourhood_idx[:,i]] -= weights[:,i]
        elif labels[i] == "D":
            # these do not change
            M[i,i] = 1
        elif labels[i] == "N" or labels[i] == "R":
            M[i,i] = 1
            for j, idx in enumerate(neighbourhood_idx[1:,i], 1):
                if labels[idx] == None:
                    M[i,idx] = -weights[j,i]
                else:
                    T[i] += weights[j,i] * rhs_vals[idx]

        else:
            raise ValueError("Invalid node label")

    # T, succ = gmres(scipy.sparse.csr_matrix(M), T, tol=1e-8)
    T = scipy.sparse.linalg.spsolve(scipy.sparse.csr_matrix(M), T)

    return T


def alternative_implicit_step(T, weights, neighbourhood_idx, labels, rhs_vals, jumps=1):
    """
    Implicit alternative step, using GMRES
    Not currently working
    """
    N = T.size
    M = scipy.sparse.lil_matrix((N, N))

    domain_idx = np.where(labels == None)[0]
    boundary_idx = np.where(labels != None)[0]

    for i in domain_idx:
        M[i,i] = 2
        for j, idx in enumerate(neighbourhood_idx[:,i]):
            if labels[idx] is None:
                M[i,idx] -= weights[j,i]
            else:
                T[i] += weights[j,i] * rhs_vals[idx]

    M = M[domain_idx]
    M = M[:,domain_idx]

    M = scipy.sparse.csr_matrix(M)
    print(M)
    res, succ = gmres(M, T[labels == None], tol=1e-18)
    T[labels == None] = res

    for i in boundary_idx:
        if labels[i] == "D":
            T[i] = rhs_vals[i]
        else:
            T[i] = 0
            # print(enumerate(neighbourhood_idx[:,]))
            for j, idx in enumerate(neighbourhood_idx[:,i]):
                if labels[idx] is None:
                    T[i] += T[idx] * weights[j,i]
                else:
                    T[i] += rhs_vals[idx] * weights[j,i]

        if T[i] < -1:
            print(i, labels[i])
            for j, idx in enumerate(neighbourhood_idx[::,i]):
                print(labels[idx])
                print(T[idx], weights[j,i])
                print(rhs_vals[idx])

    return T

def filter_boundary_vals(boundary_vals, labels):
    """
    Replaces entries [x, y] with -xy (Robin nodes)
    """
    rhs_vals = np.zeros_like(boundary_vals, dtype=np.float64)

    for i in range(boundary_vals.size):
        if labels[i] == "D" or labels[i] ==  "N":
            rhs_vals[i] = boundary_vals[i]
        elif labels[i] == "R":
            rhs_vals[i] = -boundary_vals[i][0] * boundary_vals[i][1]

    return rhs_vals
