import numpy as np
import os
import matplotlib.pyplot as plt

def fill_domain(boundary_nodes, domain_conditions, num_domain_nodes, x_nodes=100, y_nodes=100):
    """
    Assuming all boundaries have been generated, fills the domain with num_domain_nodes
    optimally placed nodes.

    Arguments:
    - boundary_nodes: array of node positions (complex format)
    - domain_conditions: list of lambdas that fully determine the shape of the domain to fill
    - num_domain_nodes: number of nodes to add
    """
    # Check if nodes have been previously generated
    h = hash(tuple(boundary_nodes)) + x_nodes + y_nodes# + sum(hash(cond) for cond in domain_conditions)
    if not os.path.exists(f"node_positions/{h}"):
        os.makedirs(f"node_positions/{h}")
    else:
        try:
            nodes = np.load(f"node_positions/{h}/{num_domain_nodes}nodes.npy")
            cut_outs = np.load(f"node_positions/{h}/{num_domain_nodes}cut_outs.npy")
            print("Node positions loaded")
            return nodes, cut_outs
        except FileNotFoundError:
            pass

    print("Generating nodes")

    x_min, x_max, y_min, y_max = np.min(boundary_nodes.real), np.max(boundary_nodes.real), np.min(boundary_nodes.imag), np.max(boundary_nodes.imag)
    x_potentials = np.linspace(x_min, x_max, x_nodes+2)[1:-1]
    y_potentials = np.linspace(y_min, y_max, y_nodes+2)[1:-1]
    x, y = np.meshgrid(x_potentials, y_potentials)
    potentials = x.ravel() + y.ravel() * 1j

    cut_outs = np.array([], dtype=np.complex128)

    for condition in domain_conditions:
        cut_outs = np.concatenate((cut_outs, potentials[np.logical_not(condition(potentials))]))
        potentials = potentials[condition(potentials)]
        print(potentials.size)

    plt.scatter(potentials.real, potentials.imag, s=3)
    plt.show()
    nodes = np.array([], dtype=np.complex128)

    for i in range(num_domain_nodes):
        print(i)
        ds = np.zeros_like(potentials, dtype=np.float64)

        # vectorize this
        max_dist = -1
        k = 0

        for j in range(potentials.size):
            # ds[j] = np.min(np.abs(np.concatenate((nodes, boundary_nodes)) - potentials[j]))
            dist = np.min(np.abs(np.concatenate((nodes, boundary_nodes)) - potentials[j]))
            if dist > max_dist:
                max_dist = dist
                k = j

        # k = np.argmax(ds)
        nodes = np.append(nodes, potentials[k])
        cartesians = np.delete(potentials, k)

    np.save(f"node_positions/{h}/{num_domain_nodes}nodes.npy", nodes)
    np.save(f"node_positions/{h}/{num_domain_nodes}cut_outs.npy", cut_outs)

    return nodes, cut_outs


def make_rectangle(x_min, x_max, y_min, y_max, num_ns_nodes, num_ew_nodes, condition_dict, value_dict, deriv_dict, inner_rect_gap=0):
    """
    Returns the node positions for the given rectangle, plus the corresponding label, boundary_vals and deriv_lambdas arrays required

    Arguments:
    - x_min, x_max, y_min, y_max: Self explanatory
    - num_ns_nodes: Number of nodes on north/south edges
    - num_ew_nodes: Number of nodes on east/west edges
    - condition_dict: Dictionary with entries eg. {"North": "N"} for a Neumann boundary on the North edge
    - value_dict: Dictionary with entries eg. {"North": 0}
    - deriv_dict: Dictionary with entries eg. {"North": lambda ...} encoding outward normal
    - inner_rect_gap: If non-zero, adds a set of nodes just inside the boundary for better Neumann performance
    """
    nodes = np.array([], dtype=np.complex128)
    boundary_vals = np.array([], dtype=object)
    labels = np.array([], dtype="<U1")
    deriv_lambdas = np.array([], dtype=object)
    inner_nodes = np.array([])

    # place nodes in a clockwise fashion to avoid double placing

    # top of rectangle
    edge_nodes = np.linspace(x_min, x_max, num_ns_nodes+1)[:-1] + y_max * 1j
    nodes = np.concatenate((nodes, edge_nodes))
    if inner_rect_gap:
        inner_nodes = np.concatenate((inner_nodes, edge_nodes[1:]-inner_rect_gap*1j))

    # right edge
    edge_nodes = x_max + np.linspace(y_max, y_min, num_ew_nodes+1)[:-1] * 1j
    nodes = np.concatenate((nodes, edge_nodes))
    if inner_rect_gap:
        inner_nodes = np.concatenate((inner_nodes, edge_nodes[1:]-inner_rect_gap))


    # bottom of rectangle
    edge_nodes = np.linspace(x_max, x_min, num_ns_nodes+1)[:-1] + y_min * 1j
    nodes = np.concatenate((nodes, edge_nodes))
    if inner_rect_gap:
        inner_nodes = np.concatenate((inner_nodes, edge_nodes[1:]+inner_rect_gap*1j))

    # left edge
    edge_nodes = x_min + np.linspace(y_min, y_max, num_ew_nodes+1)[:-1] * 1j
    nodes = np.concatenate((nodes, edge_nodes))
    if inner_rect_gap:
        inner_nodes = np.concatenate((inner_nodes, edge_nodes[1:]+inner_rect_gap))


    def fill_array(dct):
        """
        Helper function to fill all these arrays
        """
        return np.concatenate((np.full(num_ns_nodes, dct["North"]),
                                 np.full(num_ew_nodes, dct["East"]),
                                 np.full(num_ns_nodes, dct["South"]),
                                 np.full(num_ew_nodes, dct["West"])))

    labels = np.concatenate((fill_array(condition_dict), np.full(inner_nodes.size, None)))
    boundary_vals = np.concatenate((fill_array(value_dict), np.full(inner_nodes.size, None)))
    deriv_lambdas = np.concatenate((fill_array(deriv_dict), np.full(inner_nodes.size, None)))

    return np.concatenate((nodes, inner_nodes)), labels, boundary_vals, deriv_lambdas


def make_circle(centre, radius, num_nodes, condition, value, deriv, inner_gap=0):
    """
    Returns node positions for the given circle, plus the corresponding label, boundary_vals and deriv_lambdas arrays required

    Arguments self-explanatory / similar to make_rectangle
    """
    thetas = np.linspace(0, 2*np.pi, num_nodes+1)[:-1]
    nodes = centre + radius * (np.cos(thetas) + 1j * np.sin(thetas))

    labels = np.full(num_nodes, condition)
    boundary_vals = np.full(num_nodes, value)
    deriv_lambdas = np.full(num_nodes, deriv)

    if inner_gap:
        inner_nodes = centre + (radius - inner_gap) * (np.cos(thetas) + 1j * np.sin(thetas))
        nodes = np.concatenate((nodes, inner_nodes))
        labels = np.concatenate((labels, np.full(num_nodes, None)))
        boundary_vals = np.concatenate((boundary_vals, np.full(num_nodes, None)))
        deriv_lambdas = np.concatenate((deriv_lambdas, np.full(num_nodes, None)))

    return nodes, labels, boundary_vals, deriv_lambdas
