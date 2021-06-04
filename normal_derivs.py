import numpy as np
import matplotlib.pyplot as plt # delete, only for testing

def cartesian(centre_pos, positions, c, axis, direction):
    """
    Evaluates the derivative with respect to x or y at a boundary

    Arguments:
    - centre_pos: position of the centre node
    - positions: relative positions of neighbourhood nodes
    - c: shape parameter
    - axis: "x" or "y"
    - direction: "+" or "-"
    """
    rel_pos = positions - centre_pos
    distances = np.abs(rel_pos)
    cr_0_sq = (np.max(distances) * c) ** 2

    if axis == "x":
        diff = (positions.real - centre_pos.real) # PRETTY SURE THESE NEED SWITCHING!
    elif axis == "y":
        diff = (positions.imag - centre_pos.imag)
    else:
        raise ValueError("Invalid axis")

    if direction == "+":
        return diff / np.sqrt(distances ** 2 + cr_0_sq)
    elif direction == "-":
        return -diff / np.sqrt(distances ** 2 + cr_0_sq)
    else:
        raise ValueError("Invalid direction")


def radial(centre_pos, positions, c, direction="inwards", circ_centre=0):
    """
    Evaluates the derivative with respect to the inwards radial direction at a boundary
    """
    centre_pos_mod, positions_mod = centre_pos - circ_centre, positions - circ_centre

    r = np.abs(centre_pos_mod)
    theta = np.arccos(centre_pos_mod.real / r)
    if centre_pos_mod.imag < 0:
        theta *= -1

    rk = np.abs(positions_mod)
    thetak = np.arccos(positions_mod.real / rk)
    thetak[positions_mod.imag < 0] *= -1

    distances_sq = rk ** 2 + r ** 2 - 2 * r * rk * np.cos(thetak - theta)
    rel_pos = positions_mod - centre_pos_mod
    distances2 = np.abs(rel_pos)


    cr_0_sq = c**2 * np.max(distances_sq)

    deriv = -(2*r -2*rk*np.cos(thetak-theta)) / (2 * np.sqrt(distances_sq + cr_0_sq))
    if direction == "inwards":
        return deriv
    else:
        return -deriv
