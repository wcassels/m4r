import numpy as np

def cartesian(centre_pos, positions, c, axis, direction):
    """
    Function to evaluate the derivative with respect to x or y at a boundary

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
        diff = (positions.real - centre_pos.real)
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


def radial():
    """
    Now this is where the fun begins
    """
    return
