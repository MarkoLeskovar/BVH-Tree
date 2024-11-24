import numba
import numpy as np

# Define tolerance
_EPS = np.finfo(np.double).resolution

@numba.njit(cache=True)
def closest_point_on_line(point: np.ndarray, line_vertices: np.ndarray) -> tuple[np.ndarray, bool]:
    # Edge vectors
    ab = line_vertices[1, :] - line_vertices[0, :]
    ap = point - line_vertices[0, :]
    # Distance along a segment
    denom = np.dot(ab, ab) + _EPS
    proj = np.dot(ap, ab) / denom
    # Select regions
    if proj <= 0.0:
        closest_point = line_vertices[0, :]
    elif proj >= 1.0:
        closest_point = line_vertices[1, :]
    else:
        closest_point = line_vertices[0, :] + proj * ab
    # Return results
    return closest_point


# https://stackoverflow.com/questions/2924795/fastest-way-to-compute-point-to-triangle-distance-in-3d
@numba.njit(cache=True)
def closest_point_on_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, bool]:

    # Edge vectors
    ab = b - a
    ac = c - a
    ap = point - a

    # Outside-inside test
    normal = np.cross(ab, ac)
    is_outside = np.dot(ap, normal) >= 0.0

    # Case 1 - Point a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, is_outside

    # Case 2 - Point b
    bp = point - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, is_outside

    # Case 3 - Point c
    cp = point - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, is_outside

    # Case 4 - Edge ab
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        t = a + v * ab
        return t, is_outside

    # Case 5 - Edge ac
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        v = d2 / (d2 - d6)
        t = a + v * ac
        return t, is_outside

    # Case 6 - Edge bc
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        v = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        t = b + v * (c - b)
        return t, is_outside

    # Case 0 - Inside
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    t = a + v * ab + w * ac
    return t, is_outside


@numba.njit(cache=True)
def closest_point_on_box(point: np.ndarray, box_min: np.ndarray, box_max: np.ndarray) -> tuple[np.ndarray, bool]:
    closest_point = point.copy()

    # Inside-outside test
    is_inside_x = box_min[0] <= point[0] <= box_max[0]
    is_inside_y = box_min[1] <= point[1] <= box_max[1]
    is_inside_z = box_min[2] <= point[2] <= box_max[2]
    is_inside = is_inside_x and is_inside_y and is_inside_z
    is_outside = not is_inside

    # Find the closest point
    if is_outside:
        closest_point = np.maximum(box_min, np.minimum(point, box_max))
    else:
        t = (point - box_min) / (box_max - box_min)
        axis = np.abs(t - 0.5).argmax()
        if t[axis] >= 0.5:
            closest_point[axis] = box_max[axis]
        else:
            closest_point[axis] = box_min[axis]

    # Return results
    return closest_point, is_outside
