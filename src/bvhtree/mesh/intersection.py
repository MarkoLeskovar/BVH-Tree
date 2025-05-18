import numba
import numpy as np


# Adapted from "Gabe" (original in C++)
# https://stackoverflow.com/questions/4578967/cube-sphere-intersection-test
@numba.njit(cache=True)
def box_sphere_intersection(box_min, box_max, sphere_center, sphere_radius):
    r2 = sphere_radius * sphere_radius
    d_min = 0.0
    for i in range(3):
        if sphere_center[i] < box_min[i]:
            d_min += (sphere_center[i] - box_min[i]) ** 2
        elif sphere_center[i] > box_max[i]:
            d_min += (sphere_center[i] - box_max[i]) ** 2
    return d_min <= r2



# Adapted from "Btan2" (original in C#)
# https://gist.github.com/zvonicek/fe73ba9903f49d57314cf7e8e0f05dcf
# https://bronsonzgeb.com/index.php/2021/05/29/gpu-mesh-voxelizer-part-2/
_UNIT_VEC_X = np.array([1.0, 0.0, 0.0])
_UNIT_VEC_Y = np.array([0.0, 1.0, 0.0])
_UNIT_VEC_Z = np.array([0.0, 0.0, 1.0])
@numba.njit(cache=True)
def box_triangle_intersection(box_min, box_max, a, b, c):
    box_extent = 0.5 * (box_max - box_min)
    box_center = box_max - box_extent

    # Center the triangle
    cent_a = a - box_center
    cent_b = b - box_center
    cent_c = c - box_center

    # Triangle edge vectors
    ab = cent_b - cent_a
    bc = cent_c - cent_b
    ca = cent_a - cent_c

    # Normalize the edge vectors
    ab /= np.linalg.norm(ab)
    bc /= np.linalg.norm(bc)
    ca /= np.linalg.norm(ca)

    # Cross ab, bc, and ca with (1, 0, 0)
    a00 = np.asarray([0.0, -ab[2], ab[1]])
    a01 = np.asarray([0.0, -bc[2], bc[1]])
    a02 = np.asarray([0.0, -ca[2], ca[1]])

    # Cross ab, bc, and ca with (0, 1, 0)
    a10 = np.asarray([ab[2], 0.0, -ab[0]])
    a11 = np.asarray([bc[2], 0.0, -bc[0]])
    a12 = np.asarray([ca[2], 0.0, -ca[0]])

    # Cross ab, bc, and ca with (0, 0, 1)
    a20 = np.asarray([-ab[1], ab[0], 0.0])
    a21 = np.asarray([-bc[1], bc[0], 0.0])
    a22 = np.asarray([-ca[1], ca[0], 0.0])

    # Check intersection with 9 axes, 3 AABB face normals and 1 triangle face normal
    axes = [a00, a01, a02, a10, a11, a12, a20, a21, a22, _UNIT_VEC_X, _UNIT_VEC_Y, _UNIT_VEC_Z, np.cross(ab, bc)]
    for i in range(13):
        if _separating_axis_theorem(cent_a, cent_b, cent_c, box_extent, axes[i]):
            return False

    # There is an intersection
    return True


@numba.njit(cache=True)
def _separating_axis_theorem(a, b, c, box_extent, axis):
    # Project points onto the provided axis
    p0 = a.dot(axis)
    p1 = b.dot(axis)
    p2 = c.dot(axis)

    # Length of the box projection ont a single axis
    r = (box_extent[0] * abs(_UNIT_VEC_X.dot(axis)) +
         box_extent[1] * abs(_UNIT_VEC_Y.dot(axis)) +
         box_extent[2] * abs(_UNIT_VEC_Z.dot(axis)))

    # Find min and max of the projected triangle to create a line (p_min -> p_max)
    p_max = max(p0, p1, p2)
    p_min = min(p0, p1, p2)

    # Check if the line (0 -> r) overlaps the line (p_min -> p_max)
    return max(-p_max, p_min) > r
