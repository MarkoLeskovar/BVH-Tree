import numba
import numpy as np
from typing import Sequence

from bvhtree.tree import AABBTree
from bvhtree.mesh.examples import stanford_bunny_coarse, burial_urn
from bvhtree.mesh.utils import _rot_mat_3d_r

@numba.njit(cache=True)
def combine_transformations(transforms: list[np.ndarray]):
    output = np.eye(4, dtype=np.float64)
    for t in transforms:
        output = np.matmul(output, t)
    return output

@numba.njit(cache=True)
def rotation_matrix(angles: np.ndarray):
    output = np.eye(4, dtype=np.float64)
    angles = np.deg2rad(angles)
    output[0:3, 0:3] = _rot_mat_3d_r(angles)
    return output

@numba.njit(cache=True)
def translation_matrix(displacements: np.ndarray):
    output = np.eye(4, dtype=np.float64)
    output[0:3, 3] = displacements
    return output

@numba.njit(cache=True)
def scaling_matrix(scales: np.ndarray):
    output = np.eye(4, dtype=np.float64)
    output[0, 0] = scales[0]
    output[1, 1] = scales[1]
    output[2, 2] = scales[2]
    return output


def named_transform(name: str, value: Sequence[float]) -> np.ndarray:
    name = str(name)
    value = np.asarray(value).astype('float')
    # Check input
    if len(value) != 3:
        raise TypeError(f"Mag value must have 3 arguments!")
    # Select correct transformation
    if name == 'rotate':
        return rotation_matrix(value)
    elif name == 'translate':
        return translation_matrix(value)
    elif name == 'scale':
        return scaling_matrix(value)
    else:
        raise TypeError(f"Unknown transformation name!")


# This function matches the Clion input !!
if __name__ == "__main__":

    # DEBUG
    transform = named_transform(name='rotate', value=[0, 0, 90])

    # Define mesh size
    mesh_size = 50

    # Load the mesh
    mesh = burial_urn(size=0.9 * mesh_size)

    # Query point
    point = np.asarray([10, 20, 30])

    # Create a BVH class
    aabb_tree = AABBTree.from_triangle_mesh(mesh, depth_lim=16, split_lim=10)

    # Query closest point
    result = aabb_tree.query_closest_points(point)

    pass