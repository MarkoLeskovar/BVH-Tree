import numba
import numpy as np
from typing import Sequence

__all__ = ['rot_mat_3d_r', 'rot_mat_3d_d', 'translate_points', 'rotate_points', 'compute_triangle_basis_vectors',
           'compute_triangle_centers', 'compute_triangle_normals', 'compute_triangle_areas']


'''
O------------------------------------------------------------------------------O
| ROTATION MATRICES                                                            |
O------------------------------------------------------------------------------O
'''


def rot_mat_3d_r(axis_angles: Sequence[float]) -> np.ndarray:
    """
    Compute the 3D rotation matrix for given axis angles in radians.

    :param axis_angles: The axis angles in radians.
    :returns: The 3D rotation matrix of shape [3,3].
    """
    axis_angles = np.asarray(axis_angles)
    return _rot_mat_3D_r(axis_angles)


def rot_mat_3d_d(axis_angles: Sequence[float]) -> np.ndarray:
    """
    Compute the 3D rotation matrix for given axis angles in degrees.

    :param axis_angeles: The axis angles in degrees.
    :returns: The 3D rotation matrix of shape [3,3].
    """
    axis_angles = np.deg2rad(axis_angles)
    return _rot_mat_3D_r(axis_angles)


@numba.njit(cache=True)
def _rot_mat_3D_r(axis_angeles: Sequence[float]):
    cosA = np.cos(axis_angeles[0])
    cosB = np.cos(axis_angeles[1])
    cosC = np.cos(axis_angeles[2])
    sinA = np.sin(axis_angeles[0])
    sinB = np.sin(axis_angeles[1])
    sinC = np.sin(axis_angeles[2])
    rotation_matrix = np.asarray([
        [cosB*cosC, sinA*sinB*cosC - cosA*sinC, cosA*sinB*cosC + sinA*sinC],
        [cosB*sinC, sinA*sinB*sinC + cosA*cosC, cosA*sinB*sinC - sinA*cosC],
        [-sinB    , sinA*cosB                 , cosA*cosB                 ]
    ], dtype='float')
    return rotation_matrix


'''
O------------------------------------------------------------------------------O
| ROTATE AND TRANSLATE POINTS                                                  |
O------------------------------------------------------------------------------O
'''

@numba.njit(cache=True)
def translate_points(points: np.ndarray, trans_vec: Sequence[float]):
    trans_vec = np.asarray(trans_vec)
    new_points = points + trans_vec
    return new_points


@numba.njit(cache=True)
def rotate_points(points: np.ndarray, rot_mat: np.ndarray, rot_origin=None):
    if rot_origin is None:
        rot_origin = np.zeros(points.shape[1], dtype='float')
    new_points = points - rot_origin
    new_points = np.dot(rot_mat, new_points.T).T
    new_points = new_points + rot_origin
    return new_points



'''
O------------------------------------------------------------------------------O
| TRIANGLE FUNCTIONS                                                           |
O------------------------------------------------------------------------------O
'''

@numba.njit(cache=True)
def compute_triangle_basis_vectors(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    # Edge vectors
    ab = v1 - v0
    ac = v2 - v0
    # Basis vectors
    basis_vec = np.zeros(shape=(3, 3), dtype='float')
    basis_vec[0, :] = ab / np.linalg.norm(ab)
    basis_vec[2, :] = np.cross(ab, ac)
    basis_vec[2, :] = basis_vec[2, :] / np.linalg.norm(basis_vec[2, :])
    basis_vec[1, :] = np.cross(basis_vec[2, :], basis_vec[0, :])
    # Return results
    return basis_vec.transpose()


@numba.njit(cache=True)
def compute_triangle_centers(faces: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    centers = np.empty(shape=(faces.shape[0], 3), dtype='float')
    for i in range(faces.shape[0]):
        face_id = faces[i, :]
        face = vertices[face_id, :]
        # Compute face center
        centers[i, 0] = face[:, 0].mean()
        centers[i, 1] = face[:, 1].mean()
        centers[i, 2] = face[:, 2].mean()
    # Return results
    return centers


@numba.njit(cache=True)
def compute_triangle_normals(faces: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    normals = np.empty(shape=(faces.shape[0], 3), dtype='float')
    for i in range(faces.shape[0]):
        face_id = faces[i, :]
        face = vertices[face_id, :]
        # Get face vectors
        ab = face[1, :] - face[0, :]
        ac = face[2, :] - face[0, :]
        # Compute face normal
        face_normal = np.cross(ab, ac)
        normals[i, :] = face_normal / np.linalg.norm(face_normal)
    # Return results
    return normals


@numba.njit(cache=True)
def compute_triangle_areas(faces: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    areas = np.empty(shape=faces.shape[0], dtype='float')
    for i in range(faces.shape[0]):
        face_id = faces[i, :]
        face = vertices[face_id, :]
        # Get face vectors
        ab = face[1, :] - face[0, :]
        ac = face[2, :] - face[0, :]
        # Compute face area
        face_area = 0.5 * np.linalg.norm(np.cross(ab, ac))
        areas[i] = face_area
    # Return results
    return areas
