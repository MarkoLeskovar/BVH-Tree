import os
import numpy as np
import pyvista as pv
import pymeshfix as mf
from icosphere import icosphere

from .utils import rot_mat_3d_d
from .class_mesh import TriangleMesh

# Global variables
_FILE_FORMAT = '.vtk'

__all__ = ['cube', 'sphere', 'nefertiti', 'stanford_bunny_coarse', 'stanford_bunny',
           'stanford_lucy', 'stanford_dragon','armadillo', 'action_figure', 'burial_urn']


'''
O------------------------------------------------------------------------------O
| CACHED DATA                                                                  |
O------------------------------------------------------------------------------O
'''

_CACHED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'cached'))


def set_cached_dir(path: str):
    global _CACHED_DIR
    _CACHED_DIR = str(path)


def get_cached_dir() -> str:
    return _CACHED_DIR


'''
O------------------------------------------------------------------------------O
| BASIC MESHES                                                                 |
O------------------------------------------------------------------------------O
'''


def cube(size=1.0) -> TriangleMesh:
    """
    Cube mesh consisting of 5 tetrahedral elements.

    :param size: The size of the cube side.
    :returns: An instance of SurfaceMesh.
    """
    # Create nodes
    vertices = np.empty((8, 3), dtype='float')
    vertices[0, :] = [0, 1, 0]
    vertices[1, :] = [1, 1, 0]
    vertices[2, :] = [1, 1, 1]
    vertices[3, :] = [0, 1, 1]
    vertices[4, :] = [0, 0, 0]
    vertices[5, :] = [1, 0, 0]
    vertices[6, :] = [1, 0, 1]
    vertices[7, :] = [0, 0, 1]

    # TODO : Flip face normals !!
    # Create elements
    elements = np.empty((12, 3), 'int32')
    elements[0, :] = [0, 1, 2]
    elements[1, :] = [0, 2, 3]

    elements[2, :] = [4, 5, 6]
    elements[3, :] = [4, 6, 7]

    elements[4, :] = [0, 3, 7]
    elements[5, :] = [0, 7, 4]

    elements[6, :] = [1, 5, 6]
    elements[7, :] = [1, 6, 2]

    elements[8, :] = [0, 4, 5]
    elements[9, :] = [0, 5, 1]

    elements[10, :] = [3, 2, 6]
    elements[11, :] = [3, 6, 7]

    # Shift and scale the geometry
    vertices -= 0.5
    vertices *= size
    sm_mesh = TriangleMesh(elements, vertices)
    # Return results
    return sm_mesh


def sphere(diameter=1.0, nu=5) -> TriangleMesh:
    """
    Sphere mesh.

    :param diameter: The diameter of the sphere.
    :param nu: Subdivision frequency. nu = 1 returns regular unit icosahedron, and nu>1 preforms subdivision.
    :returns: An instance of SurfaceMesh.
    """
    # Create a volumetric tetrahedral mesh of a sphere
    vertices, faces = icosphere(nu=nu)
    vertices = vertices * 0.5 * float(diameter)
    sm_mesh = TriangleMesh(faces, vertices)
    return sm_mesh


'''
O------------------------------------------------------------------------------O
| PYVISTA EXAMPLE MESHES                                                       |
O------------------------------------------------------------------------------O
'''


def stanford_bunny_coarse(size=1.0) -> TriangleMesh:
    """
    Coarse Stanford bunny mesh.

    :param size: The size of the mesh measured as the approximate length of the largest diagonal.
    :returns: An instance of SurfaceMesh.
    """
    mesh = _load_and_fix_mesh(pv.examples.download_bunny_coarse, 'stanford_bunny_coarse' + _FILE_FORMAT)
    _shift_and_scale_mesh(mesh, size)
    mesh.rotate([90.0, 0.0, 180.0])
    return mesh


# TODO : Normals are facing inwards!! Fix this !!
def stanford_bunny(size=1.0) -> TriangleMesh:
    """
    Stanford bunny mesh.

    :param size: The size of the mesh measured as the approximate length of the largest diagonal.
    :returns: An instance of SurfaceMesh.
    """
    mesh = _load_and_fix_mesh(pv.examples.download_bunny, 'stanford_bunny' + _FILE_FORMAT, fix_mesh=True, flip_normals=True)
    _shift_and_scale_mesh(mesh, size)
    mesh.rotate([90.0, 0.0, 180.0])
    return mesh


def stanford_dragon(size=1.0) -> TriangleMesh:
    """
    Stanford dragon mesh.

    :param size: The size of the mesh measured as the approximate length of the largest diagonal.
    :returns: An instance of SurfaceMesh.
    """
    mesh = _load_and_fix_mesh(pv.examples.download_dragon, 'stanford_dragon' + _FILE_FORMAT, fix_mesh=True)
    _shift_and_scale_mesh(mesh, size)
    mesh.rotate([90.0, 0.0, 180.0])
    return mesh


def stanford_lucy(size=1.0) -> TriangleMesh:
    """
    Stanford Lucy mesh.

    :param size: The size of the mesh measured as the approximate length of the largest diagonal.
    :returns: An instance of SurfaceMesh.
    """
    mesh = _load_and_fix_mesh(pv.examples.download_lucy, 'stanford_lucy' + _FILE_FORMAT, fix_mesh=True)
    _shift_and_scale_mesh(mesh, size)
    return mesh


def armadillo(size=1.0) -> TriangleMesh:
    """
    Armadillo mesh.

    :param size: The size of the mesh measured as the approximate length of the largest diagonal.
    :returns: An instance of SurfaceMesh.
    """
    mesh = _load_and_fix_mesh(pv.examples.download_armadillo, 'armadillo' + _FILE_FORMAT, fix_mesh=True)
    _shift_and_scale_mesh(mesh, size)
    mesh.vertices = np.dot(rot_mat_3d_d([90.0, 0.0, 0.0]), mesh.vertices.T).T
    mesh.rotate([90.0, 0.0, 0.0])
    return mesh


def nefertiti(size=1.0) -> TriangleMesh:
    """
    Nefertiti mesh.

    :param size: The size of the mesh measured as the approximate length of the largest diagonal.
    :returns: An instance of SurfaceMesh.
    """
    mesh = _load_and_fix_mesh(pv.examples.download_nefertiti, 'nefertiti' + _FILE_FORMAT, fix_mesh=True)
    _shift_and_scale_mesh(mesh, size)
    mesh.rotate([0.0, 0.0, 180.0])
    return mesh


def action_figure(size=1.0,) -> TriangleMesh:
    """
    Action figure mesh.

    :param size: The size of the mesh measured as the approximate length of the largest diagonal.
    :returns: An instance of SurfaceMesh.
    """
    mesh = _load_and_fix_mesh(pv.examples.download_action_figure, 'action_figure' + _FILE_FORMAT)
    _shift_and_scale_mesh(mesh, size)
    mesh.rotate([90.0, 0.0, 180.0])
    return mesh


def burial_urn(size=1.0) -> TriangleMesh:
    """
    Burial urn mesh.

    :param size: The size of the mesh measured as the approximate length of the largest diagonal.
    :returns: An instance of SurfaceMesh.
    """
    mesh = _load_and_fix_mesh(pv.examples.download_urn, 'burial_urn' + _FILE_FORMAT, fix_mesh=True)
    _shift_and_scale_mesh(mesh, size)
    mesh.rotate([90.0, 0.0, 180.0])
    return mesh


def drill(size=1.0) -> TriangleMesh:
    """
    Burial urn mesh.

    :param size: The size of the mesh measured as the approximate length of the largest diagonal.
    :returns: An instance of SurfaceMesh.
    """
    mesh = _load_and_fix_mesh(pv.examples.download_drill, 'drill' + _FILE_FORMAT)
    _shift_and_scale_mesh(mesh, size)
    mesh.rotate([40.0, 0.0, -31.0])
    mesh.rotate([0.0, -90.0, -90.0])
    return mesh


'''
O------------------------------------------------------------------------------O
| PRIVATE - AUXILIARY FUNCTIONS                                                |
O------------------------------------------------------------------------------O
'''


def _load_and_fix_mesh(pyvista_download_function, filename: str, fix_mesh=False, flip_normals=False) -> TriangleMesh:
    os.makedirs(_CACHED_DIR, exist_ok=True)
    model_name = os.path.join(_CACHED_DIR, filename)
    # Check if the model already exists
    if os.path.exists(model_name):
        new_mesh = TriangleMesh.from_file(model_name)
    else:
        print(f'Saving the model "{filename}" in folder "{_CACHED_DIR}"...')
        # Download example surface from pyvista
        pyvista_mesh = pyvista_download_function().clean()
        # Flip surface normals
        if flip_normals:
            pyvista_mesh.flip_normals()
        # Fix the holes in the mesh
        if fix_mesh:
            meshfix = mf.MeshFix(pyvista_mesh)
            meshfix.repair()
            pyvista_mesh = meshfix.mesh
        # Get faces and points
        mesh_faces = pyvista_mesh.faces.reshape(-1, 4)[:, 1:]
        mesh_vertices = pyvista_mesh.points
        new_mesh = TriangleMesh(mesh_faces, mesh_vertices)
        new_mesh.save(model_name)
    # Return the model
    return new_mesh


def _shift_and_scale_mesh(mesh: TriangleMesh, target_size: float) -> None:
    # Shift and scale the geometry
    vertices = mesh.vertices
    center_point = np.average(vertices, axis=0)
    max_distance = np.max(np.square(vertices - center_point))
    scaling_factor = target_size / (2.0 * np.sqrt(max_distance))
    # Transform the mesh in-place
    mesh.translate(-center_point)
    mesh.scale([scaling_factor, scaling_factor, scaling_factor])
