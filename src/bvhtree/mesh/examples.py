import os
import numpy as np
import pyvista as pv
import pymeshfix as mf
from pyvista import examples
from icosphere import icosphere


from .utils import rot_mat_3d_d
from .class_mesh import SurfaceMesh

__all__ = ['cube', 'sphere', 'nefertiti', 'stanford_bunny_coarse', 'stanford_bunny', 'stanford_lucy', 'stanford_dragon', 'armadillo', 'action_figure', 'burial_urn']


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


def cube(size=1.0) -> SurfaceMesh:
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
    sm_mesh = SurfaceMesh(elements, vertices)
    # Return results
    return sm_mesh


def sphere(diameter=1.0, nu=5) -> SurfaceMesh:
    """
    Sphere mesh.

    :param diameter: The diameter of the sphere.
    :param nu: Subdivision frequency. nu = 1 returns regular unit icosahedron, and nu>1 preforms subdivision.
    :returns: An instance of SurfaceMesh.
    """
    # Create a volumetric tetrahedral mesh of a sphere
    vertices, faces = icosphere(nu=nu)
    vertices = vertices * 0.5 * float(diameter)
    sm_mesh = SurfaceMesh(faces, vertices)
    return sm_mesh


'''
O------------------------------------------------------------------------------O
| PYVISTA EXAMPLE MESHES                                                       |
O------------------------------------------------------------------------------O
'''


def stanford_bunny_coarse(size=1.0) -> SurfaceMesh:
    """
    Coarse Stanford bunny mesh.

    :param size: The size of the mesh measured as the approximate length of the largest diagonal.
    :returns: An instance of SurfaceMesh.
    """
    sim_mesh = _load_and_fix_mesh(pv.examples.download_bunny_coarse, 'stanford_bunny_coarse.sm.vtk')
    sim_mesh = _shift_and_scale_mesh(sim_mesh, size)
    sim_mesh.vertices = np.dot(rot_mat_3d_d([90.0, 0.0, 180.0]), sim_mesh.vertices.T).T
    return sim_mesh


# TODO : Normals are facing inwards!! Fix this !!
def stanford_bunny(size=1.0) -> SurfaceMesh:
    """
    Stanford bunny mesh.

    :param size: The size of the mesh measured as the approximate length of the largest diagonal.
    :returns: An instance of SurfaceMesh.
    """
    sm_mesh = _load_and_fix_mesh(pv.examples.download_bunny, 'stanford_bunny.sm.vtk', fix_mesh=True, flip_normals=True)
    sm_mesh = _shift_and_scale_mesh(sm_mesh, size)
    sm_mesh.vertices = np.dot(rot_mat_3d_d([90.0, 0.0, 180.0]), sm_mesh.vertices.T).T
    return sm_mesh


def stanford_dragon(size=1.0) -> SurfaceMesh:
    """
    Stanford dragon mesh.

    :param size: The size of the mesh measured as the approximate length of the largest diagonal.
    :returns: An instance of SurfaceMesh.
    """
    sm_mesh = _load_and_fix_mesh(pv.examples.download_dragon, 'stanford_dragon.sm.vtk', fix_mesh=True)
    sm_mesh = _shift_and_scale_mesh(sm_mesh, size)
    sm_mesh.vertices = np.dot(rot_mat_3d_d([90.0, 0.0, 180.0]), sm_mesh.vertices.T).T
    return sm_mesh


def stanford_lucy(size=1.0) -> SurfaceMesh:
    """
    Stanford Lucy mesh.

    :param size: The size of the mesh measured as the approximate length of the largest diagonal.
    :returns: An instance of SurfaceMesh.
    """
    sm_mesh = _load_and_fix_mesh(pv.examples.download_lucy, 'stanford_lucy.sm.vtk', fix_mesh=True)
    sm_mesh = _shift_and_scale_mesh(sm_mesh, size)
    return sm_mesh


def armadillo(size=1.0) -> SurfaceMesh:
    """
    Armadillo mesh.

    :param size: The size of the mesh measured as the approximate length of the largest diagonal.
    :returns: An instance of SurfaceMesh.
    """
    sm_mesh = _load_and_fix_mesh(pv.examples.download_armadillo, 'armadillo.sm.vtk', fix_mesh=True)
    sm_mesh = _shift_and_scale_mesh(sm_mesh, size)
    sm_mesh.vertices = np.dot(rot_mat_3d_d([90.0, 0.0, 0.0]), sm_mesh.vertices.T).T
    return sm_mesh


def nefertiti(size=1.0) -> SurfaceMesh:
    """
    Nefertiti mesh.

    :param size: The size of the mesh measured as the approximate length of the largest diagonal.
    :returns: An instance of SurfaceMesh.
    """
    sm_mesh = _load_and_fix_mesh(pv.examples.download_nefertiti, 'nefertiti.sm.vtk', fix_mesh=True)
    sm_mesh = _shift_and_scale_mesh(sm_mesh, size)
    sm_mesh.vertices = np.dot(rot_mat_3d_d([0.0, 0.0, 180.0]), sm_mesh.vertices.T).T
    return sm_mesh


def action_figure(size=1.0,) -> SurfaceMesh:
    """
    Action figure mesh.

    :param size: The size of the mesh measured as the approximate length of the largest diagonal.
    :returns: An instance of SurfaceMesh.
    """
    tet_mesh = _load_and_fix_mesh(pv.examples.download_action_figure, 'action_figure.sm.vtk')
    tet_mesh = _shift_and_scale_mesh(tet_mesh, size)
    tet_mesh.vertices = np.dot(rot_mat_3d_d([90.0, 0.0, 180.0]), tet_mesh.vertices.T).T
    return tet_mesh


def burial_urn(size=1.0) -> SurfaceMesh:
    """
    Burial urn mesh.

    :param size: The size of the mesh measured as the approximate length of the largest diagonal.
    :returns: An instance of SurfaceMesh.
    """
    sm_mesh = _load_and_fix_mesh(pv.examples.download_urn, 'urn.sm.vtk', fix_mesh=True)
    sm_mesh = _shift_and_scale_mesh(sm_mesh, size)
    sm_mesh.vertices = np.dot(rot_mat_3d_d([90.0, 0.0, 180.0]), sm_mesh.vertices.T).T
    return sm_mesh


'''
O------------------------------------------------------------------------------O
| PRIVATE - AUXILIARY FUNCTIONS                                                |
O------------------------------------------------------------------------------O
'''


def _load_and_fix_mesh(pyvista_download_function, filename: str, fix_mesh=False, flip_normals=False) -> SurfaceMesh:
    os.makedirs(_CACHED_DIR, exist_ok=True)
    model_name = os.path.join(_CACHED_DIR, filename)
    # Check if the model already exists
    if os.path.exists(model_name):
        new_mesh = SurfaceMesh.from_file(model_name)
    else:
        print(f'Saving the model "{filename}" in folder "{_CACHED_DIR}"...')
        # Download example surface from pyvista
        pyvista_mesh = pyvista_download_function()
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
        new_mesh = SurfaceMesh(mesh_faces, mesh_vertices)
        new_mesh.save(model_name)
    # Return the model
    return new_mesh


def _shift_and_scale_mesh(mesh: SurfaceMesh, target_size: float) -> SurfaceMesh:
    # Shift and scale the geometry
    vertices = mesh.vertices
    center_point = np.average(vertices, axis=0)
    max_distance = np.max(np.square(vertices - center_point))
    scaling_factor = target_size / (2.0 * np.sqrt(max_distance))
    vertices -= center_point
    vertices *= scaling_factor
    # Return the transformed mesh
    new_mesh = SurfaceMesh(mesh.faces, vertices)
    return new_mesh
