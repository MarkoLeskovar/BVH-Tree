import numpy as np
import pyvista as pv

from bvh.core import AABB, AABBNode, AABBTree_core
from ..mesh.class_mesh import TriangleMesh
from ..mesh.class_mesh import _get_query_results

'''
O------------------------------------------------------------------------------O
| CLASS - AXIS ALIGNED BOUNDING BOX - TREE                                     |
O------------------------------------------------------------------------------O
'''

class AABBTree:

    def __init__(self, mesh: TriangleMesh, depth_lim=10, split_lim=64):
        self._core = AABBTree_core(mesh._core, depth_lim, split_lim)


    # O------------------------------------------------------------------------------O
    # | PUBLIC - CONSTRUCTORS AND PROPERTIES                                         |
    # O------------------------------------------------------------------------------O

    @classmethod
    def from_pyvista_grid(cls, pyvista_grid: pv.core.pointset.PolyData, depth_lim=10, split_lim=64):
        """
        Create a class instance from PyVista PolyData.

        :param pyvista_grid: PyVista PolyData of triangular surface elements.
        :param depth_lim: Maximum depth of the BVH tree.
        :param split_lim: Minimum number of faces in each node.
        :returns: A new instance of BVH.
        """
        # Check input type
        if not isinstance(pyvista_grid, pv.core.pointset.PolyData):
            raise ValueError(f'Wrong type! Mesh is not an instance of {pv.core.pointset.PolyData}!')
        # Get the data
        faces = np.asarray(pyvista_grid.faces).reshape((-1, 4))[:, 1:4]
        vertices = pyvista_grid.points
        mesh = TriangleMesh(vertices, faces)
        # Return a new instance
        return cls(mesh, depth_lim, split_lim)

    @classmethod
    def from_triangle_mesh(cls, triangle_mesh: TriangleMesh, depth_lim=10, split_lim=64):
        """
        Create a class instance from PyVista PolyData.

        :param triangle_mesh: SurfaceMesh class containing triangular surface elements.
        :param depth_lim: Maximum depth of the BVH tree.
        :param split_lim: Minimum number of faces in each node.
        :returns: A new instance of BVH.
        """
        # Check input type
        if not isinstance(triangle_mesh, TriangleMesh):
            raise ValueError(f'Wrong type! Mesh is not an instance of {TriangleMesh}!')
        # Return a new instance
        return cls(triangle_mesh, depth_lim, split_lim)

    @property
    def faces(self) -> np.ndarray:
        return np.asarray(self._core.mesh.faces)

    @property
    def vertices(self) -> np.ndarray:
        return np.asarray(self._core.mesh.vertices)

    @property
    def max_depth(self) -> int:
        return self._core.max_depth

    @property
    def nodes(self) -> list[AABBNode]:
        return self._core.nodes

    @property
    def face_ids(self) -> np.ndarray:
        return self._core.face_ids

    @property
    def root_node(self) -> AABBNode:
        return self._core.nodes[0]


    # O------------------------------------------------------------------------------O
    # | PUBLIC - NODE RETRIVAL                                                       |
    # O------------------------------------------------------------------------------O

    def get_leaf_nodes(self) -> list[AABBNode]:
        return self._core.get_leaf_nodes()

    def get_nodes_at_depth(self, depth: int) -> list[AABBNode]:
        return self._core.get_nodes_at_depth(depth)


    # O------------------------------------------------------------------------------O
    # | PUBLIC - FACE RETRIVAL                                                       |
    # O------------------------------------------------------------------------------O

    def get_node_faces(self, node: AABBNode, mesh: TriangleMesh) -> np.ndarray:
        face_ids = self._core.face_ids[node.face_index: node.face_index + node.face_count]
        return mesh.faces[face_ids, :]


    # O------------------------------------------------------------------------------O
    # | PUBLIC - QUERY CLOSEST PRIMITIVES                                            |
    # O------------------------------------------------------------------------------O

    def query_closest_point(self, point: np.ndarray) -> tuple[np.ndarray, float, int]:
        result = self._core.query_closest_point(point)
        return result.point, result.distance, result.face_id


    def query_closest_points(self, points: np.ndarray, workers=1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(points.shape) == 1:
            points = points.reshape((-1, 3))
        # Query closest points
        results = self._core.query_closest_points(points, workers)
        return _get_query_results(results)
