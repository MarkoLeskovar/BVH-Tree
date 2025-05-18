import numba
import numpy as np
import pyvista as pv
import scipy.spatial
import matplotlib.pyplot as plt


from .class_aabb import axis_aligned_bounding_box
from .class_aabbnode import AABBNode, AABBNodeList
from ..mesh.class_mesh import SurfaceMesh
from ..mesh.utils import compute_triangle_centers
from ..mesh.distance import closest_point_on_triangle
from ..mesh.intersection import box_sphere_intersection

# TODO : Look at this website -> https://mshgrid.com/2021/01/17/aabb-tree/
# TODO : Implement Surface Area Heuristic (SAH) split strategy !
# TODO : Convert query function to support numba !



'''
O------------------------------------------------------------------------------O
| CLASS - AXIS ALIGNED BOUNDING BOX - TREE                                     |
O------------------------------------------------------------------------------O
'''

class AABBTree:

    def __init__(self, faces, vertices, depth_lim=10, split_lim=64):
        self._faces = np.asarray(faces).copy()
        self._vertices = np.asarray(vertices).copy()
        # Create an internal kd-tree of vertices
        self._vertex_kdtree = scipy.spatial.KDTree(self._vertices)
        # Build the tree
        self._data_list, self._box_list, self._face_ids, self._max_depth = _build_tree(
            self._faces, self._vertices, depth_lim, split_lim)
        # Get number of nodes
        self._num_nodes = self._data_list.shape[0]
        self._num_leaf_nodes = np.sum(self._data_list[:, 1] == -1).item()

        # Create tree nodes
        self._nodes = _create_nodes(self._data_list, self._box_list)


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
        # Return a new instance
        return cls(faces, vertices, depth_lim, split_lim)

    @classmethod
    def from_surface_mesh(cls, surface_mesh: SurfaceMesh, depth_lim=10, split_lim=64):
        """
        Create a class instance from PyVista PolyData.

        :param surface_mesh: SurfaceMesh class containing triangular surface elements.
        :param depth_lim: Maximum depth of the BVH tree.
        :param split_lim: Minimum number of faces in each node.
        :returns: A new instance of BVH.
        """
        # Check input type
        if not isinstance(surface_mesh, SurfaceMesh):
            raise ValueError(f'Wrong type! Mesh is not an instance of {SurfaceMesh}!')
        # Return a new instance
        return cls(surface_mesh.faces, surface_mesh.vertices, depth_lim, split_lim)

    @property
    def faces(self) -> np.ndarray:
        return self._faces

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @property
    def nodes(self) -> AABBNodeList:
        return self._nodes

    @property
    def face_ids(self) -> np.ndarray:
        return self._face_ids

    @property
    def root_node(self) -> AABBNode:
        return self._nodes[0]


    # O------------------------------------------------------------------------------O
    # | PUBLIC - NODE RETRIVAL                                                       |
    # O------------------------------------------------------------------------------O

    def get_leaf_nodes(self) -> AABBNodeList:
        nodes_list = AABBNodeList()
        for node in self._nodes:
            if node.is_leaf():
                nodes_list.append(node)
        return nodes_list

    def get_nodes(self, depth: int) -> AABBNodeList:
        nodes_list = AABBNodeList()
        for node in self._nodes:
            if node.depth == depth:
                nodes_list.append(node)
        return nodes_list


    # O------------------------------------------------------------------------------O
    # | PUBLIC - FACE RETRIVAL                                                       |
    # O------------------------------------------------------------------------------O

    def get_node_faces(self, node: AABBNode) -> np.ndarray:
        face_ids = self._face_ids[node.face_index: node.face_index + node.face_count]
        return self._faces[face_ids, :]



    # O------------------------------------------------------------------------------O
    # | PUBLIC - QUERY CLOSEST PRIMITIVES                                            |
    # O------------------------------------------------------------------------------O

    def query_closest_points(self, points: np.ndarray, workers=1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(points.shape) == 1:
            points = points.reshape((-1, 3))
        # Query points
        init_distances = self._vertex_kdtree.query(points, workers=workers)[0]
        numba.set_num_threads(workers)
        return _query_closest_points(self.faces, self._vertices, self._data_list, self._box_list, self._face_ids, self._max_depth, self._num_leaf_nodes, points, init_distances)




'''
O------------------------------------------------------------------------------O
| PRIVATE - TREE BUILDING                                                      |
O------------------------------------------------------------------------------O
'''

@numba.njit(cache=True)
def _build_tree(faces, vertices, depth_lim, split_lim):
    max_depth = 0
    max_nodes = 2 ** depth_lim - 1

    # Compute face centers
    all_face_centers = compute_triangle_centers(faces, vertices)
    all_face_ids = np.arange(faces.shape[0])

    # Initialize outputs
    # ... [0:3] -> box_min, [3:6] -> box_max
    box_list = np.zeros(shape=(max_nodes, 6), dtype='float')
    # ... [0] -> parent_id, [1] -> child_id, [2] -> depth, [3] -> face_index, [4] -> face_count
    data_list = np.full(shape=(max_nodes, 5), fill_value=-1, dtype='int')

    # Initialize the list with root data
    box_list[0, 0:3], box_list[0, 3:6] = _create_aabb(all_face_ids, faces, vertices)
    data_list[0, 2] = 1
    data_list[0, 3] = 0
    data_list[0, 4] = faces.shape[0]

    # Initialize the stack
    node_id_stack = np.zeros(shape=depth_lim, dtype='int')

    # Iteration counters
    list_size = 1
    stack_size = 1

    # Add the root node
    while stack_size != 0:
        stack_size -= 1
        node_id = node_id_stack[stack_size]

        # Get the current data
        node_depth = data_list[node_id, 2]
        node_face_index = data_list[node_id, 3]
        node_face_count = data_list[node_id, 4]

        # Check splitting condition
        if node_depth < depth_lim and node_face_count > split_lim:

            # Update child index
            data_list[node_id, 1] = list_size

            # Update the max depth
            new_depth = node_depth + 1
            if new_depth > max_depth:
                max_depth = new_depth

            # Split the nodes
            face_index_left, face_count_left, face_index_right, face_count_right = _split_middle(
                node_face_index, node_face_count, all_face_ids, all_face_centers)

            # Left bounding box
            face_ids_left = all_face_ids[face_index_left: face_index_left + face_count_left]
            node_left_box_min, node_left_box_max = _create_aabb(face_ids_left, faces, vertices)

            # Right bounding box
            face_ids_right = all_face_ids[face_index_right: face_index_right + face_count_right]
            node_right_box_min, node_right_box_max = _create_aabb(face_ids_right, faces, vertices)

            # Add left node
            j = list_size
            box_list[j, 0:3] = node_left_box_min
            box_list[j, 3:6] = node_left_box_max
            data_list[j, 0] = node_id
            data_list[j, 2] = new_depth
            data_list[j, 3] = face_index_left
            data_list[j, 4] = face_count_left

            # Add right node
            j = list_size + 1
            box_list[j, 0:3] = node_right_box_min
            box_list[j, 3:6] = node_right_box_max
            data_list[j, 0] = node_id
            data_list[j, 2] = new_depth
            data_list[j, 3] = face_index_right
            data_list[j, 4] = face_count_right

            # Add to stack (process left node first)
            node_id_stack[stack_size] = list_size + 1
            node_id_stack[stack_size + 1] = list_size

            # Update counter
            list_size += 2
            stack_size += 2

    # Crop the list to actual size
    data_list = data_list[:list_size, :]
    box_list = box_list[:list_size, :]

    # Return tree nodes
    return data_list, box_list, all_face_ids, max_depth


@numba.njit(cache=True)
def _create_aabb(face_ids, all_faces, all_vertices):
    faces = all_faces[face_ids, :]
    vertices_0 = all_vertices[faces[:, 0], :]
    vertices_1 = all_vertices[faces[:, 1], :]
    vertices_2 = all_vertices[faces[:, 2], :]
    vertices = np.vstack((vertices_0, vertices_1, vertices_2))
    return axis_aligned_bounding_box(vertices)


@numba.njit(cache=True)
def _split_middle(face_index, face_count, all_face_ids, all_face_centers):
    # Get active face centers
    face_ids = all_face_ids[face_index: face_index + face_count]
    face_centers = all_face_centers[face_ids, :]

    # Get the bounding box of face centroid
    box_min, box_max = axis_aligned_bounding_box(face_centers)
    box_size = box_max - box_min

    # Get split axis
    split_axis = box_size.argmax()

    # Get axis centers and split value
    split_pos = face_centers[:, split_axis].mean()

    # Initialize left and right nodes
    face_index_left = face_index
    face_count_left = 0
    face_index_right = face_index
    face_count_right = 0

    # Partial sort
    for i in range(face_index, face_index + face_count):
        if all_face_centers[all_face_ids[i], split_axis] < split_pos:
            i_swap = face_index_left + face_count_left
            if i != i_swap:
                all_face_ids[i], all_face_ids[i_swap] = all_face_ids[i_swap], all_face_ids[i]
            face_index_right += 1
            face_count_left += 1
        else:
            face_count_right += 1

    # Return results
    return face_index_left, face_count_left, face_index_right, face_count_right


# TODO : Implement this !!
@numba.njit(cache=True)
def _split_surface_area_heuristic(face_index, face_count, all_face_ids, all_face_centers):
    # Get active face centers
    face_ids = all_face_ids[face_index: face_index + face_count]
    face_centers = all_face_centers[face_ids, :]

    # Get the split axis (largest dimension of centroid bounds)
    box_min, box_max = axis_aligned_bounding_box(face_centers)
    box_size = box_max - box_min

    face_index_left = None
    face_count_left = None
    face_index_right = None
    face_count_right = None
    # Return results
    return face_index_left, face_count_left, face_index_right, face_count_right


def _create_nodes(data_list: np.ndarray, box_list: np.ndarray) -> AABBNodeList:
    nodes = AABBNodeList()
    for i in range(data_list.shape[0]):
        node = AABBNode()
        # [0:3] -> box_min, [3:6] -> box_max
        node.box_min = box_list[i, 0:3]
        node.box_max = box_list[i, 3:6]
        # [0] -> parent_id, [1] -> child_id, [2] -> depth, [3] -> face_index, [4] -> face_count
        node.parent_id = data_list[i, 0].item()
        node.child_id = data_list[i, 1].item()
        node.depth = data_list[i, 2].item()
        node.face_index = data_list[i, 3].item()
        node.face_count = data_list[i, 4].item()
        # Add to nodes
        nodes.append(node)
    # Return results
    return nodes



'''
O------------------------------------------------------------------------------O
| PRIVATE - TREE QUERY                                                         |
O------------------------------------------------------------------------------O
'''

@numba.njit(cache=True, parallel=True)
def _query_closest_points(faces, vertices, data_list, box_list, all_face_ids, max_depth, num_leaf_nodes, points, init_distances):

    # Initialize output
    min_points = np.empty_like(points)
    min_distances = np.empty(shape=points.shape[0], dtype='float')
    min_faces_ids = np.empty(shape=points.shape[0], dtype='int')

    # Loop over all points
    for i in numba.prange(points.shape[0]):
        node_ids = _ball_query(data_list, box_list, max_depth, num_leaf_nodes, points[i], init_distances[i])
        min_points[i, :], min_distances[i], min_faces_ids[i] = _find_closest_face(faces, vertices, data_list, all_face_ids, node_ids, points[i])

    # Return results
    return min_points, min_distances, min_faces_ids


@numba.njit(cache=True)
def _ball_query(data_list, box_list, max_depth, num_leaf_nodes, ball_center, ball_radius):

    # Initialize the stack
    node_id_stack = np.zeros(shape=max_depth, dtype='int')
    node_id_list = np.zeros(shape=num_leaf_nodes, dtype='int')

    # Iteration counters
    list_size = 0
    stack_size = 1

    # Add the root node
    node_id_stack[0] = 0
    while stack_size != 0:
        stack_size -= 1
        node_id = node_id_stack[stack_size]

        # Get the current data
        node_child_id = data_list[node_id, 1]

        # Check leaf node
        if node_child_id == -1:
            node_id_list[list_size] = node_id
            list_size += 1
        else:
            # Check left node
            left_node_id = node_child_id
            if box_sphere_intersection(box_list[left_node_id, 0:3], box_list[left_node_id, 3:6], ball_center, ball_radius):
                node_id_stack[stack_size] = left_node_id
                stack_size += 1
            # Check right node
            right_node_id = node_child_id + 1
            if box_sphere_intersection(box_list[right_node_id, 0:3], box_list[right_node_id, 3:6], ball_center, ball_radius):
                node_id_stack[stack_size] = right_node_id
                stack_size += 1

    # Crop the results
    node_id_list = node_id_list[:list_size]
    return node_id_list


@numba.njit(cache=True)
def _find_closest_face(faces, vertices, data_list, face_ids, node_ids, point):
    # Initialize variables
    closest_point = np.zeros_like(point)
    closest_distance = np.inf
    closest_face_id = -1

    # Loop over every node
    for i in range(len(node_ids)):
        node_id = node_ids[i]
        face_index = data_list[node_id, 3]
        face_count = data_list[node_id, 4]

        # Loop over every triangle
        for j in range(face_index, face_index + face_count):

            # Get triangle
            face_id = face_ids[j]
            face = faces[face_id, :]
            vertices_0 = vertices[face[0], :]
            vertices_1 = vertices[face[1], :]
            vertices_2 = vertices[face[2], :]

            # Get distance to triangle
            temp_point = closest_point_on_triangle(point, vertices_0, vertices_1, vertices_2)[0]
            temp_distance = np.sum(np.square(temp_point - point))

            # Update the distance
            if temp_distance < closest_distance:
                closest_point = temp_point
                closest_distance = temp_distance
                closest_face_id = face_id

    # Compute the distance
    closest_distance = np.sqrt(closest_distance)

    # Return results
    return closest_point, closest_distance, closest_face_id
