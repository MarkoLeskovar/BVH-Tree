import numba
import numpy as np
import pyvista as pv

__all__ = ['AABBNode', 'AABBNodeList']


'''
O------------------------------------------------------------------------------O
| CLASS - AXIS ALIGNED BOUNDING BOX - NODE                                     |
O------------------------------------------------------------------------------O
'''

class AABBNode:

    def __init__(self):
        self.box_min = None
        self.box_max = None
        self.parent_id = -1
        self.child_id = -1
        self.depth = 0
        self.face_index = -1
        self.face_count = -1

    @property
    def left_id(self) -> int:
        return self.child_id

    @property
    def right_id(self) -> int:
        return self.child_id + 1

    def is_leaf(self) -> bool:
        return self.child_id == -1

    def is_root(self) -> bool:
        return self.parent_id == -1


'''
O------------------------------------------------------------------------------O
| CLASS - AXIS ALIGNED BOUNDING BOX - NODE LIST                                |
O------------------------------------------------------------------------------O
'''

class AABBNodeList:

    def __init__(self):
        self._nodes = []

    # Container emulation
    def __getitem__(self, index):
        return self._nodes[index]
    def __setitem__(self, index, node: AABBNode):
        self._nodes[index] = node
    def __delitem__(self, index):
        del self._nodes[index]

    # Size and iteration
    def __len__(self):
        return len(self._nodes)
    def __iter__(self):
        return iter(self._nodes)
    def __reversed__(self):
        return reversed(self._nodes)

    # String representation
    def __repr__(self):
        return repr(self._nodes)
    def __str__(self):
        return str(self._nodes)

    # List-like methods
    def append(self, node: AABBNode):
        self._nodes.append(node)

    def pop(self, index=-1) -> AABBNode:
        return self._nodes.pop(index)

    def clear(self):
        self._nodes.clear()

    # Custom function
    def to_pyvista_faces(self):
        points, faces = _merge_box_faces(self._nodes)
        return pv.PolyData(points, faces=faces)

    def to_pyvista_lines(self):
        points, lines = _merge_box_lines(self._nodes)
        return pv.PolyData(points, lines=lines)



'''
O------------------------------------------------------------------------------O
| PRIVATE - PYVISTA VISUALIZATION                                              |
O------------------------------------------------------------------------------O
'''

def _merge_box_faces(nodes: list[AABBNode]) -> tuple[np.ndarray, np.ndarray]:
    # Extract all lines
    points = np.zeros(shape=(len(nodes) * 8, 3), dtype='float')
    faces = np.full(shape=(len(nodes) * 6, 5), fill_value=4, dtype='int')
    connectivity = _get_box_faces()
    for i in range(len(nodes)):
        # Fill points
        point_id_start = i * 8
        point_id_end = point_id_start + 8
        points[point_id_start: point_id_end] = _get_box_points(nodes[i].box_min, nodes[i].box_max)
        # Fill faces
        face_id_start = i * 6
        face_id_end = face_id_start + 6
        faces[face_id_start: face_id_end, 1:5] = connectivity + point_id_start
    # Return results
    return points, faces


def _merge_box_lines(nodes: list[AABBNode]) -> tuple[np.ndarray, np.ndarray]:
    # Extract all lines
    points = np.zeros(shape=(len(nodes) * 8, 3), dtype='float')
    lines = np.full(shape=(len(nodes) * 12, 3), fill_value=2, dtype='int')
    connectivity = _get_box_lines()
    for i in range(len(nodes)):
        # Fill points
        point_id_start = i * 8
        point_id_end = point_id_start + 8
        points[point_id_start: point_id_end] = _get_box_points(nodes[i].box_min, nodes[i].box_max)
        # Fill lines
        line_id_start = i * 12
        line_id_end = line_id_start + 12
        lines[line_id_start: line_id_end, 1:3] = connectivity + point_id_start
    # Return results
    return points, lines


@numba.njit(cache=True)
def _get_box_points(box_min, box_max):
    points = np.array([
        [box_min[0], box_min[1], box_min[2]],
        [box_min[0], box_min[1], box_max[2]],
        [box_min[0], box_max[1], box_min[2]],
        [box_min[0], box_max[1], box_max[2]],
        [box_max[0], box_min[1], box_min[2]],
        [box_max[0], box_min[1], box_max[2]],
        [box_max[0], box_max[1], box_min[2]],
        [box_max[0], box_max[1], box_max[2]],
    ], dtype='float')
    return points


@numba.njit(cache=True)
def _get_box_lines():
    lines = np.asarray([
        [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
        [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]
    ], dtype='int')
    return lines


@numba.njit(cache=True)
def _get_box_faces():
    faces = np.asarray([
        [0, 1, 5, 4],  # Front face
        [1, 3, 7, 5],  # Right face
        [3, 2, 6, 7],  # Back face
        [2, 0, 4, 6],  # Left face
        [0, 1, 3, 2],  # Bottom face
        [4, 5, 7, 6],  # Top face
    ])
    return faces
