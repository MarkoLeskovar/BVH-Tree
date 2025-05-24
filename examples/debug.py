import numba
import numpy as np
import pyvista as pv
from typing import Sequence

from bvh.tree import AABBTree
import bvh.mesh.examples as examples
from bvh.mesh.utils import _rot_mat_3d_r


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

    # Define mesh size
    mesh_size = 50

    # Load the mesh
    mesh = examples.stanford_bunny_coarse(size=0.9 * mesh_size)

    # TODO : Make this general for any triangular mesh !!

    # INFO
    # (This only hold true for watertight meshes)
    # - vertices       -> A list of vertices containing [x,y,z] coordinates
    # - faces_vertices -> A list of faces containing [v0,v1,v2] vertex indices
    # - edges_vertices -> A list of edges containing [v0,v1] vertex indices
    # - faces_edges    -> A list of faces containing [e0,e1,e2] edge indices
    # - edge_faces     -> A list of edges containing [f0, f1] connecting faces
    # - vertices_edges -> A list of vertices containing [e0, ...en] connecting elements
    # - vertices_faces ->

    # Get mesh data
    vertices = mesh.vertices
    faces_vertices = mesh.faces
    num_vertices = vertices.shape[0]
    num_faces = faces_vertices.shape[0]

    # Extract edges
    edges_vertices_dict = {}
    faces_edges = np.zeros_like(faces_vertices)
    edge_id = 0
    for i in range(num_faces):
        temp_face = faces_vertices[i].tolist()
        # Edge combinations
        temp_edges = [(temp_face[0], temp_face[1]), (temp_face[1], temp_face[2]), (temp_face[2], temp_face[0])]
        # Loop over all edges
        for j in range(3):
            normalized_edge = tuple(sorted(temp_edges[j]))
            # Add to edges to dictionary
            if normalized_edge in edges_vertices_dict:
                faces_edges[i, j] = edges_vertices_dict[normalized_edge]
            else:
                edges_vertices_dict[normalized_edge] = edge_id
                faces_edges[i, j] = edge_id
                edge_id += 1

    # NOTE : Edge will always have two vertices, so I could use a matrix here !!
    # Convert to list of lists
    edges_vertices = [[] for x in range(len(edges_vertices_dict))]
    for key, val in edges_vertices_dict.items():
        edges_vertices[val] = list(key)


    # Number of edges
    num_edges = len(edges_vertices)


    # Extract edge-face data
    flat_faces_edges = faces_edges.ravel().tolist()
    flat_face_ids = np.repeat(np.arange(num_faces), 3).tolist()
    edges_faces_dict = {}
    for i in range(len(flat_faces_edges)):
        temp_face_edge = flat_faces_edges[i]
        temp_face_id = flat_face_ids[i]
        # Check if such edge exists
        if temp_face_edge not in edges_faces_dict:
            edges_faces_dict[temp_face_edge] = [temp_face_id]
        else:
            edges_faces_dict[temp_face_edge].append(temp_face_id)

    # NOTE : Edge can have 0, 1, or 2 faces connected to it !!
    # Convert to list of lists
    edges_faces = [[] for x in range(len(edges_faces_dict))]
    for key, val in edges_faces_dict.items():
        edges_faces[key] = val


    # Extract vertices-edges data
    flat_edges_vertices = []
    flat_edge_ids = []
    n = 0
    for i in range(len(edges_vertices)):
        flat_edges_vertices.extend(edges_vertices[i])
        for j in range(len(edges_vertices[i])):
            flat_edge_ids.append(n)
        n += 1

    vertices_edges_dict = {}
    for i in range(len(flat_edges_vertices)):
        temp_edge_vertex = flat_edges_vertices[i]
        temp_edge_id = flat_edge_ids[i]
        # Check if such edge exists
        if temp_edge_vertex not in vertices_edges_dict:
            vertices_edges_dict[temp_edge_vertex] = [temp_edge_id]
        else:
            vertices_edges_dict[temp_edge_vertex].append(temp_edge_id)

    # Convert to list of lists
    vertices_edges = [[] for x in range(len(vertices_edges_dict))]
    for key, value in vertices_edges_dict.items():
        vertices_edges[key] = value


    # Extract vertices-faces data
    pass




    # Query by edges by face id
    face_id = int(np.random.random() * (num_faces - 1))
    vertex_ids = faces_edges[face_id]
    edge_ids = np.asarray(edges_vertices)[vertex_ids]
    points = vertices[edge_ids]
    # Draw the scene
    pl = pv.Plotter()
    pl.add_mesh(mesh.to_pyvista_grid())
    for temp_point in points:
        pl.add_mesh(pv.Line(temp_point[0], temp_point[1]), color='red', line_width=5)
    pl.add_points(vertices[faces_vertices[face_id]], color='green', render_points_as_spheres=True, point_size=10)
    pl.show()

    # Query by faces by edge if
    edge_id = int(np.random.random() * (num_edges - 1))
    face_ids = edges_faces[edge_id]
    # Draw the scene
    pl = pv.Plotter()
    pl.add_mesh(mesh.to_pyvista_grid())
    for face_id in face_ids:
        temp_points = vertices[faces_vertices[face_id]]
        temp_points = np.vstack((temp_points, temp_points[0]))
        pl.add_mesh(pv.MultipleLines(temp_points), color='red', line_width=5)
    temp_point = vertices[np.asarray(edges_vertices)[edge_id]]
    pl.add_mesh(pv.Line(temp_point[0], temp_point[1]), color='green', line_width=5)
    pl.show()