import time
import numba
import numpy as np
import pyvista as pv

from bvhtree import AABBTree
import bvhtree.mesh.examples as examples
from bvhtree.mesh.distance import closest_point_on_triangle


@numba.njit(cache=True)
def closest_point_on_mesh(point: np.ndarray, mesh_faces: np.ndarray, mesh_vertices: np.ndarray) -> tuple[np.ndarray, int]:
    closest_face = 0
    closest_point = point.copy()
    closest_distance = np.inf
    for i in range(mesh_faces.shape[0]):
        temp_vertices = mesh_vertices[mesh_faces[i, :], :]
        # Find closest
        temp_point, _ = closest_point_on_triangle(point, temp_vertices[0, :], temp_vertices[1, :], temp_vertices[2, :])
        temp_distance = np.sum(np.square(point - temp_point))
        # Update the distance
        if temp_distance <= closest_distance:
            closest_distance = temp_distance
            closest_point = temp_point
            closest_face = i
    # Return results
    return closest_point, closest_face


# Define main function
def main():

    # Fix random seed
    np.random.seed(1)

    # Mesh size
    mesh_size = 50

    # Load the mesh
    mesh = examples.stanford_bunny_coarse(size=mesh_size)

    # Create points on a sphere
    points = examples.sphere(diameter=2*mesh_size, nu=3).vertices

    # Create reference points
    print('RUNNING: Exhaustive closest point search...')
    t0 = time.time()

    # Find the closest points on the mesh
    reference_closest_points = np.empty_like(points)
    for i in range(points.shape[0]):
        reference_closest_points[i], _ = closest_point_on_mesh(points[i], mesh.faces, mesh.vertices)

    print(f'...done! t = {time.time() - t0:.2f}')


    # Create the AABBTree
    print('RUNNING: Initialize AABB Tree...')
    t0 = time.time()

    aabb_tree = AABBTree.from_triangle_mesh(mesh, depth_lim=16, split_lim=10)


    print(f'...done! t = {time.time() - t0:.2f}')


    # AABB Tree-based point search
    print('RUNNING: AABB Tree-based closest point search...')
    t0 = time.time()

    # Query closest points
    closest_points = aabb_tree.query_closest_points(points, workers=16)[0]

    print(f'...done! t = {time.time() - t0:.2f}')

    # Initialize the plotter
    pl = pv.Plotter()
    pl.add_axes(color='black')
    pl.add_axes_at_origin(labels_off=True)
    pl.set_background('white')

    # Add the mesh
    pl.add_mesh(mesh.to_pyvista_grid(), color='gray', opacity=1.0)

    # Add points
    pl.add_points(points, render_points_as_spheres=True, point_size=10, color='black')

    # Create lines
    lines = np.full(shape=(points.shape[0], 3), fill_value=2, dtype='int')
    lines[:, 1] = np.arange(points.shape[0])
    lines[:, 2] = np.arange(points.shape[0], 2 * points.shape[0])

    # Add the reference closest points
    pl.add_points(reference_closest_points, render_points_as_spheres=True, point_size=5, color='green')
    pl.add_mesh(pv.PolyData(np.vstack((points, reference_closest_points)), lines=lines), color='green', line_width=3)

    # Add the closest points
    pl.add_points(closest_points, render_points_as_spheres=True, point_size=5, color='red')
    pl.add_mesh(pv.PolyData(np.vstack((points, closest_points)), lines=lines), color='red', line_width=3)

    # Show everything
    pl.show()


# Call main function
if __name__ == "__main__":
    main()
