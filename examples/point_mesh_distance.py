import time
import numpy as np
import pyvista as pv

from bvh.tree import AABBTree
import bvh.mesh.examples as examples


# Define main function
def main():

    # Fix random seed
    np.random.seed(1)

    # Mesh size
    mesh_size = 50

    # Load the mesh
    mesh = examples.nefertiti(size=mesh_size)

    # Create points on a sphere
    points = examples.sphere(diameter=2*mesh_size, nu=5).vertices

    # Query closest points -> O(n^2)
    print('RUNNING: Exhaustive closest point search...')
    t0 = time.time()
    reference_closest_points = mesh.query_closest_points(points, workers=16)[0]
    print(f'...done! t = {time.time() - t0:.2f}')

    # Create the AABBTree
    print('RUNNING: Initialize AABB Tree...')
    t0 = time.time()
    aabb_tree = AABBTree.from_triangle_mesh(mesh, depth_lim=16, split_lim=10)
    print(f'...done! t = {time.time() - t0:.2f}')

    # Query closest points -> O(nlog(m))
    print('RUNNING: AABB Tree-based closest point search...')
    t0 = time.time()
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
