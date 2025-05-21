import time
import numpy as np
import pyvista as pv

from bvhtree import AABBTree
import bvhtree.mesh.examples as examples


# Define main function
def main():

    # Fix random seed
    np.random.seed(1)

    # Mesh size
    mesh_size = 50

    # Load the mesh
    mesh = examples.stanford_bunny_coarse(size=0.9 * mesh_size)

    # Define a volume
    lattice = np.asarray([50, 50, 50])
    origin = -0.5 * np.asarray([mesh_size, mesh_size, mesh_size])
    size = np.asarray([mesh_size, mesh_size, mesh_size])
    spacing = size / lattice

    # Create a volume
    pyvista_grid = pv.ImageData(dimensions=lattice, spacing=spacing, origin=origin)

    # Sampling points
    points = pyvista_grid.points


    # Create the AABBTree
    print('RUNNING: Initialize AABB Tree...')
    t0 = time.time()

    aabb_tree = AABBTree.from_triangle_mesh(mesh, depth_lim=16, split_lim=10)

    print(f'...done! t = {time.time() - t0:.2f}')


    # AABB Tree-based point search
    print('RUNNING: AABB Tree-based closest point search...')
    t0 = time.time()

    # Query closest points
    closest_points, distances, _,  = aabb_tree.query_closest_points(points, workers=16)

    print(f'...done! t = {time.time() - t0:.2f}')

    # Initialize the plotter
    pl = pv.Plotter()
    pl.add_axes(color='black')
    pl.add_axes_at_origin(labels_off=True)
    pl.set_background('white')

    # Add the mesh
    pl.add_mesh(mesh.to_pyvista_grid(), color='darkgray', opacity=1.0)

    # Add the image
    pyvista_grid['Distance'] = distances
    pl.add_mesh_clip_plane(pyvista_grid, cmap='jet')

    # Show everything
    pl.show()


# Call main function
if __name__ == "__main__":
    main()
