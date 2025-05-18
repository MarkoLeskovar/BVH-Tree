import numba
import numpy as np
import pyvista as pv

from bvhtree.mesh.distance import closest_point_on_box
from bvhtree.tree.class_aabb import axis_aligned_bounding_box


def main():

    # Create a random box
    aabb_min, aabb_max = axis_aligned_bounding_box(np.random.random(size=(6, 3)) * 10)

    # Define a point
    num_points = 100
    points = np.random.random(size=(num_points, 3)) * 10

    # Initialize the closest points
    closest_points = np.empty_like(points)

    # Find the closest point on a triangle
    for i in range(closest_points.shape[0]):
        closest_points[i] = closest_point_on_box(points[i], aabb_min, aabb_max)

    # Initialize the plotter
    pl = pv.Plotter()
    pl.add_axes(color='black')
    pl.add_axes_at_origin(labels_off=True)
    pl.set_background('white')

    # Add bounding box
    pyvista_aabb = pv.Box(np.vstack((aabb_min, aabb_max)).T.ravel())
    pl.add_mesh(pyvista_aabb, color='gray', opacity=0.1)
    pl.add_mesh(pyvista_aabb.extract_all_edges(), color='orange', line_width=5)

    # Add points
    pl.add_points(points, render_points_as_spheres=True, point_size=10, color='black')

    # Add the closest point
    for i in range(closest_points.shape[0]):
        pl.add_points(closest_points[i], render_points_as_spheres=True, point_size=10, color='green')
        pl.add_lines(np.asarray([points[i], closest_points[i]]),  color='green', width=5)

    # Show everything
    pl.show()


# Call main function
if __name__ == "__main__":
    main()
