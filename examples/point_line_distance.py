import numpy as np
import pyvista as pv

from bvhtree.mesh.distance import closest_point_on_line


# Define main function
def main():

    # Load the mesh
    line_vertices = np.random.random(size=(2, 3)) * 10

    # Define a point
    num_points = 100
    points = np.random.random(size=(num_points, 3)) * 10

    # Initialize the closest points
    closest_points = np.empty_like(points)

    # Find the closest point on a triangle
    for i in range(closest_points.shape[0]):
        closest_points[i] = closest_point_on_line(points[i], line_vertices)

    # Initialize the plotter
    pl = pv.Plotter()
    pl.add_axes(color='black')
    pl.add_axes_at_origin(labels_off=True)
    pl.set_background('white')

    # Add triangle
    pl.add_mesh(pv.Line(line_vertices[0, :], line_vertices[1, :]), color='orange', line_width=5)
    pl.add_point_labels(line_vertices, labels=['0', '1'], font_size=20, always_visible=True, shape_color='gray')

    # Add points
    pl.add_points(points, render_points_as_spheres=True, point_size=10, color='black')

    # Add the closest point
    for i in range(closest_points.shape[0]):
        pl.add_points(closest_points[i], render_points_as_spheres=True, point_size=10, color='green')
        pl.add_lines(np.asarray([points[i], closest_points[i]]), color='green', width=5)

    # Show everything
    pl.show()


# Call main function
if __name__ == "__main__":
    main()
