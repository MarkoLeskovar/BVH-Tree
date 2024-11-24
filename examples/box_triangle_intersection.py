import numpy as np
import pyvista as pv

from aabbtree.tree.class_aabb import axis_aligned_bounding_box
from aabbtree.mesh.intersection import box_triangle_intersection


def main():

    # Create a random box
    box_min, box_max = axis_aligned_bounding_box(np.random.random(size=(6, 3)) * 10)

    # Define triangles
    num_triangles = 1000
    triangles = []
    for i in range(num_triangles):
        triangles.append(np.random.random(size=(3, 3)) * 10)

    # Check intersection
    does_intersect = np.zeros(shape=num_triangles, dtype='bool')
    for i in range(num_triangles):
        tri_vertices = triangles[i]
        does_intersect[i] = box_triangle_intersection(box_min, box_max, tri_vertices[0, :], tri_vertices[1, :], tri_vertices[2, :])

    # Initialize the plotter
    pl = pv.Plotter()
    pl.add_axes(color='black')
    pl.add_axes_at_origin(labels_off=True)
    pl.set_background('white')

    # Add a box
    pyvista_aabb = pv.Box(np.vstack((box_min, box_max)).T.ravel())
    pl.add_mesh(pyvista_aabb, color='gray', opacity=0.1)
    pl.add_mesh(pyvista_aabb.extract_all_edges(), color='orange', line_width=5)
    pl.add_point_labels(pyvista_aabb.points, labels=['0', '1', '2', '3', '4', '5', '6', '7'], font_size=20, always_visible=True, shape_color='orange')

    # Add triangles
    for i in range(num_triangles):
        if does_intersect[i]:
            color = 'red'
        else:
            color = 'green'
        if not does_intersect[i]:
            tri_vertices = triangles[i]
            pl.add_mesh(pv.Triangle(tri_vertices), color='gray', opacity=0.1)
            pl.add_lines(np.vstack((tri_vertices, tri_vertices[0, :])), connected=True, color=color, width=5)

    # Show everything
    pl.show()


# Call main function
if __name__ == "__main__":
    main()
