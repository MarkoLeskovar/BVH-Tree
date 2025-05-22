import numpy as np
import pyvista as pv

from bvh.tree import AABB
from bvh.mesh import box_sphere_intersection


def main():

    # Create a random box
    aabb = AABB(np.random.random(size=(6, 3)) * 10)

    # Define spheres
    num_spheres = 1000
    spheres = []
    for i in range(num_spheres):
        temp_sphere = np.random.random(size=4)
        temp_sphere[0:3] *= 10
        spheres.append(temp_sphere)

    # Check intersection
    does_intersect = np.zeros(shape=num_spheres, dtype='bool')
    for i in range(num_spheres):
        sphere_center, sphere_radius = spheres[i][0:3], spheres[i][3]
        does_intersect[i] = box_sphere_intersection(aabb.min, aabb.max, sphere_center, sphere_radius)

    # Initialize the plotter
    pl = pv.Plotter()
    pl.add_axes(color='black')
    pl.add_axes_at_origin(labels_off=True)
    pl.set_background('white')

    # Add a box
    pyvista_aabb = pv.Box(np.vstack((aabb.min, aabb.max)).T.ravel())
    pl.add_mesh(pyvista_aabb, color='gray', opacity=0.1)
    pl.add_mesh(pyvista_aabb.extract_all_edges(), color='orange', line_width=5)
    pl.add_point_labels(pyvista_aabb.points, labels=['0', '1', '2', '3', '4', '5', '6', '7'], font_size=20, always_visible=True, shape_color='orange')

    # Add triangles
    for i in range(num_spheres):
        if does_intersect[i]:
            color = 'red'
        else:
            color = 'green'
        if not does_intersect[i]:
            sphere_center, sphere_radius = spheres[i][0:3], spheres[i][3]
            pl.add_mesh(pv.Sphere(sphere_radius, sphere_center), color=color)

    # Show everything
    pl.show()


# Call main function
if __name__ == "__main__":
    main()
