import vtk
import numpy as np
import pyvista as pv

import bvhtree.mesh.examples as examples


# Define main function
def main():


    # Load meshes
    burial_urn_1 = examples.burial_urn(size=20).to_pyvista_grid()
    burial_urn_2 = examples.burial_urn(size=20).to_pyvista_grid()

    base_vertices = burial_urn_2.points.copy()

    # Initialize the plotter
    pl = pv.Plotter()

    # Add actors
    actor_1 = pl.add_mesh(burial_urn_1, opacity=0.0)
    actor_2 = pl.add_mesh(burial_urn_2)

    flag = True
    def key_callback():
        nonlocal flag
        flag = not flag

    def callback_1(user_matrix):
        widget_1.origin = user_matrix[0:3, 3]
        if flag:
            widget_1.axes = user_matrix[0:3, 0:3].T
        else:
            widget_1.axes = np.eye(3, dtype='float')

        # Transform the points
        burial_urn_2.points = np.dot(user_matrix[0:3, 0:3], base_vertices.T).T + user_matrix[0:3, 3]




    # def callback_2(user_matrix):
    #     widget_2.origin = user_matrix[0:3, 3]
    #     widget_2.axes = user_matrix[0:3, 0:3].T




    pl.add_key_event('b', key_callback)





    # Add widgets
    widget_1 = pv.AffineWidget3D(pl, actor_1, origin=(0.0, 0.0, 0.0), interact_callback=callback_1)
    # widget_2 = pv.AffineWidget3D(pl, actor_2, interact_callback=callback_2)

    # Show everything
    pl.show()


# Call main function
if __name__ == '__main__':
    main()
