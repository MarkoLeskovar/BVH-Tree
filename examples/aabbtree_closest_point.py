import vtk
import numpy as np
import pyvista as pv

from bvhtree import AABBTree
from bvhtree.mesh import examples
from bvhtree.mesh.intersection import box_sphere_intersection
from bvhtree.tree.utils import nodes_to_pyvista_lines

# TODO : Add these two files to examples models!!
# download_drill()
# download_horse()


# Define main function
def main():

    # Load the mesh
    mesh_size = 20
    mesh = examples.nefertiti(size=mesh_size)

    # Create a BVH class
    aabb_tree = AABBTree.from_triangle_mesh(mesh, depth_lim=16, split_lim=10)

    # Initial point
    init_point = np.random.random(size=3)
    init_point = mesh_size * init_point / np.linalg.norm(init_point)

    # Show the nodes
    pl = pv.Plotter()

    # Add the mesh
    pl.add_mesh(mesh.to_pyvista_grid(), color='lightgray')

    # Sphere widget
    def sphere_widget_callback(point):

        # Initial distance guess via kd-tree
        closest_point, distance = aabb_tree.query_closest_points(np.asarray(point))[0:2]
        closest_point = closest_point[0]
        distance = distance[0]

        # Check intersection leafs
        intersecting_leaf_nodes = []
        for node in aabb_tree.get_leaf_nodes():
            if box_sphere_intersection(node.box_min, node.box_max, point, distance):
                intersecting_leaf_nodes.append(node)

        # Node list
        node_list = []
        # Node stack
        node_stack = [aabb_tree.root_node]
        # While stack has nodes
        while len(node_stack) != 0:
            node = node_stack.pop()
            # Check intersection
            if box_sphere_intersection(node.box_min, node.box_max, point, distance):
                node_list.append(node)
                # Add children
                if not node.is_leaf():
                    # Get children
                    left_node = aabb_tree.nodes[node.left_id]
                    right_node = aabb_tree.nodes[node.right_id]
                    node_stack.append(left_node)
                    node_stack.append(right_node)

        # Get node depths
        depths = np.zeros(shape=len(node_list), dtype='int')
        for i in range(len(node_list)):
            depths[i] = node_list[i].depth

        # Convert to pyvista mesh
        pyvista_mesh_nodes = nodes_to_pyvista_lines(node_list)
        pyvista_mesh_nodes['color'] = np.repeat(depths, 12)
        pyvista_mesh_nodes['opacity'] = pyvista_mesh_nodes['color'] / depths.max()

        # Add the nodes
        pl.add_mesh(pyvista_mesh_nodes, name='nodes_1', scalars='color', opacity='opacity', line_width=2, cmap='jet', show_scalar_bar=False)
        pl.add_mesh(nodes_to_pyvista_lines(intersecting_leaf_nodes), name='nodes_2', color='black', line_width=5)

        # Add the closest points
        pl.add_points(closest_point, name='closest_point', render_points_as_spheres=True, point_size=5, color='red')
        pl.add_mesh(pv.Line(point, closest_point), name='lines', color='red', line_width=3)


    # Add sphere widget
    pl.add_sphere_widget(sphere_widget_callback, center=init_point, color='darkorange', selected_color='orange',
                         radius=0.01 * mesh_size, interaction_event=vtk.vtkCommand.InteractionEvent)

    # Show everything
    pl.show()


# Call main function
if __name__ == '__main__':
    main()
