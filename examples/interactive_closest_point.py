import vtk
import numpy as np
import pyvista as pv

from bvhtree.tree import AABBTree
from bvhtree.mesh import examples
from bvhtree.mesh import box_sphere_intersection
from bvhtree.tree.utils import nodes_to_pyvista_lines

# TODO : Prevent data copy when accessing internal variables from classes !!
# TODO : Draw all the nodes in advance and dynamically change their visibility to speed up the rendering !!
# TODO : Add "download_drill()" and "download_horse()"  to examples models!!


# Define main function
def main():

    # Load the mesh
    mesh_size = 20
    mesh = examples.nefertiti(size=mesh_size)

    # Create a BVH class
    aabb_tree = AABBTree.from_triangle_mesh(mesh, depth_lim=14, split_lim=20)

    # Print info
    print(f'NUM_MESH_FACES = {mesh.num_faces}')
    print(f'NUM_MESH_VERTICES = {mesh.num_vertices}')
    print(f'NUM_TREE_NODES = {len(aabb_tree.nodes)}')

    # Get the nodes
    all_nodes = aabb_tree.nodes
    all_leaf_nodes = aabb_tree.get_leaf_nodes()

    # Initial point
    init_point = np.random.random(size=3)
    init_point = mesh_size * init_point / np.linalg.norm(init_point)

    # Show the nodes
    pl = pv.Plotter()

    # Add the mesh
    pl.add_mesh(mesh.to_pyvista_grid(), color='lightgray')

    # Non-local variables
    closest_point = None
    query_point = None
    draw_nodes_flag = False

    # Draw nodes function
    def draw_nodes(radius):
        num_box_tests = 0
        num_triangle_tests = 0
        intersecting_leaf_nodes = []
        node_list = []
        # Check intersection leafs
        for node in all_leaf_nodes:
            if box_sphere_intersection(node.aabb.min, node.aabb.max, query_point, radius):
                intersecting_leaf_nodes.append(node)
                num_triangle_tests += node.face_count
        # Node stack
        node_stack = [aabb_tree.root_node]
        # While stack has nodes
        while len(node_stack) != 0:
            node = node_stack.pop()
            # Check intersection
            if box_sphere_intersection(node.aabb.min, node.aabb.max, query_point, radius):
                node_list.append(node)
                num_box_tests += 1
                # Add children
                if not node.is_leaf():
                    # Get children
                    left_node = all_nodes[node.left_child_id()]
                    right_node = all_nodes[node.right_child_id()]
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
        # Print info
        print(f'\rBOX TESTS = {num_box_tests}, TRIANGLE TESTS = {num_triangle_tests}, TRIANGLE TESTS [%] = {num_triangle_tests / mesh.num_faces * 100:.2f} ', end='')

    def remove_nodes():
        pl.remove_actor('nodes_1')
        pl.remove_actor('nodes_2')

    def draw_line():
        pl.add_points(closest_point, name='closest_point', render_points_as_spheres=True, point_size=5, color='red')
        pl.add_mesh(pv.Line(query_point, closest_point), name='lines', color='red', line_width=3)

    def update_scene():
        nonlocal closest_point
        # Find the closest point
        closest_point, distance = aabb_tree.query_closest_point(query_point)[0:2]
        # Draw nodes
        if draw_nodes_flag:
            draw_nodes(distance)
        else:
            remove_nodes()
        # Draw line
        draw_line()

    def sphere_widget_callback(point):
        nonlocal query_point
        query_point = np.asarray(point)
        update_scene()


    def button_widget_callback(flag):
        nonlocal draw_nodes_flag
        draw_nodes_flag = flag
        update_scene()
        if not draw_nodes_flag:
            print(f'\r',  end='')


    # Add widgets
    pl.add_sphere_widget(sphere_widget_callback, center=init_point, color='black', selected_color='orange',
                         radius=0.01 * mesh_size, interaction_event=vtk.vtkCommand.InteractionEvent)
    pl.add_checkbox_button_widget(button_widget_callback, value=False, color_on='red')

    # Show everything
    pl.show()


# Call main function
if __name__ == '__main__':
    main()
