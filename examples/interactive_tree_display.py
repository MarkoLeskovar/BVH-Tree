from bvhtree.mesh import examples
from bvhtree.tree import AABBTree, AABBTreeDisplay


# Define main function
def main():

    # Load the mesh
    mesh_size = 20
    mesh = examples.action_figure(size=mesh_size)

    # Create a BVH class
    aabb_tree = AABBTree.from_triangle_mesh(mesh, depth_lim=16, split_lim=10)

    # Print info
    print(f'NUM_MESH_FACES = {mesh.num_faces}')
    print(f'NUM_MESH_VERTICES = {mesh.num_vertices}')
    print(f'NUM_TREE_NODES = {len(aabb_tree.nodes)}')

    # Show the tree
    AABBTreeDisplay(aabb_tree).show(cmap='hsv')


# Call main function
if __name__ == '__main__':
    main()
