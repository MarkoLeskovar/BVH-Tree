import aabbtree.mesh.examples as examples
from aabbtree import AABBTree, AABBTreeDisplay


# Define main function
def main():

    # Load the mesh
    mesh_size = 20
    mesh = examples.action_figure(size=mesh_size)

    # Create a BVH class
    aabb_tree = AABBTree.from_surface_mesh(mesh, depth_lim=16, split_lim=10)

    # Show the tree
    AABBTreeDisplay(aabb_tree).show(cmap='hsv')


# Call main function
if __name__ == '__main__':
    main()
