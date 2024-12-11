import numba
import numpy as np
import pyvista as pv

import bvhtree.mesh.examples as examples
from bvhtree.mesh.utils import compute_triangle_centers
from morton_codes import morton_encode, morton_decode, MAX_EPS_32, MAX_EPS_64

def _clz(n: np.uint) -> int:
    n_bits = n.nbytes * 8
    return n_bits - int(n).bit_length()


def find_split(sorted_morton_nodes, first: int, last: int):

    # Get first and last code
    first_code = sorted_morton_nodes[first]
    last_code = sorted_morton_nodes[last]

    # Identical Morton codes => split the range in the middle
    if first_code == last_code:
        return (first + last) >> 1

    # Calculate the number of highest bits that are the same for all objects
    common_prefix = _clz(first_code ^ last_code)


    pass


# Define main function
def main():

    # Create a mesh
    mesh = examples.armadillo(size=10)

    # Compute face center
    centers = compute_triangle_centers(mesh.faces, mesh.vertices)

    # Face centers data
    centers_min = np.min(centers, axis=0)
    centers_max = np.max(centers, axis=0)
    center_size = centers_max - centers_min

    # Normalize the points to 0...1
    normalized_centers = (centers - centers_min) / center_size

    # Quantize the points
    quantized_centers = (normalized_centers * (MAX_EPS_32 - 1)).astype('uint32')

    # Interleave morton codes
    morton_codes = np.zeros(shape=centers.shape[0], dtype='uint32')
    for i in range(centers.shape[0]):
        morton_codes[i] = morton_encode(quantized_centers[i, :])

    # Sort the codes
    morton_sort_ids = np.argsort(morton_codes)

    # Sort morton codes
    sorted_morton_codes = morton_codes[morton_sort_ids]

    # # Split the morton codes
    # split_id = find_split(sorted_morton_codes, 0, sorted_morton_codes.size-1)


    # Create pyvista points
    sorted_centers = centers[morton_sort_ids, :]
    pyvista_points = pv.PolyData(sorted_centers.astype('float'))
    pyvista_points['Index'] = np.arange(centers.shape[0])

    # Create pyvista mesh
    sorted_mesh = mesh.copy()
    sorted_mesh._faces = sorted_mesh.faces[morton_sort_ids]
    pyvista_mesh = sorted_mesh.to_pyvista_grid()
    pyvista_mesh['Index'] = np.arange(sorted_mesh.num_faces)

    # Show the mesh
    cmap = 'jet'
    pl = pv.Plotter()
    # pl.add_mesh(pyvista_points, render_points_as_spheres=True, point_size=10, scalars='Index', cmap=cmap,  show_scalar_bar=False)
    pl.add_mesh(pyvista_mesh, scalars='Index', cmap=cmap, show_scalar_bar=False)
    pl.show()


# Call main function
if __name__ == '__main__':
    main()
