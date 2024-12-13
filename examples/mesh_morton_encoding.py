import math
import time

import numba
import numpy as np
import pyvista as pv

import bvhtree.mesh.examples as examples
from bvhtree.mesh.utils import compute_triangle_centers
from morton_codes import morton_encode, morton_decode, MAX_EPS_32, MAX_EPS_64, MAX_BIT_WIDTH_32


@numba.njit(cache=True)
def _bit_count_leading_zeros(n: np.uint, n_bits: int) -> int:
    return n_bits - _bit_length(n)


@numba.njit(cache=True)
def _bit_length(n: np.uint) -> int:
    return int(math.ceil(math.log2(n+1)))


# @numba.njit(cache=True)
def _split_morton(sorted_morton_nodes, first_id: int, last_id: int):

    # Get first and last code
    first_code = sorted_morton_nodes[first_id]
    last_code = sorted_morton_nodes[last_id]

    # Identical Morton codes => split the range in the middle
    if first_code == last_code:
        return (first_id + last_id) >> 1

    # Calculate the number of highest bits that are the same for all objects
    common_prefix = _bit_count_leading_zeros(first_code ^ last_code, 32)

    # Binary search for the split position
    split_id = first_id
    step = last_id - first_id
    while step > 1:
        step = (step + 1) >> 1  # Exponential decrease
        new_split = split_id + step  # Proposed new position
        if new_split < last_id:
            split_code = sorted_morton_nodes[new_split]
            split_prefix = _bit_count_leading_zeros(first_code ^ split_code, 32)
            if split_prefix > common_prefix:
                split_id = new_split  # Accept proposal

    # Return results
    return split_id


@numba.njit(cache=True)
def _binary_radix_sort(array, n_bits: int):
    sorted_array = array.copy()

    # Initialize binary buckets
    bucket_a = np.zeros_like(array)
    bucket_b = np.zeros_like(array)

    # Loop over all bits
    for i in range(n_bits):
        counter_a = 0
        counter_b = 0

        # Split into buckets
        for j in range(sorted_array.size):
            if _check_bit(sorted_array[j], i):
                bucket_b[counter_b] = sorted_array[j]
                counter_b += 1
            else:
                bucket_a[counter_a] = sorted_array[j]
                counter_a += 1

        # Merge the buckets
        counter = 0
        for j in range(counter_a):
            sorted_array[counter] = bucket_a[j]
            counter += 1
        for j in range(counter_b):
            sorted_array[counter] = bucket_b[j]
            counter += 1

    # Return results
    return sorted_array


@numba.njit(cache=True)
def _check_bit(n: np.uint, bit: int) -> bool:
    return bool(n >> bit & 1)



# Define main function
def main():

    # Create a mesh
    mesh = examples.nefertiti(size=10)

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

    t0 = time.time()

    # Sort morton codes
    morton_sort_ids = np.argsort(morton_codes)
    sorted_morton_codes = morton_codes[morton_sort_ids]

    t1 = time.time()

    # Binary radix sort
    sorted_morton_codes_NEW = _binary_radix_sort(morton_codes, MAX_BIT_WIDTH_32)

    t2 = time.time()

    # DEBUG information
    print(f'Numpy sort -> t = {t1 - t0} s')
    print(f'Radix sort -> t = {t2 - t1} s')
    print(f'Are equal = {np.sum(sorted_morton_codes != sorted_morton_codes_NEW) == 0}')

    # Convert to binary representation strings
    binary_repr = np.zeros(shape=morton_codes.shape[0], dtype='<U32')
    for i in range(binary_repr.shape[0]):
        binary_repr[i] = np.binary_repr(sorted_morton_codes[i], 32)

    # Split the morton codes
    split_id = _split_morton(sorted_morton_codes, 0, sorted_morton_codes.size - 1) + 1

    # Assign colors
    colors = np.zeros(shape=centers.shape[0], dtype='int')
    colors[:split_id] = 1
    colors[split_id:] = 2

    # Create pyvista points
    sorted_centers = centers[morton_sort_ids, :]
    pyvista_points = pv.PolyData(sorted_centers.astype('float'))
    # pyvista_points['Index'] = np.arange(centers.shape[0])
    pyvista_points['Index'] = colors

    # Create pyvista mesh
    sorted_mesh = mesh.copy()
    sorted_mesh._faces = sorted_mesh.faces[morton_sort_ids]
    pyvista_mesh = sorted_mesh.to_pyvista_grid()
    pyvista_mesh['Index'] = np.arange(sorted_mesh.num_faces)

    # Show the mesh
    cmap = 'jet'
    pl = pv.Plotter()
    # pl.add_mesh(pyvista_points, render_points_as_spheres=True, point_size=10, scalars='Index', cmap=cmap,  show_scalar_bar=False)
    pl.add_mesh(pyvista_points, render_points_as_spheres=True, point_size=10, scalars='Index', cmap='plasma',  show_scalar_bar=False)
    pl.add_mesh(pyvista_mesh, scalars='Index', cmap=cmap, show_scalar_bar=False)
    pl.show()


# Call main function
if __name__ == '__main__':
    main()
