import numba
import numpy as np
import pyvista as pv

from morton_codes import morton_encode, morton_decode, MAX_EPS_32, MAX_EPS_64
import bvhtree.mesh.examples as examples


@numba.njit(cache=True)
def regular_point_lattice(num_points=(5, 5, 5), spacing=(1, 1, 1)) -> np.ndarray:
    num_points = np.asarray(num_points).astype('int')
    spacing = np.asarray(spacing).astype('float')
    # Create a list of points
    num_points_all = num_points[0] * num_points[1] * num_points[2]
    points = np.zeros((num_points_all, 3), dtype='float')
    counter = 0
    for i in range(num_points[0]):
        for j in range(num_points[1]):
            for k in range(num_points[2]):
                points[counter, :] = np.asarray([i * spacing[0], j * spacing[1], k * spacing[2]])
                counter += 1
    # Return results
    return points


# Define main function
def main():

    # Create points (lattice)
    lattice_size = 4
    points = regular_point_lattice(num_points=(lattice_size, lattice_size, lattice_size), spacing=(1, 1, 1))
    points -= np.mean(points, axis=0)

    # Points data
    points_min = np.min(points, axis=0)
    points_max = np.max(points, axis=0)
    points_size = points_max - points_min

    # Normalize the points to 0...1
    normalized_points = (points - points_min) / points_size

    # Quantize the points
    quantized_points_in = (normalized_points * (MAX_EPS_32 - 1)).astype('uint32')

    # Interleave morton codes
    morton_codes = np.zeros(shape=points.shape[0], dtype='uint32')
    for i in range(points.shape[0]):
        morton_codes[i] = morton_encode(quantized_points_in[i, :])

    # Sort the codes
    morton_codes = np.sort(morton_codes)

    # De-interleave morton codes
    quantized_points_out = np.zeros_like(quantized_points_in)
    for i in range(points.shape[0]):
        quantized_points_out[i, :] = morton_decode(morton_codes[i])

    # Create pyvista points
    pyvista_points = pv.PolyData(quantized_points_out.astype('float'))
    pyvista_points['Index'] = np.arange(quantized_points_out.shape[0])

    # Create pyvista lines
    temp_lines = np.repeat(np.arange(quantized_points_out.shape[0]), 2)[1:-1].reshape(-1, 2)
    temp_lines = np.hstack((np.full(shape=(temp_lines.shape[0], 1), fill_value=2, dtype='int'), temp_lines))
    pyvista_lines = pv.PolyData(quantized_points_out.astype('float'), lines=temp_lines)
    pyvista_lines['Index'] = np.arange(temp_lines.shape[0])

    # Initialize the plotter
    pl = pv.Plotter()

    # Add points
    cmap = 'jet'
    pl.add_mesh(pyvista_points, render_points_as_spheres=True, point_size=10, scalars='Index', cmap=cmap,  show_scalar_bar=False)
    pl.add_mesh(pyvista_lines, scalars='Index', line_width=3, cmap=cmap, show_scalar_bar=False)

    # Show everything
    pl.show()


# Call main function
if __name__ == '__main__':
    main()
