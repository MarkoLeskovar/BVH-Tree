import os
import numpy as np
import pyvista as pv

# Global variables
_FILE_FORMAT = '.sm.vtk'


'''
O------------------------------------------------------------------------------O
| CLASS - TRIANGULAR SURFACE MESH                                              |
O------------------------------------------------------------------------------O
'''

class SurfaceMesh:
    """
    Surface mesh consisting of triangular surface elements.
    """
    def __init__(self, elements: np.ndarray, vertices: np.ndarray):
        elements = np.asarray(elements)
        vertices = np.asarray(vertices)
        # Check input
        _check_faces(elements)
        _check_vertices(vertices)
        # Assign values
        self._faces = elements
        self._vertices = vertices

    @classmethod
    def from_arrays(cls, elements: np.ndarray, vertices: np.ndarray):
        """
        Create a class instance from arrays.

        :param elements: An [N,3] array of elements defined by their node indices.
        :param vertices: An [M,3] array of node coordinates of the mesh.
        :returns: A new instance of ShapeModel.
        """
        return cls(elements, vertices)

    @classmethod
    def from_pyvista_grid(cls, pyvista_grid: pv.core.pointset.PolyData):
        """
        Create a class instance from PyVista PolyData.

        :param pyvista_grid: PyVista PolyData of triangular surface elements.
        :returns: A new instance of ShapeModel.
        """
        # Check input type
        if not isinstance(pyvista_grid, pv.core.pointset.PolyData):
            raise ValueError(f'Wrong type! Mesh is not an instance of {pv.core.pointset.PolyData}!')
        # Get the data
        elements = np.asarray(pyvista_grid.faces).reshape((-1, 4))[:, 1:4]
        vertices = pyvista_grid.points
        # Return a new instance
        return cls(elements, vertices)

    @classmethod
    def from_file(cls, file: str):
        """
        Create a class instance from a ".sm.vtk" file.

        :param file: The path to the ".sm.vtk" file
        :returns: A new instance of ShapeModel.
        """
        file_format = os.path.splitext(os.path.splitext(file)[0])[1] + os.path.splitext(file)[1]
        if file_format != _FILE_FORMAT:
            raise ValueError(f'Wrong format! Please provide a "{_FILE_FORMAT}" file!')
        # Read the data from the disk
        pyvista_grid = pv.read(file)
        # Return a new instance
        return cls.from_pyvista_grid(pyvista_grid)

    def to_pyvista_grid(self) -> pv.PolyData:
        """
        Convert the class to a PyVista PolyData.

        :returns: An instance of PyVista PolyData.
        """
        padding = np.ones(shape=(self._faces.shape[0], 1), dtype='int') * 3
        faces = np.hstack((padding, self._faces))
        pyvista_grid = pv.PolyData(self._vertices, faces=faces)
        return pyvista_grid

    def save(self, file: str):
        """
        Save the model as a ".sm.vtk" file.

        :param file: The path to a ".sm.vtk" file.
        """
        # Check input type
        if not file.endswith(_FILE_FORMAT):
            raise ValueError(f'Wrong format! Please provide a "{_FILE_FORMAT}" file!')
        # Create output data and save it
        pyvista_grid = self.to_pyvista_grid()
        pyvista_grid.save(file)

    def show(self, **kwargs):
        """
        Visualize the model using PyVista Plotter.
        """
        if 'color' not in kwargs:
            kwargs['color'] = 'lightblue'
        pl = pv.Plotter()
        pl.add_axes()
        pl.add_mesh(self.to_pyvista_grid(), color='lightblue')
        pl.show()

    def copy(self):
        """
        Create a copy of the class.

        :returns: A copy of the ShapeModel instance.
        """
        obj = type(self).__new__(self.__class__)
        obj._faces = self._faces.copy()
        obj._vertices = self._vertices.copy()
        return obj

    @property
    def num_faces(self) -> int:
        """
        Get number of mesh faces.

        :returns: Number of mesh faces.
        """
        return self._faces.shape[0]

    @property
    def num_vertices(self) -> int:
        """
        Get number of mesh vertices.

        :returns: Number of mesh vertices.
        """
        return self._vertices.shape[0]

    @property
    def vertices(self) -> np.ndarray:
        """
        Get mesh vertices.

        :returns: An [M,3] array of mesh vertices.
        """
        return self._vertices

    @vertices.setter
    def vertices(self, vertices: np.ndarray):
        """
        Set mesh vertices.

        :vertices: An [M,3] array of mesh vertices.
        """
        vertices = np.asarray(vertices)
        if self._vertices.shape != vertices.shape:
            raise ValueError('Shape of vertices does not match the existing vertices!')
        self._vertices = vertices

    @property
    def faces(self) -> np.ndarray:
        """
        Get mesh faces.

        :returns: An [N,3] array of triangular mesh faces.
        """
        return self._faces


'''
O------------------------------------------------------------------------------O
| AUXILIARY FUNCTIONS                                                          |
O------------------------------------------------------------------------------O
'''

def _check_faces(faces):
    if faces.shape[1] != 3:
        raise ValueError('Wrong shape! Faces are not triangles!')
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError('Wrong type! Faces are not integer values!')

def _check_vertices(vertices):
    if vertices.shape[1] != 3:
        raise ValueError('Wrong shape! Vertices are not three-dimensional!')
