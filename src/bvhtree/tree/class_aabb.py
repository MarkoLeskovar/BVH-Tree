import numba
import numpy as np

__all__ = ['AABB', 'axis_aligned_bounding_box']


'''
O------------------------------------------------------------------------------O
| CLASS - AXIS ALIGNED BOUNDING BOX                                            |
O------------------------------------------------------------------------------O
'''


class AABB:

    def __init__(self, points: np.ndarray):
        self._n_dim = points.shape[1]
        self._min, self._max = axis_aligned_bounding_box(points)

    @property
    def min(self) -> np.ndarray:
        return self._min

    @property
    def max(self) -> np.ndarray:
        return self._max

    @property
    def n_dim(self) -> int:
        return self._n_dim

    def diagonal(self) -> float:
        return float(np.linalg.norm(self._max - self._min))

    def size(self) -> np.ndarray:
        return self._max - self._min

    def center(self) -> np.ndarray:
        size = self._max - self._min
        return self._min + 0.5 * size

    def inflate(self, delta: float):
        self._min -= 0.5 * delta
        self._max += 0.5 * delta

    def largest_axis(self) -> int:
        size = self._max - self._min
        return int(np.argmax(size))

    def grow(self, points: np.ndarray):
        # Check shape
        if points.shape[1] != self._n_dim:
            raise ValueError('Dimension of input points does not match the AABB!')
        # Grow the box
        for i in range(self._n_dim):
            self._min[i] = np.minimum(self._min[i], np.min(points[:, i]))
            self._max[i] = np.maximum(self._max[i], np.max(points[:, i]))

    def copy(self):
        obj = type(self).__new__(self.__class__)
        obj._n_dim = self._n_dim
        obj._min = self._min.copy()
        obj._max = self._max.copy()
        return obj


'''
O------------------------------------------------------------------------------O
| PUBLIC - AUXILIARY FUNCTIONS                                                 |
O------------------------------------------------------------------------------O
'''


@numba.njit(cache=True)
def axis_aligned_bounding_box(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ndim = points.shape[1]
    box_min = np.zeros(shape=ndim, dtype='float')
    box_max = np.zeros(shape=ndim, dtype='float')
    # Loop over all dimensions
    for i in range(ndim):
        box_min[i] = points[:, i].min()
        box_max[i] = points[:, i].max()
    # Return results
    return box_min, box_max
