"""
Adapted from pymorton (https://github.com/trevorprater/pymorton)
"""
import numba
import numpy as np

MAX_EPS_32 = 2**(32 // 3)
MAX_EPS_64 = 2**(64 // 3)
MAX_BIT_WIDTH_32 = 3 * (32 // 3)
MAX_BIT_WIDTH_64 = 3 * (64 // 3)


def morton_encode(point: np.ndarray) -> np.uint:
    if point.dtype == np.uint32:
        return morton_encode_uint32(point)
    elif point.dtype == np.uint64:
        return morton_encode_uint64(point)
    else:
        raise ValueError(f'Data type must be "uint32" or "uint64"!')


def morton_decode(n: np.uint) -> np.ndarray:
    if n.dtype == np.uint32:
        return morton_decode_uint32(n)
    elif n.dtype == np.uint64:
        return morton_decode_uint64(n)
    else:
        raise ValueError(f'Data must be "uint32" or "uint64"!')


@numba.njit(cache=True)
def morton_encode_uint32(point: np.ndarray) -> np.uint:
    return _part_uint32(point[0]) | (_part_uint32(point[1]) << 1) | (_part_uint32(point[2]) << 2)


@numba.njit(cache=True)
def morton_decode_uint32(n: np.uint) -> np.ndarray:
    return np.asarray([_unpart_uint32(n), _unpart_uint32(n >> 1), _unpart_uint32(n >> 2)])


@numba.njit(cache=True)
def morton_encode_uint64(point: np.ndarray) -> np.uint:
    return _part_uint64(point[0]) | (_part_uint64(point[1]) << 1) | (_part_uint64(point[2]) << 2)


@numba.njit(cache=True)
def morton_decode_uint64(n: np.uint) -> np.ndarray:
    return np.asarray([_unpart_uint64(n), _unpart_uint64(n >> 1), _unpart_uint64(n >> 2)])


@numba.njit(cache=True)
def _part_uint32(n):
    n &= 0x000003ff                   # base10: 1023,       binary: 1111111111,                       len: 10
    n = (n ^ (n << 16)) & 0xff0000ff  # base10: 4278190335, binary: 11111111000000000000000011111111, len: 32
    n = (n ^ (n << 8))  & 0x0300f00f  # base10: 50393103,   binary: 11000000001111000000001111,       len: 26
    n = (n ^ (n << 4))  & 0x030c30c3  # base10: 51130563,   binary: 11000011000011000011000011,       len: 26
    n = (n ^ (n << 2))  & 0x09249249  # base10: 153391689,  binary: 1001001001001001001001001001,     len: 28
    return n


@numba.njit(cache=True)
def _unpart_uint32(n):
    n &= 0x09249249                   # base10: 153391689,  binary: 1001001001001001001001001001,     len: 28
    n = (n ^ (n >> 2))  & 0x030c30c3  # base10: 51130563,   binary: 11000011000011000011000011,       len: 26
    n = (n ^ (n >> 4))  & 0x0300f00f  # base10: 50393103,   binary: 11000000001111000000001111,       len: 26
    n = (n ^ (n >> 8))  & 0xff0000ff  # base10: 4278190335, binary: 11111111000000000000000011111111, len: 32
    n = (n ^ (n >> 16)) & 0x000003ff  # base10: 1023,       binary: 1111111111,                       len: 10
    return n


@numba.njit(cache=True)
def _part_uint64(n):
    n &= 0x1fffff                             # binary: 111111111111111111111,                                         len: 21
    n = (n | (n << 32)) & 0x1f00000000ffff    # binary: 11111000000000000000000000000000000001111111111111111,         len: 53
    n = (n | (n << 16)) & 0x1f0000ff0000ff    # binary: 11111000000000000000011111111000000000000000011111111,         len: 53
    n = (n | (n << 8))  & 0x100f00f00f00f00f  # binary: 1000000001111000000001111000000001111000000001111000000001111, len: 61
    n = (n | (n << 4))  & 0x10c30c30c30c30c3  # binary: 1000011000011000011000011000011000011000011000011000011000011, len: 61
    n = (n | (n << 2))  & 0x1249249249249249  # binary: 1001001001001001001001001001001001001001001001001001001001001, len: 61
    return n


@numba.njit(cache=True)
def _unpart_uint64(n):
    n &= 0x1249249249249249                   # binary: 1001001001001001001001001001001001001001001001001001001001001, len: 61
    n = (n ^ (n >> 2))  & 0x10c30c30c30c30c3  # binary: 1000011000011000011000011000011000011000011000011000011000011, len: 61
    n = (n ^ (n >> 4))  & 0x100f00f00f00f00f  # binary: 1000000001111000000001111000000001111000000001111000000001111, len: 61
    n = (n ^ (n >> 8))  & 0x1f0000ff0000ff    # binary: 11111000000000000000011111111000000000000000011111111,         len: 53
    n = (n ^ (n >> 16)) & 0x1f00000000ffff    # binary: 11111000000000000000000000000000000001111111111111111,         len: 53
    n = (n ^ (n >> 32)) & 0x1fffff            # binary: 111111111111111111111,                                         len: 21
    return n
