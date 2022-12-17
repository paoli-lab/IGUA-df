# coding: utf-8
# cython: language_level=3, linetrace=True

"""A fast implementation pairwise Manhattan distances of CSR matrices.

The original code from ``scikit-learn`` was adapted to fill a condensed
distance vector rather than the full matrix to save additional space. 

"""

from cython cimport floating, integral
from cython.parallel cimport prange
from libc.math cimport fabs, ceil, sqrt


def sparse_manhattan(integral[:] data, int[:] indices, int[:] indptr, floating[:] D, int threads = 1):
    """Pairwise L1 distances for a CSR matrix.
    """
    cdef ssize_t px, py, i, j, k, ix, iy
    
    cdef floating d = 0.0
    cdef int n = <int> ceil(sqrt(D.shape[0] * 2))
    cdef int X_indptr_end = 0
    cdef int Y_indptr_end = 0

    for px in prange(n-1, nogil=True, num_threads=threads):
        X_indptr_end = indptr[px + 1]
        for py in range(px+1, n):
            Y_indptr_end = indptr[py + 1]
            i = indptr[px]
            j = indptr[py]
            d = 0.0
            while i < X_indptr_end and j < Y_indptr_end:
                ix = indices[i]
                iy = indices[j]

                if ix == iy:
                    d = d + fabs(data[i] - data[j])
                    i = i + 1
                    j = j + 1
                elif ix < iy:
                    d = d + fabs(data[i])
                    i = i + 1
                else:
                    d = d + fabs(data[j])
                    j = j + 1

            if i == X_indptr_end:
                while j < Y_indptr_end:
                    d = d + fabs(data[j])
                    j = j + 1
            else:
                while i < X_indptr_end:
                    d = d + fabs(data[i])
                    i = i + 1

            k = n*px - px*(px+1)//2 + py - 1 - px
            D[k] = d