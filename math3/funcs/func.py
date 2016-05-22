# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals
from ..objects import Vector3
import numpy as np


def normal(v1, v2, v3, normalise=True):
    """
    Generates a normal vector for 3 vertices.

    The result is a normalised vector.

    It is assumed the ordering is counter-clockwise starting
    at v1, v2 then v3::

        v1      v3
          \    /
            v2

    The vertices are Nd arrays and may be 1d or Nd.
    As long as the final axis is of size 3.

    For 1d arrays::
        >>> v1 = numpy.array( [ 1.0, 0.0, 0.0 ] )
        >>> v2 = numpy.array( [ 0.0, 0.0, 0.0 ] )
        >>> v3 = numpy.array( [ 0.0, 1.0, 0.0 ] )
        >>> vector.generate_normals( v1, v2, v3 )
        array([ 0.,  0., -1.])

    For Nd arrays::
        >>> v1 = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 1.0, 0.0, 0.0 ] ] )
        >>> v2 = numpy.array( [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ] )
        >>> v3 = numpy.array( [ [ 0.0, 1.0, 0.0 ], [ 0.0, 1.0, 0.0 ] ] )
        >>> vector.generate_normals( v1, v2, v3 )
        array([[ 0.,  0., -1.],
               [ 0.,  0., -1.]])

    :param numpy.array v1: an Nd array with the final dimension
        being size 3. (a vector)
    :param numpy.array v2: an Nd array with the final dimension
        being size 3. (a vector)
    :param numpy.array v3: an Nd array with the final dimension
        being size 3. (a vector)
    :param boolean normalise_result: Specifies if the result should
        be normalised before being returned.
    """
    # make vectors relative to v2
    # we assume opengl counter-clockwise ordering
    a = v1 - v2
    b = v3 - v2
    n = Vector3(np.cross(b, a))
    if normalise:
        n.normalize()
    return n


def cross(v1, v2):
    """Calculates the cross-product of two vectors.

    :param numpy.array v1: an Nd array with the final dimension
        being size 3. (a vector)
    :param numpy.array v2: an Nd array with the final dimension
        being size 3. (a vector)
    :rtype: A np.array with shape v1.shape.
    """
    return np.cross(v1, v2)


def matrix3_view(mat):
    """
    Returns a view into the matrix in Matrix3 format.

    This is different from Matrix3.from_matrix4, in that
    changes to the returned matrix will also alter the original matrix.

    :rtype: numpy.array
    :return: A view into the matrix in the format of a matrix3 (shape (3,3)).
    """
    return mat[0:3, 0:3]

