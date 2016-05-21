# -*- coding: utf-8 -*-
"""Represents a 3x3 Matrix.

The Matrix3 class provides a number of convenient functions and
conversions.
::

    import numpy as np
    from math3 import Quaternion, Matrix3, Matrix4, Vector3

    m = Matrix3()
    m = Matrix3([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])

    # copy constructor
    m = Matrix4(Matrix4())

    # explicit creation
    m = Matrix3.identity()
    m = Matrix3.from_matrix44(Matrix4())

    # inferred conversions
    m = Matrix3(Quaternion())
    m = Matrix3(Matrix4())

    # multiply matricies together
    m = Matrix3() * Matrix3()
    m = Matrix3() * Matrix4()

    # extract a quaternion from a matrix
    q = m.quaternion

    # convert from quaternion back to matrix
    m = q.matrix33
    m = Matrix3(q)

    # rotate a matrix by a quaternion
    m = Matrix3.identity() * Quaternion()

    # rotate a vector 3 by a matrix
    v = Matrix3.from_x_rotation(np.pi) * Vector3([1.,2.,3.])

    # undo a rotation
    m = Matrix3.from_x_rotation(np.pi)
    v = m * Vector3([1.,1.,1.])
    # ~m is the same as m.inverse
    v = ~m * v

    # access specific parts of the matrix
    # first row
    m1 = m.m1
    # first element, first row
    m11 = m.m11
    # third element, third row
    m33 = m.m33
    # first row, same as m1
    r1 = m.r1
    # first column
    c1 = m.c1
"""
from __future__ import absolute_import
from numbers import Number
import numpy as np
from multipledispatch import dispatch
from .base import BaseObject, BaseMatrix, BaseMatrix3, BaseQuaternion, \
    BaseVector, NpProxy
from .. import matrix33


class Matrix3(BaseMatrix3):
    _module = matrix33
    _shape = (3, 3,)

    # m<c> style access
    #: The first row of this Matrix as a numpy.ndarray.
    m1 = NpProxy(0)
    #: The second row of this Matrix as a numpy.ndarray.
    m2 = NpProxy(1)
    #: The third row of this Matrix as a numpy.ndarray.
    m3 = NpProxy(2)

    # m<r><c> access
    #: The [0,0] value of this Matrix.
    m11 = NpProxy((0, 0))
    #: The [0,1] value of this Matrix.
    m12 = NpProxy((0, 1))
    #: The [0,2] value of this Matrix.
    m13 = NpProxy((0, 2))
    #: The [1,0] value of this Matrix.
    m21 = NpProxy((1, 0))
    #: The [1,1] value of this Matrix.
    m22 = NpProxy((1, 1))
    #: The [1,2] value of this Matrix.
    m23 = NpProxy((1, 2))
    #: The [2,0] value of this Matrix.
    m31 = NpProxy((2, 0))
    #: The [2,1] value of this Matrix.
    m32 = NpProxy((2, 1))
    #: The [2,2] value of this Matrix.
    m33 = NpProxy((2, 2))

    # rows
    #: The first row of this Matrix as a numpy.ndarray. This is the same as m1.
    r1 = NpProxy(0)
    #: The second row of this Matrix as a numpy.ndarray. This is the same as m2.
    r2 = NpProxy(1)
    #: The third row of this Matrix as a numpy.ndarray. This is the same as m3.
    r3 = NpProxy(2)

    # columns
    #: The first column of this Matrix as a numpy.ndarray.
    c1 = NpProxy((slice(0, 3), 0))
    #: The second column of this Matrix as a numpy.ndarray.
    c2 = NpProxy((slice(0, 3), 1))
    #: The third column of this Matrix as a numpy.ndarray.
    c3 = NpProxy((slice(0, 3), 2))

    ########################
    # Creation
    @classmethod
    def from_matrix44(cls, matrix, dtype=None):
        """Creates a Matrix3 from a Matrix4.

        The Matrix4 translation will be lost.
        """
        return cls(matrix33.create_from_matrix44(matrix, dtype))

    def __new__(cls, value=None, dtype=None):
        if value is not None:
            obj = value
            if not isinstance(value, np.ndarray):
                obj = np.array(value, dtype=dtype)

            # matrix44
            if obj.shape == (4, 4) or isinstance(obj, Matrix4):
                obj = matrix33.create_from_matrix44(obj, dtype=dtype)
            # quaternion
            elif obj.shape == (4,) or isinstance(obj, Quaternion):
                obj = matrix33.create_from_quaternion(obj, dtype=dtype)
        else:
            obj = np.zeros(cls._shape, dtype=dtype)
        obj = obj.view(cls)
        return super(Matrix3, cls).__new__(cls, obj)

    ########################
    # Basic Operators

    @dispatch(BaseObject)
    def __add__(self, other):
        self._unsupported_type('add', other)

    @dispatch(BaseObject)
    def __sub__(self, other):
        self._unsupported_type('subtract', other)

    @dispatch(BaseObject)
    def __mul__(self, other):
        self._unsupported_type('multiply', other)

    @dispatch(BaseObject)
    def __truediv__(self, other):
        self._unsupported_type('divide', other)

    @dispatch(BaseObject)
    def __div__(self, other):
        self._unsupported_type('divide', other)

    def __invert__(self):
        return self.inverse

    ########################
    # Matrices
    @dispatch((BaseMatrix, np.ndarray, list))
    def __add__(self, other):
        return Matrix3(super(Matrix3, self).__add__(Matrix3(other)))

    @dispatch((BaseMatrix, np.ndarray, list))
    def __sub__(self, other):
        return Matrix3(super(Matrix3, self).__sub__(Matrix3(other)))

    @dispatch((BaseMatrix, np.ndarray, list))
    def __mul__(self, other):
        return Matrix3(matrix33.multiply(self, Matrix3(other)))

    ########################
    # Quaternions
    @dispatch(BaseQuaternion)
    def __mul__(self, other):
        m = other.matrix33
        return self * m

    ########################
    # Vectors
    @dispatch(BaseVector)
    def __mul__(self, other):
        return type(other)(matrix33.apply_to_vector(self, other))

    ########################
    # Number
    @dispatch((Number, np.number))
    def __add__(self, other):
        return Matrix3(super(Matrix3, self).__add__(other))

    @dispatch((Number, np.number))
    def __sub__(self, other):
        return Matrix3(super(Matrix3, self).__sub__(other))

    @dispatch((Number, np.number))
    def __mul__(self, other):
        return Matrix3(super(Matrix3, self).__mul__(other))

    @dispatch((Number, np.number))
    def __truediv__(self, other):
        return Matrix3(super(Matrix3, self).__truediv__(other))

    @dispatch((Number, np.number))
    def __div__(self, other):
        return Matrix3(super(Matrix3, self).__div__(other))

    ########################
    # Methods and Properties
    @property
    def matrix33(self):
        """Returns the Matrix3.

        This can be handy if you're not sure what type of Matrix class you have
        but require a Matrix3.
        """
        return self

    @property
    def matrix44(self):
        """Returns a Matrix4 representing this matrix.
        """
        return Matrix4(self)

    @property
    def quaternion(self):
        """Returns a Quaternion representing this matrix.
        """
        return Quaternion(self)


from .matrix4 import Matrix4
from .quaternion import Quaternion
