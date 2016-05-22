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
    m = Matrix3.from_matrix4(Matrix4())

    # inferred conversions
    m = Matrix3(Quaternion())
    m = Matrix3(Matrix4())

    # multiply matricies together
    m = Matrix3() * Matrix3()
    m = Matrix3() * Matrix4()

    # extract a quaternion from a matrix
    q = m.quaternion

    # convert from quaternion back to matrix
    m = q.mat3
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
from .basecls import BaseObject, BaseMatrix, BaseMatrix3, BaseQuaternion, \
    BaseVector, NpProxy
from .quaternion import Quaternion
from .matrix4 import Matrix4
from ..utils import parameters_as_numpy_arrays


class Matrix3(BaseMatrix3):

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
    def from_matrix4(cls, matrix, dtype=None):
        """Creates a Matrix3 from a Matrix4.

        The Matrix4 translation will be lost.
        """
        return cls(np.array(matrix[0:3, 0:3], dtype=dtype))

    def __new__(cls, value=None, dtype=None):
        if value is not None:
            obj = value
            if not isinstance(value, np.ndarray):
                obj = np.array(value, dtype=dtype)

            # matrix4
            if obj.shape == (4, 4) or isinstance(obj, Matrix4):
                obj = cls.create_from_matrix4(obj, dtype=dtype)
            # quaternion
            elif obj.shape == (4,) or isinstance(obj, Quaternion):
                obj = cls.from_quaternion(obj, dtype=dtype)
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
        return Matrix3(self.multiply(self, Matrix3(other)))

    ########################
    # Quaternions
    @dispatch(BaseQuaternion)
    def __mul__(self, other):
        m = other.mat3
        return self * m

    ########################
    # Vectors
    @dispatch(BaseVector)
    def __mul__(self, other):
        return type(other)(self.apply_to_vector(self, other))

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
    def mat3(self):
        """Returns the Matrix3.

        This can be handy if you're not sure what type of Matrix class you have
        but require a Matrix3.
        """
        return self

    @property
    def mat4(self):
        """Returns a Matrix4 representing this matrix.
        """
        return Matrix4(self)

    @property
    def quat(self):
        """Returns a Quaternion representing this matrix.
        """
        return Quaternion(self)

    @property
    def inverse(self):
        """
        Returns the inverse of the matrix.

        This is essentially a wrapper around numpy.linalg.inv.

        :param numpy.array m: A matrix.
        :rtype: numpy.array
        :return: The inverse of the specified matrix.

        .. seealso:: http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.inv.html
        """
        return np.linalg.inv(self)

    @classmethod
    def identity(cls, dtype=None):
        """
        Creates an identity Matrix.
        """
        return cls(np.identity(3, dtype=dtype))

    # Class methods
    @classmethod
    def from_euler(cls, euler, dtype=None):
        """
        Creates a matrix from the specified Euler rotations.

        :param numpy.array euler: A set of euler rotations in the format
            specified by the euler modules.
        :rtype: numpy.array
        :return: A matrix with shape (3,3) with the euler's rotation.
        """
        dtype = dtype or euler.dtype

        pitch, roll, yaw = euler.pitch(euler), euler.roll(euler), euler.yaw(
                euler)

        sP = np.sin(pitch)
        cP = np.cos(pitch)
        sR = np.sin(roll)
        cR = np.cos(roll)
        sY = np.sin(yaw)
        cY = np.cos(yaw)

        mat = np.array(
                [
                    # m1
                    [
                        cY * cP,
                        -cY * sP * cR + sY * sR,
                        cY * sP * sR + sY * cR,
                    ],
                    # m2
                    [
                        sP,
                        cP * cR,
                        -cP * sR,
                    ],
                    # m3
                    [
                        -sY * cP,
                        sY * sP * cR + cY * sR,
                        -sY * sP * sR + cY * cR,
                    ]
                ],
                dtype=dtype
        )

        return cls(mat)

    @classmethod
    def from_quaternion(cls, quaternion, dtype=None):
        """Creates a matrix with the same rotation as a quaternion.

        :param quaternion: The quaternion to create the matrix from.
        :rtype: numpy.array
        :return: A matrix with shape (3,3) with the quaternion's rotation.
        """
        dtype = dtype or quaternion.dtype
        # the quaternion must be normalised
        if not np.isclose(np.linalg.norm(quaternion), 1.):
            quaternion = fquat.normalise(quaternion)

        x, y, z, w = quaternion

        y2 = y ** 2
        x2 = x ** 2
        z2 = z ** 2
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        mat = np.array(
                [
                    # m1
                    [
                        # m11 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                        1.0 - 2.0 * (y2 + z2),
                        # m21 = 2.0 * (q.x * q.y - q.w * q.z)
                        2.0 * (xy - wz),
                        # m31 = 2.0 * (q.x * q.z + q.w * q.y)
                        2.0 * (xz + wy),
                    ],
                    # m2
                    [
                        # m12 = 2.0 * (q.x * q.y + q.w * q.z)
                        2.0 * (xy + wz),
                        # m22 = 1.0 - 2.0 * (q.x * q.x + q.z * q.z)
                        1.0 - 2.0 * (x2 + z2),
                        # m32 = 2.0 * (q.y * q.z - q.w * q.x)
                        2.0 * (yz - wx),
                    ],
                    # m3
                    [
                        # m13 = 2.0 * (q.x * q.z - q.w * q.y)
                        2.0 * (xz - wy),
                        # m23 = 2.0 * (q.y * q.z + q.w * q.x)
                        2.0 * (yz + wx),
                        # m33 = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
                        1.0 - 2.0 * (x2 + y2),
                    ],
                ],
                dtype=dtype
        )

        return cls(mat)

    @classmethod
    def from_inverse_of_quaternion(cls, quaternion, dtype=None):
        """Creates a matrix with the inverse rotation of a quaternion.

        :param numpy.array quaternion: The quaternion to make the matrix from (shape 4).
        :rtype: numpy.array
        :return: A matrix with shape (3,3) that respresents the inverse of
            the quaternion.
        """
        dtype = dtype or quaternion.dtype

        x, y, z, w = quaternion

        x2 = x ** 2
        y2 = y ** 2
        z2 = z ** 2
        wx = w * x
        wy = w * y
        xy = x * y
        wz = w * z
        xz = x * z
        yz = y * z

        mat = np.array(
                [
                    # m1
                    [
                        # m11 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                        1.0 - 2.0 * (y2 + z2),
                        # m21 = 2.0 * (q.x * q.y + q.w * q.z)
                        2.0 * (xy + wz),
                        # m31 = 2.0 * (q.x * q.z - q.w * q.y)
                        2.0 * (xz - wy),
                    ],
                    # m2
                    [
                        # m12 = 2.0 * (q.x * q.y - q.w * q.z)
                        2.0 * (xy - wz),
                        # m22 = 1.0 - 2.0 * (q.x * q.x + q.z * q.z)
                        1.0 - 2.0 * (x2 + z2),
                        # m32 = 2.0 * (q.y * q.z + q.w * q.x)
                        2.0 * (yz + wx),
                    ],
                    # m3
                    [
                        # m13 = 2.0 * ( q.x * q.z + q.w * q.y)
                        2.0 * (xz + wy),
                        # m23 = 2.0 * (q.y * q.z - q.w * q.x)
                        2.0 * (yz - wx),
                        # m33 = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
                        1.0 - 2.0 * (x2 + y2),
                    ]
                ],
                dtype=dtype
        )

        return cls(mat)

    @classmethod
    def from_scale(cls, scale, dtype=None):
        """
        Creates an identity matrix with the scale set.

        :param numpy.array scale: The scale to apply as a vector (shape 3).
        :rtype: numpy.array
        :return: A matrix with shape (3,3) with the scale
            set to the specified vector.
        """
        # apply the scale to the values diagonally
        # down the matrix
        m = np.diagflat(scale)
        if dtype:
            m = m.astype(dtype)
        return cls(m)

    @classmethod
    def from_x_rotation(cls, theta, dtype=None):
        """
        Creates a matrix with the specified rotation about the X axis.

        :param float theta: The rotation, in radians, about the X-axis.
        :rtype: numpy.array
        :return: A matrix with the shape (3,3) with the specified rotation about
            the X-axis.

        .. seealso:: http://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        """
        cosT = np.cos(theta)
        sinT = np.sin(theta)

        mat = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, cosT, -sinT],
                    [0.0, sinT, cosT]
                ],
                dtype=dtype
        )

        return cls(mat)

    @classmethod
    def from_y_rotation(cls, theta, dtype=None):
        """
        Creates a matrix with the specified rotation about the Y axis.

        :param float theta: The rotation, in radians, about the Y-axis.
        :rtype: numpy.array
        :return: A matrix with the shape (3,3) with the specified rotation about
            the Y-axis.

        .. seealso:: http://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        """
        cosT = np.cos(theta)
        sinT = np.sin(theta)

        mat = np.array(
                [
                    [cosT, 0.0, sinT],
                    [0.0, 1.0, 0.0],
                    [-sinT, 0.0, cosT]
                ],
                dtype=dtype
        )

        return mat

    @classmethod
    def from_z_rotation(cls, theta, dtype=None):
        """
        Creates a matrix with the specified rotation about the Z axis.

        :param float theta: The rotation, in radians, about the Z-axis.
        :rtype: numpy.array
        :return: A matrix with the shape (3,3) with the specified rotation about
            the Z-axis.

        .. seealso:: http://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        """
        cosT = np.cos(theta)
        sinT = np.sin(theta)

        mat = np.array(
                [
                    [cosT, -sinT, 0.0],
                    [sinT, cosT, 0.0],
                    [0.0, 0.0, 1.0]
                ],
                dtype=dtype
        )

        return cls(mat)

    @classmethod
    def multiply(cls, mat1, mat2):
        """
        Multiplication of two matrices
        """
        return np.dot(mat1, mat2)

    @classmethod
    @parameters_as_numpy_arrays('vec')
    def apply_to_vector(cls, mat, vec):
        """Apply a matrix to a vector.

        The matrix's rotation are applied to the vector.
        Supports multiple matrices and vectors.

        :param numpy.array mat: The rotation / translation matrix.
            Can be a list of matrices.
        :param numpy.array vec: The vector to modify.
            Can be a list of vectors.
        :rtype: numpy.array
        :return: The vectors rotated by the specified matrix.
        """
        if vec.size == 3:
            return np.dot(mat, vec)
        else:
            raise ValueError("Vector size unsupported")

    @classmethod
    @parameters_as_numpy_arrays('axis')
    def from_axis_rotation(cls, axis, theta, dtype=None):
        """
        Creates a matrix from the specified theta rotation around an axis.

        :param numpy.array axis: A (3,) vec specifying the axis of rotation.
        :param float theta: A rotation specified in radians.
        :rtype: numpy.array
        :return: A matrix with shape (3,3).
        """
        dtype = dtype or axis.dtype

        axis = axis.normorlized
        x, y, z = axis

        s = np.sin(theta)
        c = np.cos(theta)
        t = 1 - c

        # Construct the elements of the rotation matrix
        mat = np.array(
                [
                    [x * x * t + c, y * x * t + z * s, z * x * t - y * s],
                    [x * y * t - z * s, y * y * t + c, z * y * t + x * s],
                    [x * z * t + y * s, y * z * t - x * s, z * z * t + c]
                ],
                dtype=dtype
        )

        return cls(mat)

    @classmethod
    def create_direction_scale(cls, direction, scale):
        """
        Creates a matrix which can apply a directional scaling to a set of vectors.

        An example usage for this is to flatten a mesh against a
        single plane.

        :param numpy.array direction: a numpy.array of shape (3,) of the direction to scale.
        :param float scale: a float value for the scaling along the specified direction.
            A scale of 0.0 will flatten the vertices into a single plane with the direction being the
            plane's normal.
        :rtype: numpy.array
        :return: The scaling matrix.
        """
        """
        scaling is defined as:

        [p'][1 + (k - 1)n.x^2, (k - 1)n.x n.y^2, (k - 1)n.x n.z   ]
        S(n,k) = [q'][(k - 1)n.x n.y,   1 + (k - 1)n.y,   (k - 1)n.y n.z   ]
        [r'][(k - 1)n.x n.z,   (k - 1)n.y n.z,   1 + (k - 1)n.z^2 ]

        where:
        v' is the resulting vector after scaling
        v is the vector to scale
        n is the direction of the scaling
        n.x is the x component of n
        n.y is the y component of n
        n.z is the z component of n
        k is the scaling factor
        """
        if not np.isclose(np.linalg.norm(direction), 1.):
            direction = direction.normalized

        x, y, z = direction

        x2 = x ** 2.
        y2 = y ** 2.
        z2 = z ** 2

        scaleMinus1 = scale - 1.
        mat = np.array(
                [
                    # m1
                    [
                        # m11 = 1 + (k - 1)n.x^2
                        1. + scaleMinus1 * x2,
                        # m12 = (k - 1)n.x n.y^2
                        scaleMinus1 * x * y2,
                        # m13 = (k - 1)n.x n.z
                        scaleMinus1 * x * z
                    ],
                    # m2
                    [
                        # m21 = (k - 1)n.x n.y
                        scaleMinus1 * x * y,
                        # m22 = 1 + (k - 1)n.y
                        1. + scaleMinus1 * y,
                        # m23 = (k - 1)n.y n.z
                        scaleMinus1 * y * z
                    ],
                    # m3
                    [
                        # m31 = (k - 1)n.x n.z
                        scaleMinus1 * x * z,
                        # m32 = (k - 1)n.y n.z
                        scaleMinus1 * y * z,
                        # m33 = 1 + (k - 1)n.z^2
                        1. + scaleMinus1 * z2
                    ]
                ]
        )

        return cls(mat)

