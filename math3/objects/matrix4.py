# -*- coding: utf-8 -*-
"""
Represents a 4x4 Matrix.

The Matrix4 class provides a number of convenient functions and
conversions.
::

    import numpy as np
    from math3 import Quaternion, Matrix33, Matrix4, Vector4

    m = Matrix4()
    m = Matrix4([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])

    # copy constructor
    m = Matrix4(Matrix4())

    # explicit creation
    m = Matrix4.identity()
    m = Matrix4.from_mat4(Matrix4())

    # inferred conversions
    m = Matrix4(Quaternion())
    m = Matrix4(Matrix33())

    # multiply matricies together
    m = Matrix4() * Matrix4()

    # extract a quaternion from a matrix
    q = m.quaternion

    # convert from quaternion back to matrix
    m = q.mat4
    m = Matrix4(q)

    # rotate a matrix by a quaternion
    m = Matrix4.identity() * Quaternion()

    # rotate a vector 4 by a matrix
    v = Matrix4.from_x_rotation(np.pi) * Vector4([1.,2.,3.,1.])

    # undo a rotation
    m = Matrix4.from_x_rotation(np.pi)
    v = m * Vector4([1.,1.,1.,1.])
    # ~m is the same as m.inverse
    v = ~m * v

    # access specific parts of the matrix
    # first row
    m1 = m.m1
    # first element, first row
    m11 = m.m11
    # fourth element, fourth row
    m44 = m.m44
    # first row, same as m1
    r1 = m.r1
    # first column
    c1 = m.c1
"""
from __future__ import absolute_import, unicode_literals
from numbers import Number
import numpy as np
from multipledispatch import dispatch
from .basecls import BaseObject, BaseMatrix, BaseMatrix4, BaseQuaternion, BaseVector, NpProxy
from ..utils import parameters_as_numpy_arrays
from .matrix3 import Matrix3
from .quaternion import Quaternion


class Matrix4(BaseMatrix4):

    _shape = (4, 4,)

    # m<c> style access
    #: The first row of this Matrix as a numpy.ndarray.
    m1 = NpProxy(0)
    #: The second row of this Matrix as a numpy.ndarray.
    m2 = NpProxy(1)
    #: The third row of this Matrix as a numpy.ndarray.
    m3 = NpProxy(2)
    #: The fourth row of this Matrix as a numpy.ndarray.
    m4 = NpProxy(3)

    # m<r><c> access
    #: The [0,0] value of this Matrix.
    m11 = NpProxy((0,0))
    #: The [0,1] value of this Matrix.
    m12 = NpProxy((0,1))
    #: The [0,2] value of this Matrix.
    m13 = NpProxy((0,2))
    #: The [0,3] value of this Matrix.
    m14 = NpProxy((0,3))
    #: The [1,0] value of this Matrix.
    m21 = NpProxy((1,0))
    #: The [1,1] value of this Matrix.
    m22 = NpProxy((1,1))
    #: The [1,2] value of this Matrix.
    m23 = NpProxy((1,2))
    #: The [1,3] value of this Matrix.
    m24 = NpProxy((1,3))
    #: The [2,0] value of this Matrix.
    m31 = NpProxy((2,0))
    #: The [2,1] value of this Matrix.
    m32 = NpProxy((2,1))
    #: The [2,2] value of this Matrix.
    m33 = NpProxy((2,2))
    #: The [2,3] value of this Matrix.
    m34 = NpProxy((2,3))
    #: The [3,0] value of this Matrix.
    m41 = NpProxy((3,0))
    #: The [3,1] value of this Matrix.
    m42 = NpProxy((3,1))
    #: The [3,2] value of this Matrix.
    m43 = NpProxy((3,2))
    #: The [3,3] value of this Matrix.
    m44 = NpProxy((3,3))

    # rows
    #: The first row of this Matrix as a numpy.ndarray. This is the same as m1.
    r1 = NpProxy(0)
    #: The second row of this Matrix as a numpy.ndarray. This is the same as m2.
    r2 = NpProxy(1)
    #: The third row of this Matrix as a numpy.ndarray. This is the same as m3.
    r3 = NpProxy(2)
    #: The fourth row of this Matrix as a numpy.ndarray. This is the same as m4.
    r4 = NpProxy(3)

    # columns
    #: The first column of this Matrix as a numpy.ndarray.
    c1 = NpProxy((slice(0,4),0))
    #: The second column of this Matrix as a numpy.ndarray.
    c2 = NpProxy((slice(0,4),1))
    #: The third column of this Matrix as a numpy.ndarray.
    c3 = NpProxy((slice(0,4),2))
    #: The fourth column of this Matrix as a numpy.ndarray.
    c4 = NpProxy((slice(0,4),3))

    ########################
    # Creation
    @classmethod
    def from_matrix3(cls, matrix, dtype=None):
        """
        Creates a Matrix4 from a Matrix3.

        The translation will be 0,0,0.

        :rtype: numpy.array
        :return: A matrix with shape (4,4) with the input matrix rotation.
        """
        mat4 = np.identity(4, dtype=dtype)
        mat4[0:3, 0:3] = matrix

        return cls(mat4)

    @classmethod
    def perspective_projection(cls, fovy, aspect, near, far, dtype=None):
        """
        Creates perspective projection matrix.

        .. seealso:: http://www.opengl.org/sdk/docs/man2/xhtml/gluPerspective.xml
        .. seealso:: http://www.geeks3d.com/20090729/howto-perspective-projection-matrix-in-opengl/

        :param float fovy: field of view in y direction in degrees
        :param float aspect: aspect ratio of the view (width / height)
        :param float near: distance from the viewer to the near clipping plane (only positive)
        :param float far: distance from the viewer to the far clipping plane (only positive)
        :rtype: numpy.array
        :return: A projection matrix representing the specified perspective.
        """
        ymax = near * np.tan(fovy * np.pi / 360.0)
        xmax = ymax * aspect
        return cls.create_perspective_projection_matrix_from_bounds(-xmax, xmax,
                                                                    -ymax, ymax,
                                                                    near, far)

    @classmethod
    def perspective_projection_bounds(cls, left, right, top, bottom, near, far, dtype=None):
        """
        Creates a perspective projection matrix using the specified near
        plane dimensions.

        :param float left: The left of the near plane relative to the plane's centre.
        :param float right: The right of the near plane relative to the plane's centre.
        :param float top: The top of the near plane relative to the plane's centre.
        :param float bottom: The bottom of the near plane relative to the plane's centre.
        :param float near: The distance of the near plane from the camera's origin.
            It is recommended that the near plane is set to 1.0 or above to avoid rendering issues
            at close range.
        :param float far: The distance of the far plane from the camera's origin.
        :rtype: numpy.array
        :return: A projection matrix representing the specified perspective.

        .. seealso:: http://www.gamedev.net/topic/264248-building-a-projection-matrix-without-api/
        .. seealso:: http://www.glprogramming.com/red/chapter03.html
        """

        """
        E 0 A 0
        0 F B 0
        0 0 C D
        0 0-1 0

        A = (right+left)/(right-left)
        B = (top+bottom)/(top-bottom)
        C = -(far+near)/(far-near)
        D = -2*far*near/(far-near)
        E = 2*near/(right-left)
        F = 2*near/(top-bottom)
        """
        A = (right + left) / (right - left)
        B = (top + bottom) / (top - bottom)
        C = -(far + near) / (far - near)
        D = -2. * far * near / (far - near)
        E = 2. * near / (right - left)
        F = 2. * near / (top - bottom)

        mat = np.array((
            (E, 0., 0., 0.),
            (0., F, 0., 0.),
            (A, B, C, -1.),
            (0., 0., D, 0.),
        ), dtype=dtype)

        return cls(mat)

    @classmethod
    def orthogonal_projection(cls, left, right, top, bottom, near, far, dtype=None):
        """
        Creates an orthogonal projection matrix.

        :param float left: The left of the near plane relative to the plane's centre.
        :param float right: The right of the near plane relative to the plane's centre.
        :param float top: The top of the near plane relative to the plane's centre.
        :param float bottom: The bottom of the near plane relative to the plane's centre.
        :param float near: The distance of the near plane from the camera's origin.
            It is recommended that the near plane is set to 1.0 or above to avoid rendering issues
            at close range.
        :param float far: The distance of the far plane from the camera's origin.
        :rtype: numpy.array
        :return: A projection matrix representing the specified orthogonal perspective.

        .. seealso:: http://msdn.microsoft.com/en-us/library/dd373965(v=vs.85).aspx
        """

        """
        A 0 0 Tx
        0 B 0 Ty
        0 0 C Tz
        0 0 0 1

        A = 2 / (right - left)
        B = 2 / (top - bottom)
        C = -2 / (far - near)

        Tx = (right + left) / (right - left)
        Ty = (top + bottom) / (top - bottom)
        Tz = (far + near) / (far - near)
        """
        rml = right - left
        tmb = top - bottom
        fmn = far - near

        A = 2. / rml
        B = 2. / tmb
        C = -2. / fmn
        Tx = -(right + left) / rml
        Ty = -(top + bottom) / tmb
        Tz = -(far + near) / fmn

        mat = np.array((
            (A, 0., 0., 0.),
            (0., B, 0., 0.),
            (0., 0., C, 0.),
            (Tx, Ty, Tz, 1.),
        ), dtype=dtype)

        return cls(mat)

    @classmethod
    @parameters_as_numpy_arrays('vec')
    def from_translation(cls, vec, dtype=None):
        """Creates an identity matrix with the translation set.

        :param numpy.array vec: The translation vector (shape 3 or 4).
        :rtype: numpy.array
        :return: A matrix with shape (4,4) that represents a matrix
            with the translation set to the specified vector.
        """
        dtype = dtype or vec.dtype
        mat = cls.identity(dtype)
        mat[:3, 3] = vec[:3]
        return mat

    def __new__(cls, value=None, dtype=None):
        if value is not None:
            obj = value
            if not isinstance(value, np.ndarray):
                obj = np.array(value, dtype=dtype)

            # matrix3
            if obj.shape == (3,3) or isinstance(obj, Matrix3):
                obj = Matrix4.from_matrix3(obj, dtype=dtype)
            # quaternion
            elif obj.shape == (4,) or isinstance(obj, Quaternion):
                obj = Matrix4.from_quaternion(obj, dtype=dtype)
        else:
            obj = np.zeros(cls._shape, dtype=dtype)
        obj = obj.view(cls)
        return super(Matrix4, cls).__new__(cls, obj)

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
        return Matrix4(super(Matrix4, self).__add__(Matrix4(other)))

    @dispatch((BaseMatrix, np.ndarray, list))
    def __sub__(self, other):
        return Matrix4(super(Matrix4, self).__sub__(Matrix4(other)))

    @dispatch((BaseMatrix, np.ndarray, list))
    def __mul__(self, other):
        return self.multiply(Matrix4(other))

    ########################
    # Quaternions
    @dispatch(BaseQuaternion)
    def __mul__(self, other):
        m = other.mat4
        return self * m

    ########################
    # Vectors
    @dispatch(BaseVector)
    def __mul__(self, other):
        return type(other)(self.apply_to_vector(other))

    ########################
    # Number
    @dispatch((Number, np.number))
    def __add__(self, other):
        return Matrix4(super(Matrix4, self).__add__(other))

    @dispatch((Number, np.number))
    def __sub__(self, other):
        return Matrix4(super(Matrix4, self).__sub__(other))

    @dispatch((Number, np.number))
    def __mul__(self, other):
        return Matrix4(super(Matrix4, self).__mul__(other))

    @dispatch((Number, np.number))
    def __truediv__(self, other):
        return Matrix4(super(Matrix4, self).__truediv__(other))

    @dispatch((Number, np.number))
    def __div__(self, other):
        return Matrix4(super(Matrix4, self).__div__(other))

    ########################
    # Methods and Properties
    @property
    def mat3(self):
        """Returns a Matrix33 representing this matrix.
        """
        return Matrix3(self)

    @property
    def mat4(self):
        """Returns the Matrix4.

        This can be handy if you're not sure what type of Matrix class you have
        but require a Matrix4.
        """
        return self

    @property
    def quat(self):
        """Returns a Quaternion representing this matrix.
        """
        return Quaternion(self)

    # Class method
    @classmethod
    def identity(cls, dtype=None):
        """
        Creates a new matrix44 and sets it to an identity matrix.

        :rtype: numpy.array
        :return: A matrix representing an identity matrix with shape (4,4).
        """
        return cls(np.identity(4, dtype=dtype))

    @classmethod
    def from_euler(cls, euler, dtype=None):
        """
        Creates a matrix from the specified Euler rotations.

        :param numpy.array euler: A set of euler rotations in the format
            specified by the euler modules.
        :rtype: numpy.array
        :return: A matrix with shape (4,4) with the euler's rotation.
        """
        dtype = dtype or euler.dtype
        # set to identity matrix
        # this will populate our extra rows for us
        mat = cls.identity(dtype)

        # we'll use Matrix33 for our conversion
        mat[0:3, 0:3] = Matrix3.from_euler(euler, dtype)
        return cls(mat)

    @classmethod
    def from_quaternion(cls, quaternion, dtype=None):
        """
        Creates a matrix with the same rotation as a quaternion.

        :param quaternion: The quaternion to create the matrix from.
        :rtype: numpy.array
        :return: A matrix with shape (4,4) with the quaternion's rotation.
        """
        dtype = dtype or quaternion.dtype
        # set to identity matrix
        # this will populate our extra rows for us
        mat = cls.identity(dtype)

        # we'll use Matrix33 for our conversion
        mat[0:3, 0:3] = Matrix3.from_quaternion(quaternion, dtype)

        return cls(mat)

    @classmethod
    def from_inverse_of_quaternion(cls, quaternion, dtype=None):
        """
        Creates a matrix with the inverse rotation of a quaternion.

        This can be used to go from object space to intertial space.

        :param numpy.array quat: The quaternion to make the matrix from (shape 4).
        :rtype: numpy.array
        :return: A matrix with shape (4,4) that respresents the inverse of
        the quaternion.
        """
        dtype = dtype or quaternion.dtype
        # set to identity matrix
        # this will populate our extra rows for us
        mat = cls.identity(dtype)

        # we'll use Matrix33 for our conversion
        mat[0:3, 0:3] = Matrix3.from_inverse_of_quaternion(quaternion, dtype)

        return cls(mat)

    @classmethod
    def from_scale(cls, scale, dtype=None):
        """
        Creates an identity matrix with the scale set.

        :param numpy.array scale: The scale to apply as a vector (shape 3).
        :rtype: numpy.array
        :return: A matrix with shape (4,4) with the scale
            set to the specified vector.
        """
        # we need to expand 'scale' into it's components
        # because numpy isn't flattening them properly.
        m = np.diagflat([scale[0], scale[1], scale[2], 1.0])

        if dtype:
            m = m.astype(dtype)

        return cls(m)

    @classmethod
    def from_x_rotation(cls, theta, dtype=None):
        """
        Creates a matrix with the specified rotation about the X axis.

        :param float theta: The rotation, in radians, about the X-axis.
        :rtype: numpy.array
        :return: A matrix with the shape (4,4) with the specified rotation about
             the X-axis.

        .. seealso:: http://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        """
        mat = cls.identity(dtype)
        mat[0:3, 0:3] = Matrix3.from_x_rotation(theta, dtype)

        return cls(mat)

    @classmethod
    def from_y_rotation(cls, theta, dtype=None):
        """
        Creates a matrix with the specified rotation about the Y axis.

        :param float theta: The rotation, in radians, about the Y-axis.
        :rtype: numpy.array
        :return: A matrix with the shape (4,4) with the specified rotation about
                the Y-axis.

        .. seealso:: http://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        """
        mat = cls.identity(dtype)
        mat[0:3, 0:3] = Matrix3.from_y_rotation(theta, dtype)

        return cls(mat)

    @classmethod
    def from_z_rotation(cls, theta, dtype=None):
        """
        Creates a matrix with the specified rotation about the Z axis.

        :param float theta: The rotation, in radians, about the Z-axis.
        :rtype: numpy.array
        :return: A matrix with the shape (4,4) with the specified rotation about
                the Z-axis.

        .. seealso:: http://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        """
        mat = cls.identity(dtype)
        mat[0:3, 0:3] = Matrix3.create_from_z_rotation(theta, dtype)

        return cls(mat)

    @classmethod
    def from_axis_rotation(cls, axis, theta, dtype=None):
        """Creates a matrix from the specified rotation theta around an axis.

        :param numpy.array axis: A (3,) vector.
        :param float theta: A rotation in radians.

        :rtype: numpy.array
        :return: A matrix with shape (4,4).
        """
        dtype = dtype or axis.dtype
        # set to identity matrix
        # this will populate our extra rows for us
        mat = cls.identity(dtype)

        # we'll use Matrix33 for our conversion
        mat[0:3, 0:3] = Matrix3.from_axis_rotation(axis, theta, dtype)

        return cls(mat)

    def apply_to_vector(self, vec):
        """Apply a matrix to a vector.

        The matrix's rotation and translation are applied to the vector.
        Supports multiple matrices and vectors.

        :param numpy.array mat: The rotation / translation matrix.
            Can be a list of matrices.
        :param numpy.array vec: The vector to modify.
            Can be a list of vectors.
        :rtype: numpy.array
        :return: The vectors rotated by the specified matrix.
        """
        if vec.size == 3:
            # convert to a vec4
            vec4 = np.array([vec[0], vec[1], vec[2], 1.], dtype=vec.dtype)
            vec4 = np.dot(self, vec4)
            if np.allclose(vec4[3], 0.):
                vec4[:] = [np.inf, np.inf, np.inf, np.inf]
            else:
                vec4 /= vec4[3]
            return vec4[:3]
        elif vec.size == 4:
            return np.dot(self, vec)
        else:
            raise ValueError("Vector size unsupported")

    def multiply(self, mat):
        """
        Multiply two matricies, m1 . m2.

        This is essentially a wrapper around
        numpy.dot(m1, m2)

        :param numpy.array mat: The second matrix.
        :rtype: numpy.array
        :return: A matrix that results from multiplying m1 by m2.
        """
        return np.dot(self, mat)

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
