# -*- coding: utf-8 -*-
"""
Represents a Quaternion rotation.

The Quaternion class provides a number of convenient functions and
conversions.
::

    import numpy as np
    from math3 import Quaternion, Matrix3, Matrix4, Vector3, Vector4

    q = Quaternion()

    # explicit creation
    q = Quaternion.from_x_rotation(np.pi / 2.0)
    q = Quaternion.from_matrix(Matrix3.identity())
    q = Quaternion.from_matrix(Matrix4.identity())

    # inferred conversions
    q = Quaternion(Quaternion())
    q = Quaternion(Matrix3.identity())
    q = Quaternion(Matrix4.identity())

    # apply one quaternion to another
    q1 = Quaternion.from_y_rotation(np.pi / 2.0)
    q2 = Quaternion.from_x_rotation(np.pi / 2.0)
    q3 = q1 * q2

    # extract a matrix from the quaternion
    m33 = q3.matrix33
    m44 = q3.matrix44

    # convert from matrix back to quaternion
    q4 = Quaternion(m44)

    # rotate a quaternion by a matrix
    q = Quaternion() * Matrix3.identity()
    q = Quaternion() * Matrix4.identity()

    # apply quaternion to a vector
    v3 = Quaternion() * Vector3()
    v4 = Quaternion() * Vector4()

    # undo a rotation
    q = Quaternion.from_x_rotation(np.pi / 2.0)
    v = q * Vector3([1.,1.,1.])
    # ~q is the same as q.conjugate
    original = ~q * v
    assert np.allclose(original, v)

    # get the dot product of 2 Quaternions
    dot = Quaternion() | Quaternion.from_x_rotation(np.pi / 2.0)
"""

from __future__ import absolute_import, unicode_literals
from __future__ import division, print_function
import numpy as np
from multipledispatch import dispatch
from .basecls import BaseObject, BaseQuaternion, BaseMatrix, BaseVector, NpProxy
from .vector4 import Vector4
from .vector3 import Vector3
from .matrix3 import Matrix3
from .matrix4 import Matrix4


class Quaternion(BaseQuaternion):

    _shape = (4,)

    #: The X value of this Quaternion.
    x = NpProxy(0)
    #: The Y value of this Quaternion.
    y = NpProxy(1)
    #: The Z value of this Quaternion.
    z = NpProxy(2)
    #: The W value of this Quaternion.
    w = NpProxy(3)
    #: The X,Y value of this Quaternion as a numpy.ndarray.
    xy = NpProxy([0, 1])
    #: The X,Y,Z value of this Quaternion as a numpy.ndarray.
    xyz = NpProxy([0, 1, 2])
    #: The X,Y,Z,W value of this Quaternion as a numpy.ndarray.
    xyzw = NpProxy([0, 1, 2, 3])
    #: The X,Z value of this Quaternion as a numpy.ndarray.
    xz = NpProxy([0, 2])
    #: The X,Z,W value of this Quaternion as a numpy.ndarray.
    xzw = NpProxy([0, 2, 3])
    #: The X,Y,W value of this Quaternion as a numpy.ndarray.
    xyw = NpProxy([0, 1, 3])
    #: The X,W value of this Quaternion as a numpy.ndarray.
    xw = NpProxy([0, 3])

    # Creation
    @classmethod
    def from_x_rotation(cls, theta, dtype=None):
        """
        Creates a new Quaternion with a rotation around the X-axis.
        """
        thetaOver2 = theta * 0.5

        quat = np.array(
                [
                    np.sin(thetaOver2),
                    0.0,
                    0.0,
                    np.cos(thetaOver2)
                ],
                dtype=dtype
        )

        return cls(quat)

    @classmethod
    def from_y_rotation(cls, theta, dtype=None):
        """
        Creates a new Quaternion with a rotation around the Y-axis.
        """
        thetaOver2 = theta * 0.5

        quat = np.array(
                [
                    0.0,
                    np.sin(thetaOver2),
                    0.0,
                    np.cos(thetaOver2)
                ],
                dtype=dtype
        )

        return cls(quat)

    @classmethod
    def from_z_rotation(cls, theta, dtype=None):
        """
        Creates a new Quaternion with a rotation around the Z-axis.
        """
        thetaOver2 = theta * 0.5

        quat = np.array(
                [
                    0.0,
                    0.0,
                    np.sin(thetaOver2),
                    np.cos(thetaOver2)
                ],
                dtype=dtype
        )

        return cls(quat)

    @classmethod
    def from_axis_rotation(cls, axis, theta, dtype=None):
        """
        Creates a new Quaternion with a rotation around the specified axis.
        """
        dtype = dtype or axis.dtype
        # make sure the vector is normalised
        if not np.isclose(np.linalg.norm(axis), 1.):
            axis = Vector3.normalise(axis)

        thetaOver2 = theta * 0.5
        sinThetaOver2 = np.sin(thetaOver2)

        quat = np.array(
                [
                    sinThetaOver2 * axis[0],
                    sinThetaOver2 * axis[1],
                    sinThetaOver2 * axis[2],
                    np.cos(thetaOver2)
                ],
                dtype=dtype
        )

        return cls(quat)

    @classmethod
    def from_matrix(cls, matrix, dtype=None):
        """
        Creates a Quaternion from the specified Matrix (Matrix3 or Matrix4).
        """
        # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
        dtype = dtype or matrix.dtype

        quat = np.array(
                [
                    np.sqrt(np.maximum(0., 1. + matrix[0][0] - matrix[1][1] - matrix[2][
                        2])) / 2.,
                    np.sqrt(np.maximum(0., 1. - matrix[0][0] + matrix[1][1] - matrix[2][
                        2])) / 2.,
                    np.sqrt(np.maximum(0., 1. - matrix[0][0] - matrix[1][1] + matrix[2][
                        2])) / 2.,
                    np.sqrt(np.maximum(0., 1. + matrix[0][0] + matrix[1][1] + matrix[2][
                        2])) / 2.,
                ],
                dtype=dtype
        )

        # the method suggests this, but it produces the wrong results
        # if np.sign(quat[0]) != np.sign(mat[2][1]-mat[1][2]):
        #    quat[0] *= -1
        # if np.sign(quat[1]) != np.sign(mat[0][2]-mat[2][0]):
        #    quat[1] *= -1
        # if np.sign(quat[2]) != np.sign(mat[1][0]-mat[0][1]):
        #    quat[2] *= -1

        return cls(quat)

    @classmethod
    def from_euler(cls, euler, dtype=None):
        """
        Creates a quaternion from a set of Euler angles.

        Euler are an array of length 3 in the following order::
        [yaw, pitch, roll]
        """
        dtype = dtype or euler.dtype

        pitch, yaw, roll = euler

        halfPitch = pitch * 0.5
        sP = np.sin(halfPitch)
        cP = np.cos(halfPitch)

        halfRoll = roll * 0.5
        sR = np.sin(halfRoll)
        cR = np.cos(halfRoll)

        halfYaw = yaw * 0.5
        sY = np.sin(halfYaw)
        cY = np.cos(halfYaw)

        quat = np.array(
                [
                    # x = -cy * sp * cr - sy * cp * sr
                    (-cY * sP * cR) - (sY * cP * sR),
                    # y = cy * sp * sr - sy * cp * cr
                    (cY * sP * sR) - (sY * cP * cR),
                    # z = sy * sp * cr - cy * cp * sr
                    (sY * sP * cR) - (cY * cP * sR),
                    # w = cy * cp * cr + sy * sp * sr
                    (cY * cP * cR) + (sY * sP * sR),
                ],
                dtype=dtype
        )

        return cls(quat)

    @classmethod
    def from_inverse_of_euler(cls, euler, dtype=None):
        """
        Creates a quaternion from the inverse of a set of Euler angles.

        Euler are an array of length 3 in the following order::
            [yaw, pitch, roll]
        """
        dtype = dtype or euler.dtype

        pitch, roll, yaw = euler.pitch(euler), euler.roll(euler), euler.yaw(
                euler)

        halfRoll = roll * 0.5
        sinRoll = np.sin(halfRoll)
        cosRoll = np.cos(halfRoll)

        halfPitch = pitch * 0.5
        sinPitch = np.sin(halfPitch)
        cosPitch = np.cos(halfPitch)

        halfYaw = yaw * 0.5
        sinYaw = np.sin(halfYaw)
        cosYaw = np.cos(halfYaw)

        quat = np.array(
                [
                    # x = cy * sp * cr + sy * cp * sr
                    (cosYaw * sinPitch * cosRoll) + (
                    sinYaw * cosPitch * sinRoll),
                    # y = -cy * sp * sr + sy * cp * cr
                    (-cosYaw * sinPitch * sinRoll) + (
                    sinYaw * cosPitch * cosRoll),
                    # z = -sy * sp * cr + cy * cp * sr
                    (-sinYaw * sinPitch * cosRoll) + (
                    cosYaw * cosPitch * sinRoll),
                    # w = cy * cp * cr + sy * sp * sr
                    (cosYaw * cosPitch * cosRoll) + (
                    sinYaw * sinPitch * sinRoll)
                ],
                dtype=dtype
        )

        return cls(quat)

    def __new__(cls, x=None, y=0., z=0., w=0., dtype=None):
        if isinstance(x, list) and len(x) == 4:
            obj = x
            if not isinstance(x, np.ndarray):
                obj = np.array(x, dtype=dtype)

            # matrix3, matrix4
            if obj.shape in ((4, 4,), (3, 3,)) or isinstance(obj, (
                    Matrix3, Matrix4)):
                obj = Quaternion.from_matrix(obj, dtype=dtype)
        elif x is not None:
            obj = np.array([x, y, z, w], dtype)
        else:
            obj = np.array([x, y, z, w], dtype=dtype)

        obj = obj.view(cls)
        return super(Quaternion, cls).__new__(cls, obj)
    
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
    
    # Quaternions
    @dispatch((BaseQuaternion, np.ndarray, list))
    def __sub__(self, other):
        return Quaternion(super(Quaternion, self).__sub__(other))

    @dispatch((BaseQuaternion, list))
    def __mul__(self, other):
        return self.cross(other)

    @dispatch((BaseQuaternion, list))
    def __or__(self, other):
        return self.dot(other)

    def __invert__(self):
        return self.conjugate
    
    # Matrices
    @dispatch(BaseMatrix)
    def __mul__(self, other):
        return self * Quaternion(other)
    
    # Vectors
    @dispatch(BaseVector)
    def __mul__(self, other):
        return type(other)(self.apply_to_vector(other))
    
    # Methods and Properties
    @property
    def length(self):
        """
        Calculates the length of a quaternion.

        :param numpy.array quat: The quaternion to measure.
        :rtype: float, numpy.array
        :return: If a 1d array was passed, it will be a scalar.
            Otherwise the result will be an array of scalars with shape
            vec.ndim with the last dimension being size 1.
        """
        return np.sqrt(np.sum(self ** 2, axis=-1))

    def normalise(self):
        """Ensure a quaternion is unit length (length ~= 1.0).

        The quaternion is **not** changed in place.

        :rtype: numpy.array
        :return: The normalised quaternion(s).
        """
        self[:] = Vector4.normalise(self)

    @property
    def normalised(self):
        """Returns a normalised version of this Quaternion as a new Quaternion.
        """
        return Quaternion(Vector4.normalise(self))

    @property
    def angle(self):
        """
        Calculates the rotation around the quaternion's axis.

        :rtype: float.
        :return: The quaternion's rotation about the its axis in radians.
        """
        # extract the W component
        thetaOver2 = np.arccos(self[3])

        return thetaOver2 * 2.0

    @property
    def axis(self):
        """
        Calculates the axis of the quaternion's rotation.

        :rtype: numpy.array.
        :return: The quaternion's rotation axis.
        """
        # extract W component
        sinThetaOver2Sq = 1.0 - (self[3] ** 2)

        # check for zero before we sqrt
        if sinThetaOver2Sq <= 0.0:
            # identity quaternion or numerical imprecision.
            # return a valid vector
            # we'll treat -Z as the default
            return np.array([0.0, 0.0, -1.0], dtype=self.dtype)

        oneOverSinThetaOver2 = 1.0 / np.sqrt(sinThetaOver2Sq)

        # we use the x,y,z values
        return np.array(
                [
                    self[0] * oneOverSinThetaOver2,
                    self[1] * oneOverSinThetaOver2,
                    self[2] * oneOverSinThetaOver2
                ],
                dtype=self.dtype
        )

    def cross(self, other):
        """
        Returns the cross-product of the two quaternions.

        Quaternions are **not** communicative. Therefore, order is important.

        This is NOT the same as a vector cross-product.
        Quaternion cross-product is the equivalent of matrix multiplication.
        """
        q1x, q1y, q1z, q1w = self
        q2x, q2y, q2z, q2w = other

        quat = np.array(
                [
                    q1x * q2w + q1y * q2z - q1z * q2y + q1w * q2x,
                    -q1x * q2z + q1y * q2w + q1z * q2x + q1w * q2y,
                    q1x * q2y - q1y * q2x + q1z * q2w + q1w * q2z,
                    -q1x * q2x - q1y * q2y - q1z * q2z + q1w * q2w,
                ],
                dtype=self.dtype
        )

        return Quaternion(quat)

    def dot(self, other):
        """
        Calculate the dot product of quaternions.

        This is the same as a vector dot product.

        :param numpy.array quat1: The first quaternion(s).
        :param numpy.array quat2: The second quaternion(s).
        :rtype: float, numpy.array
        :return: If a 1d array was passed, it will be a scalar.
            Otherwise the result will be an array of scalars with shape
            vec.ndim with the last dimension being size 1.
        """
        return Vector4.dot(self, other)

    @property
    def conjugate(self):
        """
        Calculates a quaternion with the opposite rotation.

        :rtype: numpy.array.
        :return: A quaternion representing the conjugate.
        """

        # invert x,y,z and leave w as is
        quat = np.array(
                [
                    -self[0],
                    -self[1],
                    -self[2],
                    self[3]
                ],
                dtype=self.dtype
        )

        return Quaternion(quat)

    @property
    def inverse(self):
        """
        Calculates the inverse quaternion.

        The inverse of a quaternion is defined as
        the conjugate of the quaternion divided
        by the magnitude of the original quaternion.

        :param numpy.array quat: The quaternion to invert.
        :rtype: numpy.array.
        :return: The inverse of the quaternion.
        """
        return Quaternion(self.conjugate(self) / self.length(self))

    def power(self, exponent):
        """
        Multiplies the quaternion by the exponent.

        The quaternion is **not** changed in place.

        :param numpy.array quat: The quaternion.
        :param float scalar: The exponent.
        :rtype: numpy.array.
        :return: A quaternion representing the original quaternion
            to the specified power.
        """
        # check for identify quaternion
        if np.fabs(self[3]) > 0.9999:
            # assert for the time being
            assert False
            print("rotation axis was identity")

            return quat

        alpha = np.arccos(self[3])
        newAlpha = alpha * exponent
        multi = np.sin(newAlpha) / np.sin(alpha)

        quat = np.array(
                [
                    self[0] * multi,
                    self[1] * multi,
                    self[2] * multi,
                    np.cos(newAlpha)
                ],
                dtype=self.dtype
        )

        return Quaternion(quat)

    @property
    def negative(self):
        """Calculates the negated quaternion.

        This is essentially the quaternion * -1.0.

        :rtype: numpy.array
        :return: The negated quaternion.
        """
        return Quaternion(self * -1.0)

    @property
    def is_identity(self):
        """Returns True if the Quaternion has no rotation (0.,0.,0.,1.).
        """
        return np.allclose(self, [0., 0., 0., 1.])

    @property
    def mat4(self):
        """Returns a Matrix4 representation of this Quaternion.
        """
        return Matrix4.from_quaternion(self)

    @property
    def mat3(self):
        """Returns a Matrix3 representation of this Quaternion.
        """
        return Matrix3.from_quaternion(self)

    @property
    def squared_length(self):
        """
        Calculates the squared length of a quaternion.

        Useful for avoiding the performance penalty of
        the square root function.

        :param numpy.array quat: The quaternion to measure.
        :rtype: float, numpy.array
        :return: If a 1d array was passed, it will be a scalar.
                Otherwise the result will be an array of scalars with shape
                vec.ndim with the last dimension being size 1.
        """
        length = np.sum(self ** 2., axis=-1)

        return length

    def apply_to_vector(self, vec):
        """Rotates a vector by a quaternion.

        :param numpy.array self: The quaternion.
        :param numpy.array vec: The vector.
        :rtype: numpy.array
        :return: The vector rotated by the quaternion.
        :raise ValueError: raised if the vector is an unsupported size

        .. seealso:: http://content.gpwiki.org/index.php/OpenGL:Tutorials:Using_Quaternions_to_represent_rotation
        """

        def apply(quat, vec4):
            """
            v = numpy.array(vec)
            return v + 2.0 * vector.cross(
                self[:-1],
                vector.cross(self[:-1], v) + (self[-1] * v)
               )
            """
            length = vec.length(vec4)
            vec4[:] = vec.normalise(vec4)

            # self * vec * self^-1
            result = quat.cross(vec4.cross(quat.conjugate))
            result *= length
            return result

        if vec.size == 3:
            # convert to vector4
            # ignore w component by setting it to 0.
            vec = np.array([vec[0], vec[1], vec[2], 0.0], dtype=vec.dtype)
            vec = apply(self, vec)
            vec = vec[:3]
            return vec
        elif vec.size == 4:
            vec = apply(self, vec)
            return vec
        else:
            raise ValueError("Vector size unsupported")