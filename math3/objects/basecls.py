# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np


class NpProxy(object):
    def __init__(self, index):
        self._index = index

    def __get__(self, obj, cls):
        return obj[self._index]

    def __set__(self, obj, value):
        obj[self._index] = value


class BaseObject(np.ndarray):

    _shape = None

    def __new__(cls, obj):
        # ensure the object matches the required shape
        obj.shape = cls._shape
        return obj

    def _unsupported_type(self, method, other):
        raise ValueError('Cannot {} a {} to a {}'.format(method, type(other).__name__, type(self).__name__))

    # Redirect assignment operators
    def __iadd__(self, other):
        self[:] = self.__add__(other)
        return self

    def __isub__(self, other):
        self[:] = self.__sub__(other)
        return self

    def __imul__(self, other):
        self[:] = self.__mul__(other)
        return self

    def __idiv__(self, other):
        self[:] = self.__div__(other)
        return self


class BaseMatrix(BaseObject):

    @classmethod
    def identity(cls, dtype=None):
        """Creates an identity Matrix.
        """
        return cls(cls._module.create_identity(dtype), dtype)

    @classmethod
    def from_euler(cls, euler, dtype=None):
        """Creates a Matrix from the specified Euler angles.
        """
        return cls(cls._module.create_from_eulers(euler, dtype=dtype))

    @classmethod
    def from_quaternion(cls, quat, dtype=None):
        """Creates a Matrix from a Quaternion.
        """
        return cls(cls._module.create_from_quaternion(quat, dtype=dtype))

    @classmethod
    def from_inverse_of_quaternion(cls, quat, dtype=None):
        """Creates a Matrix from the inverse of the specified Quaternion.
        """
        return cls(cls._module.create_from_inverse_of_quaternion(quat, dtype=dtype))

    @classmethod
    def from_scale(cls, scale, dtype=None):
        return cls(cls._module.create_from_scale(scale, dtype=dtype))

    @classmethod
    def from_x_rotation(cls, theta, dtype=None):
        """Creates a Matrix with a rotation around the X-axis.
        """
        return cls(cls._module.create_from_x_rotation(theta, dtype=dtype))

    @classmethod
    def from_y_rotation(cls, theta, dtype=None):
        return cls(cls._module.create_from_y_rotation(theta, dtype=dtype))

    @classmethod
    def from_z_rotation(cls, theta, dtype=None):
        """Creates a Matrix with a rotation around the Z-axis.
        """
        return cls(cls._module.create_from_z_rotation(theta, dtype=dtype))

    @property
    def inverse(self):
        """Returns the inverse of this matrix.
        """
        return type(self)(self._module.inverse(self))


class BaseVector(BaseObject):
    @classmethod
    def from_matrix44_translation(cls, matrix, dtype=None):
        return cls(cls._module.create_from_matrix44_translation(matrix, dtype))

    def normalize(self):
        self[:] = self.normalised

    @property
    def normalized(self):
        """
        Normalizes an Nd list of vectors or a single vector to unit length.

        The vector is **not** changed in place.

        For zero-length vectors, the result will be np.nan.

                numpy.array([ x, y, z ])

            Or an NxM array::

                numpy.array([
                    [x1, y1, z1],
                    [x2, y2, z2]
                ]).

        :rtype: A numpy.array the normalised value
        """
        return type(self)(self.T / np.sqrt(np.sum(self ** 2, axis=-1)))

    @property
    def squared_length(self):
        """
        Calculates the squared length of a vector.

        Useful when trying to avoid the performance
        penalty of a square root operation.

        :rtype: If one vector is supplied, the result with be a scalar.
                Otherwise the result will be an array of scalars with shape
                vec.ndim with the last dimension being size 1.
        """
        lengths = np.sum(self ** 2., axis=-1)

        return lengths

    @property
    def length(self):
        """
        Returns the length of an Nd list of vectors
        or a single vector.

            Single vector::

                numpy.array([ x, y, z ])

            Nd array::

                numpy.array([
                    [x1, y1, z1],
                    [x2, y2, z2]
                ]).

        :rtype: If a 1d array was passed, it will be a scalar.
            Otherwise the result will be an array of scalars with shape
            vec.ndim with the last dimension being size 1.
        """
        return np.sqrt(np.sum(self ** 2, axis=-1))

    @length.setter
    def length(self, length):
        """
        Resize a Nd list of vectors or a single vector to 'length'.

        The vector is changed in place.

            Single vector::
                numpy.array([ x, y, z ])

            Nd array::
                numpy.array([
                    [x1, y1, z1],
                    [x2, y2, z2]
                ]).
        """
        # calculate the length
        # this is a duplicate of length(vec) because we
        # always want an array, even a 0-d array.

        self[:] = (self.T / np.sqrt(np.sum(self ** 2, axis=-1)) * length).T

    def dot(self, other):
        """Calculates the dot product of two vectors.

        :param numpy.array other: an Nd array with the final dimension
            being size 3 (a vector)
        :rtype: If a 1d array was passed, it will be a scalar.
            Otherwise the result will be an array of scalars with shape
            vec.ndim with the last dimension being size 1.
        """
        return np.sum(self * other, axis=-1)

    def cross(self, other):
        return type(self)(np.cross(self[:3], other[:3]))

    def interpolate(self, other, delta):
        """
        Interpolates between 2 arrays of vectors (shape = N,3)
        by the specified delta (0.0 <= delta <= 1.0).

        :param numpy.array other: an Nd array with the final dimension
            being size 3. (a vector)
        :param float delta: The interpolation percentage to apply,
            where 0.0 <= delta <= 1.0.
            When delta is 0.0, the result will be v1.
            When delta is 1.0, the result will be v2.
            Values inbetween will be an interpolation.
        :rtype: A numpy.array with shape v1.shape.
        """
        # scale the difference based on the time
        # we must do it this 'unreadable' way to avoid
        # loss of precision.
        # the 'readable' method (f_now = f_0 + (f1 - f0) * delta)
        # causes floating point errors due to the small values used
        # in md2 files and the values become corrupted.
        # this horrible code curtousey of this comment:
        # http://stackoverflow.com/questions/5448322/temporal-interpolation-in-numpy-matplotlib
        return self + ((other - self) * delta)
        # return v1 * (1.0 - delta ) + v2 * delta
        t = delta
        t0 = 0.0
        t1 = 1.0
        delta_t = t1 - t0
        return (t1 - t) / delta_t * v1 + (t - t0) / delta_t * v2


class BaseQuaternion(BaseObject):
    pass


# pre-declarations to prevent circular imports
class BaseMatrix3(BaseMatrix):
    pass


class BaseMatrix4(BaseMatrix):
    pass


class BaseVector3(BaseVector):
    pass


class BaseVector4(BaseVector):
    pass
