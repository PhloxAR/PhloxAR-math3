# -*- coding: utf-8 -*-
"""
Represents a 3 dimensional Vector.

The Vector3 class provides a number of convenient functions and
conversions.
::

    import numpy as np
    from math3 import Quaternion, Matrix3, Matrix4, Vector3

    v = Vector3()
    v = Vector3([1.,2.,3.])

    # copy constructor
    v = Vector3(Vector3())

    # add / subtract vectors
    v = Vector3([1.,2.,3.]) + Vector3([4.,5.,6.])

    # rotate a vector by a Matrix
    v = Matrix3.identity() * Vector3([1.,2.,3.])
    v = Matrix4.identity() * Vector3([1.,2.,3.])

    # rotate a vector by a Quaternion
    v = Quaternion() * Vector3([1.,2.,3.])

    # get the dot-product of 2 vectors
    d = Vector3([1.,0.,0.]) | Vector3([0.,1.,0.])

    # get the cross-product of 2 vectors
    x = Vector3([1.,0.,0.]) ^ Vector3([0.,1.,0.])

    # access specific parts of the vector
    # x value
    x,y,z = v.x, v.y, v.z

    # access groups of values as np.ndarray's
    xy = v.xy
    xz = v.xz
    xyz = v.xyz
"""
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from numbers import Number
import numpy as np
from multipledispatch import dispatch
from .basecls import BaseObject, BaseVector3, BaseMatrix4, NpProxy
from ..utils import parameters_as_numpy_arrays


# TODO: add < <= > >= == != operators

class Vector3(BaseVector3):
    _shape = (3,)

    #: The X value of this Vector.
    x = NpProxy(0)
    #: The Y value of this Vector.
    y = NpProxy(1)
    #: The Z value of this Vector.
    z = NpProxy(2)
    #: The X,Y values of this Vector as a numpy.ndarray.
    xy = NpProxy([0, 1])
    #: The X,Y,Z values of this Vector as a numpy.ndarray.
    xyz = NpProxy([0, 1, 2])
    #: The X,Z values of this Vector as a numpy.ndarray.
    xz = NpProxy([0, 2])
    
    # Creation
    @classmethod
    def from_vector4(cls, vector, dtype=None):
        """
        Create a Vector3 from a Vector4.

        Returns the Vector3 and the W component as a tuple.
        """
        dtype = dtype or vector.dtype
        v = np.array([vector[0], vector[1], vector[2]], dtype=dtype)
        return cls(v), vector[3]

    def __new__(cls, x=None, y=0., z=0., w=0.0, dtype=None):
        if isinstance(x, list) and len(x) == 3:
            obj = x
            if not isinstance(x, np.ndarray):
                obj = np.array(x, dtype=dtype)

            # matrix4
            if obj.shape in (4, 4,) or isinstance(obj, BaseMatrix4):
                obj = cls.from_matrix4_translation(obj, dtype=dtype)
        elif x is not None:
            obj = np.array((x, y, z), dtype)

            if obj.shape in (4, 4,) or isinstance(obj, BaseMatrix4):
                obj = cls.from_matrix4_translation(obj, dtype=dtype)
        else:
            obj = np.zeros(cls._shape, dtype=dtype)

        obj = obj.view(cls)
        return super(Vector3, cls).__new__(cls, obj)
    
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

    @dispatch((BaseObject, Number, np.number))
    def __xor__(self, other):
        self._unsupported_type('XOR', other)

    @dispatch((BaseObject, Number, np.number))
    def __or__(self, other):
        self._unsupported_type('OR', other)

    @dispatch((BaseObject, Number, np.number))
    def __ne__(self, other):
        self._unsupported_type('NE', other)

    @dispatch((BaseObject, Number, np.number))
    def __eq__(self, other):
        self._unsupported_type('EQ', other)
    
    # Vectors
    @dispatch((BaseVector3, np.ndarray, list))
    def __add__(self, other):
        return Vector3(super(Vector3, self).__add__(other))

    @dispatch((BaseVector3, np.ndarray, list))
    def __sub__(self, other):
        return Vector3(super(Vector3, self).__sub__(other))

    @dispatch((BaseVector3, np.ndarray, list))
    def __mul__(self, other):
        return Vector3(super(Vector3, self).__mul__(other))

    @dispatch((BaseVector3, np.ndarray, list))
    def __truediv__(self, other):
        return Vector3(super(Vector3, self).__truediv__(other))

    @dispatch((BaseVector3, np.ndarray, list))
    def __div__(self, other):
        return Vector3(super(Vector3, self).__div__(other))

    @dispatch((BaseVector3, np.ndarray, list))
    def __xor__(self, other):
        return self.cross(other)

    @dispatch((BaseVector3, np.ndarray, list))
    def __or__(self, other):
        return self.dot(other)

    @dispatch((BaseVector3, np.ndarray, list))
    def __ne__(self, other):
        return bool(np.any(super(Vector3, self).__ne__(other)))

    @dispatch((BaseVector3, np.ndarray, list))
    def __eq__(self, other):
        return bool(np.all(super(Vector3, self).__eq__(other)))
    
    # Number
    @dispatch((Number, np.number))
    def __add__(self, other):
        return Vector3(super(Vector3, self).__add__(other))

    @dispatch((Number, np.number))
    def __sub__(self, other):
        return Vector3(super(Vector3, self).__sub__(other))

    @dispatch((Number, np.number))
    def __mul__(self, other):
        return Vector3(super(Vector3, self).__mul__(other))

    @dispatch((Number, np.number))
    def __truediv__(self, other):
        return Vector3(super(Vector3, self).__truediv__(other))

    @dispatch((Number, np.number))
    def __div__(self, other):
        return Vector3(super(Vector3, self).__div__(other))
    
    # Methods and Properties
    @property
    def inverse(self):
        """Returns the opposite of this vector.
        """
        return Vector3(-self)

    @property
    def vec3(self):
        return self
    
    # Class methods
    @classmethod
    def create(cls, x=0., y=0., z=0., dtype=None):
        if isinstance(x, (list, np.ndarray)):
            raise ValueError('Function requires non-list arguments')
        return np.array([x, y, z], dtype=dtype)

    @classmethod
    def unit_x(cls, dtype=None):
        return np.array([1.0, 0.0, 0.0], dtype=dtype)

    @classmethod
    def unit_y(cls, dtype=None):
        return np.array([0.0, 1.0, 0.0], dtype=dtype)

    @classmethod
    def unit_z(cls, dtype=None):
        return np.array([0.0, 0.0, 1.0], dtype=dtype)

    @classmethod
    @parameters_as_numpy_arrays('mat')
    def from_matrix4_translation(cls, mat, dtype=None):
        return np.array(mat[:3, 3], dtype=dtype)

