# -*- coding: utf-8 -*-
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from . import base, matrix3, matrix4, quaternion, vector3, vector4
from .matrix3 import Matrix3
from .matrix4 import Matrix4
from .quaternion import Quaternion
from .vector3 import Vector3
from .vector4 import Vector4

vec3 = Vector3
vec4 = Vector4
quat = Quaternion
mat3 = Matrix3
mat4 = Matrix4

__all__ = [
    'base',
    'matrix3',
    'matrix4',
    'quaternion',
    'vector3',
    'vector4',
    'vec3',
    'vec4',
    'mat3',
    'mat4',
    'quat'
]

