# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

# the version of software
# this is used by the setup.py script
from .version import __version__


from .objects import (
    Matrix3,
    Matrix4,
    Quaternion,
    Vector3,
    Vector4,
)

vec3 = Vector3
vec4 = Vector4
quat = Quaternion
mat3 = Matrix3
mat4 = Matrix4

__all__ = [
    'Vector3',
    'Vector4',
    'Matrix3',
    'Matrix4',
    'Quaternion',
    'mat4',
    'mat3',
    'vec3',
    'vec4',
]

