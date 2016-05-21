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
    vec3,
    vec4,
    mat3,
    mat4,
    quat
)

from . import funcs, objects

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
    'funcs',
    'objects'
]

