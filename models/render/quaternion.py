# Copyright Philipp Jund (jundp@cs.uni-freiburg.de) and Eldar Insafutdinov, 2018.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Website: https://github.com/PhilJd/tf-quaternion

import numpy as np
import torch
import torch.nn.functional as F


def validate_shape(x):
    """Raise a value error if x.shape ist not (..., 4)."""
    error_msg = ("Can't create a quaternion from a tensor with shape {}."
                 "The last dimension must be 4.")
    # Check is performed during graph construction. If your dimension
    # is unknown, tf.reshape(x, (-1, 4)) might work.
    if x.shape[-1] != 4:
        raise ValueError(error_msg.format(x.shape))


def vector3d_to_quaternion(x):
    """Convert a tensor of 3D vectors to a quaternion.
    Prepends a 0 to the last dimension, i.e. [[1,2,3]] -> [[0,1,2,3]].
    Args:
        x: A `tf.Tensor` of rank R, the last dimension must be 3.
    Returns:
        A `Quaternion` of Rank R with the last dimension being 4.
    Raises:
        ValueError, if the last dimension of x is not 3.
    """
    # x = tf.convert_to_tensor(x)
    if x.shape[-1] != 3:
        raise ValueError("The last dimension of x must be 3.")
    # return torch.tensor(np.pad(x, ((0, 0), (0, 0), (1, 0)), 'constant'))
    return F.pad(x, (1, 0), 'constant', 0)


def _prepare_tensor_for_div_mul(x):
    """Prepare the tensor x for division/multiplication.
    This function
    a) converts x to a tensor if necessary,
    b) prepends a 0 in the last dimension if the last dimension is 3,
    c) validates the type and shape.
    """
    # x = tf.convert_to_tensor(x)
    if x.shape[-1] == 3:
        x = vector3d_to_quaternion(x)
    validate_shape(x)
    return x


def quaternion_multiply(a, b):
    """Multiply two quaternion tensors.
    Note that this differs from tf.multiply and is not commutative.
    Args:
        a, b: A `tf.Tensor` with shape (..., 4).
    Returns:
        A `Quaternion`.
    """
    a = _prepare_tensor_for_div_mul(a)
    b = _prepare_tensor_for_div_mul(b)
    # print(a.shape)
    # print(b.shape)
    w1, x1, y1, z1 = torch.unbind(a, dim=-1)
    w2, x2, y2, z2 = torch.unbind(b, dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.stack((w, x, y, z), dim=-1)


def quaternion_conjugate(q):
    """Compute the conjugate of q, i.e. [q.w, -q.x, -q.y, -q.z]."""
    return torch.mul(q, torch.Tensor([1.0, -1.0, -1.0, -1.0]).cuda())


def quaternion_rotate(pc, q, inverse=False):
    """rotates a set of 3D points by a rotation,
    represented as a quaternion
    Args:
        pc: [B,N,3] point cloud
        q: [B,4] rotation quaternion
    Returns:
        q * pc * q'
    """
    
    q = torch.tensor(q).cuda()
    q_norm = torch.unsqueeze(torch.norm(q, dim=1), axis=-1)
    q /= q_norm
    q = torch.unsqueeze(q, axis=1)  # [B,1,4]
    q_ = quaternion_conjugate(q)
    qmul = quaternion_multiply
    if not inverse:
        wxyz = qmul(qmul(q, pc), q_)  # [B,N,4]
    else:
        wxyz = qmul(qmul(q_, pc), q)  # [B,N,4]
    if len(wxyz.shape) == 2: # bug with batch size of 1
        wxyz = torch.unsqueeze(wxyz, axis=0)
    xyz = wxyz[:, :, 1:4]  # [B,N,3]
    # print(f'xyz shape: {xyz.shape}')
    return xyz
