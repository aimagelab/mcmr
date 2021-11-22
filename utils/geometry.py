import math
from typing import Union

import torch
import numpy as np


def x_rot(alpha: float,
          clockwise: bool=False,
          pytorch: bool=False
          ) -> Union[np.ndarray, torch.Tensor]:
    """
    Compose a rotation matrix around X axis (default: counter-clockwise).
    :param alpha: Rotation angle in radians.
    :param clockwise: Default rotation convention is counter-clockwise.
     In case the `clockwise` flag is set, the sign of `sin(alpha)` is reversed
     to rotate clockwise.
    :param pytorch: In case the `pytorch` flag is set, all operation are
     between torch tensors and a torch.Tensor is returned .
    :return rot: Rotation matrix around X axis.
    """
    if pytorch:
        cx = torch.cos(alpha)
        sx = torch.sin(alpha)
    else:
        cx = np.cos(alpha)
        sx = np.sin(alpha)

    if clockwise:
        sx *= -1

    if pytorch:
        zero = torch.zeros(1)
        one = torch.ones(1)
        rot = torch.cat([torch.stack([one, zero, zero], dim=1),
                         torch.stack([zero, cx, -sx], dim=1),
                         torch.stack([zero, sx, cx], dim=1)], dim=0)
    else:
        rot = np.asarray([[1., 0., 0.],
                          [0., cx, -sx],
                          [0., sx, cx]], dtype=np.float32)
    return rot


def y_rot(alpha: float,
          clockwise: bool=False,
          pytorch: bool=False
          ) -> Union[np.ndarray, torch.Tensor]:
    """
    Compose a rotation matrix around Y axis (default: counter-clockwise).
    :param alpha: Rotation angle in radians.
    :param clockwise: Default rotation convention is counter-clockwise.
     In case the `clockwise` flag is set, the sign of `sin(alpha)` is reversed
     to rotate clockwise.
    :param pytorch: In case the `pytorch` flag is set, all operation are
     between torch tensors and a torch.Tensor is returned .
    :return rot: Rotation matrix around Y axis.
    """
    if pytorch:
        cy = torch.cos(alpha)
        sy = torch.sin(alpha)
    else:
        cy = np.cos(alpha)
        sy = np.sin(alpha)

    if clockwise:
        sy *= -1

    if pytorch:
        zero = torch.zeros(1)
        one = torch.ones(1)
        rot = torch.cat([torch.stack([cy, zero, sy], dim=1),
                         torch.stack([zero, one, zero], dim=1),
                         torch.stack([-sy, zero, cy], dim=1)], dim=0)
    else:
        rot = np.asarray([[cy, 0., sy],
                          [0., 1., 0.],
                          [-sy, 0., cy]], dtype=np.float32)
    return rot


def z_rot(alpha: float,
          clockwise: bool=False,
          pytorch: bool=False
          ) -> Union[np.ndarray, torch.Tensor]:
    """
    Compose a rotation matrix around Z axis (default: counter-clockwise).
    :param alpha: Rotation angle in radians.
    :param clockwise: Default rotation convention is counter-clockwise.
     In case the `clockwise` flag is set, the sign of `sin(alpha)` is reversed
     to rotate clockwise.
    :param pytorch: In case the `pytorch` flag is set, all operation are
     between torch tensors and a torch.Tensor is returned .
    :return rot: Rotation matrix around Z axis.
    """
    if pytorch:
        cz = torch.cos(alpha)
        sz = torch.sin(alpha)
    else:
        cz = np.cos(alpha)
        sz = np.sin(alpha)

    if clockwise:
        sz *= -1

    if pytorch:
        zero = torch.zeros(1)
        one = torch.ones(1)
        rot = torch.cat([torch.stack([cz, -sz, zero], dim=1),
                         torch.stack([sz, cz, zero], dim=1),
                         torch.stack([zero, zero, one], dim=1)], dim=0)
    else:
        rot = np.asarray([[cz, -sz, 0.],
                          [sz, cz, 0.],
                          [0., 0., 1.]], dtype=np.float32)

    return rot


def intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Return intrinsics camera matrix with square pixel and no skew.
    :param focal: Focal length
    :param cx: X coordinate of principal point
    :param cy: Y coordinate of principal point
    :return K: intrinsics matrix of shape (3, 3)
    """
    return np.asarray([[fx, 0., cx],
                       [0., fy, cy],
                       [0., 0., 1.]])


def pascal_vpoint_to_extrinsics(az_deg: float,
                                el_deg: float,
                                radius: float):
    """
    Convert Pascal viewpoint to a camera extrinsic matrix which
     we can use to project 3D points from the CAD
    :param az_deg: Angle of rotation around X axis (degrees)
    :param el_deg: Angle of rotation around Y axis (degrees)
    :param radius: Distance from the origin
    :return extrinsic: Extrinsic matrix of shape (4, 4)
    """
    az_ours = np.radians(az_deg - 90)
    el_ours = np.radians(90 - el_deg)

    # Compose the rototranslation for a camera with look-at at the origin
    Rc = z_rot(az_ours) @ y_rot(el_ours)
    Rc[:, 0], Rc[:, 1] = Rc[:, 1].copy(), Rc[:, 0].copy()
    z_dir = Rc[:, -1] / np.linalg.norm(Rc[:, -1])
    Rc[:, -1] *= -1  # right-handed -> left-handed
    t = np.expand_dims(radius * z_dir, axis=-1)

    # Invert camera roto-translation to get the extrinsic
    #  see: http://ksimek.github.io/2012/08/22/extrinsic/
    extrinsic = np.concatenate([Rc.T, -Rc.T @ t], axis=1)
    return extrinsic


def project_points(points_3d: np.array,
                   intrinsic: np.array,
                   extrinsic: np.array,
                   scale: float = 1.) -> np.array:
    """
    Project 3D points in 2D according to pinhole camera model.

    :param points_3d: 3D points to be projected (n_points, 3)
    :param intrinsic: Intrinsics camera matrix
    :param extrinsic: Extrinsics camera matrix
    :param scale: Object scale (default: 1.0)
    :return projected: 2D projected points (n_points, 2)
    """
    n_points = points_3d.shape[0]

    assert points_3d.shape == (n_points, 3)
    assert extrinsic.shape == (3, 4) or extrinsic.shape == (4, 4)
    assert intrinsic.shape == (3, 3)

    if extrinsic.shape == (4, 4):
        if not np.all(extrinsic[-1, :] == np.asarray([0, 0, 0, 1])):
            raise ValueError('Format for extrinsic not valid')
        extrinsic = extrinsic[:3, :]

    points3d_h = np.concatenate([points_3d, np.ones(shape=(n_points, 1))], 1)

    points3d_h[:, :-1] *= scale
    projected = intrinsic @ (extrinsic @ points3d_h.T)
    projected /= projected[2, :]
    projected = projected.T
    return projected[:, :2]


def project_points_tensor(points_3d: torch.Tensor, intrinsic: torch.Tensor, extrinsic: torch.Tensor) -> torch.Tensor:
    """
    Project 3D points in 2D according to pinhole camera model.

    :param points_3d: 3D points to be projected (n_points, 3)
    :param intrinsic: Intrinsic camera matrix
    :param extrinsic: Extrinsic camera matrix
    :return projected: 2D projected points (n_points, 2)
    """
    n_points = points_3d.shape[1]

    assert points_3d.shape == (points_3d.shape[0], n_points, 3)
    assert extrinsic.shape[1:] == (3, 4) or extrinsic.shape[1:] == (4, 4)
    assert intrinsic.shape[1:] == (3, 3)

    if extrinsic.shape[1:] == (4, 4):
        if not torch.all(extrinsic[:, -1, :] == torch.FloatTensor([0, 0, 0, 1])):
            raise ValueError('Format for extrinsic not valid')
        extrinsic = extrinsic[:, 3, :]

    points3d_h = torch.cat([points_3d, torch.ones(points_3d.shape[0], n_points, 1).to(points_3d.device)], 2)

    projected = intrinsic @ extrinsic @ points3d_h.transpose(1, 2)
    projected = projected / projected[:, 2, :][:, None, :]
    projected = projected.transpose(1, 2)

    return projected[:, :, :2]


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                      [m01+m10,     m11-m00-m22, 0.0,         0.0],
                      [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                      [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def hamilton_product(qa, qb):
    """Multiply qa by qb.
    Args:
        qa: B X N X 4 quaternions
        qb: B X N X 4 quaternions
    Returns:
        q_mult: B X N X 4
    """
    qa_0 = qa[..., 0]
    qa_1 = qa[..., 1]
    qa_2 = qa[..., 2]
    qa_3 = qa[..., 3]

    qb_0 = qb[..., 0]
    qb_1 = qb[..., 1]
    qb_2 = qb[..., 2]
    qb_3 = qb[..., 3]

    # See https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    q_mult_0 = qa_0 * qb_0 - qa_1 * qb_1 - qa_2 * qb_2 - qa_3 * qb_3
    q_mult_1 = qa_0 * qb_1 + qa_1 * qb_0 + qa_2 * qb_3 - qa_3 * qb_2
    q_mult_2 = qa_0 * qb_2 - qa_1 * qb_3 + qa_2 * qb_0 + qa_3 * qb_1
    q_mult_3 = qa_0 * qb_3 + qa_1 * qb_2 - qa_2 * qb_1 + qa_3 * qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2, q_mult_3], dim=-1)


def axisangle2quat(axis, angle):
    """
    axis: B x 3: [axis]
    angle: B: [angle]
    returns quaternion: B x 4
    """
    axis = torch.nn.functional.normalize(axis, dim=-1)
    angle = angle.unsqueeze(-1) / 2
    quat = torch.cat([angle.cos(), angle.sin() * axis], dim=-1)
    return quat
