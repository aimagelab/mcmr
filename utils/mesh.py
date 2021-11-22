import numpy as np
import torch


def convert_3d_to_uv_coordinates(X):
    """
    X : N,3
    Returns UV: N,2 normalized to [-1, 1]
    U: Azimuth: Angle with +X [-pi,pi]
    V: Inclination: Angle with +Z [0,pi]
    """
    if type(X) == torch.Tensor:
        eps = 1e-4
        rad = torch.norm(X, dim=-1).clamp(min=eps)
        theta = torch.acos((X[..., 2] / rad).clamp(min=-1+eps, max=1-eps))  # Inclination: Angle with +Z [0,pi]
        phi = torch.atan2(X[..., 1], X[..., 0])  # Azimuth: Angle with +X [-pi,pi]
        vv = (theta / np.pi) * 2 - 1
        uu = ((phi + np.pi) / (2 * np.pi)) * 2 - 1
        uv = torch.stack([uu, vv], dim=-1)
    else:
        rad = np.linalg.norm(X, axis=-1)
        rad = np.clip(rad, 1e-12, None)
        theta = np.arccos(X[..., 2] / rad)      # Inclination: Angle with +Z [0,pi]
        phi = np.arctan2(X[..., 1], X[..., 0])  # Azimuth: Angle with +X [-pi,pi]
        vv = (theta / np.pi) * 2 - 1
        uu = ((phi + np.pi) / (2*np.pi)) * 2 - 1
        uv = np.stack([uu, vv], -1)
    return uv


def compute_uvsampler_softras(verts_sphere, faces, tex_size=2, convert_3d_to_uv=True, shift_uv=False):
    """
    For this mesh, pre-computes the UV coordinates for
    F x T x T points.
    Returns F x T x T x 2
    """
    # alpha_beta_txtx2x2[i,j,0,0] = i + 1/3
    # alpha_beta_txtx2x2[i,j,0,1] = j + 1/3
    # alpha_beta_txtx2x2[i,j,1,0] = i + 2/3
    # alpha_beta_txtx2x2[i,j,1,1] = j + 2/3
    alpha_beta_txtx2x2 = np.zeros((tex_size, tex_size, 2, 2))  # last dim for alpha beta
    alpha_beta_txtx2x2[:, :, :, 0] += np.arange(tex_size).reshape(tex_size, 1, 1)
    alpha_beta_txtx2x2[:, :, :, 1] += np.arange(tex_size).reshape(1, tex_size, 1)
    alpha_beta_txtx2x2[:, :, 0, :] += 1/3
    alpha_beta_txtx2x2[:, :, 1, :] += 2/3
    alpha_beta_txtx2x2 = alpha_beta_txtx2x2 / tex_size

    lower_half = alpha_beta_txtx2x2[:, :, 0, :]
    # upper_half = np.transpose(alpha_beta_txtx2x2[:,:,1,:], (1,0,2))[::-1,::-1,:]
    upper_half = alpha_beta_txtx2x2[::-1, ::-1, 1, :]
    upper_half = np.ascontiguousarray(upper_half)
    coords_txtx2 = np.where(lower_half.sum(axis=-1, keepdims=True) < 1, lower_half, upper_half)

    # coords_txtx2 = np.transpose(coords_txtx2, (1, 0, 2))
    # coords_txtx2 = np.flip(coords_txtx2,axis=0)
    # coords_txtx2 = np.flip(coords_txtx2,axis=1)
    coords_txtx2 = np.ascontiguousarray(coords_txtx2)

    vs = verts_sphere[faces]
    v0 = vs[:, 2]
    v0v1 = vs[:, 1] - vs[:, 2]
    v0v2 = vs[:, 0] - vs[:, 2]
    samples_Fx3xtxt = np.inner(np.dstack([v0v1, v0v2]), coords_txtx2) + v0.reshape(faces.shape[0],
                                                                                   verts_sphere.shape[-1], 1, 1)
    samples_Fxtxtx3 = np.transpose(samples_Fx3xtxt, (0, 2, 3, 1))

    # Now convert these to uv.
    if convert_3d_to_uv:
        uv_Fxtxtx2 = convert_3d_to_uv_coordinates(samples_Fxtxtx3)

        if shift_uv:
            # u -> u+0.5
            uv_Fxtxtx2[..., 0] = uv_Fxtxtx2[..., 0] + 0.5
            uv_Fxtxtx2 = np.where(uv_Fxtxtx2 >= 1, uv_Fxtxtx2 - 2 + 1e-12, uv_Fxtxtx2)
        return uv_Fxtxtx2
    else:
        return samples_Fxtxtx3
