import torch

from utils.geometry import hamilton_product
from utils.laplacian import GraphLaplacian


##############################
# SHAPE AND VIEWPOINT LOSSES #
##############################

def camera_loss(cam_pred, cam_gt, margin):
    """
    cam_* are B x 7, [sc, tx, ty, quat]
    Losses are in similar magnitude so one margin is ok.
    """
    rot_pred = cam_pred[:, -4:]
    rot_gt = cam_gt[:, -4:]

    rot_loss = hinge_loss(quat_loss_geodesic(rot_pred, rot_gt), margin)
    # Scale and trans.
    st_loss = (cam_pred[:, :3] - cam_gt[:, :3])**2
    st_loss = hinge_loss(st_loss.view(-1), margin)

    return rot_loss.mean() + st_loss.mean()


def hinge_loss(loss, margin):
    # Only penalize if loss > margin
    zeros = torch.autograd.Variable(torch.zeros(1).cuda(), requires_grad=False)

    return torch.max(loss - margin, zeros)


def quat_loss_geodesic(q1, q2):
    """
    Geodesic rotation loss.
    
    Args:
        q1: N X 4
        q2: N X 4
    Returns:
        loss : N x 1
    """
    q1 = torch.unsqueeze(q1, 1)
    q2 = torch.unsqueeze(q2, 1)
    q2_conj = torch.cat([ q2[:, :, [0]] , -1*q2[:, :, 1:4] ], dim=-1)
    q_rel = hamilton_product(q1, q2_conj)
    q_loss = 1 - torch.abs(q_rel[:, :, 0])

    # we can also return q_loss*q_loss
    return q_loss


def quat_reg(q):
    q = ((q ** 2).sum(dim=1) - 1) ** 2

    return q.mean()


def deform_l2reg(V):
    """
    l2 norm on V = B x N x 3
    """
    V = V.view(-1, V.size(2))

    return torch.mean(torch.norm(V, p=2, dim=1))


def kp_l2_loss(kp_pred, kp_gt):
    """
    L2 loss between visible keypoints.

    \Sum_i [0.5 * vis[i] * (kp_gt[i] - kp_pred[i])^2] / (|vis|)
    """
    criterion = torch.nn.MSELoss()

    vis = (kp_gt[:, :, 2, None] > 0).float()
    # vis = torch.ones(kp_gt.shape[0], kp_gt.shape[1], 1).to(kp_gt.device)

    # This always has to be (output, target), not (target, output)
    return criterion(vis * kp_pred, vis * kp_gt[:, :, :2])


class GraphLaplacianLoss(torch.nn.Module):
    """
    Encourages vertices to lie close to mean of neighbours
    """
    def __init__(self, faces, numV):
        # Input:
        # faces: B x F x 3
        super(GraphLaplacianLoss, self).__init__()
        self.laplacian = GraphLaplacian(faces, numV)

    def forward(self, verts):
        Lx = self.laplacian(verts)  # B,V
        return Lx.mean()
