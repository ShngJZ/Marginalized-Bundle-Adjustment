import torch
import torch.nn.functional as F

def wrap_to_parameter(tensor, requires_grad):
    return torch.nn.Parameter(torch.clone(tensor).detach(), requires_grad=requires_grad)

def pose_to_r6_t(pose: torch.Tensor):
    """Converts rotation matrix to 9D representation.

    We take the two first ROWS of the rotation matrix,
    along with the translation vector.
    ATTENTION: row or vector needs to be consistent from pose_to_d9 and r6d2mat
    """
    nbatch = pose.shape[0]
    R, t = pose[:, :3, :3], pose[:, :3, 3:4]
    r6 = R[:, :2, :3].reshape(nbatch, -1)  # [N, 6]

    return r6, t

def r6d2mat(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6). Here corresponds to the two
            first two rows of the rotation matrix.
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)  # corresponds to row

