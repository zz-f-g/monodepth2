import random
import torch
from posegeometry import pose_vec2mat
from typing import Tuple


def make_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def mc(
    H: int,
    W: int,
    Hrange: Tuple[int, int] | None = None,
    Wrange: Tuple[int, int] | None = None,
):
    """
    The function `make_coord` generates normalized coordinates at pixel centers
    for a image given its height and width, with optional ranges for the height
    and width.

    :param H: The parameter `H` represents the height of the grid
    :type H: int
    :param W: The parameter `W` represents the width of the grid
    :type W: int
    :param Hrange: The `Hrange` parameter is a tuple that specifies the range of
    values for the height (H) dimension. It allows you to select a subset of
    rows from the grid. The first element of the tuple represents the starting
    index (inclusive) and the second element represents the ending index
    (exclusive)
    :type Hrange: Tuple[int, int] | None
    :param Wrange: The `Wrange` parameter is a tuple that specifies the range of
    values for the width (W) dimension. It allows you to select a subset of the
    width values to generate coordinates for. The first element of the tuple
    represents the starting value (inclusive) and the second element represents
    the ending value (exclusive)
    :type Wrange: Tuple[int, int] | None
    :return: The function `make_coord` returns a tensor of shape `(H, W, 2)` containing
    the normalized coordinates of each pixel center in the image where each
    coordinates in [-1, 1].
    """
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(Hrange[0], Hrange[1]) if Hrange else torch.arange(H),
            torch.arange(Wrange[0], Wrange[1]) if Wrange else torch.arange(W),
            indexing="ij",
        )[::-1],
        dim=-1,
    ).to(torch.float32)
    coords += 0.5
    coords[..., 0] /= W
    coords[..., 1] /= H
    coords *= 2
    coords -= 1
    return coords


def get_tgt_coord(
    B: int,
    H: int,
    W: int,
    mode="train",
    size=48,
    Hrange: Tuple[int, int] | None = None,
    Wrange: Tuple[int, int] | None = None,
):
    if mode == "train":
        start_h = random.randint(0, H - size)
        start_w = random.randint(0, W - size)
        return (
            mc(H, W, (start_h, start_h + size), (start_w, start_w + size))
            .unsqueeze(0)
            .repeat(B, 1, 1, 1)
        )
    elif mode == "val":
        return mc(H, W, Hrange=Hrange, Wrange=Wrange).unsqueeze(0).repeat(B, 1, 1, 1)
    else:
        raise NotImplementedError


def get_ref_coord_depth(
    tgt_coord: torch.Tensor,
    tgt_depth: torch.Tensor,
    pose: torch.Tensor,
    intrinsic: torch.Tensor,
    inv_intrinsic: torch.Tensor,
    H: int,
    W: int,
):
    B, N, _ = tgt_coord.shape
    assert tgt_coord.shape == (B, N, 2)
    assert tgt_depth.shape == (B, N, 1)
    assert pose.shape == (B, 6)
    assert intrinsic.shape == (B, 4, 4)
    assert inv_intrinsic.shape == (B, 4, 4)

    tgt_coord_homogenous = tgt_coord.clone()
    tgt_coord_homogenous += 1
    tgt_coord_homogenous /= 2
    tgt_coord_homogenous[..., 0] *= W
    tgt_coord_homogenous[..., 1] *= H
    tgt_coord_homogenous = torch.cat(
        [
            tgt_coord_homogenous,
            torch.ones(B, N, 2).type_as(tgt_coord),
        ],
        dim=-1,
    ).permute(
        0, 2, 1
    )  # [B, 4, N] (u, v, 1, 1)

    tgt_world_point = torch.matmul(
        inv_intrinsic, tgt_coord_homogenous
    )  # [B, 4, N] (x, y, 1, 1)
    tgt_world_point[:, :3, :] *= tgt_depth.permute(0, 2, 1)  # [B, 4, N] (x, y, z, 1)

    T = pose_vec2mat(pose)  # [B, 3, 4]
    # T = (
    #     torch.cat(
    #         [
    #             torch.eye(3, dtype=torch.float32),
    #             torch.zeros([3, 1], dtype=torch.float32),
    #         ],
    #         dim=1,
    #     )
    #     .cuda()
    #     .unsqueeze(0)
    #     .repeat([B, 1, 1])
    # )
    P = torch.matmul(intrinsic[:, :3, :3], T)
    ref_cam_point = torch.matmul(P, tgt_world_point)  # [B, 3, N] (u * d, v * d, d)
    ref_coord = (ref_cam_point[:, :2, :] / (ref_cam_point[:, 2:, :] + 1e-7)).permute(
        0, 2, 1
    )  # [B, N, 2] (u, v)
    ref_coord[:, :, 0] /= W
    ref_coord[:, :, 1] /= H
    ref_coord *= 2
    ref_coord -= 1
    ref_computed_depth = ref_cam_point[:, 2, :].unsqueeze(-1)  # [B, N]
    return ref_coord, ref_computed_depth


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    """from liif"""
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == "benchmark":
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == "div2k":
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)
