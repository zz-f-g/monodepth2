import torch
import torch.nn.functional as F
from layers import BackprojectDepth, Project3D
from typing import Tuple


def make_qp_params_random(
    batch_size: int,
    img_size: Tuple[int, int],
    qp_size: int,
    res_range: Tuple[float, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert isinstance(img_size, tuple) and 2 == len(img_size)
    H, W = img_size
    assert isinstance(res_range, tuple) and 2 == len(res_range)
    res_min, res_max = res_range
    resolution = res_min + (res_max - res_min) * torch.rand(batch_size)
    start_point = (
        torch.tensor([W, H]).view(1, -1) - qp_size * resolution.view(-1, 1)
    ) * torch.rand([batch_size, 2])
    assert bool((start_point > 0.0).all()), "qpsize too large or resolution too large"
    return (resolution, start_point)


def make_query_points(
    batch_size: int,
    img_size: Tuple[int, int],
    qpsize: int,
    qpresolution: torch.Tensor,
    start_point: torch.Tensor,
) -> torch.Tensor:
    assert isinstance(img_size, tuple) and 2 == len(img_size)
    H, W = img_size
    qpts = torch.stack(
        torch.meshgrid(
            torch.tensor(range(qpsize)), torch.tensor(range(qpsize)), indexing="ij"
        )[::-1],
        dim=2,
    ).view(1, qpsize, qpsize, 2)

    assert qpresolution.shape == (batch_size,)
    assert start_point.shape == (batch_size, 2)
    qpts = qpts * qpresolution.view(-1, 1, 1, 1) + start_point.view(batch_size, 1, 1, 2)
    return qpts.view(batch_size, -1, 2)


def warp_query_rgb(
    ref_rgb: torch.Tensor,
    query_points: torch.Tensor,
    query_depth: torch.Tensor,
    K: torch.Tensor,
    inv_K: torch.Tensor,
    T: torch.Tensor,
    qpsize: int,
):
    assert isinstance(ref_rgb, torch.Tensor) and ref_rgb.shape[1] == 3
    batch_size, _, H, W = ref_rgb.shape
    backproj = BackprojectDepth(batch_size, H, W, query_points).cuda()
    cam_points = backproj(query_depth, inv_K)
    proj3d = Project3D(batch_size, H, W, qpsize=qpsize).cuda()
    pix_coords = proj3d(cam_points, K, T)

    # outputs[("sample", frame_id, scale)] = pix_coords

    warp_res = F.grid_sample(
        ref_rgb,
        pix_coords,
        padding_mode="border",
        align_corners=True,
    )
    # warp_res = torch.ones(batch_size, 3, 48, 48)
    return warp_res


if __name__ == "__main__":
    torch.manual_seed(42)
    from PIL import Image
    import numpy as np
    from torchvision import transforms

    img_path = "assets/test_image.jpg"
    with open(img_path, "rb") as f:
        with Image.open(f) as img:
            img = transforms.ToTensor()(img.convert("RGB"))
    img = img.view(1, *img.shape)
    resolution, start_point = make_qp_params_random(
        1, tuple(img.shape[-2:]), 48, (0.5, 2)
    )
    grid = make_query_points(1, tuple(img.shape[-2:]), 48, resolution, start_point)
    img_sample = F.grid_sample(
        img, grid.view(1, 48, 48, 2), "bilinear", align_corners=False
    )
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1)
    axes[0].imshow(img[0].permute(1, 2, 0).numpy())
    axes[1].imshow(img_sample[0].permute(1, 2, 0).numpy())
    fig.show()
    input()
