import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def align_img(grid, x, mode="bilinear"):
    return F.grid_sample(
        x,
        grid=grid,
        mode=mode,
        padding_mode="border",
        align_corners=False,
    )

def uniform_norm_grid(grid_shape, dim=3):
    if dim == 2:
        x = torch.linspace(-1, 1, grid_shape[2])
        y = torch.linspace(-1, 1, grid_shape[3])
        grid = torch.meshgrid(x, y, indexing="ij")
    else:
        x = torch.linspace(-1, 1, grid_shape[2])
        y = torch.linspace(-1, 1, grid_shape[3])
        z = torch.linspace(-1, 1, grid_shape[4])
        grid = torch.meshgrid(x, y, z, indexing="ij")
    grid = torch.stack(grid, dim=-1).float()
    return grid

def rescale_intensity(array, out_range=(0, 1), percentiles=(0, 100)):
    if isinstance(array, torch.Tensor):
        array = array.float()

    if percentiles != (0, 100):
        cutoff = np.percentile(array, percentiles)
        np.clip(array, *cutoff, out=array)  # type: ignore[call-overload]
    in_min = array.min()
    in_range = array.max() - in_min
    out_min = out_range[0]
    out_range = out_range[1] - out_range[0]

    array -= in_min
    array /= in_range
    array *= out_range
    array += out_min
    return array

def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]