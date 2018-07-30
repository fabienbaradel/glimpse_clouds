from PIL import Image, ImageDraw
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import random

def zoom_ST(x, loc, zoom, H_prim, W_prim, cuda: bool = True):
    """
    Spatial Transformer operation apply on images given the zoom and the location
    Args:
        x (Variable): input images (B x C x H x W)
        loc (Variable): location of the focus point -- height and width location (B, 2)
        zoom (Variable): zoom for each image -- zoom for height and width (B, 2)
        H_prim (int): height output size
        W_prim (int): width output size
        cuda (bool): on GPU or not
    Returns:
        grid_sample (Variable): output Tensor of size (B x C x H' x W')
    """
    # Create the theta matrix
    B = x.size(0)
    theta = torch.zeros(B, 2, 3)
    theta[:, 0, 0] = zoom[:, 0].data  # zoom on height
    theta[:, 1, 1] = zoom[:, 1].data  # zoom on width
    theta[:, :, -1] = loc.data

    # Get the affine grid (2D flow fields)
    C = x.size(1)
    output_size = torch.Size((B, C, H_prim, W_prim))
    affine_grid = nn.functional.affine_grid(theta, output_size)
    if cuda:
        affine_grid = affine_grid.cuda()
    grid_sample = nn.functional.grid_sample(x, affine_grid)

    return grid_sample