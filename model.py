"""" This is the file for creating the model """
import random
import torch
import torch.nn as nn
import torch.nn.parallel

from torchsummary import summary

#from MRI_Classification.models.dcn.vox_resnet import VoxRes
#from MRI_Classification.models.dcn.vox_resnet import VoxResNet

# NOTE: These VoxRes blocks included from these imports are not identical to those used
# in the 3D deformable convolutions for MRI classification paper. Blocks used
# in said paper can be found below.

# 3DDeformableBlock from https://github.com/kondratevakate/3DDeformableConvolutions
from models.deformable_block_3d import DeformConv3d


class FullModel(nn.Module):
    """Replica of the dVoxResNet model from "3D Deformable Convolutions for MRI classification"
    https://arxiv.org/pdf/1911.01898.pdf.
    Keyword Args:
        n_f: Depth of first convolutional block (default 32).
        n_classes: Number of output classes (default 2).
        p: Probability of dropout
    """

    def __init__(self, n_f=32, n_classes=2, p=0.5):
        super(FullModel, self).__init__()

        # Fully convolutional backend
        self.main = nn.Sequential(
            # Conv3d x2
            ConvBlock3d(1, n_f),
            nn.Dropout(p=p),
            ConvBlock3d(n_f, n_f),
            nn.Dropout(p=p),
            # Conv3d w/ VoxRes
            ConvBlock3d(n_f, 2 * n_f),
            nn.Dropout(p=p),
            VoxResBlock3d(2 * n_f),
            # Conv3d w/ dVoxResNet x3
            # DeformConv3d layers inserted into baseline model in place of Conv3d layers 4-6
            # stride automatically maintains shape (so kept for VRN use),
            # so change to 2 to follow padding 1, stride 2, kernel 3 paper
            # convention
            DeformConv3d(2 * n_f, 2 * n_f, stride=2),
            nn.Dropout(p=p),
            DeformVoxResBlock3d(2 * n_f),
            DeformConv3d(2 * n_f, 4 * n_f, stride=2),
            nn.Dropout(p=p),
            DeformVoxResBlock3d(4 * n_f),
            DeformConv3d(4 * n_f, 4 * n_f, stride=2),
            nn.Dropout(p=p),
            DeformVoxResBlock3d(4 * n_f),
            # could be dropout2d to get rid of entire channels
            nn.Flatten(),
            # Hard-coded based on input size of 176*256*256
            nn.Linear(128*3*4*4, n_classes)
        )

    def forward(self, input):
        return self.main(input)


# =================================================================
# Block Classes
# =================================================================


class ConvBlock3d(nn.Module):
    """Standard 3D conv block as implemented in "3D Deformable Convolutions for MRI classification"
    https://arxiv.org/pdf/1911.01898.pdf.
    Args:
        in_channels: int, number of input channels.
        out_channels: int, number of output channels.
    Keyword Args:
        kernel_size: int or 3-tuple of ints, size of convolutional or d-conv kernel (default 3).
        stride: int or 3-tuple of ints, stride of convolutional or d-conv kernel (default 1). -> changed 2
        padding: int or 3-tuple of ints, size of input padding (default 0). -> changed 1
        bias: bool, Whether or not to include bias (default False).
    """
    # changed stride and padding defaults to match author's
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False):
        super(ConvBlock3d, self).__init__()
        self.main = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.main(x)


class VoxResBlock3d(nn.Module):
    """Standard 3D VoxRes block as implemented in "3D Deformable Convolutions for MRI classification"
    https://arxiv.org/pdf/1911.01898.pdf.
    Note that the output of the VoxRes block has the same depth as its input to allow for the residual
    connection.
    Args:
        in_channels: int, number of input channels.
    Keyword Args:
        kernel_size: int or 3-tuple of ints, size of convolutional or d-conv kernel (default 3).
        stride: int or 3-tuple of ints, stride of convolutional or d-conv kernel (default 1).
        padding: int or 3-tuple of ints, size of input padding (default 0).
        bias: bool, Whether or not to include bias (default False).
    """

    def __init__(self, in_channels, **kwargs):
        super(VoxResBlock3d, self).__init__()
        # changed stride, st the convolution doesn't change the size
        self.main = ConvBlock3d(in_channels, in_channels, stride=1, **kwargs)

    def forward(self, x):
        return x + self.main(x)

class DeformVoxResBlock3d(nn.Module):
    """Deformable 3D VoxRes block as implemented in "3D Deformable Convolutions for MRI classification"
    https://arxiv.org/pdf/1911.01898.pdf.
    Note that the output of the VoxRes block has the same depth as its input to allow for the residual
    connection.
    Args:
        in_channels: int, number of input channels.
    Keyword Args:
        kernel_size: int or 3-tuple of ints, size of convolutional or d-conv kernel (default 3).
        stride: int or 3-tuple of ints, stride of convolutional or d-conv kernel (default 1).
        padding: int or 3-tuple of ints, size of input padding (default 0).
        bias: bool, Whether or not to include bias (default False).
    """

    def __init__(self, in_channels, **kwargs):
        super(DeformVoxResBlock3d, self).__init__()
        self.main = nn.Sequential(
            # imported from 3DDeformableBlock, recreating non-imported
            # DeformBasicBlock code how we understand it
            DeformConv3d(in_channels, in_channels, **kwargs),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return x + self.main(x)


if __name__ == "__main__":
    """If run as script, prints model information."""

    model = FullModel()
    input_size = (1, 176, 256, 256)
    summary(model.cuda(), input_size)