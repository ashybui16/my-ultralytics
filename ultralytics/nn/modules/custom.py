import torch
import torch.nn as nn
import math

from ultralytics.nn.modules.conv import autopad, Conv
from ultralytics.nn.modules.block import C3

__all__ = (
    "PConv",
    "SPDConv",
    "Res2Block",
    "CSPRes2B",
    "LEAF",
    "LEAFT",
    "ELAN",
)


class PConv(nn.Module):
    """FasterNet partial convolution."""

    def __init__(self, c1: int, k: int = 3, n_div: int = 4):
        super().__init__()

        self.c_partial = c1 // n_div
        self.c_untouched = c1 - self.c_partial
        self.partial_conv = nn.Conv2d(self.c_partial, self.c_partial, k, 1, autopad(k), bias=False)
        self.channel_mixer = Conv(c1, c1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.forward_split_cat(x)
        return self.forward_slicing(x)

    def forward_split_cat(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, (self.c_partial, self.c_untouched), dim=1)
        return self.channel_mixer(torch.cat((self.partial_conv(x1), x2), dim=1))

    def forward_slicing(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        y[:, : self.c_partial] = self.partial_conv(x[:, : self.c_partial])
        return self.channel_mixer(y)


class SPDConv(nn.Module):
    """Downsample by space-to-depth followed by a non-strided convolution."""

    def __init__(self, c1, c2, k=3, scale=2):
        super().__init__()
        self.scale = scale

        c_ = c1 * scale**2
        self.conv = Conv(c_, c2, k)

    def forward(self, x):
        s = self.scale
        x = torch.cat(
            [x[..., row::s, col::s] for col in range(s) for row in range(s)],
            dim=1,
        )
        return self.conv(x)


class Res2Block(nn.Module):
    """Res2Block from LEAF-YOLO."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        base_width: int = 16,
        scale: int = 4,
    ):
        super().__init__()
        self.scale = scale
        self.add = shortcut and c1 == c2

        width = int(math.floor(c2 * (base_width/64.0)))
        self.conv1 = Conv(c1, width*scale, 1)

        self.nums = 1 if scale == 1 else scale - 1
        self.convs = nn.ModuleList(Conv(width, width, 3) for _ in range(self.nums))
        
        self.conv2 = Conv(width*scale, c2, 1, act=False)
        self.act = nn.SiLU(inplace=True)
        self.scale = scale
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.add else None

        projected = self.conv1(x)
        splits = torch.split(projected, self.width, dim=1)

        out = None
        branch = None
        for i in range(self.nums):
            branch = splits[i] if i == 0 else branch + splits[i]
            branch = self.convs[i](branch)
            out = branch if i == 0 else torch.cat((out, branch), dim=1)

        if self.scale != 1:
            out = torch.cat((out, splits[self.nums]), dim=1)

        out = self.conv2(out)
        if residual is not None:
            out = out + residual
        return self.silu(out)


class CSPRes2B(C3):
    """CSPRes2B from LEAF-YOLO."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, baseWidth=16, scale=4):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Res2Block(c_, c_, shortcut, baseWidth, scale) for _ in range(n)))


class LEAF(nn.Module):
    """LEAF from LEAF-YOLO."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        n_div: int = 4,
        baseWidth: int = 16,
        scale: int = 4,
    ):
        super().__init__()

        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.pconvs = nn.ModuleList(PConv(c_, k=3, n_div=n_div) for _ in range(4))
        self.csp = CSPRes2B(
            4 * c_,
            c2,
            n=n,
            shortcut=shortcut,
            g=g,
            e=e,
            baseWidth=baseWidth,
            scale=scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.cv1(x)
        second = self.cv2(x)
        states = [second]
        for pconv in self.pconvs:
            states.append(pconv(states[-1]))
        
        return self.csp(torch.cat((first, states[0], states[2], states[4]), dim=1))


class LEAFT(nn.Module):
    """LEAF-T from LEAF-YOLO."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        baseWidth: int = 16,
        scale: int = 4,
    ):
        super().__init__()

        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.convs = nn.ModuleList(Conv(c_, c_, 3) for _ in range(2))
        self.csp = CSPRes2B(
            4 * c_,
            c2,
            n=n,
            shortcut=shortcut,
            g=g,
            e=e,
            baseWidth=baseWidth,
            scale=scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.cv1(x)
        second = self.cv2(x)
        states = [second]
        for conv in self.convs:
            states.append(conv(states[-1]))

        return self.csp(torch.cat((first, *states), dim=1))

class ELAN(C3):
    """ELAN from LEAF-YOLO."""

    def __init__(self, c1, c2, n=2, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Conv(c_, c_, 3) for _ in range(n)))