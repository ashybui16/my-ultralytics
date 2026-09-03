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
    "GhostPConv",
    "CoordAtt",
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
    ):
        super().__init__()
        self.add = shortcut and c1 == c2

        self.c_ =  c1 // 4
        self.convs = nn.ModuleList(Conv(self.c_, self.c_, 3) for _ in range(3))
        
        self.conv2 = Conv(c1, c2, 1, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.add else None

        splits = torch.split(x, self.c_, dim=1)

        out = None
        branch = None
        for i in range(3):
            branch = splits[i] if i == 0 else branch + splits[i]
            branch = self.convs[i](branch)
            out = branch if i == 0 else torch.cat((out, branch), dim=1)

        out = torch.cat((out, splits[3]), dim=1)

        out = self.conv2(out)
        if residual is not None:
            out = out + residual
        return self.act(out)


class CSPRes2B(C3):
    """CSPRes2B from LEAF-YOLO."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Res2Block(c_, c_, shortcut) for _ in range(n)))


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
        n_div: int = 4,
    ):
        super().__init__()

        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.pconvs = nn.ModuleList(PConv(c_, k=3, n_div=n_div) for _ in range(2))
        self.csp = CSPRes2B(
            4 * c_,
            c2,
            n=n,
            shortcut=shortcut,
            g=g,
            e=e,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.cv1(x)
        second = self.cv2(x)
        states = [second]
        for pconv in self.pconvs:
            states.append(pconv(states[-1]))

        return self.csp(torch.cat((first, *states), dim=1))

class ELAN(C3):
    """ELAN from LEAF-YOLO."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, n_div=4):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(PConv(c_, 3, n_div=n_div) for _ in range(n)))


class GhostPConv(nn.Module):
    """GhostConv using PConv."""

    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s)
        self.cv2 = PConv(c_, k, n_div=4)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class CoordAtt(nn.Module):
    """Coordinate Attention."""
    def __init__(self, c1, r=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        c_ = max(8, c1 // r)

        self.conv1 = nn.Conv2d(c1, c_, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(c_)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(c_, c1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(c_, c1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        _, _, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out