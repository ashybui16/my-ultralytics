import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules.block import C3, C2f
from ultralytics.nn.modules.conv import Conv, GhostConv, autopad

__all__ = (
    "CARAFE",
    "ELAN",
    "LEAF",
    "LEAFT",
    "MGC",
    "SPPRFEM",
    "CSPRes2B",
    "CoordAtt",
    "CoordBlock",
    "DySample",
    "NeXt",
    "NeXtC2f",
    "PConv",
    "Res2Block",
    "SPDConv",
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
        x1, x2 = torch.split(x, (self.c_partial, self.c_untouched), dim=1)
        return self.channel_mixer(torch.cat((self.partial_conv(x1), x2), dim=1))


class SPDConv(nn.Module):
    """Downsample by space-to-depth followed by a non-strided convolution."""

    def __init__(self, c1, c2, k=3, scale=2, ds=False):
        super().__init__()
        self.scale = scale

        c_ = c1 * scale**2
        if ds:
            self.conv = nn.Sequential(
                Conv(c_, c2, 1),
                Conv(c2, c2, k, g=c2),
            )
        else:
            self.conv = Conv(c_, c2, k)

    def forward(self, x):
        s = self.scale
        x = torch.cat(
            [x[..., row::s, col::s] for col in range(s) for row in range(s)],
            dim=1,
        )
        return self.conv(x)


class Res2Block(nn.Module):
    """Res2Net bottleneck from LEAF-YOLO."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        base_width: int = 8,
        scale: int = 5,
    ):
        super().__init__()
        self.add = shortcut and c1 == c2

        self.c_ = c2 * base_width // 64
        self.conv1 = Conv(c1, self.c_ * scale, 1)
        self.convs = nn.ModuleList(Conv(self.c_, self.c_, 3) for _ in range(scale - 1))
        self.conv2 = Conv(self.c_ * scale, c2, 1, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        splits = torch.split(self.conv1(x), self.c_, dim=1)

        out = None
        branch = None
        for i, conv in enumerate(self.convs):
            branch = splits[i] if i == 0 else branch + splits[i]
            branch = conv(branch)
            out = branch if i == 0 else torch.cat((out, branch), dim=1)

        out = self.conv2(torch.cat((out, splits[-1]), dim=1))

        return self.act(out + x if self.add else out)


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


class ELAN(nn.Module):
    """ELAN from LEAF-YOLO.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int): Unused here; it is the CSP repeat count for the LEAFT subclass.
        shortcut (bool): Unused here; forwarded to CSPRes2B by the LEAFT subclass.
        g (int): Unused here; forwarded to CSPRes2B by the LEAFT subclass.
        e (float): Hidden channel expansion.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()

        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.convs = nn.ModuleList(Conv(c_, c_, 3) for _ in range(2))
        self.transition = Conv(4 * c_, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states = [self.cv1(x), self.cv2(x)]
        for conv in self.convs:
            states.append(conv(states[-1]))

        return self.transition(torch.cat(states, dim=1))


class LEAFT(ELAN):
    """LEAF-T from LEAF-YOLO: ELAN with a CSPRes2B transition."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.transition = CSPRes2B(4 * int(c2 * e), c2, n, shortcut, g, e)


class RFEM(nn.Module):
    """Receptive field enhancement with weight-shared dilated convolutions, from LEAF-YOLO."""

    def __init__(self, c1: int, e: float = 0.5, dilations: tuple = (1, 2, 3)):
        super().__init__()

        c_ = int(c1 * e)
        self.dilations = dilations
        self.weight1 = nn.Parameter(torch.empty(c_, c1, 1, 1))
        self.weight2 = nn.Parameter(torch.empty(c1, c_, 3, 3))
        nn.init.kaiming_uniform_(self.weight1, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.weight2, nonlinearity="relu")

        self.bn1 = nn.BatchNorm2d(c_)
        self.bn2 = nn.BatchNorm2d(c1)
        self.bn3 = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(F.conv2d(x, self.weight1)))

        out = x
        for d in self.dilations:
            out = out + self.act(self.bn2(F.conv2d(y, self.weight2, padding=d, dilation=d)) + x)

        return self.act(self.bn3(out))


class SPPRFEM(nn.Module):
    """SPPF with an added receptive field enhancement branch, from LEAF-YOLO."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()

        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, None, 2)
        self.cv2 = Conv(c_ * 5, c2, 1, 1, None, 2)
        self.m = nn.MaxPool2d(k, 1, k // 2)
        self.rfe = RFEM(c_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y = [x]
        y.extend(self.m(y[-1]) for _ in range(3))
        y.append(self.rfe(x))
        return self.cv2(torch.cat(y, dim=1))


class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        batch_size, _, dim_y, dim_x = input_tensor.shape
        device = input_tensor.device
        dtype = input_tensor.dtype

        y_coord = torch.linspace(-1, 1, dim_y, device=device, dtype=dtype)
        x_coord = torch.linspace(-1, 1, dim_x, device=device, dtype=dtype)

        y_grid, x_grid = torch.meshgrid(y_coord, x_coord, indexing="ij")

        y_grid = y_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        x_grid = x_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)

        coords = [input_tensor, y_grid, x_grid]

        if self.with_r:
            r_grid = torch.sqrt(torch.pow(x_grid, 2) + torch.pow(y_grid, 2))
            coords.append(r_grid)

        return torch.cat(coords, dim=1)


class CoordConv(nn.Module):
    """Coordinate Convolution."""

    def __init__(self, c1, c2, k=1, s=1, with_r=False):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        c1 += 2
        if with_r:
            c1 += 1

        self.conv = Conv(c1, c2, k, s)

    def forward(self, x):
        x = self.addcoords(x)
        x = self.conv(x)
        return x


class CoordAtt(nn.Module):
    """Coordinate Attention."""

    def __init__(self, c1, r=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        c_ = max(8, c1 // r)

        self.conv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(c_, c1, 1, 1)
        self.conv_w = nn.Conv2d(c_, c1, 1, 1)

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


class CoordBlock(nn.Module):
    """CoordBlock from LEAF-YOLO."""

    def __init__(self, c1, c2, k=1, s=1, with_r=False):
        super().__init__()
        self.coordconv = CoordConv(c1, c2, k, s, with_r)
        self.coordatt = CoordAtt(c2)

    def forward(self, x):
        return self.coordatt(self.coordconv(x))


class MGC(nn.Module):
    """MGC from LEAF-YOLO."""

    def __init__(self, c1, c2):
        super().__init__()
        self.c_ = c2 // 2

        self.mp = nn.MaxPool2d(2, 2)
        self.conv1 = Conv(c1, self.c_, 1)

        self.conv2 = Conv(c1, self.c_, 1)
        self.conv3 = GhostConv(self.c_, self.c_, 3, 2)

    def forward(self, x):
        y1 = self.mp(x)
        y1 = self.conv1(y1)

        y2 = self.conv2(x)
        y2 = self.conv3(y2)

        return torch.cat((y1, y2), dim=1)


class NeXt(nn.Module):
    """Inspired by ConvNeXt and FastViT"""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.dwconv = nn.Conv2d(c1, c1, kernel_size=7, padding=3, groups=c1)
        self.norm = nn.BatchNorm2d(c1)
        self.pwconv1 = nn.Conv2d(c1, 2 * c2, kernel_size=1)
        self.act = nn.SiLU(inplace=True)
        self.pwconv2 = nn.Conv2d(2 * c2, c2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pwconv2(self.act(self.pwconv1(self.norm(self.dwconv(x)))))


class NeXtC2f(C2f):
    """C2f using NeXt."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(NeXt(c_, c_) for _ in range(n)))


class DySample(nn.Module):
    """Dynamic upsampling by learned point sampling."""

    def __init__(self, c1, scale=2, style="lp", groups=4, dyscope=False):
        """Initialize DySample.

        Args:
            c1 (int): Number of input and output channels.
            scale (int): Upsampling scale factor.
            style (str): Sampling style, either "lp" or "pl".
            groups (int): Number of channel groups.
            dyscope (bool): Whether to learn a dynamic offset scope.
        """
        super().__init__()
        if style not in {"lp", "pl"}:
            raise ValueError(f"DySample style must be 'lp' or 'pl', not {style!r}.")
        if c1 % groups:
            raise ValueError(f"Input channels {c1} must be divisible by groups {groups}.")
        if style == "pl" and c1 % scale**2:
            raise ValueError(f"Input channels {c1} must be divisible by scale² ({scale**2}) for style='pl'.")

        self.scale = scale
        self.style = style
        self.groups = groups

        offset_channels = c1 // scale**2 if style == "pl" else c1
        output_channels = 2 * groups if style == "pl" else 2 * groups * scale**2

        self.offset = nn.Conv2d(offset_channels, output_channels, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.constant_(self.offset.bias, 0)

        if dyscope:
            self.scope = nn.Conv2d(offset_channels, output_channels, 1, bias=False)
            nn.init.constant_(self.scope.weight, 0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        """Create the initial regular sampling positions."""
        coordinate = (
            torch.arange(
                (-self.scale + 1) / 2,
                (self.scale - 1) / 2 + 1,
            )
            / self.scale
        )

        position = torch.stack(
            (
                coordinate.repeat(self.scale, 1),
                coordinate.view(-1, 1).repeat(1, self.scale),
            )
        )
        return position.repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def _sample(self, x, offset):
        """Sample input features using learned offsets."""
        batch, _, height, width = offset.shape
        offset = offset.view(batch, 2, -1, height, width)

        coordinates_w = torch.arange(width, dtype=x.dtype, device=x.device) + 0.5
        coordinates_h = torch.arange(height, dtype=x.dtype, device=x.device) + 0.5
        coordinates = torch.stack(
            (
                coordinates_w.view(1, width).expand(height, width),
                coordinates_h.view(height, 1).expand(height, width),
            )
        )
        coordinates = coordinates.unsqueeze(0).unsqueeze(2)
        normalizer = x.new_tensor((width, height)).view(1, 2, 1, 1, 1)
        coordinates = 2 * (coordinates + offset) / normalizer - 1

        coordinates = F.pixel_shuffle(
            coordinates.view(batch, -1, height, width),
            self.scale,
        )
        coordinates = (
            coordinates.view(
                batch,
                2,
                -1,
                height * self.scale,
                width * self.scale,
            )
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )

        return F.grid_sample(
            x.reshape(batch * self.groups, -1, height, width),
            coordinates,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).view(batch, -1, height * self.scale, width * self.scale)

    def forward(self, x):
        """Upsample the input feature map."""
        if self.style == "pl":
            shuffled = F.pixel_shuffle(x, self.scale)
            offset = self.offset(shuffled)

            if hasattr(self, "scope"):
                offset = offset * self.scope(shuffled).sigmoid() * 0.5
            else:
                offset = offset * 0.25

            offset = F.pixel_unshuffle(offset, self.scale) + self.init_pos
        else:
            offset = self.offset(x)

            if hasattr(self, "scope"):
                offset = offset * self.scope(x).sigmoid() * 0.5
            else:
                offset = offset * 0.25

            offset = offset + self.init_pos

        return self._sample(x, offset)


class CARAFE(nn.Module):
    """Content-aware reassembly of features upsampler."""

    def __init__(self, c1, c_mid=64, scale=2, k_up=5, k_enc=3):
        """Initialize CARAFE.

        Args:
            c1 (int): Number of input and output channels.
            c_mid (int): Number of compressed channels.
            scale (int): Upsampling scale factor.
            k_up (int): Reassembly kernel size.
            k_enc (int): Content encoder kernel size.
        """
        super().__init__()
        if k_up % 2 == 0 or k_enc % 2 == 0:
            raise ValueError("CARAFE kernel sizes must be odd.")

        self.scale = scale
        self.k_up = k_up

        self.compressor = Conv(c1, c_mid, 1, act=nn.ReLU(inplace=True))
        self.encoder = Conv(
            c_mid,
            scale**2 * k_up**2,
            k_enc,
            act=False,
        )
        self.unfold = nn.Unfold(
            kernel_size=k_up,
            dilation=scale,
            padding=k_up // 2 * scale,
        )

    def forward(self, x):
        """Upsample the input feature map."""
        batch, channels, height, width = x.shape
        output_height = height * self.scale
        output_width = width * self.scale

        weights = self.encoder(self.compressor(x))
        weights = F.pixel_shuffle(weights, self.scale)
        weights = weights.softmax(dim=1)

        features = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        features = self.unfold(features)
        features = features.view(
            batch,
            channels,
            self.k_up**2,
            output_height,
            output_width,
        )

        return (features * weights.unsqueeze(1)).sum(dim=2)
