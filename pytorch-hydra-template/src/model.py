import torch
import math
from torch import nn, Tensor
import torch.nn.functional as F
import warnings
from torch.nn import functional as F
from einops import rearrange


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Copied from timm
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """

    def __init__(self, p: float = None):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0.0 or not self.training:
            return x
        kp = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = kp + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(kp) * random_tensor


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        k, v = (
            self.kv(x)
            .reshape(B, -1, 2, self.head, C // self.head)
            .permute(2, 0, 3, 1, 4)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: Tensor, H, W) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(
            c1, c2, patch_size, stride, patch_size // 2
        )  # padding=(ps[0]//2, ps[1]//2)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


mit_settings = {
    "B0": [[32, 64, 160, 256], [2, 2, 2, 2]],  # [embed_dims, depths]
    "B1": [[64, 128, 320, 512], [2, 2, 2, 2]],
    "B2": [[64, 128, 320, 512], [3, 4, 6, 3]],
    "B3": [[64, 128, 320, 512], [3, 4, 18, 3]],
    "B4": [[64, 128, 320, 512], [3, 8, 27, 3]],
    "B5": [[64, 128, 320, 512], [3, 6, 40, 3]],
}


class MiT(nn.Module):
    def __init__(self, model_name: str = "B0"):
        super().__init__()
        assert (
            model_name in mit_settings.keys()
        ), f"MiT model name should be in {list(mit_settings.keys())}"
        embed_dims, depths = mit_settings[model_name]
        drop_path_rate = 0.1
        self.channels = embed_dims

        # patch_embed
        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        self.block1 = nn.ModuleList(
            [Block(embed_dims[0], 1, 8, dpr[cur + i]) for i in range(depths[0])]
        )
        self.norm1 = nn.LayerNorm(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList(
            [Block(embed_dims[1], 2, 4, dpr[cur + i]) for i in range(depths[1])]
        )
        self.norm2 = nn.LayerNorm(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList(
            [Block(embed_dims[2], 5, 2, dpr[cur + i]) for i in range(depths[2])]
        )
        self.norm3 = nn.LayerNorm(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList(
            [Block(embed_dims[3], 8, 1, dpr[cur + i]) for i in range(depths[3])]
        )
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x1 = self.norm1(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # stage 2
        x, H, W = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, H, W)
        x2 = self.norm2(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # stage 3
        x, H, W = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, H, W)
        x3 = self.norm3(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # stage 4
        x, H, W = self.patch_embed4(x3)
        for blk in self.block4:
            x = blk(x, H, W)
        x4 = self.norm4(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        return x1, x2, x3, x4


if __name__ == "__main__":
    model = MiT("B0")
    x = torch.zeros(1, 3, 224, 224)
    outs = model(x)
    for y in outs:
        print(y.shape)


class BaseModel(nn.Module):
    def __init__(self, backbone: str = "MiT-B0", num_classes: int = 19) -> None:
        super().__init__()
        backbone, variant = backbone.split("-")
        self.backbone = eval(backbone)(variant)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(
                torch.load(pretrained, map_location="cpu"), strict=False
            )


class MLP(nn.Module):
    def __init__(self, dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_ch=3, dim=96, type="pool") -> None:
        super().__init__()
        self.patch_size = patch_size
        self.type = type
        self.dim = dim

        if type == "conv":
            self.proj = nn.Conv2d(
                in_ch, dim, patch_size, patch_size, groups=patch_size * patch_size
            )
        else:
            self.proj = nn.ModuleList(
                [
                    nn.MaxPool2d(patch_size, patch_size),
                    nn.AvgPool2d(patch_size, patch_size),
                ]
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        if W % self.patch_size != 0:
            x = F.pad(x, (0, self.patch_size - W % self.patch_size))
        if H % self.patch_size != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size - H % self.patch_size))

        if self.type == "conv":
            x = self.proj(x)
        else:
            x = 0.5 * (self.proj[0](x) + self.proj[1](x))
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).view(-1, self.dim, Wh, Ww)
        return x


class LawinAttn(nn.Module):
    def __init__(self, in_ch=512, head=4, patch_size=8, reduction=2) -> None:
        super().__init__()
        self.head = head

        self.position_mixing = nn.ModuleList(
            [
                nn.Linear(patch_size * patch_size, patch_size * patch_size)
                for _ in range(self.head)
            ]
        )

        self.inter_channels = max(in_ch // reduction, 1)
        self.g = nn.Conv2d(in_ch, self.inter_channels, 1)
        self.theta = nn.Conv2d(in_ch, self.inter_channels, 1)
        self.phi = nn.Conv2d(in_ch, self.inter_channels, 1)
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_ch, 1, bias=False), nn.BatchNorm2d(in_ch)
        )

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        B, C, H, W = context.shape
        context = context.reshape(B, C, -1)
        context_mlp = []

        for i, pm in enumerate(self.position_mixing):
            context_crt = context[
                :, (C // self.head) * i : (C // self.head) * (i + 1), :
            ]
            context_mlp.append(pm(context_crt))

        context_mlp = torch.cat(context_mlp, dim=1)
        context = context + context_mlp
        context = context.reshape(B, C, H, W)

        g_x = self.g(context).view(B, self.inter_channels, -1)
        g_x = rearrange(g_x, "b (h dim) n -> (b h) dim n", h=self.head)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(query).view(B, self.inter_channels, -1)
        theta_x = rearrange(theta_x, "b (h dim) n -> (b h) dim n", h=self.head)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(context).view(B, self.inter_channels, -1)
        phi_x = rearrange(phi_x, "b (h dim) n -> (b h) dim n", h=self.head)

        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= theta_x.shape[-1] ** 0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)

        y = torch.matmul(pairwise_weight, g_x)
        y = rearrange(y, "(b h) n dim -> b n (h dim)", h=self.head)
        y = (
            y.permute(0, 2, 1)
            .contiguous()
            .reshape(B, self.inter_channels, *query.shape[-2:])
        )

        output = query + self.conv_out(y)
        return output


class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)  # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))


class LawinHead(nn.Module):
    def __init__(self, in_channels: list, embed_dim=512, num_classes=19) -> None:
        super().__init__()
        for i, dim in enumerate(in_channels):
            self.add_module(f"linear_c{i+1}", MLP(dim, 48 if i == 0 else embed_dim))

        self.lawin_8 = LawinAttn(embed_dim, 64)
        self.lawin_4 = LawinAttn(embed_dim, 16)
        self.lawin_2 = LawinAttn(embed_dim, 4)
        self.ds_8 = PatchEmbed(8, embed_dim, embed_dim)
        self.ds_4 = PatchEmbed(4, embed_dim, embed_dim)
        self.ds_2 = PatchEmbed(2, embed_dim, embed_dim)

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), ConvModule(embed_dim, embed_dim)
        )
        self.linear_fuse = ConvModule(embed_dim * 3, embed_dim)
        self.short_path = ConvModule(embed_dim, embed_dim)
        self.cat = ConvModule(embed_dim * 5, embed_dim)

        self.low_level_fuse = ConvModule(embed_dim + 48, embed_dim)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def get_lawin_att_feats(self, x: Tensor, patch_size: int):
        _, _, H, W = x.shape
        query = F.unfold(x, patch_size, stride=patch_size)
        query = rearrange(
            query,
            "b (c ph pw) (nh nw) -> (b nh nw) c ph pw",
            ph=patch_size,
            pw=patch_size,
            nh=H // patch_size,
            nw=W // patch_size,
        )
        outs = []

        for r in [8, 4, 2]:
            context = F.unfold(
                x,
                patch_size * r,
                stride=patch_size,
                padding=int((r - 1) / 2 * patch_size),
            )
            context = rearrange(
                context,
                "b (c ph pw) (nh nw) -> (b nh nw) c ph pw",
                ph=patch_size * r,
                pw=patch_size * r,
                nh=H // patch_size,
                nw=W // patch_size,
            )
            context = getattr(self, f"ds_{r}")(context)
            output = getattr(self, f"lawin_{r}")(query, context)
            output = rearrange(
                output,
                "(b nh nw) c ph pw -> b c (nh ph) (nw pw)",
                ph=patch_size,
                pw=patch_size,
                nh=H // patch_size,
                nw=W // patch_size,
            )
            outs.append(output)
        return outs

    def forward(self, features):
        B, _, H, W = features[1].shape
        outs = [
            self.linear_c2(features[1])
            .permute(0, 2, 1)
            .reshape(B, -1, *features[1].shape[-2:])
        ]

        for i, feature in enumerate(features[2:]):
            cf = (
                eval(f"self.linear_c{i+3}")(feature)
                .permute(0, 2, 1)
                .reshape(B, -1, *feature.shape[-2:])
            )
            outs.append(
                F.interpolate(cf, size=(H, W), mode="bilinear", align_corners=False)
            )

        feat = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        B, _, H, W = feat.shape

        ## Lawin attention spatial pyramid pooling
        feat_short = self.short_path(feat)
        feat_pool = F.interpolate(
            self.image_pool(feat), size=(H, W), mode="bilinear", align_corners=False
        )
        feat_lawin = self.get_lawin_att_feats(feat, 8)
        output = self.cat(torch.cat([feat_short, feat_pool, *feat_lawin], dim=1))

        ## Low-level feature enhancement
        c1 = (
            self.linear_c1(features[0])
            .permute(0, 2, 1)
            .reshape(B, -1, *features[0].shape[-2:])
        )
        output = F.interpolate(
            output, size=features[0].shape[-2:], mode="bilinear", align_corners=False
        )
        fused = self.low_level_fuse(torch.cat([output, c1], dim=1))

        seg = self.linear_pred(self.dropout(fused))
        return seg


class Lawin(BaseModel):
    """
    Notes::::: This implementation has larger params and FLOPs than the results reported in the paper.
    Will update the code and weights if the original author releases the full code.
    """

    def __init__(self, backbone: str = "MiT-B0", num_classes: int = 19) -> None:
        super().__init__(backbone, num_classes)
        self.decode_head = LawinHead(
            self.backbone.channels, 256 if "B0" in backbone else 512, num_classes
        )
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)  # 4x reduction in image size
        y = F.interpolate(
            y, size=x.shape[2:], mode="bilinear", align_corners=False
        )  # to original image shape
        return y
