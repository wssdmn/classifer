import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange,Reduce

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding nn.LayerNorm
    """
    def __init__(self, patch_num=8, img_size=128, patch_size=16, in_c=1, embed_dim=16*16, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.tran = Rearrange('b c h w -> b (h w) c' )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x)
        x = self.tran(x)
        x = self.norm(x)
        return x


class CPE(nn.Module):
    # (b n*n c*h*w) -> (b*c*h*w 1 n n) -> (b n*n c*h*w)
    def __init__(self, patch_num = 8, patch_size = 16):
        super(CPE, self).__init__()
        # self.net = nn.Sequential(
        #     Rearrange('b (nh nw) chw -> (b chw) 1 nh nw', nw=patch_num, nh=patch_num),
        #     nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
        #     Rearrange('(b chw) 1 nh nw -> b (nh nw) chw', nw=patch_num, nh=patch_num, chw=patch_size ** 2)
        # )
        dim = patch_size ** 2
        self.net = nn.Sequential(
            Rearrange('b (nh nw) chw -> b chw nh nw', nw=patch_num, nh=patch_num),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            Rearrange('b chw nh nw -> b (nh nw) chw')
        )
    def forward(self, x):
        out = x + self.net(x)
        return out


class EMLP(nn.Module):
    # (b n*n c*h*w)
    def __init__(self, patch_num, patch_size, expandRatio = 4, dropRatio = 0.5):
        super(EMLP, self).__init__()
        dim, expand_dim = patch_size ** 2, patch_size ** 2 * expandRatio
        self.expandingLinear = nn.Linear(dim, expand_dim)
        self.expandedLinear = nn.Linear(expand_dim, expand_dim)
        self.shrinkingLinear = nn.Linear(expand_dim, dim)
        self.drop1 = nn.Dropout(dropRatio) if dropRatio > 0 else nn.Identity()
        self.drop2 = nn.Dropout(dropRatio) if dropRatio > 0 else nn.Identity()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
    def forward(self, x):
        x = self.drop1(self.relu1(self.expandingLinear(x)))
        x = self.drop2(self.relu2(self.expandedLinear(x)))
        return self.shrinkingLinear(x)


class MHSA(nn.Module):
    # (b n*n chw*3) -> (b n*n 3 head chw//head) -> (3 b head n*n chw//head) -> (b n*n chw)
    def __init__(self, patch_num, patch_size, head = 4, dropRatio = 0.5):
        super(MHSA, self).__init__()
        self.n = patch_num * patch_num
        self.dim = patch_size ** 2
        self.head_dim = self.dim // head
        self.head = head
        assert head * self.head_dim == self.dim
        self.scale = self.head_dim ** -0.5
        self.norm = nn.LayerNorm(self.dim)
        self.qkv = nn.Linear(self.dim, self.dim * 3)
        self.linear = nn.Linear(self.dim, self.dim)
        self.drop = nn.Dropout(dropRatio) if dropRatio > 0 else nn.Identity()
    def forward(self, x):
        qkv = self.qkv(self.norm(x)).reshape(-1, self.n, 3, self.head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        attn = (attn @ v).transpose(1, 2).reshape(-1, self.n, self.dim)
        out = self.drop(self.linear(attn))
        return out


class LFFN(nn.Module):
    # (b n*n chw)
    def __init__(self, patch_num, patch_size, expandRatio = 4):
        super(LFFN, self).__init__()
        dim = patch_size ** 2
        scale_dim = expandRatio * dim
        self.po_s2i = nn.Sequential(
            Rearrange('b (nh nw) chw -> b chw nh nw', nw=patch_num, nh=patch_num),
            nn.Conv2d(dim, scale_dim, kernel_size=1),
            nn.BatchNorm2d(scale_dim),
            nn.ReLU(),

            nn.Conv2d(scale_dim, scale_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(scale_dim),
            nn.ReLU()
        )
        self.se = nn.Sequential(
            Reduce('b c h w -> b c 1 1', 'mean'),
            nn.Conv2d(scale_dim, dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim, scale_dim, kernel_size=1),
            nn.ReLU()
        )
        self.po_i2s = nn.Sequential(
            nn.Conv2d(scale_dim, dim, kernel_size=1),
            Rearrange('b chw nh nw -> b (nh nw) chw')
        )
    def forward(self, x):
        p1out = self.po_s2i(x)
        return self.po_i2s(self.se(p1out) * p1out)


class ELM(nn.Module):
    # (b (nh nw) chw) -> (b (nh//2 nw//2) chw)
    def __init__(self, patch_num, patch_size, shrinkImg=False):
        super(ELM, self).__init__()
        # self.net = nn.Sequential(
        #     Rearrange('b (nh nw) chw -> (b chw) 1 nh nw', nw=patch_num, nh=patch_num),
        #     nn.Conv2d(1, 1, kernel_size=2, stride=2, bias=False),
        #     Rearrange('(b chw) 1 nh nw -> b (nh nw) chw', nw=patch_num//2, nh=patch_num//2, chw=patch_size ** 2)
        # )
        dim = patch_size ** 2
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b (nh nw) chw -> b chw nh nw', nw=patch_num, nh=patch_num),
            nn.Conv2d(dim, dim//4, kernel_size=3, stride=1, padding=1, bias=False) if shrinkImg else \
                nn.Conv2d(dim, dim, kernel_size=2, stride=2, bias=False),
            Rearrange('b chw nh nw -> b (nh nw) chw')
        )
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    # (b c n*h n*w) -> (b n*n c*h*w)
    def __init__(self, patch_num, patch_size, dropRatio = 0.5, shrinkImg = False):
        super(Block, self).__init__()
        self.net = nn.Sequential(
            CPE(patch_num, patch_size),
            EMLP(patch_num, patch_size, dropRatio=dropRatio),
            MHSA(patch_num, patch_size, dropRatio=dropRatio),
            LFFN(patch_num, patch_size),
            ELM(patch_num, patch_size, shrinkImg = shrinkImg)
        )
    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    # (b c n*h n*w) -> (b n*n c*h*w)
    def __init__(self, dim, num_class, dropRatio):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropRatio) if dropRatio > 0 else nn.Identity(),
            nn.Linear(dim, num_class),
            # nn.LogSoftmax()
        )
    def forward(self, x):
        return self.net(x)


from torchvision.transforms import Resize
class Net(nn.Module):
    def __init__(self, num_class = 100, dropRatio = 0):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            PatchEmbed(8),
            Block(8, 16, shrinkImg=True, dropRatio = dropRatio),
            Block(8, 8, dropRatio = dropRatio),
            Block(4, 8, dropRatio = dropRatio),
            Block(2, 8, dropRatio = dropRatio),
        )
        self.classifier = Classifier(8 ** 2, num_class = num_class, dropRatio = dropRatio)
        self.apply(_init_vit_weights)
    def forward(self, x):
        assert x.shape[1]== 1
        if x.shape[-1] != 128 or x.shape[-2] != 128:
            x = Resize([128,128])(x)
        feature = self.net(x).squeeze(1)
        out = self.classifier(feature)
        return out, feature
def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

if __name__ == '__main__':
    img = torch.arange(0, 8 * 1 * 128 * 128, dtype = torch.float).reshape(-1, 1, 128, 128)
    label = torch.flip(torch.arange(8), dims = [0])

    model = Net(10)
    out, feature = model(img)
    print(img.shape), print(feature.shape), print(label.shape)
    print(model)
