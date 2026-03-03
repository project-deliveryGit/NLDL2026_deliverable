import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_model(config, pretrained_path=None, device="cpu"):
    model = smp.Unet(
        encoder_name=config["encoder_name"],
        encoder_weights=config.get("encoder_weights", "imagenet"),
        in_channels=config["in_channels"],
        classes=config["classes"],
        activation=None,
    ).to(device)
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    return model


# --------------- LoRA ---------------

class LoRAConv2d(nn.Module):
    def __init__(self, original_conv, r, alpha):
        super().__init__()
        self.original = original_conv
        self.scaling = alpha / r
        in_c, out_c = original_conv.in_channels, original_conv.out_channels
        stride = original_conv.stride

        self.A = nn.Conv2d(in_c, r, kernel_size=1, stride=1, bias=False)
        self.B = nn.Conv2d(r, out_c, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(stride) if stride not in [(1, 1), 1] else None

        nn.init.kaiming_uniform_(self.A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x):
        lora = self.A(x)
        if self.pool is not None:
            lora = self.pool(lora)
        return self.original(x) + self.B(lora) * self.scaling


def apply_lora(model, r, alpha):
    targets = [
        name for name, m in model.named_modules()
        if isinstance(m, nn.Conv2d)
        and ("encoder" in name or "decoder" in name)
        and m.kernel_size != (1, 1)
        and m.out_channels >= 32
    ]
    for name in targets:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        orig = getattr(parent, parts[-1])
        setattr(parent, parts[-1], LoRAConv2d(orig, r=r, alpha=alpha).to(orig.weight.device))

    for n, p in model.named_parameters():
        p.requires_grad = any(k in n for k in (".A.", ".B.", "segmentation_head"))
    return model


# --------------- Cross-Attention ---------------

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, downsample_ratio=1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj  = nn.Conv2d(dim, dim, 1, bias=False)
        self.k_proj  = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_proj  = nn.Conv2d(dim, dim, 1, bias=False)
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.norm    = nn.BatchNorm2d(dim)
        self.dropout = nn.Dropout2d(0.1)
        self.pool    = nn.AvgPool2d(downsample_ratio) if downsample_ratio > 1 else None

    def forward(self, query, kv):
        B, C, H, W = query.shape
        identity = query

        q = self.pool(query) if self.pool else query
        k = self.pool(kv)    if self.pool else kv
        Hd, Wd = q.shape[2], q.shape[3]

        def reshape(t):
            return self.q_proj(t).reshape(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)

        Q = self.q_proj(q).reshape(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)
        K = self.k_proj(k).reshape(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)
        V = self.v_proj(k).reshape(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)

        chunks = [
            torch.matmul(
                torch.softmax(torch.matmul(Q[:, :, i:i+256], K.transpose(-2, -1)) * self.scale, dim=-1),
                V
            )
            for i in range(0, Q.shape[2], 256)
        ]
        out = torch.cat(chunks, dim=2).permute(0, 1, 3, 2).reshape(B, C, Hd, Wd)

        if self.pool:
            out = nn.functional.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        return self.norm(self.dropout(self.out_proj(out)) + identity)


class CrossAttentionUNet(nn.Module):
    def __init__(self, patient_model, config, device="cpu"):
        super().__init__()
        self.patient_model = patient_model
        for p in self.patient_model.parameters():
            p.requires_grad = False
        self.patient_model.eval()

        self.phantom_model = build_model(config, device=device)
        self.attention_layers = config["attention_layers"]
        self.patient_feats = {}

        self.cross_attns = nn.ModuleDict()
        for idx in self.attention_layers:
            block = self.phantom_model.decoder.blocks[idx]
            channels = block.conv2[0].out_channels
            downsample = 1 if idx <= 1 else (2 if idx == 2 else config.get("attention_downsample", 4))
            self.cross_attns[f"layer_{idx}"] = CrossAttentionBlock(
                dim=channels,
                num_heads=config["attention_heads"],
                downsample_ratio=downsample,
            )
        self._register_hooks()

    def _register_hooks(self):
        for i, block in enumerate(self.patient_model.decoder.blocks):
            def patient_hook(m, inp, out, idx=i):
                self.patient_feats[f"layer_{idx}"] = out.detach()
            block.register_forward_hook(patient_hook)

        for i, block in enumerate(self.phantom_model.decoder.blocks):
            def phantom_hook(m, inp, out, idx=i):
                if idx in self.attention_layers:
                    feat = self.patient_feats.get(f"layer_{idx}")
                    if feat is not None:
                        out = self.cross_attns[f"layer_{idx}"](out, feat)
                return out
            block.register_forward_hook(phantom_hook)

    def forward(self, x):
        self.patient_feats.clear()
        with torch.no_grad():
            self.patient_model(x)
        return self.phantom_model(x)


# --------------- Loaders (used by evaluate.py) ---------------

def load_standard_model(path, config, device):
    return build_model({**config, "encoder_weights": None}, pretrained_path=path, device=device)


def load_lora_model(path, config, device):
    model = build_model(config, device=device)
    model = apply_lora(model, r=config["lora_r"], alpha=config["lora_alpha"])
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def load_cross_attention_model(path, config, patient_path, device):
    patient_model = build_model(
        {**config, "encoder_weights": None}, pretrained_path=patient_path, device=device
    )
    model = CrossAttentionUNet(patient_model, config, device=device).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model