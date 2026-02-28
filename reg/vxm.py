from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class VxmConfig:
    device: str = "auto"  # auto|cpu|cuda
    scale: int = 1  # downsample factor for VXM (>=1)
    smooth_weight: float = 0.05
    lr: float = 1e-4
    epochs: int = 0  # if >0 and no weights provided, train
    steps_per_epoch: int = 100
    batch_size: int = 1
    weights_path: Optional[str] = None
    norm: str = "ct_hu"  # ct_hu|percentile|minmax
    ct_clip: Tuple[float, float] = (-1000.0, 400.0)
    pct: Tuple[float, float] = (1.0, 99.0)
    sim: str = "mse"  # mse|ncc
    ncc_win: int = 9


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "VoxelMorph backend requires PyTorch. Install torch and rerun "
            "(see requirements.txt GPU section)."
        ) from e


def _get_device(device: str):
    import torch

    d = str(device).lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d in {"cuda", "cpu"}:
        return torch.device(d)
    raise ValueError("device must be one of: auto, cpu, cuda")


def _normalize_np_to_0_1(x: np.ndarray, *, mode: str, ct_clip: Tuple[float, float], pct: Tuple[float, float]) -> np.ndarray:
    """
    Normalize a 3D numpy volume to [0, 1] for network input.

    - ct_hu: clamp to ct_clip then scale
    - percentile: clamp to per-volume percentiles then scale
    - minmax: scale using per-volume min/max
    """
    mode = str(mode).lower()
    v = x.astype(np.float32, copy=False)
    if mode == "ct_hu":
        lo, hi = map(float, ct_clip)
        v = np.clip(v, lo, hi)
        return (v - lo) / (hi - lo + 1e-6)
    if mode == "percentile":
        p_lo, p_hi = map(float, pct)
        lo = float(np.percentile(v, p_lo))
        hi = float(np.percentile(v, p_hi))
        if hi <= lo:
            hi = lo + 1.0
        v = np.clip(v, lo, hi)
        return (v - lo) / (hi - lo + 1e-6)
    if mode == "minmax":
        lo = float(np.min(v))
        hi = float(np.max(v))
        if hi <= lo:
            hi = lo + 1.0
        return (v - lo) / (hi - lo + 1e-6)
    raise ValueError("norm must be one of: ct_hu, percentile, minmax")


def _ncc_loss_3d(a, b, *, win: int = 9, eps: float = 1e-5, mask=None):
    """
    Local normalized cross-correlation loss (negative NCC) for 3D volumes.
    a, b: [B, 1, Z, Y, X]
    mask: optional [B, 1, Z, Y, X] binary mask applied to the final NCC map.
    """
    import torch
    import torch.nn.functional as F

    win = int(win)
    if win <= 1:
        # Global NCC
        a0 = a - a.mean(dim=(-3, -2, -1), keepdim=True)
        b0 = b - b.mean(dim=(-3, -2, -1), keepdim=True)
        num = (a0 * b0).mean(dim=(-3, -2, -1))
        den = torch.sqrt((a0 * a0).mean(dim=(-3, -2, -1)) * (b0 * b0).mean(dim=(-3, -2, -1)) + eps)
        ncc = num / den
        return -ncc.mean()

    k = win
    pad = k // 2
    filt = torch.ones((1, 1, k, k, k), device=a.device, dtype=a.dtype)

    def conv(x):
        return F.conv3d(x, filt, padding=pad)

    a2 = a * a
    b2 = b * b
    ab = a * b
    sum_a = conv(a)
    sum_b = conv(b)
    sum_a2 = conv(a2)
    sum_b2 = conv(b2)
    sum_ab = conv(ab)

    win_size = float(k * k * k)
    mean_a = sum_a / win_size
    mean_b = sum_b / win_size

    cross = sum_ab - mean_b * sum_a - mean_a * sum_b + mean_a * mean_b * win_size
    var_a = sum_a2 - 2 * mean_a * sum_a + mean_a * mean_a * win_size
    var_b = sum_b2 - 2 * mean_b * sum_b + mean_b * mean_b * win_size

    ncc_map = (cross * cross) / (var_a * var_b + eps)
    if mask is not None:
        ncc_map = ncc_map * mask.to(ncc_map.dtype)
        denom = mask.to(ncc_map.dtype).sum().clamp_min(1.0)
        return -ncc_map.sum() / denom
    return -ncc_map.mean()


class _ConvBlock3d:
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1):
        import torch.nn as nn

        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def __call__(self, x):
        return self.net(x)


def _build_unet_3d(in_ch: int = 2, feat: Tuple[int, ...] = (16, 32, 32, 32, 32)):
    """
    Lightweight 3D U-Net similar in spirit to VoxelMorph.
    Returns a torch.nn.Module with forward(x)->features.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.ModuleList()
            self.dec = nn.ModuleList()

            ch = in_ch
            for f in feat:
                self.enc.append(_ConvBlock3d(ch, f, stride=2 if len(self.enc) > 0 else 1).net)
                ch = f

            # decoder: upsample + conv
            dec_feat = list(reversed(feat[:-1]))
            for f in dec_feat:
                self.dec.append(
                    nn.Sequential(
                        nn.Conv3d(ch + f, f, kernel_size=3, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv3d(f, f, kernel_size=3, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                    )
                )
                ch = f

            self.out_ch = ch

        def forward(self, x):
            skips = []
            for i, layer in enumerate(self.enc):
                x = layer(x)
                skips.append(x)
            # last skip is deepest; pop it
            x = skips.pop()
            for layer in self.dec:
                if skips:
                    s = skips.pop()
                    # Upsample to the skip's spatial size to handle odd/non-multiple-of-2 input sizes.
                    if x.shape[-3:] != s.shape[-3:]:
                        x = F.interpolate(x, size=s.shape[-3:], mode="trilinear", align_corners=False)
                    # As a last resort, match skip to x (should be rare).
                    if s.shape[-3:] != x.shape[-3:]:
                        s = F.interpolate(s, size=x.shape[-3:], mode="trilinear", align_corners=False)
                    x = torch.cat([x, s], dim=1)
                else:
                    # Fallback if skip bookkeeping changes: standard 2x upsample.
                    x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
                x = layer(x)
            return x

    return UNet()


def _grid_sample_3d(src, flow_zyx_vox, *, mode: str, padding_mode: str):
    """
    src: [B, C, Z, Y, X]
    flow_zyx_vox: [B, 3, Z, Y, X] in voxel units, component order Z,Y,X.
    """
    import torch
    import torch.nn.functional as F

    b, _, z, y, x = src.shape
    device = src.device
    dtype = src.dtype

    zz = torch.arange(z, device=device, dtype=dtype)
    yy = torch.arange(y, device=device, dtype=dtype)
    xx = torch.arange(x, device=device, dtype=dtype)
    Z, Y, X = torch.meshgrid(zz, yy, xx, indexing="ij")

    z_m = Z[None, ...] + flow_zyx_vox[:, 0]
    y_m = Y[None, ...] + flow_zyx_vox[:, 1]
    x_m = X[None, ...] + flow_zyx_vox[:, 2]

    # Normalize to [-1, 1] for grid_sample; order is x,y,z.
    x_n = 2.0 * (x_m / max(x - 1, 1) - 0.5)
    y_n = 2.0 * (y_m / max(y - 1, 1) - 0.5)
    z_n = 2.0 * (z_m / max(z - 1, 1) - 0.5)
    grid = torch.stack([x_n, y_n, z_n], dim=-1)  # [B, Z, Y, X, 3]

    return F.grid_sample(src, grid, mode=mode, padding_mode=padding_mode, align_corners=True)


def _grad_loss(flow):
    """
    Smoothness loss on flow in voxel units.
    flow: [B, 3, Z, Y, X]
    """
    dz = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
    dy = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
    dx = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
    return (dz.abs().mean() + dy.abs().mean() + dx.abs().mean())


class VoxelMorphDense:
    """
    Minimal VoxelMorph-style dense regressor:
      flow = CNN([moving, fixed])
      warped = ST(moving, flow)
    """

    def __init__(self):
        _require_torch()
        import torch.nn as nn

        self.unet = _build_unet_3d(in_ch=2)
        self.flow = nn.Conv3d(self.unet.out_ch, 3, kernel_size=3, padding=1)
        # VoxelMorph-style initialization: start near identity (small flows).
        # This prevents early "warp everything out of bounds" collapse in unsupervised training.
        nn.init.normal_(self.flow.weight, mean=0.0, std=1e-5)
        if self.flow.bias is not None:
            nn.init.constant_(self.flow.bias, 0.0)

    def to(self, device):
        self.unet.to(device)
        self.flow.to(device)
        return self

    def parameters(self):
        yield from self.unet.parameters()
        yield from self.flow.parameters()

    def state_dict(self):
        return {"unet": self.unet.state_dict(), "flow": self.flow.state_dict()}

    def load_state_dict(self, sd):
        self.unet.load_state_dict(sd["unet"])
        self.flow.load_state_dict(sd["flow"])

    def train(self, mode: bool = True):
        self.unet.train(mode)
        self.flow.train(mode)

    def eval(self):
        self.train(False)

    def __call__(self, moving, fixed):
        import torch

        x = torch.cat([moving, fixed], dim=1)
        feats = self.unet(x)
        flow_zyx = self.flow(feats)
        # Guardrail: bound extreme flows in voxel units to avoid out-of-bounds sampling.
        # Uses a smooth clamp (tanh) so gradients don't disappear abruptly.
        z, y, x_ = moving.shape[-3:]
        max_disp = float(0.5 * max(1, min(z, y, x_)))
        flow_zyx = max_disp * torch.tanh(flow_zyx / max_disp)
        # Safety: make sure flow matches moving grid (odd sizes can otherwise drift).
        if flow_zyx.shape[-3:] != moving.shape[-3:]:
            flow_zyx = _upsample_flow(flow_zyx, tuple(moving.shape[-3:]))
        warped = _grid_sample_3d(moving, flow_zyx, mode="bilinear", padding_mode="border")
        return warped, flow_zyx


def save_weights(model: VoxelMorphDense, path: str | Path) -> None:
    _require_torch()
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_weights(model: VoxelMorphDense, path: str | Path, device: str = "auto") -> None:
    _require_torch()
    import torch

    dev = _get_device(device)
    sd = torch.load(Path(path), map_location=dev)
    model.load_state_dict(sd)


def _downsample_if_needed(vol, scale: int):
    import torch.nn.functional as F

    if int(scale) <= 1:
        return vol
    z, y, x = vol.shape[-3:]
    return F.interpolate(vol, size=(z // scale, y // scale, x // scale), mode="trilinear", align_corners=False)


def _downsample_nearest_if_needed(vol, scale: int):
    """
    Downsample a 5D tensor [B,C,Z,Y,X] using nearest-neighbor. Intended for label maps.
    """
    import torch.nn.functional as F

    if int(scale) <= 1:
        return vol
    z, y, x = vol.shape[-3:]
    return F.interpolate(vol, size=(z // scale, y // scale, x // scale), mode="nearest")


def _hard_dice_from_onehot(pred_oh, target_oh, *, exclude_background: bool = True, eps: float = 1e-6):
    """
    pred_oh, target_oh: [B, C, Z, Y, X] float tensors (soft/hard).
    Returns mean hard Dice over batch (and classes), computed from argmax.
    """
    import torch

    if pred_oh.shape != target_oh.shape:
        raise ValueError("pred_oh and target_oh shapes must match")
    b, c = pred_oh.shape[:2]
    pred = torch.argmax(pred_oh, dim=1)  # [B, Z, Y, X]
    targ = torch.argmax(target_oh, dim=1)
    cls = list(range(c))
    if exclude_background and c > 1:
        cls = cls[1:]
    dices = []
    for k in cls:
        pk = (pred == k).to(torch.float32)
        tk = (targ == k).to(torch.float32)
        inter = 2.0 * (pk * tk).sum(dim=(-3, -2, -1))
        den = (pk.sum(dim=(-3, -2, -1)) + tk.sum(dim=(-3, -2, -1)) + eps)
        dices.append((inter / den))
    if not dices:
        return torch.tensor(0.0, device=pred_oh.device)
    return torch.stack(dices, dim=0).mean()


def _upsample_flow(flow_zyx, out_shape_zyx: Tuple[int, int, int]):
    import torch
    import torch.nn.functional as F

    z, y, x = out_shape_zyx
    flow_up = F.interpolate(flow_zyx, size=(z, y, x), mode="trilinear", align_corners=False)

    # Scale flow because voxel units change with resizing.
    z0, y0, x0 = flow_zyx.shape[-3:]
    sz = (z / max(z0, 1))
    sy = (y / max(y0, 1))
    sx = (x / max(x0, 1))
    scale = torch.tensor([sz, sy, sx], device=flow_up.device, dtype=flow_up.dtype).view(1, 3, 1, 1, 1)
    return flow_up * scale


def infer_flow_and_warp(
    *,
    model: VoxelMorphDense,
    inhale_ct_zyx: np.ndarray,
    exhale_ct_zyx: np.ndarray,
    device: str = "auto",
    scale: int = 1,
    norm: str = "ct_hu",
    ct_clip: Tuple[float, float] = (-1000.0, 400.0),
    pct: Tuple[float, float] = (1.0, 99.0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      exhale_warped_zyx (float32)
      disp_vox_3zyx (float32) in Z,Y,X component order on inhale grid
    """
    _require_torch()
    import torch

    dev = _get_device(device)
    model.to(dev)
    model.eval()

    with torch.no_grad():
        inh_hu = torch.from_numpy(inhale_ct_zyx[None, None].astype(np.float32)).to(dev)
        exh_hu = torch.from_numpy(exhale_ct_zyx[None, None].astype(np.float32)).to(dev)
        inh_n = torch.from_numpy(_normalize_np_to_0_1(inhale_ct_zyx, mode=norm, ct_clip=ct_clip, pct=pct)[None, None]).to(dev)
        exh_n = torch.from_numpy(_normalize_np_to_0_1(exhale_ct_zyx, mode=norm, ct_clip=ct_clip, pct=pct)[None, None]).to(dev)

        inh_ds = _downsample_if_needed(inh_n, int(scale))
        exh_ds = _downsample_if_needed(exh_n, int(scale))

        warped_ds, flow_ds = model(exh_ds, inh_ds)
        if tuple(warped_ds.shape[-3:]) != tuple(inh_n.shape[-3:]):
            flow_full = _upsample_flow(flow_ds, tuple(inh_n.shape[-3:]))
            # Safety clamp on full-res grid.
            z, y, x = inh_n.shape[-3:]
            max_disp = float(0.5 * max(1, min(int(z), int(y), int(x))))
            flow_full = torch.nan_to_num(flow_full, nan=0.0, posinf=0.0, neginf=0.0).clamp(-max_disp, max_disp)
            warped_full = _grid_sample_3d(exh_hu, flow_full, mode="bilinear", padding_mode="border")
        else:
            flow_full = flow_ds
            z, y, x = inh_n.shape[-3:]
            max_disp = float(0.5 * max(1, min(int(z), int(y), int(x))))
            flow_full = torch.nan_to_num(flow_full, nan=0.0, posinf=0.0, neginf=0.0).clamp(-max_disp, max_disp)
            warped_full = _grid_sample_3d(exh_hu, flow_full, mode="bilinear", padding_mode="border")

        exhale_warped = warped_full[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        disp_vox = flow_full[0].detach().cpu().numpy().astype(np.float32, copy=False)
        return exhale_warped, disp_vox


def warp_mask_nearest(
    *,
    mask_zyx: np.ndarray,
    disp_vox_3zyx: np.ndarray,
    device: str = "auto",
) -> np.ndarray:
    _require_torch()
    import torch

    dev = _get_device(device)
    m = torch.from_numpy(mask_zyx[None, None].astype(np.float32)).to(dev)
    f = torch.from_numpy(np.nan_to_num(disp_vox_3zyx, nan=0.0, posinf=0.0, neginf=0.0)[None].astype(np.float32)).to(dev)
    z, y, x = mask_zyx.shape
    max_disp = float(0.5 * max(1, min(int(z), int(y), int(x))))
    f = f.clamp(-max_disp, max_disp)
    with torch.no_grad():
        warped = _grid_sample_3d(m, f, mode="nearest", padding_mode="zeros")
    return (warped[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)


def warp_labels_nearest(
    *,
    labels_zyx: np.ndarray,
    disp_vox_3zyx: np.ndarray,
    device: str = "auto",
) -> np.ndarray:
    """
    Nearest-neighbor warping for integer label maps.
    Returns uint16 labels on the fixed grid.
    """
    _require_torch()
    import torch

    dev = _get_device(device)
    lab = torch.from_numpy(np.asarray(labels_zyx, dtype=np.float32)[None, None]).to(dev)
    f = torch.from_numpy(np.nan_to_num(np.asarray(disp_vox_3zyx, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)[None]).to(dev)
    z, y, x = labels_zyx.shape
    max_disp = float(0.5 * max(1, min(int(z), int(y), int(x))))
    f = f.clamp(-max_disp, max_disp)
    with torch.no_grad():
        warped = _grid_sample_3d(lab, f, mode="nearest", padding_mode="zeros")
    out = warped[0, 0].detach().cpu().numpy()
    return np.asarray(np.rint(out), dtype=np.uint16)


def warp_image_bilinear(
    *,
    volume_zyx: np.ndarray,
    disp_vox_3zyx: np.ndarray,
    device: str = "auto",
    padding_mode: str = "border",
) -> np.ndarray:
    """
    Bilinear warping for scalar images.
    Returns float32 image on the fixed grid.
    """
    _require_torch()
    import torch

    dev = _get_device(device)
    vol = torch.from_numpy(np.asarray(volume_zyx, dtype=np.float32)[None, None]).to(dev)
    f = torch.from_numpy(
        np.nan_to_num(np.asarray(disp_vox_3zyx, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)[None]
    ).to(dev)
    z, y, x = volume_zyx.shape
    max_disp = float(0.5 * max(1, min(int(z), int(y), int(x))))
    f = f.clamp(-max_disp, max_disp)
    with torch.no_grad():
        warped = _grid_sample_3d(vol, f, mode="bilinear", padding_mode=str(padding_mode))
    return warped[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)


def _dice_loss_onehot(
    *,
    pred: "torch.Tensor",
    target: "torch.Tensor",
    eps: float = 1e-6,
    exclude_background: bool = True,
) -> "torch.Tensor":
    """
    Soft Dice loss for one-hot/soft segmentations.
      pred, target: [B, C, Z, Y, X]
    """
    import torch

    if pred.shape != target.shape:
        raise ValueError("pred and target shapes must match for dice loss")
    b, c = pred.shape[:2]
    dims = tuple(range(2, pred.ndim))
    num = 2.0 * torch.sum(pred * target, dim=dims)
    den = torch.sum(pred * pred, dim=dims) + torch.sum(target * target, dim=dims) + eps
    dice = num / den
    if exclude_background and c > 1:
        dice = dice[:, 1:]
    return 1.0 - dice.mean()


def train_supervised_seg(
    *,
    model: VoxelMorphDense,
    images_zyx_list: Iterable[np.ndarray],
    labels_zyx_list: Iterable[np.ndarray],
    cfg: VxmConfig = VxmConfig(),
    n_labels: int | None = None,
) -> VoxelMorphDense:
    """
    Supervised training loop using segmentation Dice loss.

    Trains on random ordered pairs (moving, fixed) sampled from the provided lists.
    """
    _require_torch()
    import torch
    import torch.nn.functional as F

    dev = _get_device(cfg.device)
    model.to(dev)
    model.train(True)
    opt = torch.optim.Adam(list(model.parameters()), lr=float(cfg.lr))

    imgs = list(images_zyx_list)
    labs = list(labels_zyx_list)
    if len(imgs) != len(labs):
        raise ValueError("images_zyx_list and labels_zyx_list must have the same length")
    n = len(imgs)
    if n < 2:
        raise ValueError("Need at least 2 labeled cases for supervised training")

    if n_labels is None:
        mx = 0
        for a in labs:
            mx = max(mx, int(np.max(a)))
        n_labels = int(mx + 1)
    n_labels = int(max(2, n_labels))

    rng = np.random.default_rng(0)
    scale = int(cfg.scale)
    for _epoch in range(int(cfg.epochs)):
        loss_sum = 0.0
        dice_sum = 0.0
        n_steps = 0
        for _step in range(int(cfg.steps_per_epoch)):
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n - 1))
            if j >= i:
                j += 1

            fixed_img = _normalize_np_to_0_1(imgs[i], mode=cfg.norm, ct_clip=cfg.ct_clip, pct=cfg.pct)
            moving_img = _normalize_np_to_0_1(imgs[j], mode=cfg.norm, ct_clip=cfg.ct_clip, pct=cfg.pct)
            fixed_lab = np.asarray(labs[i], dtype=np.int64)
            moving_lab = np.asarray(labs[j], dtype=np.int64)

            fixed_t = torch.from_numpy(fixed_img[None, None].astype(np.float32)).to(dev)
            moving_t = torch.from_numpy(moving_img[None, None].astype(np.float32)).to(dev)
            fixed_t = _downsample_if_needed(fixed_t, scale)
            moving_t = _downsample_if_needed(moving_t, scale)

            # Labels: downsample using nearest (via interpolation on float then round).
            fixed_lab_t = torch.from_numpy(fixed_lab[None, None].astype(np.float32)).to(dev)
            moving_lab_t = torch.from_numpy(moving_lab[None, None].astype(np.float32)).to(dev)
            fixed_lab_t = _downsample_nearest_if_needed(fixed_lab_t, scale)
            moving_lab_t = _downsample_nearest_if_needed(moving_lab_t, scale)
            fixed_lab_t = torch.round(fixed_lab_t).to(torch.int64)
            moving_lab_t = torch.round(moving_lab_t).to(torch.int64)

            # One-hot (hard) then warp with bilinear to keep gradients.
            fixed_oh = F.one_hot(fixed_lab_t[:, 0].clamp(min=0, max=n_labels - 1), num_classes=n_labels).permute(0, 4, 1, 2, 3).to(torch.float32)
            moving_oh = F.one_hot(moving_lab_t[:, 0].clamp(min=0, max=n_labels - 1), num_classes=n_labels).permute(0, 4, 1, 2, 3).to(torch.float32)

            warped_img, flow = model(moving_t, fixed_t)
            moving_oh_warped = _grid_sample_3d(moving_oh, flow, mode="bilinear", padding_mode="zeros")

            seg_loss = _dice_loss_onehot(pred=moving_oh_warped, target=fixed_oh, exclude_background=True)
            loss = seg_loss + float(cfg.smooth_weight) * _grad_loss(flow)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                dsc = _hard_dice_from_onehot(moving_oh_warped, fixed_oh, exclude_background=True).item()
            loss_sum += float(loss.detach().cpu().item())
            dice_sum += float(dsc)
            n_steps += 1

        if n_steps > 0:
            print(
                f"[VM supervised] epoch {(_epoch+1)}/{int(cfg.epochs)} "
                f"loss={loss_sum/n_steps:.4f} hard_dice={dice_sum/n_steps:.4f} "
                f"(scale={scale}, smooth_w={float(cfg.smooth_weight):.3g})"
            )

    model.eval()
    return model


def train_hybrid(
    *,
    model: VoxelMorphDense,
    images_zyx_list: Iterable[np.ndarray],
    labels_zyx_list: Iterable[np.ndarray],
    cfg: VxmConfig = VxmConfig(),
    n_labels: int | None = None,
) -> VoxelMorphDense:
    """
    Hybrid training: supervised Dice + unsupervised similarity + smoothness.
    """
    _require_torch()
    import torch
    import torch.nn.functional as F

    dev = _get_device(cfg.device)
    model.to(dev)
    model.train(True)
    opt = torch.optim.Adam(list(model.parameters()), lr=float(cfg.lr))

    imgs = list(images_zyx_list)
    labs = list(labels_zyx_list)
    if len(imgs) != len(labs):
        raise ValueError("images_zyx_list and labels_zyx_list must have the same length")
    n = len(imgs)
    if n < 2:
        raise ValueError("Need at least 2 labeled cases for hybrid training")

    if n_labels is None:
        mx = 0
        for a in labs:
            mx = max(mx, int(np.max(a)))
        n_labels = int(mx + 1)
    n_labels = int(max(2, n_labels))

    rng = np.random.default_rng(0)
    scale = int(cfg.scale)
    for _epoch in range(int(cfg.epochs)):
        loss_sum = 0.0
        seg_sum = 0.0
        sim_sum = 0.0
        dice_sum = 0.0
        n_steps = 0
        for _step in range(int(cfg.steps_per_epoch)):
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n - 1))
            if j >= i:
                j += 1

            fixed_img = _normalize_np_to_0_1(imgs[i], mode=cfg.norm, ct_clip=cfg.ct_clip, pct=cfg.pct)
            moving_img = _normalize_np_to_0_1(imgs[j], mode=cfg.norm, ct_clip=cfg.ct_clip, pct=cfg.pct)
            fixed_lab = np.asarray(labs[i], dtype=np.int64)
            moving_lab = np.asarray(labs[j], dtype=np.int64)

            fixed_t = torch.from_numpy(fixed_img[None, None].astype(np.float32)).to(dev)
            moving_t = torch.from_numpy(moving_img[None, None].astype(np.float32)).to(dev)
            fixed_t = _downsample_if_needed(fixed_t, scale)
            moving_t = _downsample_if_needed(moving_t, scale)

            fixed_lab_t = torch.from_numpy(fixed_lab[None, None].astype(np.float32)).to(dev)
            moving_lab_t = torch.from_numpy(moving_lab[None, None].astype(np.float32)).to(dev)
            fixed_lab_t = _downsample_nearest_if_needed(fixed_lab_t, scale)
            moving_lab_t = _downsample_nearest_if_needed(moving_lab_t, scale)
            fixed_lab_t = torch.round(fixed_lab_t).to(torch.int64)
            moving_lab_t = torch.round(moving_lab_t).to(torch.int64)

            fixed_oh = F.one_hot(fixed_lab_t[:, 0].clamp(min=0, max=n_labels - 1), num_classes=n_labels).permute(0, 4, 1, 2, 3).to(torch.float32)
            moving_oh = F.one_hot(moving_lab_t[:, 0].clamp(min=0, max=n_labels - 1), num_classes=n_labels).permute(0, 4, 1, 2, 3).to(torch.float32)

            warped_img, flow = model(moving_t, fixed_t)
            moving_oh_warped = _grid_sample_3d(moving_oh, flow, mode="bilinear", padding_mode="zeros")

            seg_loss = _dice_loss_onehot(pred=moving_oh_warped, target=fixed_oh, exclude_background=True)
            if str(cfg.sim).lower() == "mse":
                sim_loss = F.mse_loss(warped_img, fixed_t)
            elif str(cfg.sim).lower() == "ncc":
                sim_loss = _ncc_loss_3d(fixed_t, warped_img, win=int(cfg.ncc_win), mask=None)
            else:
                raise ValueError("sim must be one of: mse, ncc")

            loss = seg_loss + sim_loss + float(cfg.smooth_weight) * _grad_loss(flow)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                dsc = _hard_dice_from_onehot(moving_oh_warped, fixed_oh, exclude_background=True).item()
            loss_sum += float(loss.detach().cpu().item())
            seg_sum += float(seg_loss.detach().cpu().item())
            sim_sum += float(sim_loss.detach().cpu().item())
            dice_sum += float(dsc)
            n_steps += 1

        if n_steps > 0:
            print(
                f"[VM hybrid] epoch {(_epoch+1)}/{int(cfg.epochs)} "
                f"loss={loss_sum/n_steps:.4f} seg={seg_sum/n_steps:.4f} sim={sim_sum/n_steps:.4f} "
                f"hard_dice={dice_sum/n_steps:.4f} (sim={str(cfg.sim).lower()}, win={int(cfg.ncc_win)}, scale={scale})"
            )

    model.eval()
    return model


def train_unsupervised(
    *,
    model: VoxelMorphDense,
    inhale_ct_zyx_list: Iterable[np.ndarray],
    exhale_ct_zyx_list: Iterable[np.ndarray],
    inhale_mask_zyx_list: Optional[Iterable[np.ndarray]] = None,
    cfg: VxmConfig = VxmConfig(),
) -> VoxelMorphDense:
    """
    Unsupervised training loop. This is intentionally minimal:
      loss = MSE(fixed, warp(moving)) + smooth_weight * grad_loss(flow)
    Optionally masks the MSE to inhale lung region.
    """
    _require_torch()
    import torch
    import torch.nn.functional as F

    dev = _get_device(cfg.device)
    model.to(dev)
    model.train(True)

    opt = torch.optim.Adam(list(model.parameters()), lr=float(cfg.lr))

    inh_list = list(inhale_ct_zyx_list)
    exh_list = list(exhale_ct_zyx_list)
    if len(inh_list) != len(exh_list):
        raise ValueError("Inhale/exhale lists must match length")

    mask_list = None
    if inhale_mask_zyx_list is not None:
        mask_list = list(inhale_mask_zyx_list)
        if len(mask_list) != len(inh_list):
            raise ValueError("Mask list must match inhale list length")

    rng = np.random.default_rng(0)
    n = len(inh_list)
    if n == 0:
        raise ValueError("Empty training set")

    scale = int(cfg.scale)
    for _epoch in range(int(cfg.epochs)):
        for _step in range(int(cfg.steps_per_epoch)):
            idx = int(rng.integers(0, n))
            inh_n = torch.from_numpy(
                _normalize_np_to_0_1(inh_list[idx], mode=cfg.norm, ct_clip=cfg.ct_clip, pct=cfg.pct)[None, None]
            ).to(dev)
            exh_n = torch.from_numpy(
                _normalize_np_to_0_1(exh_list[idx], mode=cfg.norm, ct_clip=cfg.ct_clip, pct=cfg.pct)[None, None]
            ).to(dev)

            inh_ds = _downsample_if_needed(inh_n, scale)
            exh_ds = _downsample_if_needed(exh_n, scale)

            warped, flow = model(exh_ds, inh_ds)

            if mask_list is not None:
                m = torch.from_numpy(mask_list[idx][None, None].astype(np.float32)).to(dev)
                m = _downsample_if_needed(m, scale)
                m = (m > 0.5).to(warped.dtype)
                if str(cfg.sim).lower() == "mse":
                    sim_loss = ((inh_ds - warped) ** 2 * m).sum() / (m.sum() + 1e-6)
                elif str(cfg.sim).lower() == "ncc":
                    sim_loss = _ncc_loss_3d(inh_ds, warped, win=int(cfg.ncc_win), mask=m)
                else:
                    raise ValueError("sim must be one of: mse, ncc")
            else:
                if str(cfg.sim).lower() == "mse":
                    sim_loss = F.mse_loss(warped, inh_ds)
                elif str(cfg.sim).lower() == "ncc":
                    sim_loss = _ncc_loss_3d(inh_ds, warped, win=int(cfg.ncc_win), mask=None)
                else:
                    raise ValueError("sim must be one of: mse, ncc")

            loss = sim_loss + float(cfg.smooth_weight) * _grad_loss(flow)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    return model
