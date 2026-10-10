"""Inference-only fast path for CREStereo (general: any input size, no per-size compilation / tuning).

    prepare_inference(model, half=True, cl=True, hacc=True, kern=True)
    flow = forward_fast(model, image1, image2, flow_init=None, iters=20)   # same output as model.forward (test_mode)

Changes vs. upstream forward (same math unless noted):
  - work that does not change the result is skipped: upsample mask + convex upsampling only after the last iteration of
    each stage (upstream: every iteration, plus an unused full-res interpolation per iteration), the 1/16 cross
    attention once per pass (upstream: every iteration on the same inputs), one grid_sample for the 4 correlation
    groups (identical coordinates), bilinear sampling with F.grid_sample / a fused kernel instead of the ONNX-export
    gather implementation
  - GRU: z|r convs merged; the constant context part (inp channels) of all six GRU convs computed once per stage
  - feature encoder per image (instance norm is per sample -> identical), halves the full-res activation peak
  - half: pure fp16 weights/activations, fp32 flow state, fp32 attention (1/16 res) and convex upsampling
  - cl: channels_last; hacc: fp16-accumulate cuDNN convs (precompiled engines only, nets/cudnn_conv.py)
  - kern: fused warp + group correlation kernel (no warped feature map, no 36 mul/mean passes) and fused GRU gates
  - fused (CRE_FUSED=1, default with kern): CUTLASS implicit-GEMM convs with the epilogue applied from registers
    (nets/cutlass_conv.cu, port of RAFT-Stereo exp 002 v8, compiled once, any size): SepConvGRU z|r conv + gates and
    q conv + state update for the 1x5 / 5x1 convs, conv + bias + relu for >= 128 output channels
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import PositionEncodingSine

_K = [None]
FUSED = os.environ.get("CRE_FUSED", "1") == "1"
_FC = [None]


def fused():
    if _FC[0] is None:
        from .cutlass_conv import EpiArgs, FusedConvs

        _FC[0] = (FusedConvs.get(), EpiArgs)
    return _FC[0]


def fused_cfg(P):
    """Fixed size rule (results/2026-10-09_fused_probe.txt): 128x128x32/3st tile, 64x128x64/4st for small problems."""
    return 0 if P >= 7000 else 1


def _conv_fused(m, x, w, b, act, out=None, out2=None, kmin=128):
    """conv + bias + act as one CUTLASS kernel, or None (not applicable -> cuDNN path). out / out2: (tensor, channel
    offset) destinations inside wider channels-last buffers (out2: second copy, same row stride as out)."""
    if (
        m is None
        or m.groups != 1
        or tuple(m.dilation) != (1, 1)
        or m.stride[0] != m.stride[1]
        or x.shape[0] != 1
        or w.shape[0] < kmin
        or w.shape[0] % 8
        or w.shape[1] % 8
        or not isinstance(m.padding, tuple)
        or not x.is_contiguous(memory_format=torch.channels_last)
        or getattr(m, "_nofuse", False)
    ):
        return None
    fc, E = fused()
    _, C, H, W = x.shape
    K = w.shape[0]
    (R, S), (ph, pw), st = w.shape[2:], m.padding, m.stride[0]
    Ho, Wo = (H + 2 * ph - R) // st + 1, (W + 2 * pw - S) // st + 1
    b32 = None
    if b is not None:
        b32 = m.__dict__.get("_b32")
        if b32 is None or b32[0] is not b:
            m._b32 = b32 = (b, b.float().contiguous())
        b32 = b32[1]
    if out is None:
        y = torch.empty(
            (1, K, Ho, Wo), device=x.device, dtype=x.dtype, memory_format=torch.channels_last
        )
        yp, ys = y.data_ptr(), K
    else:
        y, c0 = out
        assert y.is_contiguous(memory_format=torch.channels_last) and tuple(y.shape[2:]) == (Ho, Wo)
        yp, ys = y.data_ptr() + 2 * c0, y.shape[1]
    y2p = None if out2 is None else out2[0].data_ptr() + 2 * out2[1]
    e = E(
        2,
        int(act == "relu"),
        None,
        0,
        yp,
        ys,
        None,
        ys,
        y2p,
        None,
        None,
        None if b32 is None else b32.data_ptr(),
        None,
        0,
        0,
    )
    rc = fc.run(fused_cfg(Ho * Wo), x, w, H, W, C, K, e, padding=(ph, pw), stride=st)
    if rc == 1:
        m._nofuse = True
        return None
    if rc:
        raise RuntimeError(f"fused conv failed ({rc})")
    return y


def kern():
    if _K[0] is None:
        from .kernels import Kernels

        _K[0] = Kernels.get()
    return _K[0]


def _cl(x, o):
    return x.contiguous(memory_format=torch.channels_last) if o["cl"] else x.contiguous()


def conv(m, x, o, w=None, b="default", act=None):
    """conv2d of module m (or explicit weight w), bias, optional in-place activation."""
    w = m.weight if w is None else w
    b = (m.bias if m is not None else None) if isinstance(b, str) else b
    if o.get("fused") and act in (None, "relu") and x.dim() == 4 and x.dtype == torch.float16:
        y = _conv_fused(m, x, w, b, act)
        if y is not None:
            return y
    if o["hacc"]:
        from .cudnn_conv import conv_half_acc

        if o["kern"] and b is not None and w.shape[0] % 8 == 0:
            y = conv_half_acc(x, w, None, list(m.padding), list(m.stride))
            if y.is_contiguous(memory_format=torch.channels_last):
                k = kern()
                P = y.numel() // y.shape[1]
                k.launch(
                    "bias_act",
                    k.linear(P * y.shape[1] // 8),
                    y,
                    b,
                    P,
                    y.shape[1],
                    int(act == "relu"),
                )
                return y
            y.add_(b.view(1, -1, 1, 1))
        else:
            y = conv_half_acc(x, w, b, list(m.padding), list(m.stride))
    else:
        y = F.conv2d(x, w, b, m.stride, m.padding)
        y = _cl(y, o)
    if act == "relu":
        y.relu_()
    return y


# ------------------------------------------------------------------------------------------- feature encoder
def _inorm_(x, eps=1e-5):
    var, mean = torch.var_mean(
        x, dim=(2, 3), keepdim=True, unbiased=False
    )  # fp32 accumulation internally
    return x.sub_(mean).mul_(torch.rsqrt(var.float() + eps).to(x.dtype))


def _resblock(blk, x, o):
    y = _inorm_(conv(blk.conv1, x, o)).relu_()
    y = _inorm_(conv(blk.conv2, y, o)).relu_()
    x = _inorm_(conv(blk.downsample[0], x, o))
    return y.add_(x).relu_()


def encoder(enc, x, o):
    x = _inorm_(conv(enc.conv1, x, o)).relu_()
    for layer in (enc.layer1, enc.layer2, enc.layer3):
        for blk in layer:
            x = _resblock(blk, x, o)
    return conv(enc.conv2, x, o)


# ------------------------------------------------------------------------------------------- correlation
def _offsets(small):
    return [(o // 3 - 1, o % 3 - 1) for o in range(9)] if small else [(0, o - 4) for o in range(9)]


def _norm_grid(
    coords, H, W
):  # coords [N, h, w, 2] pixel units -> grid_sample grid (upstream bilinear_sampler)
    return torch.stack([2 * coords[..., 0] / (W - 1) - 1, 2 * coords[..., 1] / (H - 1) - 1], -1)


def corr_torch(fm1, fm2, flow, extra, small):
    """Reference (upstream semantics, F.grid_sample in fp32). fm1/fm2 [N, 256, H, W]; returns [N, 36, H, W] fp32."""
    N, C, H, W = fm1.shape
    ys, xs = torch.meshgrid(
        torch.arange(H, device=fm1.device), torch.arange(W, device=fm1.device), indexing="ij"
    )
    base = torch.stack([xs, ys], -1).float()[None] + flow.permute(0, 2, 3, 1)  # [N, H, W, 2]
    left_features = fm1.float().reshape(N, 4, C // 4, 1, H, W)
    offs = _offsets(small)
    if extra is None:
        warped = F.grid_sample(fm2.float(), _norm_grid(base, H, W), align_corners=True)
        pad = (1, 1, 1, 1) if small else (4, 4, 0, 0)
        wp = F.pad(warped, pad, mode="replicate")
        py, px = pad[2], pad[0]
        r = torch.stack(
            [wp[:, :, py + dy : py + dy + H, px + dx : px + dx + W] for dy, dx in offs], 2
        )  # [N, C, 9, H, W]
    else:
        e = extra.reshape(N, 9, 2, H, W).permute(0, 1, 3, 4, 2)  # [N, 9, H, W, 2]
        d = torch.tensor(
            [[dx, dy] for dy, dx in offs], device=fm1.device, dtype=torch.float32
        ).view(1, 9, 1, 1, 2)
        coords = (base[:, None] + d + e).reshape(N, 9 * H, W, 2)
        r = F.grid_sample(fm2.float(), _norm_grid(coords, H, W), align_corners=True).reshape(
            N, C, 9, H, W
        )
    return (left_features * r.reshape(N, 4, C // 4, 9, H, W)).mean(2).reshape(N, 36, H, W)


def corr(fm1, fm2, flow, extra, small, o):
    if o["kern"]:
        N, C, H, W = fm1.shape
        assert (
            C == 256
            and fm1.is_contiguous(memory_format=torch.channels_last)
            and fm2.is_contiguous(memory_format=torch.channels_last)
        )
        out = torch.empty(
            (N, 40, H, W), device=fm1.device, dtype=torch.half, memory_format=torch.channels_last
        )
        k = kern()
        nwarp = N * H * W
        k.launch(
            "agcl_corr",
            ((min((nwarp + 7) // 8, 65535 * 8), 1, 1), (256, 1, 1)),
            fm1,
            fm2,
            flow.contiguous(),
            0 if extra is None else extra.contiguous(),
            out,
            N,
            H,
            W,
            40,
            int(small),
        )
        return out
    c = corr_torch(fm1, fm2, flow, extra, small).to(fm1.dtype)
    return _cl(F.pad(c, (0, 0, 0, 0, 0, 4)), o)  # 36 -> 40 channels (convc1 weight zero-padded)


# ------------------------------------------------------------------------------------------- update block
def stage_context(model, inp, o):
    """constant part of the six GRU convs: conv(inp, W[:, 128:256]) + bias, per direction (z|r merged, q)."""
    g = model.update_block.gru
    ctx = []
    for d in (1, 2):
        czr = conv(getattr(g, f"convz{d}"), inp, o, w=g._w[f"zr_i{d}"], b=g._w[f"zr_b{d}"])
        cq = conv(getattr(g, f"convq{d}"), inp, o, w=g._w[f"q_i{d}"], b=g._w[f"q_b{d}"])
        ctx.append((czr, cq))
    return ctx


def update(model, net, ctx, corr_feat, flow, o):
    """one update step: returns (net, delta_flow). net updated in place when kern."""
    ub = model.update_block
    enc, g = ub.encoder, ub.gru
    flow_h = flow.to(net.dtype)
    if (
        o.get("fused")
        and net.shape[0] == 1
        and net.is_contiguous(memory_format=torch.channels_last)
    ):
        r = _update_fused(model, net, ctx, corr_feat, flow_h, o)
        if r is not None:
            return r
    cor = conv(enc.convc1, corr_feat, o, w=enc._wc1, act="relu")
    cor = conv(enc.convc2, cor, o, act="relu")
    flo = conv(enc.convf1, _cl(flow_h, o), o, act="relu")
    flo = conv(enc.convf2, flo, o, act="relu")
    mo = conv(enc.conv, torch.cat([cor, flo], 1), o, act="relu")  # 126 ch
    del cor, flo
    N, Hd, H, W = net.shape
    hx = torch.empty(
        (N, 2 * Hd, H, W),
        device=net.device,
        dtype=net.dtype,
        memory_format=torch.channels_last if o["cl"] else torch.contiguous_format,
    )
    hx[:, :Hd] = net
    hx[:, Hd : Hd + 126] = mo
    hx[:, Hd + 126 :] = flow_h
    del mo
    for d in (1, 2):
        czr, cq = ctx[d - 1]
        zr = conv(getattr(g, f"convz{d}"), hx, o, w=g._w[f"zr_h{d}"], b=None)
        if o["kern"]:
            k = kern()
            P = N * H * W
            k.launch("gru_zr", k.linear(P * Hd // 8), zr, czr, net, hx, 2 * Hd, P, Hd)
            q = conv(getattr(g, f"convq{d}"), hx, o, w=g._w[f"q_h{d}"], b=None)
            k.launch("gru_update", k.linear(P * Hd // 8), q, cq, zr, net, hx, 2 * Hd, P, Hd)
        else:
            zr = torch.sigmoid(zr + czr)
            z, r = zr[:, :Hd], zr[:, Hd:]
            hx[:, :Hd] = r * net
            q = torch.tanh(conv(getattr(g, f"convq{d}"), hx, o, w=g._w[f"q_h{d}"], b=None) + cq)
            net = _cl((1 - z) * net + z * q, o)
            hx[:, :Hd] = net
    fh = ub.flow_head
    delta = conv(fh.conv2, conv(fh.conv1, net, o, act="relu"), o)
    return net, delta[:, :2].float()


def _update_fused(model, net, ctx, corr_feat, flow_h, o):
    """update() with CUTLASS fused-epilogue convs: convc2 writes straight into the [cor | flo] buffer (no cat),
    the motion conv (zero-padded to 128 outputs) writes into the x slots of both GRU buffers A = [h | x] (read by
    the z|r convs) and B = [r*h | x] (read by the q convs, r*h written by the z|r epilogue); the q epilogue updates net
    in place and (direction 1) h in A for the 5x1 z|r conv. Returns None if a conv geometry is unsupported."""
    ub = model.update_block
    enc, g = ub.encoder, ub.gru
    fc, E = fused()
    N, Hd, H, W = net.shape
    cl = torch.channels_last
    cor = conv(enc.convc1, corr_feat, o, w=enc._wc1, act="relu")
    cf = torch.empty((N, 256, H, W), device=net.device, dtype=net.dtype, memory_format=cl)
    if (
        _conv_fused(
            enc.convc2, cor, enc.convc2.weight, enc.convc2.bias, "relu", out=(cf, 0), kmin=64
        )
        is None
    ):
        return None
    del cor
    flo = conv(enc.convf1, _cl(flow_h, o), o, act="relu")
    cf[:, 192:] = conv(
        enc.convf2, flo, o, act="relu"
    )  # 64-ch conv: cuDNN + bias_act + copy (fused was slower)
    del flo
    hx = torch.empty((N, 2 * Hd, H, W), device=net.device, dtype=net.dtype, memory_format=cl)
    rhx = torch.empty_like(hx)
    if (
        _conv_fused(enc.conv, cf, enc._w128, enc._b128, "relu", out=(hx, Hd), out2=(rhx, Hd))
        is None
    ):
        return None
    del cf
    hx[:, :Hd] = net
    hx[:, 2 * Hd - 2 :] = flow_h
    rhx[:, 2 * Hd - 2 :] = flow_h
    P = H * W
    zr = torch.empty((N, 2 * Hd, H, W), device=net.device, dtype=net.dtype, memory_format=cl)
    ctot = 2 * Hd
    for d in (1, 2):
        czr, cq = ctx[d - 1]
        pad = getattr(g, f"convz{d}").padding
        e0 = E(
            0,
            0,
            czr.data_ptr(),
            2 * Hd,
            zr.data_ptr(),
            2 * Hd,
            hx.data_ptr(),
            ctot,
            rhx.data_ptr(),
            None,
            None,
            None,
            None,
            0,
            0,
        )
        e1 = E(
            1,
            0,
            cq.data_ptr(),
            Hd,
            zr.data_ptr(),
            2 * Hd,
            net.data_ptr(),
            Hd,
            None,
            hx.data_ptr() if d == 1 else None,
            None,
            None,
            None,
            0,
            ctot,
        )
        if fc.run(0, hx, g._w[f"zr_h{d}"], H, W, ctot, 2 * Hd, e0, padding=pad) or fc.run(
            fused_cfg(P), rhx, g._w[f"q_h{d}"], H, W, ctot, Hd, e1, padding=pad
        ):
            raise RuntimeError("fused GRU conv failed")
    fh = ub.flow_head
    delta = conv(fh.conv2, conv(fh.conv1, net, o, act="relu"), o)
    return net, delta[:, :2].float()


def up_mask(model, net, o):
    m = model.update_block.mask
    return (0.25 * conv(m[2], conv(m[0], net, o, act="relu"), o)).float()


# ------------------------------------------------------------------------------------------- prepare / forward
@torch.no_grad()
def prepare_inference(model, half=True, cl=True, hacc=True, kern=True):
    assert not (hacc or kern) or (half and cl), "hacc/kern need half + cl"
    model.eval()
    dt = torch.float16 if half else torch.float32
    mf = torch.channels_last if cl else torch.contiguous_format
    for mod in (model.fnet, model.update_block, model.conv_offset_8, model.conv_offset_16):
        mod.to(dt)
        for m in mod.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = m.weight.data.contiguous(memory_format=mf)
    ub = model.update_block
    enc, g = ub.encoder, ub.gru
    w = enc.convc1.weight
    enc._wc1 = F.pad(w, (0, 0, 0, 0, 0, 4)).contiguous(memory_format=mf)  # 36 -> 40 input channels
    g._w = {}
    for d in (1, 2):
        wz, wr, wq = (getattr(g, f"conv{n}{d}").weight for n in "zrq")
        bz, br, bq = (getattr(g, f"conv{n}{d}").bias for n in "zrq")
        wzr = torch.cat([wz, wr], 0)
        g._w[f"zr_h{d}"] = torch.cat([wzr[:, :128], wzr[:, 256:]], 1).contiguous(memory_format=mf)
        g._w[f"zr_i{d}"] = wzr[:, 128:256].contiguous(memory_format=mf)
        g._w[f"zr_b{d}"] = torch.cat([bz, br]).contiguous()
        g._w[f"q_h{d}"] = torch.cat([wq[:, :128], wq[:, 256:]], 1).contiguous(memory_format=mf)
        g._w[f"q_i{d}"] = wq[:, 128:256].contiguous(memory_format=mf)
        g._w[f"q_b{d}"] = bq.contiguous()
    # motion conv zero-padded to 128 outputs for the fused path (the 2 extra channels are overwritten by the flow)
    enc._w128 = F.pad(enc.conv.weight, (0, 0, 0, 0, 0, 0, 0, 2)).contiguous(memory_format=mf)
    enc._b128 = F.pad(enc.conv.bias, (0, 2)).contiguous()
    model._fast = dict(half=half, cl=cl, hacc=hacc, kern=kern, dt=dt, fused=FUSED and kern and hacc)
    if kern:
        globals()["kern"]()
    return model


def convex_upsample(flow, mask, rate=4):
    N, _, H, W = flow.shape
    mask = torch.softmax(mask.view(N, 1, 9, rate, rate, H, W), dim=2)
    up = F.unfold(rate * flow, [3, 3], padding=1).view(N, 2, 9, 1, 1, H, W)
    up = torch.sum(mask * up, dim=2).permute(0, 1, 4, 2, 5, 3)
    return up.reshape(N, 2, rate * H, rate * W)


def _interp(x, h, w):
    return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)


def _attn_feats(model, f, H16, W16):
    """avg-pooled 1/16 features + sine position encoding -> [N, L, 256] fp32."""
    pe = PositionEncodingSine(d_model=256, max_shape=(H16, W16)).pe.to(f.device)
    x = f.float() + pe[:, :, : f.shape[2], : f.shape[3]]
    return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])


def _run_stage(model, fm1, fm2, net, inp, flow, extra, iters, o):
    net = _cl(net, o)
    ctx = stage_context(model, _cl(inp, o), o)
    for itr in range(iters):
        c = corr(fm1, fm2, flow, extra, itr % 2 == 1, o)
        net, delta = update(model, net, ctx, c, flow, o)
        flow = flow + delta
    return flow, up_mask(model, net, o)


@torch.no_grad()
def forward_fast(model, image1, image2, flow_init=None, iters=20):
    o = model._fast
    dt = o["dt"]
    hdim = model.hidden_dim
    # feature encoder, one image at a time
    fm = [
        encoder(model.fnet, _cl((2 * (im / 255.0) - 1.0).to(dt), o), o) for im in (image1, image2)
    ]
    fmap1, fmap2 = fm
    N, _, H4, W4 = fmap1.shape
    net, inp = fmap1[:, :hdim].tanh(), fmap1[:, hdim:].relu()
    if flow_init is not None:
        scale = H4 / flow_init.shape[2]
        flow = -scale * _interp(flow_init, H4, W4)
    else:
        H16, W16 = image1.shape[2] // 16, image1.shape[3] // 16
        # 1/16 stage: self attention, then cross attention (upstream re-runs it every iteration on the same input)
        f1, f2 = F.avg_pool2d(fmap1, 4, stride=4), F.avg_pool2d(fmap2, 4, stride=4)
        off16 = (
            model.range_16
            * (torch.sigmoid(conv(model.conv_offset_16, _cl(f1, o), o).float()) - 0.5)
            * 2.0
        )
        a1, a2 = model.self_att_fn(
            _attn_feats(model, f1, H16, W16), _attn_feats(model, f2, H16, W16)
        )
        a1, a2 = model.cross_att_fn(a1, a2)
        h16, w16 = f1.shape[2:]
        a1, a2 = [_cl(x.reshape(N, h16, w16, -1).permute(0, 3, 1, 2).to(dt), o) for x in (a1, a2)]
        flow16 = torch.zeros((N, 2, h16, w16), device=fmap1.device)
        flow16, mask = _run_stage(
            model,
            a1,
            a2,
            F.avg_pool2d(net, 4, stride=4),
            F.avg_pool2d(inp, 4, stride=4),
            flow16,
            off16,
            iters // 2,
            o,
        )
        del a1, a2
        flow = convex_upsample(flow16, mask, rate=4)
        # 1/8 stage
        f1, f2 = F.avg_pool2d(fmap1, 2, stride=2), F.avg_pool2d(fmap2, 2, stride=2)
        f1, f2 = _cl(f1, o), _cl(f2, o)
        off8 = model.range_8 * (torch.sigmoid(conv(model.conv_offset_8, f1, o).float()) - 0.5) * 2.0
        h8, w8 = f1.shape[2:]
        flow8 = -(h8 / flow.shape[2]) * _interp(flow, h8, w8)
        flow8, mask = _run_stage(
            model,
            f1,
            f2,
            F.avg_pool2d(net, 2, stride=2),
            F.avg_pool2d(inp, 2, stride=2),
            flow8,
            off8,
            iters // 2,
            o,
        )
        del f1, f2
        flow = convex_upsample(flow8, mask, rate=4)
        flow = -(H4 / flow.shape[2]) * _interp(flow, H4, W4)
    flow, mask = _run_stage(model, fmap1, fmap2, net, inp, flow.contiguous(), None, iters, o)
    return -convex_upsample(flow, mask, rate=4)
