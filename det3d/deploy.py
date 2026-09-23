"""PointPillars 部署：导出 ONNX + ONNXRuntime / TensorRT 推理封装（把训好的模型落地）。

训练是纯 PyTorch，这里把**神经网络这一段**(VFE→scatter→2D 骨干→SSD 头)导成可移植 ONNX，
喂进 ONNXRuntime(CPU/CUDA) 或 TensorRT 引擎；数值对齐 + 保 AP，量化 FP16 测时延。
预处理(voxelize.points_to_pillars, numpy) 与后处理(infer.postprocess) 沿用原栈不动。

导出口径：batch=1 单帧实时推理。输入 (pillars[P,32,9], mask[P,32], coords[P,2]=(iy,ix))，
动态维 P；输出三张预测图 (cls,box,dir)，与 SSDHead 完全一致，因此权重可原样加载、结果可对齐。
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from .voxelize import NX, NY, points_to_pillars


class PPExport(nn.Module):
    """batch=1 导出包装：复用训练好的 vfe/backbone/head 子模块，只改成 ONNX 友好的算子。

    与 PointPillars.forward 的唯一区别：屏蔽柱内无效点用**有限大负值**(而非 -inf)，
    这样 FP16 下 max 也稳；每个柱至少 1 个有效点，故 max 结果与原实现逐比特一致。
    """
    NEG = 1.0e4

    def __init__(self, net):
        super().__init__()
        self.vfe, self.backbone, self.head = net.vfe, net.backbone, net.head

    def forward(self, pillars, mask, coords):
        x = self.vfe.linear(pillars)                       # [P,M,64]
        P, M, C = x.shape
        x = torch.relu(self.vfe.bn(x.reshape(-1, C)).reshape(P, M, C))
        x = x + (mask.unsqueeze(-1) - 1.0) * self.NEG      # 无效点 → 极负，不会赢 max
        feats = x.max(dim=1)[0]                            # [P,64]
        canvas = feats.new_zeros((NY, NX, C))              # channels-last：索引连续才可导出 ONNX
        canvas[coords[:, 0].long(), coords[:, 1].long(), :] = feats     # scatter → 伪图(柱 iy,ix 唯一)
        bev = canvas.permute(2, 0, 1).unsqueeze(0)         # → [1,C,NY,NX]
        out = self.head(self.backbone(bev))
        return out['cls'], out['box'], out['dir']


def prep(pts):
    """点云 → (pillars[P,32,9] f32, mask[P,32] f32, coords[P,2] i64=(iy,ix))，喂任意后端。"""
    pil, coords, npt = points_to_pillars(pts)
    maxpts = pil.shape[1]
    mask = (np.arange(maxpts)[None] < npt[:, None]).astype(np.float32)
    return pil.astype(np.float32), mask, coords.astype(np.int64)


def fuse_bn(net):
    """折叠 BN：Linear+BN1d、Conv/Deconv2d+BN2d 融进权重（部署标准优化）。返回融合后的副本。

    推理期 BN 是仿射变换，可解析并入前一层，省算子、也绕开某些运行时的 BN 内核限制。
    数学等价，输出与原模型逐点一致(见 verify)。
    """
    import copy

    from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval
    net = copy.deepcopy(net).eval()
    net.vfe.linear = fuse_linear_bn_eval(net.vfe.linear, net.vfe.bn)
    net.vfe.bn = nn.Identity()
    for seq in net.backbone.children():                # b1..b3 / u1..u3 都是 Sequential
        if not isinstance(seq, nn.Sequential):
            continue
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            if isinstance(a, (nn.Conv2d, nn.ConvTranspose2d)) and isinstance(b, nn.BatchNorm2d):
                seq[i] = fuse_conv_bn_eval(a, b, transpose=isinstance(a, nn.ConvTranspose2d))
                seq[i + 1] = nn.Identity()
    return net


def export_onnx(net, path, opset=17):
    """折叠 BN 后导出端到端 ONNX（动态柱数 P）。返回 path。"""
    wrap = PPExport(fuse_bn(net)).eval().cpu()
    g = torch.Generator().manual_seed(0)
    P = 3000
    pillars = torch.randn(P, 32, 9, generator=g)
    mask = (torch.arange(32)[None] < torch.randint(1, 33, (P, 1), generator=g)).float()
    coords = torch.stack([torch.randint(0, NY, (P,), generator=g),
                          torch.randint(0, NX, (P,), generator=g)], 1)
    dyn = {'pillars': {0: 'P'}, 'mask': {0: 'P'}, 'coords': {0: 'P'}}
    kw = dict(input_names=['pillars', 'mask', 'coords'], output_names=['cls', 'box', 'dir'],
              dynamic_axes=dyn, opset_version=opset, do_constant_folding=True)
    try:
        torch.onnx.export(wrap, (pillars, mask, coords), str(path), dynamo=False, **kw)
    except TypeError:                                      # 老/新 torch 无 dynamo 形参
        torch.onnx.export(wrap, (pillars, mask, coords), str(path), **kw)
    return path


class TorchRunner:
    """PyTorch 后端（fp32/fp16, cuda/cpu），暴露 (pillars,mask,coords)->preds dict。"""
    def __init__(self, net, device='cuda', half=False):
        import copy
        self.device, self.half = device, half
        self.wrap = PPExport(copy.deepcopy(net)).eval().to(device)   # 深拷贝：各后端互不干扰
        if half:
            self.wrap = self.wrap.half()

    def prepare(self, pillars, mask, coords):
        """把输入预置到目标设备（基准计时时隔离 host↔device 拷贝）。"""
        dt = torch.float16 if self.half else torch.float32
        return (torch.from_numpy(pillars).to(self.device, dt),
                torch.from_numpy(mask).to(self.device, dt),
                torch.from_numpy(coords).to(self.device))

    def run_prepared(self, staged):
        with torch.no_grad():
            self.wrap(*staged)

    def __call__(self, pillars, mask, coords):
        with torch.no_grad():
            c, b, d = self.wrap(*self.prepare(pillars, mask, coords))
        return {'cls': c[0].float().cpu().numpy(),
                'box': b[0].float().cpu().numpy(),
                'dir': d[0].float().cpu().numpy()}


class OrtRunner:
    """ONNXRuntime 后端。providers 例：['CUDAExecutionProvider','CPUExecutionProvider']。"""
    def __init__(self, path, providers):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=providers)
        self.provider = self.sess.get_providers()[0]

    def prepare(self, pillars, mask, coords):
        return {'pillars': pillars, 'mask': mask, 'coords': coords}

    def run_prepared(self, feed):
        self.sess.run(['cls', 'box', 'dir'], feed)

    def __call__(self, pillars, mask, coords):
        o = self.sess.run(['cls', 'box', 'dir'], self.prepare(pillars, mask, coords))
        return {'cls': o[0][0], 'box': o[1][0], 'dir': o[2][0]}


def bench(runner, inputs, warmup=15, iters=100, cuda=False):
    """平均单帧毫秒（NN 计算）。输入先预置到设备再计时，隔离 host I/O；cuda=True 用同步对齐。"""
    staged = runner.prepare(*inputs)
    for _ in range(warmup):
        runner.run_prepared(staged)
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        runner.run_prepared(staged)
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def max_abs_diff(a, b):
    """两个 preds dict 的最大逐元素绝对差（数值对齐用）。"""
    return max(float(np.abs(a[k] - b[k]).max()) for k in ('cls', 'box', 'dir'))
