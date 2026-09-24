"""PointPillars 部署与基准：导出 ONNX、数值/AP 对齐、多后端时延对比（自研模型落地推理引擎）。

  export  训练权重 → reports/pp.onnx
  verify  同一帧 torch vs ONNXRuntime 最大绝对差（数值对齐，证明导出无损）
  bench   多后端单帧 NN 时延对比 + 柱状图 reports/det3d_deploy_bench.png
  eval    经 ONNX 后端在 KITTI val 上重算 Car BEV AP，确认落地不掉点

用法: python scripts/det3d_deploy.py --do all --bench-iters 100 --eval-frames 400
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from det3d import deploy as D
from det3d import kitti_det as kd
from det3d.anchors import _corners, generate_anchors  # noqa: F401 (_corners 供 eval 间接用)
from det3d.infer import postprocess
from det3d.pointpillars import PointPillars

ONNX = ROOT / 'reports' / 'pp.onnx'
CKPT = ROOT / 'reports' / 'pp_ckpt.pth'
BASELINE = {0.5: 80.46, 0.7: 70.06}      # README 里 PyTorch 原栈的 val AP


def load_net(dev='cpu'):
    net = PointPillars().to(dev)
    ck = torch.load(CKPT, map_location=dev)
    net.load_state_dict(ck['model'])
    net.eval()
    print(f'loaded ckpt @ it {ck.get("it", "?")}')
    return net


def sample_inputs():
    """一帧真实点云的 (pillars,mask,coords)；无 KITTI 时退合成点云（保证脚本可跑）。"""
    try:
        pts = kd.read_velodyne('training', kd.load_split('val')[0])
    except Exception:
        rng = np.random.default_rng(0)
        pts = rng.uniform([0, -39, -3, 0], [69, 39, 1, 1], (80000, 4)).astype(np.float32)
    return D.prep(pts)


def do_verify(net, inp):
    ref = D.TorchRunner(net, 'cpu')(*inp)
    ort_out = D.OrtRunner(ONNX, ['CPUExecutionProvider'])(*inp)
    d = D.max_abs_diff(ref, ort_out)
    print(f'  torch vs ONNXRuntime max|Δ| on (cls,box,dir): {d:.2e}  '
          f'{"OK (lossless)" if d < 1e-3 else "WARN"}')
    return d


def make_backends(net):
    """[(name, runner, cuda_sync)]；缺失的后端(无 GPU / EP 起不来)自动跳过并如实打印原因。"""
    bks = []
    if torch.cuda.is_available():
        bks.append(('PyTorch FP32 (CUDA)', D.TorchRunner(net, 'cuda', False), True))
        bks.append(('PyTorch FP16 (CUDA)', D.TorchRunner(net, 'cuda', True), True))
    if ONNX.exists():
        try:
            import onnxruntime as ort
            avail = ort.get_available_providers()
            if 'TensorrtExecutionProvider' in avail:
                try:
                    r = D.OrtRunner(ONNX, [('TensorrtExecutionProvider', {'trt_fp16_enable': True}),
                                           'CUDAExecutionProvider', 'CPUExecutionProvider'])
                    if r.provider == 'TensorrtExecutionProvider':
                        bks.append(('ONNXRuntime-TensorRT FP16', r, False))
                    else:
                        print(f'  TensorRT EP fell back to {r.provider} (skipped)')
                except Exception as e:
                    print('  TensorRT EP unavailable:', str(e).splitlines()[0][:110])
            if 'CUDAExecutionProvider' in avail:
                try:
                    r = D.OrtRunner(ONNX, ['CUDAExecutionProvider', 'CPUExecutionProvider'])
                    if r.provider == 'CUDAExecutionProvider':
                        bks.append(('ONNXRuntime (CUDA)', r, False))
                    else:
                        print(f'  CUDA EP fell back to {r.provider} (skipped)')
                except Exception as e:
                    print('  CUDA EP unavailable:', str(e).splitlines()[0][:110])
            bks.append(('ONNXRuntime (CPU)', D.OrtRunner(ONNX, ['CPUExecutionProvider']), False))
        except Exception as e:
            print('  ONNXRuntime unavailable:', e)
    bks.append(('PyTorch FP32 (CPU)', D.TorchRunner(net, 'cpu', False), False))
    return bks


def do_bench(net, inp, iters, out):
    rows = []
    for name, runner, cuda in make_backends(net):
        try:
            ms = D.bench(runner, inp, warmup=15, iters=iters, cuda=cuda)
        except Exception as e:
            print(f'  {name:26s} FAILED: {str(e).splitlines()[0][:90]}')
            continue
        rows.append((name, ms))
        print(f'  {name:26s} {ms:8.2f} ms/frame  ({1e3/ms:7.1f} FPS)')
    if not rows:
        print('  no backend ran'); return rows
    ref_name, ref = next(((n, m) for n, m in rows if 'FP32 (CUDA)' in n), rows[0])
    print(f'  speedup vs {ref_name}:')
    for n, m in rows:
        print(f'    {n:26s} {ref/m:5.2f}x')
    _plot(rows, out)
    return rows


def _run_eval(runner, anchors, frames, score):
    from scripts.det3d_eval import ap_r40, bev_iou
    res = {0.5: [[], []], 0.7: [[], []]}
    n_gt = 0
    for k, idx in enumerate(frames):
        preds = runner(*D.prep(kd.read_velodyne('training', idx)))
        boxes, scores = postprocess(preds, anchors, score_thr=score)
        gt = kd.read_labels('training', idx, kd.read_calib('training', idx))
        n_gt += len(gt)
        order = scores.argsort()[::-1]
        for thr in (0.5, 0.7):
            matched = np.zeros(len(gt), bool)
            for i in order:
                best, bj = thr, -1
                for j in range(len(gt)):
                    if matched[j]:
                        continue
                    iou = bev_iou(boxes[i], gt[j])
                    if iou >= best:
                        best, bj = iou, j
                res[thr][0].append(scores[i]); res[thr][1].append(1 if bj >= 0 else 0)
                if bj >= 0:
                    matched[bj] = True
        if (k + 1) % 100 == 0:
            print(f'    {k+1}/{len(frames)} frames, {n_gt} GT', flush=True)
    return {thr: ap_r40(res[thr][0], res[thr][1], n_gt) for thr in (0.5, 0.7)}, n_gt


def do_eval(anchors, frames, score):
    import onnxruntime as ort
    provs = (['CUDAExecutionProvider', 'CPUExecutionProvider']
             if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider'])
    runner = D.OrtRunner(ONNX, provs)
    print(f'  backend: ONNXRuntime [{runner.provider}], {len(frames)} val frames')
    aps, n_gt = _run_eval(runner, anchors, frames, score)
    print(f'  KITTI val Car BEV AP (R40, {n_gt} GT)  [baseline = PyTorch original stack]:')
    for thr in (0.5, 0.7):
        print(f'    AP@{thr}: {aps[thr]:6.2f}   (baseline {BASELINE[thr]:.2f}, Δ {aps[thr]-BASELINE[thr]:+.2f})')
    return aps


def _plot(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from kitti_slam import plotstyle as ps
    names = [n for n, _ in rows][::-1]
    ms = [m for _, m in rows][::-1]
    fig, ax = plt.subplots(figsize=(9.8, 0.62 * len(rows) + 1.9))
    fig.patch.set_facecolor(ps.BG)
    fastest = min(range(len(ms)), key=lambda i: ms[i])          # 最快后端高亮成青
    colors = [ps.GT if i == fastest else ps.EST for i in range(len(ms))]
    bars = ax.barh(names, ms, color=colors, alpha=0.92, zorder=3)
    for b, m in zip(bars, ms):
        ax.text(b.get_width() + max(ms) * 0.012, b.get_y() + b.get_height() / 2,
                f'{m:.1f} ms · {1e3 / m:.0f} FPS', va='center', color=ps.FG, fontsize=9)
    ax.set_xlabel('single-frame NN latency (ms, lower = better)')
    ax.set_title('PointPillars deployment — NN inference latency by backend (RTX 5090)')
    ax.set_xlim(0, max(ms) * 1.30)
    ps.style_ax(ax)
    ax.yaxis.grid(False)                                        # 横条图只留竖向网格更干净
    ps.savefig(fig, out)
    print(f'  saved {out}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--do', default='all', choices=['all', 'export', 'verify', 'bench', 'eval'])
    ap.add_argument('--bench-iters', type=int, default=100)
    ap.add_argument('--eval-frames', type=int, default=400)
    ap.add_argument('--score', type=float, default=0.1)
    ap.add_argument('--opset', type=int, default=17)
    a = ap.parse_args(argv)
    steps = ['export', 'verify', 'bench', 'eval'] if a.do == 'all' else [a.do]

    net = load_net('cpu')
    if 'export' in steps:
        print('== export ONNX ==')
        D.export_onnx(net, ONNX, opset=a.opset)
        import onnx
        onnx.checker.check_model(onnx.load(str(ONNX)))
        print(f'  saved {ONNX}  ({ONNX.stat().st_size/1e6:.1f} MB, checker OK)')
    inp = sample_inputs()
    if 'verify' in steps:
        print('== verify (torch vs ONNXRuntime) ==')
        do_verify(net, inp)
    if 'bench' in steps:
        print('== bench (single-frame NN latency) ==')
        do_bench(net, inp, a.bench_iters, ROOT / 'reports' / 'det3d_deploy_bench.png')
    if 'eval' in steps:
        print('== eval (AP via ONNX backend) ==')
        do_eval(generate_anchors(248, 216), kd.load_split('val')[:a.eval_frames], a.score)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


