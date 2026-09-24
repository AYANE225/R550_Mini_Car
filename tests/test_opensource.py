"""集成开源 WaffleIron 适配器单测。

适配器只在 __init__ 里才 import/加载 WaffleIron,故模块级导入与常量恒可测;
需真实第三方代码+权重+CUDA 的前向对齐测试,在缺失时自动跳过(CI 不带第三方代码)。
"""
import numpy as np
import pytest

pytest.importorskip('torch')             # opensource 适配器模块级依赖 torch;CI 无 torch 时跳过

from semseg import data as D
from semseg import opensource as O


def test_module_constants():
    assert O.WI_ROOT.name == 'WaffleIron'
    assert str(O.WI_CKPT).endswith('ckpt_last.pth')
    assert O.WI_CONFIG.suffix in ('.yaml', '.yml')


@pytest.mark.skipif(not O.WI_CKPT.exists() or not O.WI_ROOT.exists(),
                    reason='WaffleIron 第三方代码/权重未获取(见 .gitignore)')
def test_class_order_matches_ours():
    """WaffleIron 的 19 类顺序须与自研 semseg 一致,预测方可无需 remap 直接对照。"""
    import sys
    O._ensure_waffleiron_on_path()
    from datasets import SemanticKITTI                     # WaffleIron 本地包
    assert list(SemanticKITTI.CLASS_NAME) == list(D.CLASS_NAMES)
    sys.path[:] = [p for p in sys.path if p != str(O.WI_ROOT)]


@pytest.mark.skipif(not O.WI_CKPT.exists() or not O.WI_ROOT.exists(),
                    reason='WaffleIron 第三方代码/权重未获取(见 .gitignore)')
def test_predict_frame_aligns_with_raw_points():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('WaffleIron 推理需 CUDA')
    wi = O.WaffleIronSeg(dev='cuda')
    f = wi.frames()[0]
    pr = wi.predict_frame(f)
    scan = D.read_scan('08', f)
    assert pr.shape == (scan.shape[0],)                    # 逐点、原始点序、等长
    assert pr.min() >= 0 and pr.max() <= D.N_CLASSES - 1   # 训练类 id 0..18
