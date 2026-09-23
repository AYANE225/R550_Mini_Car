"""GIF 导出/压缩工具的纯逻辑测试（合成数据，不依赖 KITTI/GPU）。

只测数据集无关的自研逻辑：plotstyle.optimize_gif 全帧共享调色板复存后仍是合法多帧 GIF；
plotstyle.save_gif 能把一个极小的 matplotlib 动画落成可回放的 GIF。
"""
import numpy as np
import pytest

from kitti_slam import plotstyle as ps


def _make_gif(path, n=6, size=(64, 48)):
    """合成一个 n 帧、每帧一个移动方块的 GIF（RGB→P）。"""
    Image = pytest.importorskip("PIL.Image")
    frames = []
    for k in range(n):
        arr = np.zeros((size[1], size[0], 3), np.uint8)
        arr[:, :, 2] = 20                                    # 深色底
        x = 4 + k * 6
        arr[16:32, x:x + 8] = (255, 120, 60)                 # 移动方块
        frames.append(Image.fromarray(arr).convert('P', palette=Image.ADAPTIVE))
    frames[0].save(path, save_all=True, append_images=frames[1:], loop=0, duration=80)
    return n


def test_optimize_gif_keeps_frames_and_valid(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    p = tmp_path / "a.gif"
    n = _make_gif(p)
    ps.optimize_gif(str(p), colors=32, disposal=1)
    im = Image.open(p)
    assert im.n_frames == n                                  # 帧数不变
    assert im.mode == 'P' and im.size == (64, 48)            # 仍是合法调色板 GIF


def test_save_gif_writes_playable_gif(tmp_path):
    pytest.importorskip("PIL.Image")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from PIL import Image

    fig, ax = plt.subplots(figsize=(2, 2)); fig.patch.set_facecolor(ps.BG)
    ln, = ax.plot([], []); ax.set_xlim(0, 5); ax.set_ylim(0, 5)
    ks = list(range(5))

    def update(k):
        ln.set_data(range(k + 1), range(k + 1))
        return (ln,)

    anim = FuncAnimation(fig, update, frames=ks, interval=100)
    out = tmp_path / "anim.gif"
    ps.save_gif(anim, str(out), fps=8, dpi=40, colors=16, disposal=1)
    plt.close(fig)
    im = Image.open(out)
    assert im.n_frames == len(ks)
    assert out.stat().st_size > 0
