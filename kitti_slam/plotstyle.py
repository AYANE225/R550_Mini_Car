"""统一的深色绘图风格：让 README 画廊里所有图与首图(金字塔/nuScenes)一致。

用法：
    from kitti_slam.plotstyle import BG, GT, EST, style_ax, style_legend, savefig
    fig, ax = plt.subplots(...); fig.patch.set_facecolor(BG)
    style_ax(ax); ...; style_legend(ax.legend()); savefig(fig, path)
"""
from __future__ import annotations

# —— 调色板（对齐金字塔/nuScenes 首图）——
BG = '#0b0b12'      # 画布/坐标背景
PANEL = '#05050a'   # 点云面板（更深）
FG = '#e8e8f0'      # 标题/主文字
MUTED = '#8899aa'   # 刻度/轴标
GRID = '#334155'    # 网格/边框
GT = '#5ad1ff'      # 真值/参考（青）
EST = '#ff7043'     # 估计（橙）
ACCENT = '#c9a0ff'  # 次要强调（紫）
START = '#7CFC00'   # 起点标记（绿）
OCC = 'bone'        # 占据栅格 colormap（深色友好）


def style_ax(ax, grid=True, spines=True):
    """把单个 Axes 调成深色主题。"""
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelcolor=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(FG)
    for s in ax.spines.values():
        s.set_visible(spines)
        s.set_color(GRID)
    if grid:
        ax.grid(alpha=0.18, color=GRID, lw=0.7)
    return ax


def style_legend(leg):
    """图例调成深色主题。"""
    if leg is None:
        return leg
    fr = leg.get_frame()
    fr.set_facecolor('#16324f')
    fr.set_edgecolor('#39d0e0')
    fr.set_alpha(0.9)
    for t in leg.get_texts():
        t.set_color('w')
    return leg


def savefig(fig, path, dpi=130):
    """存图时带上深色背景（否则边框会是白的）。"""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches='tight')
    return path


def optimize_gif(path, colors=112, disposal=2):
    """PIL 复存：全帧共用一套调色板 + optimize 帧间裁剪，显著压小 README 里的 GIF 体积。
    调色板取末帧（信息最全的一帧）以免 turbo 等渐变被截断；dither 关掉让静态背景逐帧一致。
    disposal=1（只增长、无移动 artist 的动画，如轨迹铺开）可让 optimize 只编码变化区域，
    体积骤降；disposal=2（有移动小车/游标需擦除）则逐帧全画。"""
    from PIL import Image
    im = Image.open(path)
    rgb = []
    try:
        while True:
            rgb.append(im.convert('RGB')); im.seek(im.tell() + 1)
    except EOFError:
        pass
    dur = im.info.get('duration', 70)
    pal = rgb[-1].convert('P', palette=Image.ADAPTIVE, colors=colors)
    frames = [f.quantize(palette=pal, dither=Image.NONE) for f in rgb]
    frames[0].save(path, save_all=True, append_images=frames[1:], loop=0,
                   duration=dur, optimize=True, disposal=disposal)
    return path


def save_gif(anim, path, fps=14, dpi=54, colors=112, disposal=2):
    """存 matplotlib 动画为深色 GIF 并压缩体积。anim=FuncAnimation。返回 path。"""
    from pathlib import Path
    from matplotlib.animation import PillowWriter
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi, savefig_kwargs={'facecolor': BG})
    optimize_gif(path, colors=colors, disposal=disposal)
    return path
