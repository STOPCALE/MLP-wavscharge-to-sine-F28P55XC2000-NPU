# -*- coding: utf-8 -*-
"""
交互式推理演示 (PC 端可视化工具): 手动输入波形种类与参数 -> 模型推理 -> 对比可视化

用法 (conda 环境 torch):
    # 1) 交互模式 (弹窗; 输入种类/参数 -> 点"推理/刷新"):
    conda run --no-capture-output -n torch python visualize.py
    # 2) 静态出图 (命令行给参数, 保存 PNG 便于查看/分享):
    conda run --no-capture-output -n torch python visualize.py --cls square --A 0.8 --k 2.0 --phi 1.0 --offset 0.1 --duty 0.35 --snr 30 --save demo.png

界面显示 (三块对比, 体现"输入 vs 输出"):
    [输入] 观测波形 (128 点 = 4 个窗宽, 灰点=带噪观测, 蓝线=真值, 橙区=送入模型的 32 点窗)
    [推理窗] 32 点: 灰点=模型看到的东西 | 蓝线=真值 | 红虚线=模型参数重建
    [对比] 整段 128 点: 蓝线=真值参数, 红虚线=模型重建参数 (一眼看出波形吻合度)
    底部: 参数表 (真值 | 预测 | 误差) + 分类结果 / 置信度 / 窗内重建 RMSE

注意: 输入参数会被限制 (clamp) 到训练域:
    A∈[0.2,1]  k∈[0.5,4]  φ∈[0,2π)  offset∈[-0.3,0.3]  duty∈[0.2,0.8]
    超出训练域的输入模型没学过, 结果不可信 (界面会用红字提示)。
"""

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import CFG, CLASS_NAMES, PARAM_RANGES     # noqa: E402
from model import MultiTaskNet                        # noqa: E402
from wavegen import denormalize_params, render        # noqa: E402

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

RUN_NAME = "push6_h512e300_d150k"      # 当前最终模型; 换模型只改这一行 (hidden 同步)
HIDDEN = 512
N_WIN = 32          # 模型输入窗 (训练分布: 32 点, 窗内 k 个周期)
N_LONG = 128        # 展示波形 (4 个窗宽, 与窗口同采样率)
WIN_START = 32      # 推理窗在展示波形中的起点 (第 2 个窗)
_TWO_PI = 2.0 * math.pi
PARAM_ORDER = ["A", "k", "phi", "offset", "duty"]
WAVE_CHOICES = [c for c in CLASS_NAMES if c != "noise"]


# ----------------- 模型与渲染 -----------------
def load_model():
    ckpt = torch.load(os.path.join(ROOT, "runs", RUN_NAME, "best.pt"),
                      map_location="cpu", weights_only=False)
    model = MultiTaskNet(n_points=N_WIN, hidden=HIDDEN, n_layers=2, n_params=6)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt["epoch"]


def render_long(cls_id, params, n_points, n_win=N_WIN):
    """与 wavegen.render 完全同公式, 仅时间轴按"窗"延长 (t = n/n_win, 采样率不变)。

    n_points = n_win 时与 render() 数值等价 —— 见 assert_render_consistency() 自检。
    """
    p = params.view(1, 5)
    A, k, phi, off, duty = (p[:, i:i + 1] for i in range(5))
    t = torch.arange(n_points, dtype=torch.float32) / n_win
    theta = _TWO_PI * k * t.unsqueeze(0) + phi
    frac = torch.remainder(theta, _TWO_PI) / _TWO_PI
    if cls_id == 0:
        w = torch.sin(theta)
    elif cls_id == 1:
        w = torch.where(frac < duty, torch.ones_like(frac), -torch.ones_like(frac))
    elif cls_id == 2:
        w = (2.0 / math.pi) * torch.asin(torch.sin(theta))
    else:
        w = 2.0 * frac - 1.0
    return (A * w + off).squeeze(0)


def render_sine(params, n_points, n_win=N_WIN):
    """参数 -> 等效正弦合成 (忽略类别): x = A·sin(2πk·t + φ) + offset

    与 wavegen 的四类波形同一相位约定 (θ=2πk·t+φ, 与 sin 同相) ——
    因此对任意输入波形, 合成正弦都是其"等效基波"(同相)。
    """
    p = params.view(1, 5)
    A, k, phi, off = p[:, 0:1], p[:, 1:2], p[:, 2:3], p[:, 3:4]
    t = torch.arange(n_points, dtype=torch.float32) / n_win
    return (A * torch.sin(_TWO_PI * k * t.unsqueeze(0) + phi) + off).squeeze(0)


def assert_render_consistency():
    """自检: render_long 在单窗上与 wavegen.render 一致 (防止公式抄错)"""
    p = torch.tensor([0.7, 2.3, 1.1, 0.1, 0.4])
    for cid in range(4):
        a = render_long(cid, p, N_WIN)
        b = render(torch.tensor([cid]), p.view(1, 5), N_WIN)[0]
        d = (a - b).abs().max().item()
        assert d < 1e-6, f"render_long 与 render 不一致 (cls={cid}, diff={d})"


def clamp_params(vals):
    out, clamped = [], False
    for v, name in zip(vals, PARAM_ORDER):
        lo, hi = PARAM_RANGES[name]
        c = min(max(v, lo), hi)
        if abs(c - v) > 1e-9:
            clamped = True
        out.append(c)
    return out, clamped


def make_case(cls_name, params, snr_db, seed=0):
    """按参数渲染长波形 + 加噪 -> (cid, 真值波形, 观测, 推理窗, sigma)"""
    cid = CLASS_NAMES.index(cls_name)
    pt = torch.tensor(params, dtype=torch.float32)
    x_true = render_long(cid, pt, N_LONG)
    rms = x_true.pow(2).mean().sqrt()
    sigma = (rms / (10 ** (snr_db / 20.0))).item()
    g = torch.Generator().manual_seed(seed)
    x_obs = x_true + torch.randn(N_LONG, generator=g) * sigma
    x_win = x_obs[WIN_START:WIN_START + N_WIN]
    return cid, pt, x_true, x_obs, x_win, sigma


def infer(model, x_win):
    with torch.no_grad():
        logits, pred = model(x_win.view(1, -1))
        prob = F.softmax(logits, dim=1)[0]
    conf, pc = prob.max(dim=0)
    return int(pc.item()), float(conf.item()), denormalize_params(pred, CFG.phase_repr)[0]


# ----------------- 绘图 -----------------
def make_layout(fig, with_widgets=True):
    """5 行布局: 观测 / 推理窗 / 对比 / 正弦转换 / 参数表; (可选) 底部留给控件"""
    if with_widgets:
        gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 0.8],
                              top=0.955, bottom=0.30, hspace=0.55)
    else:
        gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 0.8],
                              top=0.955, bottom=0.045, hspace=0.45)
    return [fig.add_subplot(gs[i]) for i in range(5)]


def plot_case(axes, cls_name, params, snr_db, model, clamped=False):
    cid, pt, x_true, x_obs, x_win, sigma = make_case(cls_name, params, snr_db)
    pc, conf, pp = infer(model, x_win)
    x_pred_long = render_long(pc, pp, N_LONG)
    x_pred_win = render_long(pc, pp, N_WIN)
    x_sin_true = render_sine(pt, N_LONG)      # 真值参数 -> 正弦 (参照)
    x_sin_pred = render_sine(pp, N_LONG)      # 模型输出参数 -> 正弦 (要用的结果)
    k = params[1]

    def np_(t):
        return t.detach().numpy()

    # ---- [输入] 观测波形 (4 个窗宽, 高亮推理窗) ----
    ax = axes[0]
    ax.axvspan(WIN_START, WIN_START + N_WIN, color="orange", alpha=0.18, zorder=0)
    ax.plot(np_(x_obs), "o", ms=3.5, color="0.55", label="带噪观测 (模拟 ADC)")
    ax.plot(np_(x_true), "-", color="tab:blue", lw=1.6, label="真值波形 (无噪)")
    ax.text(WIN_START + N_WIN / 2, 0.94, "↓ 送入模型", ha="center", fontsize=8.5,
            color="darkorange", transform=ax.get_xaxis_transform())
    ax.set_title(f"[输入] 观测波形: 128 点 = 4 个窗宽 ≈ {4 * k:.1f} 个周期 | "
                 f"SNR {snr_db:g} dB (σ={sigma:.3f})", fontsize=10)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(alpha=0.3)

    # ---- [推理窗] 观测 vs 真值 vs 模型重建 ----
    ax = axes[1]
    ax.plot(np_(x_win), "o", ms=6, color="0.55", label="模型输入 (32 点观测)")
    ax.plot(np_(x_true[WIN_START:WIN_START + N_WIN]), "-", color="tab:blue", lw=2, label="真值")
    ax.plot(np_(x_pred_win), "--", color="tab:red", lw=2, label="模型重建")
    ax.set_title("[推理窗] 模型看到什么 vs 网络认为是什么 (32 点)", fontsize=10)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(alpha=0.3)

    # ---- [对比] 整段: 真值 vs 重建 ----
    ax = axes[2]
    ax.plot(np_(x_true), "-", color="tab:blue", lw=2, label=f"真值参数 ({cls_name})")
    ax.plot(np_(x_pred_long), "--", color="tab:red", lw=2,
            label=f"模型重建参数 ({CLASS_NAMES[pc]})")
    ax.axvspan(WIN_START, WIN_START + N_WIN, color="orange", alpha=0.12, zorder=0)
    ax.set_title("[对比] 输出波形 vs 输入真值波形 (整段 128 点)", fontsize=10)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)

    # ---- [正弦转换] 用识别出的参数合成等效正弦 (与输入同相) ----
    ax = axes[3]
    ax.plot(np_(x_obs), "o", ms=3, color="0.78", label="输入观测 (带噪)")
    ax.plot(np_(x_sin_true), "-", color="tab:blue", lw=1.8, label="真值参数 -> 正弦")
    ax.plot(np_(x_sin_pred), "--", color="tab:red", lw=2.0, label="模型输出参数 -> 正弦")
    ax.axvspan(WIN_START, WIN_START + N_WIN, color="orange", alpha=0.12, zorder=0)
    ax.set_title("[正弦转换] 参数 -> 正弦合成: A·sin(2πk·t + φ) + offset (128 点)", fontsize=10)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(alpha=0.3)

    # ---- 参数表 ----
    ax = axes[4]
    ax.axis("off")
    k_rel = abs(pp[1].item() - pt[1].item()) / pt[1].item()
    dphi = abs(((pp[2].item() - pt[2].item() + math.pi) % _TWO_PI) - math.pi)
    recon = (x_pred_win - x_win).pow(2).mean().sqrt().item()
    ok = pc == cid

    hdr = f"分类:  真值 {cls_name:<8s} | 预测 {CLASS_NAMES[pc]:<8s} | 置信度 {conf:.3f}"
    if not ok:
        hdr += "   ✗ 分类错误"
    zh = {"A": "幅值", "k": "周期数", "phi": "相位", "offset": "偏置", "duty": "占空比"}
    rows = [hdr, ""]
    for i, name in enumerate(PARAM_ORDER):
        if name == "duty" and cls_name != "square":
            rows.append(f"{name:<7s}({zh[name]})  真值 {pt[i].item():8.4f}   预测 —         误差 —  (仅方波适用)")
            continue
        ev = f"{abs(pp[i].item() - pt[i].item()):.4f}"
        if name == "k":
            ev = f"{k_rel * 100:.2f}% (相对)"
        elif name == "phi":
            ev = f"{dphi:.4f} rad ({math.degrees(dphi):.1f}°)"
        rows.append(f"{name:<7s}({zh[name]})  真值 {pt[i].item():8.4f}   预测 {pp[i].item():8.4f}   误差 {ev}")
    rows.append("")
    rows.append(f"窗内重建 RMSE: {recon:.4f}    等效频率: f = k × fs/32 (fs=ADC 采样率, 当前 k={pp[1].item():.4f})")
    color = "tab:red" if not ok else "black"
    ax.text(0.0, 1.0, "\n".join(rows), va="top", ha="left", fontsize=10.5,
            color=color, transform=ax.transAxes)
    if clamped:
        ax.text(0.72, 0.55, "⚠ 输入超出训练域\n   已截断: A[0.2,1] k[0.5,4]\n   φ[0,2π] offset[±0.3] duty[0.2,0.8]",
                va="top", ha="left", fontsize=9, color="tab:red", transform=ax.transAxes)

    fig = axes[0].figure
    fig.suptitle("波形参数回归: 输入 -> 模型推理 -> 输出对比", fontsize=12.5, y=0.99)


# ----------------- 交互模式 -----------------
def build_interactive(model):
    fig = plt.figure(figsize=(9.5, 14.2))
    axes = make_layout(fig, with_widgets=True)
    state = dict(cls=WAVE_CHOICES[0])

    ax_radio = fig.add_axes([0.03, 0.02, 0.115, 0.245])
    ax_radio.set_title("波形种类", fontsize=9)
    radio = RadioButtons(ax_radio, WAVE_CHOICES, active=0)

    def on_cls(label):
        state["cls"] = label
    radio.on_clicked(on_cls)

    defaults = dict(A="0.80", k="2.0", phi="1.00", offset="0.00", duty="0.35", snr="30")
    boxes = {}
    for i, (key, dv) in enumerate(defaults.items()):
        axb = fig.add_axes([0.60 + 0.20 * (i // 3), 0.20 - 0.055 * (i % 3), 0.085, 0.038])
        boxes[key] = TextBox(axb, key + " ", initial=dv)
        boxes[key].label.set_fontsize(9)
    fig.text(0.60, 0.275, "参数输入 (回车确认; 越界自动截断)", fontsize=9)
    fig.text(0.60, 0.245, "snr = 观测信噪比 (dB)", fontsize=8.5, color="0.4")

    ax_btn = fig.add_axes([0.19, 0.055, 0.15, 0.055])
    btn = Button(ax_btn, "推理 / 刷新", color="0.85", hovercolor="0.75")
    ax_rnd = fig.add_axes([0.37, 0.055, 0.15, 0.055])
    btn_rnd = Button(ax_rnd, "随机参数", color="0.85", hovercolor="0.75")

    def read_inputs():
        vals = []
        for key in PARAM_ORDER:
            try:
                vals.append(float(boxes[key].text))
            except ValueError:
                vals.append(PARAM_RANGES[key][0])
        try:
            snr = float(boxes["snr"].text)
        except ValueError:
            snr = 30.0
        vals, clamped = clamp_params(vals)
        return state["cls"], vals, snr, clamped

    def refresh(event=None):
        cls_name, vals, snr, clamped = read_inputs()
        for ax in axes:
            ax.clear()
        plot_case(axes, cls_name, vals, snr, model, clamped)
        fig.canvas.draw_idle()

    def on_random(event):
        g = torch.Generator().manual_seed(int(torch.randint(0, 10 ** 6, (1,)).item()))
        for key in PARAM_ORDER:
            lo, hi = PARAM_RANGES[key]
            v = torch.rand(1, generator=g).item() * (hi - lo) + lo
            boxes[key].set_val(f"{v:.3f}")
        refresh()

    btn.on_clicked(refresh)
    btn_rnd.on_clicked(on_random)
    refresh()
    return fig


# ----------------- 入口 -----------------
def main():
    ap = argparse.ArgumentParser(description="波形参数回归交互演示")
    ap.add_argument("--cls", default=None, choices=WAVE_CHOICES, help="波形种类 (不给则开交互窗口)")
    ap.add_argument("--A", type=float, default=0.8)
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--phi", type=float, default=1.0)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--duty", type=float, default=0.35)
    ap.add_argument("--snr", type=float, default=30.0)
    ap.add_argument("--save", default=None, help="保存 PNG 路径 (命令行模式)")
    ap.add_argument("--smoke", action="store_true", help="自检: Agg 下建交互布局并出图")
    args = ap.parse_args()

    model, epoch = load_model()
    assert_render_consistency()
    print(f"模型: runs/{RUN_NAME} (best epoch {epoch}, hidden={HIDDEN})")

    if args.smoke:
        fig = build_interactive(model)
        out = os.path.join(ROOT, "_viz_smoke.png")
        fig.savefig(out, dpi=110)
        print(f"SMOKE-OK (交互布局+推理已跑通) -> {out}")
        return

    if args.cls is None:
        fig = build_interactive(model)
        print("交互模式: 输入参数 -> 点 [推理/刷新]; 关闭窗口退出。")
        plt.show()
        return

    vals, clamped = clamp_params([args.A, args.k, args.phi, args.offset, args.duty])
    fig = plt.figure(figsize=(9, 13))
    axes = make_layout(fig, with_widgets=False)
    plot_case(axes, args.cls, vals, args.snr, model, clamped)
    if args.save:
        fig.savefig(args.save, dpi=130)
        print(f"已保存: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
