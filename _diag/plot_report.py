# -*- coding: utf-8 -*-
"""生成研究报告用图 (用 ti-npu env 的 matplotlib; torch env 的 mpl 在此机器上会崩)"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

F = np.loadtxt(r"E:\desk\CCS\npu_and_charge\_diag\diag_frames.csv", delimiter=",")
cls = np.round(F[:, 43]).astype(int)
OUT = r"E:\desk\CCS\npu_and_charge\_diag"
names = ["sine", "square", "triangle", "saw"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# ---------- 图1: 4 类单帧 "输入窗 vs 重建正弦" ----------
fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
for c in range(4):
    idx = np.where(cls == c)[0]
    f = F[idx[len(idx) // 2]]
    axes[0, c].plot(np.arange(32), f[0:32], ".-", color=colors[c])
    axes[0, c].set_title("class=%s  input CH0..31" % names[c])
    axes[0, c].grid(alpha=0.3)
    axes[1, c].plot(np.arange(32), f[46:78], ".-", color="tab:red")
    axes[1, c].set_title("recon sine CH46..77")
    axes[1, c].grid(alpha=0.3)
fig.suptitle(u"输入窗(上) vs 参数重建正弦(下) —— 修复后 (每类取 1 帧)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT + r"\fig_report_wave.png", dpi=110)
plt.close(fig)

# ---------- 图2: 流动时序 (全部帧, 色带=类别) ----------
n = F.shape[0]
x = np.arange(n)
fig, axs = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
for ax, ch, ttl in [(axs[0], 0, u"CH0 输入流 (长信号采样)"), (axs[1], 46, u"CH46 重建正弦流")]:
    ax.plot(x, F[:, ch], lw=0.8, color="C0" if ch == 0 else "C3")
    for c in range(4):
        m = (cls == c)
        if m.any():
            ax.axvspan(x[m][0], x[m][-1], alpha=0.07, color=colors[c])
            ax.text(x[m].mean(), ax.get_ylim()[0], names[c], ha="center", fontsize=9, color=colors[c])
    ax.set_ylabel(ttl)
    ax.grid(alpha=0.3)
axs[1].set_xlabel(u"帧号 (35 帧/s)")
fig.suptitle(u"滑窗流动模式: 每个通道在时间轴上呈现连续波形 (色带 = 4 类轮换, 每类 256 帧)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT + r"\fig_report_flow.png", dpi=110)
plt.close(fig)

# ---------- 图3: 性能 / 置信度 / 分类 ----------
fig, axs = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
axs[0].plot(F[:, 45], lw=0.8)
axs[0].set_ylabel(u"CH45 推理耗时 [us]")
axs[0].grid(alpha=0.3)
axs[0].axhline(F[:, 45].mean(), color="r", ls="--", lw=0.8,
               label=u"mean=%.1f us" % F[:, 45].mean())
axs[0].legend(loc="center right")
axs[1].plot(F[:, 44], lw=0.8, color="C2")
axs[1].set_ylabel(u"CH44 置信度")
axs[1].grid(alpha=0.3)
axs[2].plot(cls, ".", ms=2, color="C4")
axs[2].set_ylabel(u"CH43 分类")
axs[2].set_xlabel(u"帧号")
axs[2].grid(alpha=0.3)
fig.suptitle(u"性能 / 置信度 / 分类 (golden 自检偏差 = 0.00000)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT + r"\fig_report_perf.png", dpi=110)
plt.close(fig)

print("ALL FIGS DONE ->", OUT)
