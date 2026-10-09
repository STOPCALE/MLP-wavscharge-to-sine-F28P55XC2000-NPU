# -*- coding: utf-8 -*-
"""滑窗流动模式验证:
  1) 重建正弦 vs 参数直算 (修复验证, 应 maxdiff ~ 0)
  2) CH0 输入流 / CH46 重建流 的波形形状 (人读数值序列)
  3) 类别段/切换点 + conf 低谷帧位置
  4) 各类均值形状特征
"""
import numpy as np

CSV = r"E:\desk\CCS\npu_and_charge\_diag\diag_frames.csv"
F = np.loadtxt(CSV, delimiter=",")
print("shape:", F.shape)
cls = np.round(F[:, 43]).astype(int)

# ---------------- 1) rec vs calc ----------------
print("== 1) 重建正弦 vs 参数直算 (修复后 maxdiff 应 ~0) ==")
for c in range(4):
    idx = np.where(cls == c)[0]
    for pick in [idx[10], idx[-30]]:
        f = F[pick]
        A = f[37] * 0.8 + 0.2
        k = f[38] * 3.5 + 0.5
        co = f[39] * 2 - 1
        si = f[40] * 2 - 1
        ph = np.arctan2(si, co) % (2 * np.pi)
        off = f[41] * 0.6 - 0.3
        t = np.arange(32) / 32.0
        calc = A * np.sin(2 * np.pi * k * t + ph) + off
        rec = f[46:78]
        print("  class%d frame#%-4d maxdiff=%.4f | rec range [%.2f, %.2f]"
              % (c, pick, np.abs(rec - calc).max(), rec.min(), rec.max()))

# ---------------- 2) 流动波形形状 ----------------
print("== 2) 流动波形 (CH0 输入流 24 个采样值 / CH46 重建流 24 个采样值) ==")
for c in range(4):
    idx = np.where(cls == c)[0]
    seg = idx[20:20 + 24]
    x = F[seg, 0]
    y = F[seg, 46]
    print("  class%d CH0 : %s" % (c, np.array2string(x, precision=2, suppress_small=True)))
    print("        CH46: %s" % (np.array2string(y, precision=2, suppress_small=True)))

# ---------------- 3) 类别段与 conf ----------------
print("== 3) 类别段 ==")
chg = np.where(np.abs(np.diff(cls)) > 0.5)[0]
print("  切换点(帧号):", chg[:10], " 共", len(chg), "次")
low = np.where(F[:, 44] < 0.9)[0]
print("  conf<0.9 帧数:", len(low))
for i in low[:8]:
    print("    frame#%d: cls=%d conf=%.3f us=%.1f  (前后各一帧 cls: %s)"
          % (i, cls[i], F[i, 44], F[i, 45], cls[max(0, i-1):i+2].tolist()))

# ---------------- 4) 各类均值形状 ----------------
print("== 4) 各类窗(32帧均值) 与重建(均值) 特征 ==")
for c in range(4):
    idx = np.where(cls == c)[0]
    winavg = F[idx][:, 0:32].mean(axis=0)
    recavg = F[idx][:, 46:78].mean(axis=0)
    print("  class%d winavg std=%.3f range[%.2f,%.2f] | recavg std=%.3f range[%.2f,%.2f]"
          % (c, winavg.std(), winavg.min(), winavg.max(),
             recavg.std(), recavg.min(), recavg.max()))

# ---------------- 5) 参数随帧变化 (滑窗中应缓变) ----------------
print("== 5) 参数随帧缓变检查 (每类前 40 帧的 A',k',cos',sin') ==")
for c in range(4):
    idx = np.where(cls == c)[0][:40]
    print("  class%d A' [%.3f..%.3f]  k' [%.3f..%.3f]  cos' [%.3f..%.3f]  sin' [%.3f..%.3f]"
          % (c, F[idx, 37].min(), F[idx, 37].max(), F[idx, 38].min(), F[idx, 38].max(),
             F[idx, 39].min(), F[idx, 39].max(), F[idx, 40].min(), F[idx, 40].max()))
