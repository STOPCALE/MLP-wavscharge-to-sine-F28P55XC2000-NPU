# -*- coding: utf-8 -*-
"""纯 numpy 验证: 重建正弦 vs 参数重算 + 通道错位排查 (不依赖 matplotlib)"""
import numpy as np
import sys

CSV = r"E:\desk\CCS\npu_and_charge\_diag\diag_frames.csv"
F = np.loadtxt(CSV, delimiter=",")
print("shape:", F.shape)
cls = np.round(F[:, 43]).astype(int)

for c in range(4):
    idx = np.where(cls == c)[0]
    print("=" * 74)
    print("class %d  nframes %d" % (c, len(idx)))
    for pick in [idx[0], idx[len(idx) // 2], idx[-1]]:
        f = F[pick]
        A = f[37] * 0.8 + 0.2
        k = f[38] * 3.5 + 0.5
        co = f[39] * 2 - 1
        si = f[40] * 2 - 1
        ph = np.arctan2(si, co)
        ph = ph % (2 * np.pi)
        off = f[41] * 0.6 - 0.3
        t = np.arange(32) / 32.0
        calc = A * np.sin(2 * np.pi * k * t + ph) + off
        rec = f[46:78]
        print("frame#%-4d A=%.3f k=%.3f phi=%.3f off=%.3f  (duty'=%.3f)"
              % (pick, A, k, ph, off, f[42]))
        print("   rec  min/max/std = %7.3f %7.3f %7.3f" % (rec.min(), rec.max(), rec.std()))
        print("   calc min/max/std = %7.3f %7.3f %7.3f" % (calc.min(), calc.max(), calc.std()))
        print("   rec  [0:8]:", np.round(rec[:8], 3))
        print("   calc [0:8]:", np.round(calc[:8], 3))
        print("   maxdiff = %.4f" % np.abs(rec - calc).max())

print("=" * 74)
S = F[:, 46:78].std(axis=1)
W = F[:, 0:32].std(axis=1)
print("all frames: rec std  min=%.3f mean=%.3f max=%.3f" % (S.min(), S.mean(), S.max()))
print("all frames: win std  min=%.3f mean=%.3f max=%.3f" % (W.min(), W.mean(), W.max()))

# 相邻帧之间 rec 的相似度: 同一波形类别的连续帧, rec 应该一样吗?
# (板端每帧生成"同一窗参数"的波形按类别轮换 -> 同类帧间隔 4 帧, rec 应相同)
for c in range(4):
    idx = np.where(cls == c)[0]
    recs = F[idx][:, 46:78]
    d01 = np.abs(recs[0] - recs[1]).max() if len(recs) > 1 else -1
    print("class %d: |rec(frame_i) - rec(frame_i+1)| max = %.4f (同类相邻两帧)" % (c, d01))

# 检查 rec 是否等于"其他类别帧"的 rec (错位/延迟排查)
print("=" * 74)
i0 = np.where(cls == 0)[0][0]          # 第一帧 sine
rec0 = F[i0, 46:78]
diffs = np.abs(F[:, 46:78] - rec0).max(axis=1)
best = np.argsort(diffs)[:5]
print("rec(sine frame#%d) 最接近的前 5 帧: %s  差值: %s"
      % (i0, best, np.round(diffs[best], 4)))
