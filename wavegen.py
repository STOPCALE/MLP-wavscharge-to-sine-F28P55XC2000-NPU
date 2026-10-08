# -*- coding: utf-8 -*-
"""
波形工厂: 参数 <-> 窗口波形 (数据引擎 + 重建器)

同一套公式同时用于两处, 这是刻意设计 (单一事实来源):
  1) 生成训练数据: 随机采样 (类别, 参数) -> 渲染波形
  2) 训练后的重建验证: 模型预测的 (类别, 参数) -> 用同一渲染器重建波形
若两处公式不一致, "训练看到的波形" 和 "评估重建的波形" 就不是一回事, 实验会失真.

波形定义 (t = n/N ∈ [0,1) 为窗口内归一化时间, θ = 2πk·t + φ):
    sine     : x = A·sin(θ) + offset
    square   : x = A·(相位位于周期前 duty 比例内 ? +1 : -1) + offset
    triangle : x = A·(2/π)·asin(sin θ) + offset      # 与 sin 同相的三角波
    saw      : x = A·(2·frac(θ/2π) - 1) + offset      # 从 -1 升到 +1 的锯齿
    noise    : 纯白噪声 N(0, σ²), σ 随机 —— 无法由参数渲染, 回归标签被掩掉

相位监督表示可切换 (CFG.phase_repr):
    raw    : 直接回归 φ ∈ [0,2π)
    sincos : 回归 (cosφ, sinφ) 对偶 —— 消除 0/2π 边界"波形几乎相同、目标相差 2π"的冲突
"""

import math

import torch

from config import CFG, CLASS_NAMES, PARAM_NAMES, PARAM_RANGES

_TWO_PI = 2.0 * math.pi
NOISE_CLASS = CLASS_NAMES.index("noise")      # = 4


def _bounds():
    lo = torch.tensor([PARAM_RANGES[n][0] for n in PARAM_NAMES])
    hi = torch.tensor([PARAM_RANGES[n][1] for n in PARAM_NAMES])
    return lo, hi


def normalize_params(raw, phase_repr="raw"):
    """物理参数 [n,5] -> 监督目标 (必须去量纲, 否则各参数梯度尺度失衡):

    phase_repr="raw"    : 返回 [n,5]  (A, k, φ, offset, duty) 映射到 [0,1]
    phase_repr="sincos" : 返回 [n,6]  (A, k, cosφ, sinφ, offset, duty)
                          cos/sin 由 [-1,1] 线性映射到 [0,1]
    """
    lo, hi = _bounds()
    A = (raw[:, 0:1] - lo[0]) / (hi[0] - lo[0])
    k = (raw[:, 1:2] - lo[1]) / (hi[1] - lo[1])
    o = (raw[:, 3:4] - lo[3]) / (hi[3] - lo[3])
    d = (raw[:, 4:5] - lo[4]) / (hi[4] - lo[4])
    if phase_repr == "raw":
        phi = (raw[:, 2:3] - lo[2]) / (hi[2] - lo[2])
        return torch.cat([A, k, phi, o, d], dim=1)
    c = (torch.cos(raw[:, 2:3]) + 1.0) / 2.0
    s = (torch.sin(raw[:, 2:3]) + 1.0) / 2.0
    return torch.cat([A, k, c, s, o, d], dim=1)


def denormalize_params(norm, phase_repr="raw"):
    """监督目标 -> 物理参数 [n,5]. 评估/重建时转回真实单位, 指标才有物理意义"""
    lo, hi = _bounds()
    A = norm[:, 0:1] * (hi[0] - lo[0]) + lo[0]
    k = norm[:, 1:2] * (hi[1] - lo[1]) + lo[1]
    if phase_repr == "raw":
        phi = norm[:, 2:3] * (hi[2] - lo[2]) + lo[2]
        o = norm[:, 3:4] * (hi[3] - lo[3]) + lo[3]
        d = norm[:, 4:5] * (hi[4] - lo[4]) + lo[4]
    else:
        c = norm[:, 2:3] * 2.0 - 1.0
        s = norm[:, 3:4] * 2.0 - 1.0
        phi = torch.remainder(torch.atan2(s, c), _TWO_PI)
        o = norm[:, 4:5] * (hi[3] - lo[3]) + lo[3]
        d = norm[:, 5:6] * (hi[4] - lo[4]) + lo[4]
    return torch.cat([A, k, phi, o, d], dim=1)


def render(cls, params_raw, n_points):
    """按类别公式渲染波形.

    cls: [n] 类别编号; params_raw: [n,5] 物理参数; 返回 x: [n, n_points]
    注意: 噪声类 (cls=4) 的行返回全 0 (噪声不可由参数渲染), 由调用方自行处理.
    """
    n = cls.shape[0]
    A    = params_raw[:, 0:1]
    k    = params_raw[:, 1:2]
    phi  = params_raw[:, 2:3]
    off  = params_raw[:, 3:4]
    duty = params_raw[:, 4:5]

    t = torch.arange(n_points, dtype=torch.float32) / n_points     # [N] ∈ [0,1)
    theta = _TWO_PI * k * t.unsqueeze(0) + phi                     # [n,N]
    frac = torch.remainder(theta, _TWO_PI) / _TWO_PI               # [n,N] ∈ [0,1)

    sine     = torch.sin(theta)
    square   = torch.where(frac < duty, torch.ones_like(frac), -torch.ones_like(frac))
    triangle = (2.0 / math.pi) * torch.asin(torch.sin(theta))
    saw      = 2.0 * frac - 1.0

    x = torch.zeros(n, n_points)
    for cid, waveshape in ((0, sine), (1, square), (2, triangle), (3, saw)):
        m = cls == cid
        if m.any():
            x[m] = A[m] * waveshape[m] + off[m]
    return x


def sample_dataset(n, cfg=CFG, generator=None):
    """生成 n 个样本的固定数据集 (传入带种子的 generator 保证可复现)

    返回 dict:
      x          [n,N]   窗口波形 (周期类含测量噪声)
      cls        [n]     类别 (0~4)
      params_raw [n,5]   物理参数 (噪声类的值是随机的, 但被掩码, 不参与监督)
      mask       [n,5或6] 1 = 该样本该目标有定义/参与损失; 0 = 不参与 (列数随 phase_repr)
    """
    N = cfg.wave.n_points
    g = generator

    # 1) 类别与参数随机采样
    cls = torch.randint(0, len(CLASS_NAMES), (n,), generator=g)
    raw = torch.empty(n, len(PARAM_NAMES))
    for j, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_RANGES[name]
        raw[:, j] = torch.rand(n, generator=g) * (hi - lo) + lo

    # 2) 渲染周期波形
    x = render(cls, raw, N)

    # 3) "noise/未知"类: 白噪声
    noise_rows = cls == NOISE_CLASS
    if noise_rows.any():
        s_lo, s_hi = cfg.wave.noise_sigma_range
        sigma = torch.rand(n, generator=g) * (s_hi - s_lo) + s_lo
        x[noise_rows] = torch.randn(int(noise_rows.sum()), N, generator=g) * sigma[noise_rows].unsqueeze(1)

    # 4) 周期类附加测量噪声: 按每样本随机 SNR 计算标准差 (模拟真实 ADC 噪声)
    periodic = ~noise_rows
    if cfg.wave.snr_db_range is not None and periodic.any():
        lo, hi = cfg.wave.snr_db_range
        snr = torch.rand(n, generator=g) * (hi - lo) + lo
        rms = x[periodic].pow(2).mean(dim=1).sqrt().clamp(min=1e-6)
        sigma_m = rms / torch.pow(torch.tensor(10.0), snr[periodic] / 20.0)
        x[periodic] = x[periodic] + torch.randn(int(periodic.sum()), N, generator=g) * sigma_m.unsqueeze(1)

    # 5) 掩码: 噪声类无回归标签; duty 只对方波有定义
    #    (列布局随相位表示不同: raw=[A,k,φ,off,duty] / sincos=[A,k,cos,sin,off,duty])
    if cfg.phase_repr == "raw":
        mask = torch.ones(n, 5)
        mask[noise_rows] = 0.0
        mask[cls != CLASS_NAMES.index("square"), 4] = 0.0
    else:
        mask = torch.ones(n, 6)
        mask[noise_rows] = 0.0
        mask[cls != CLASS_NAMES.index("square"), 5] = 0.0

    return {"x": x, "cls": cls, "params_raw": raw, "mask": mask}


if __name__ == "__main__":
    # 自检: 生成 4 个样本, 看形状与数值范围 (改了公式先用这个快速验证)
    d = sample_dataset(4, generator=torch.Generator().manual_seed(0))
    for key in d:
        print(key, tuple(d[key].shape))
    print("x range:", d["x"].min().item(), "~", d["x"].max().item())
