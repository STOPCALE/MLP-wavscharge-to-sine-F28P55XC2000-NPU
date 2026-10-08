# -*- coding: utf-8 -*-
"""
波形工厂: 参数 <-> 窗口波形 (数据引擎 + 重建器)

同一套公式同时用于两处, 这是刻意设计 (单一事实来源):
  1) 生成训练数据: 随机采样 (类别, 参数) -> 渲染波形
  2) 训练后的重建验证: 模型预测的 (类别, 参数) -> 用同一渲染器重建波形
若两处公式不一致, "训练看到的波形" 和 "评估重建的波形" 就不是一回事, 实验会失真.

【教学】本文件是全项目的"物理定律": 参数 -> 波形的唯一定义。三个必懂点:
  a. θ = 2πk·t + φ 的双重角色: t ∈ [0,1) 是"窗内归一化时间", k 是窗内周期数 ——
     四类波形共用同一相位轴 (含 φ 与 sin 同相约定), 否则同一 φ 在不同类里"一词多义"
  b. 评估链必须可微或可替代: normalize/反演贯穿训练+评估; 硬方波不可微时用 render_soft_square 替代
  c. 掩码(mask) 与渲染无关, 是"监督有效性"层: duty 只对方波有意义, 噪声类无回归目标

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


def _bounds(device=None):
    lo = torch.tensor([PARAM_RANGES[n][0] for n in PARAM_NAMES], device=device)
    hi = torch.tensor([PARAM_RANGES[n][1] for n in PARAM_NAMES], device=device)
    return lo, hi


def normalize_params(raw, phase_repr="raw"):
    """物理参数 [n,5] -> 监督目标 (必须去量纲, 否则各参数梯度尺度失衡):

    phase_repr="raw"    : 返回 [n,5]  (A, k, φ, offset, duty) 映射到 [0,1]
    phase_repr="sincos" : 返回 [n,6]  (A, k, cosφ, sinφ, offset, duty)
                          cos/sin 由 [-1,1] 线性映射到 [0,1]

    【教学】两条原则:
      * 一切进损失的目标必须同量级 ([0,1]): 否则 φ 的 6.28 会压倒 offset 的 0.6,
        梯度尺度失衡 -> 小量级参数"学不动" (回归任务大忌, 术语: normalization 不是 quantization)
      * sincos 两项从 [-1,1] 再线性映到 [0,1]: 保持同标度; 评估时用 atan2 反推 φ,
        天然处理 0/2π 循环 (不存在"边界跳变"目标)
    """
    lo, hi = _bounds(raw.device)
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
    """监督目标 -> 物理参数 [n,5]. 评估/重建时转回真实单位, 指标才有物理意义

    【教学】与 normalize_params 是严格的互逆变换 —— 改任何一边必须对拍验证
    (往返测试: denorm(norm(x)) ≈ x); 否则训练目标与评估口径悄悄错位, 指标全失真
    """
    lo, hi = _bounds(norm.device)
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

    t = torch.arange(n_points, dtype=torch.float32, device=params_raw.device) / n_points     # [N] ∈ [0,1)
    theta = _TWO_PI * k * t.unsqueeze(0) + phi                     # [n,N]
    frac = torch.remainder(theta, _TWO_PI) / _TWO_PI               # [n,N] ∈ [0,1)

    # 【教学】θ 是四类波形的公共时间轴: sine 直接用它; triangle 用 asin(sinθ) (与 sin 同相);
    # square/saw 用 frac (每周期 0→1 的锯齿坐标: 方波砍成 +1/−1, 锯齿线性拉成 ±1)
    # 设计意图: 同一 (k, φ) 在四类里的"相位含义"完全一致 —— 分类换类不改相位含义, 回归目标不分叉
    sine     = torch.sin(theta)
    square   = torch.where(frac < duty, torch.ones_like(frac), -torch.ones_like(frac))
    triangle = (2.0 / math.pi) * torch.asin(torch.sin(theta))
    saw      = 2.0 * frac - 1.0

    x = torch.zeros(n, n_points, device=params_raw.device)
    for cid, waveshape in ((0, sine), (1, square), (2, triangle), (3, saw)):
        m = cls == cid
        if m.any():
            x[m] = A[m] * waveshape[m] + off[m]
    return x


def render_soft_square(params_raw, n_points, sharp=12.0):
    """可微"软方波" (训练损失专用; 与 render 的 square 定义同形, 仅把边沿平滑化).

        u = frac/duty                    (frac <  duty)
        u = 1 + (frac-duty)/(1-duty)     (frac >= duty)
        x = A * tanh(sharp * sin(pi*u)) + off

    几何意义: sin(pi*u) 的符号 == [frac < duty] (硬方波的取值), 幅度平滑过渡;
    sharp -> 无穷 时 tanh 饱和 => 等价于硬方波.
    为什么要软化: 硬方波用 where/sign, 对 duty/phi/k 的梯度处处为 0 (分段常数),
    重建损失无法训练; tanh 版除边沿窄带外处处可微, 梯度恰好集中在边沿附近
    (= "边沿感知", 正对上方波最敏感的跳变位置). 评估仍用硬版 render, 不受影响.
    """
    A    = params_raw[:, 0:1]
    k    = params_raw[:, 1:2]
    phi  = params_raw[:, 2:3]
    off  = params_raw[:, 3:4]
    duty = params_raw[:, 4:5].clamp(1e-3, 1.0 - 1e-3)

    t = torch.arange(n_points, dtype=torch.float32, device=params_raw.device) / n_points
    theta = _TWO_PI * k * t.unsqueeze(0) + phi
    frac = torch.remainder(theta, _TWO_PI) / _TWO_PI               # [n,N] ∈ [0,1)

    u = torch.where(frac < duty,
                    frac / duty,
                    1.0 + (frac - duty) / (1.0 - duty))
    sq = torch.tanh(sharp * torch.sin(math.pi * u))
    return A * sq + off


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
    #    【教学】SNR(dB) 定义: 20·log10(rms / σ)  ==>  σ = rms / 10^(SNR/20)
    #    rms 用该样本波形自身计算(逐样本): 幅度小的样本噪声也自动小, 相对难度保持一致
    periodic = ~noise_rows
    if cfg.wave.snr_db_range is not None and periodic.any():
        lo, hi = cfg.wave.snr_db_range
        snr = torch.rand(n, generator=g) * (hi - lo) + lo
        rms = x[periodic].pow(2).mean(dim=1).sqrt().clamp(min=1e-6)
        sigma_m = rms / torch.pow(torch.tensor(10.0), snr[periodic] / 20.0)
        x[periodic] = x[periodic] + torch.randn(int(periodic.sum()), N, generator=g) * sigma_m.unsqueeze(1)

    # 5) 掩码: 噪声类无回归标签; duty 只对方波有定义
    #    (列布局随相位表示不同: raw=[A,k,φ,off,duty] / sincos=[A,k,cos,sin,off,duty])
    #    【教学】无定义的目标不进损失: 不掩码的话模型会被逼拟合"随机垃圾值", 梯度被污染;
    #    掩码让每个样本只对"有意义的监督项"负责 (损失分母也同步用 mask 加权, 见 train.py)
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
