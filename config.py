# -*- coding: utf-8 -*-
"""
模板配置: 所有可调参数的唯一入口

实验时只改这个文件 (不要散落在各处改代码).
本文件里的每一项都对应 README 实验路线里的一个可研究变量:
    数据范围 (WaveConfig) / 网络规模 (ModelConfig) / 损失权重与训练 (TrainConfig)
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

# ----------------- 任务定义 -----------------
# 分类类别 (顺序即类别编号 0~4; 已训练 checkpoint 会依赖这个顺序, 别随意改)
CLASS_NAMES = ["sine", "square", "triangle", "saw", "noise"]

# 回归参数 (顺序即回归头输出的 5 个通道):
#   A      : 幅值
#   k      : 窗口内周期数 —— 与采样率解耦的频率表示 (Hz = k * fs / N)
#   phi    : 初相 (弧度, [0, 2pi))
#   offset : 直流偏置
#   duty   : 方波占空比 (只有方波有定义 —— 其他类别在损失里被"掩码"掉)
PARAM_NAMES = ["A", "k", "phi", "offset", "duty"]

# 每个参数的物理取值范围 (lo, hi): 随机采样用; 也是归一化 (去量纲) 的锚点
PARAM_RANGES = {
    "A":      (0.2, 1.0),
    "k":      (0.5, 4.0),
    "phi":    (0.0, 2.0 * math.pi),
    "offset": (-0.3, 0.3),
    "duty":   (0.2, 0.8),
}


@dataclass
class WaveConfig:
    """数据引擎参数"""
    n_points: int = 32                        # 窗口长度 (ADC 点数, 题目给定); 部署时是首层通道数, 保持 4 的倍数
    snr_db_range: Optional[Tuple[float, float]] = (20.0, 60.0)
    # 周期类信号附加的测量噪声 SNR (dB, 每个样本在该范围内随机); 设 None = 不加噪声 (M4 实验用)
    noise_sigma_range: Tuple[float, float] = (0.05, 0.4)
    # "noise/未知"类的白噪声标准差范围


@dataclass
class ModelConfig:
    """网络结构"""
    hidden: int = 128     # 共享骨干宽度 (64 太窄, 精度上不去; 保持 4 的倍数以兼容 NPU PWCONV)
    n_layers: int = 2     # 骨干隐藏层数


@dataclass
class TrainConfig:
    """训练与损失"""
    seed: int = 0
    n_train: int = 50000      # 固定训练集大小 (生成一次, 固定种子 => 可复现、实验间可比)
    n_val: int = 5000
    batch: int = 256
    epochs: int = 60
    lr: float = 1e-3
    cosine: bool = True       # 学习率余弦衰减 (精细回归需要小学习率收尾, 见 _diag 结论)
    eta_min: float = 1e-4     # 余弦衰减的终点学习率
    lambda_cls: float = 1.0   # 分类损失权重 (M3 实验: 与 lambda_reg 一起扫描)
    lambda_reg: float = 1.0   # 回归损失权重
    reg_weights: Tuple[float, float, float, float, float] = (1.0, 2.0, 2.0, 1.0, 1.0)
    # 各回归参数权重 (A, k, phi, offset, duty); sincos 模式时 phi 权重同时作用于 cos/sin 两项
    run_name: str = "m1_sincos_w" # 产物目录名 runs/<run_name>/


@dataclass
class Config:
    wave: WaveConfig = field(default_factory=WaveConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    phase_repr: str = "sincos"
    # 相位监督表示 (B 线实验开关):
    #   "raw"    = 直接回归 φ (简单, 但 φ≈0 与 φ≈2π 的样本波形几乎相同、目标相差 2π, 边界冲突)
    #   "sincos" = 回归 (cosφ, sinφ) 对偶 (消除边界冲突; 回归头 6 维自适应, 评估时 atan2 还原)


CFG = Config()
