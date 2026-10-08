# -*- coding: utf-8 -*-
"""
模板配置: 所有可调参数的唯一入口

实验时只改这个文件 (不要散落在各处改代码).
本文件里的每一项都对应 README 实验路线里的一个可研究变量:
    数据范围 (WaveConfig) / 网络规模 (ModelConfig) / 损失权重与训练 (TrainConfig)

【教学】读这份文件的三个层次 (自检用):
  1. 任务层: 让网络学什么? —— 5 类波形分类 + 5 个物理参数回归 (下方任务定义)
  2. 数据层: 训练样本长什么样? —— 32 点窗 + 每样本随机 SNR + 参数在给定范围随机
  3. 训练层: 如何把网络行为调向目标? —— 损失权重 / 学习率调度 / 容量 / 数据量
  改任何值之前先回答: "预期它影响哪个指标? 为什么?" (铁律: 一次只改一个变量)
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

# ----------------- 任务定义 -----------------
# 分类类别 (顺序即类别编号 0~4; 已训练 checkpoint 会依赖这个顺序, 别随意改)
# 【教学】类别顺序 = 标签编码 = softmax/混淆矩阵的下标; 换顺序 = 旧模型权重"张冠李戴"
CLASS_NAMES = ["sine", "square", "triangle", "saw", "noise"]

# 回归参数 (顺序即回归头输出的 5 个通道):
#   A      : 幅值
#   k      : 窗口内周期数 —— 与采样率解耦的频率表示 (Hz = k * fs / N)
#   phi    : 初相 (弧度, [0, 2pi))
#   offset : 直流偏置
#   duty   : 方波占空比 (只有方波有定义 —— 其他类别在损失里被"掩码"掉)
# 【教学】为什么频率用"窗内周期数 k"而不是 Hz?
#   k 无量纲、与采样率无关: 换 ADC 采样率只需按 Hz = k·fs/N 换算, 模型不用重训 —— 数据与硬件解耦的关键设计
# 【教学】φ 有两套表示 (raw 直接回归 / sincos 对偶回归, 见文件底部); 后者是本项目最大一次精度提升 (phi_circ −63%)
PARAM_NAMES = ["A", "k", "phi", "offset", "duty"]

# 每个参数的物理取值范围 (lo, hi): 随机采样用; 也是归一化 (去量纲) 的锚点
# 【教学】范围是"双重身份":
#   ① 数据生成时的均匀采样区间 —— 网络只会学到区间内的规律, 超范围 = 分布外, 结果不可信
#   ② 归一化公式 (x−lo)/(hi−lo) 的锚点 —— 改范围会同时改变数据分布和标签编码, 旧模型失效
PARAM_RANGES = {
    "A":      (0.2, 1.0),
    "k":      (0.5, 4.0),
    "phi":    (0.0, 2.0 * math.pi),
    "offset": (-0.3, 0.3),
    "duty":   (0.2, 0.8),
}


@dataclass
class WaveConfig:
    """数据引擎参数 (决定"模型看到的任务有多难")"""
    n_points: int = 32                        # 窗口长度 (ADC 点数, 题目给定); 部署时是首层通道数, 保持 4 的倍数
    # 【教学】窗口长度 = 信息量上限: 32 点里能装下多少"方波边沿"就决定方波参数误差的下限
    # (实测方波误差大头来自这个观测信息下限, 不是模型不够大 —— 见 docs/07 §2)
    snr_db_range: Optional[Tuple[float, float]] = (20.0, 60.0)
    # 周期类信号附加的测量噪声 SNR (dB, 每个样本在该范围内随机); 设 None = 不加噪声 (M4 实验用)
    # 【教学】每样本随机 SNR = 数据增广: 逼网络"在任意噪声水平下都尽量准", 而不是记住某个固定的噪声强度
    noise_sigma_range: Tuple[float, float] = (0.05, 0.4)
    # "noise/未知"类的白噪声标准差范围
    # 【教学】这一类的意义: 给模型一个"拒识出口"—— 不要求它把噪声拟合出参数, 而是学会识别"无法处理"


@dataclass
class ModelConfig:
    """网络结构 (容量旋钮)"""
    hidden: int = 128     # 共享骨干宽度 (64 太窄, 精度上不去; 保持 4 的倍数以兼容 NPU PWCONV)
    # 【教学】参数量手算: 32×h+h + h×h+h + (5h+5) + (6h+6) (过程见 model.py); h=128→22155, h=512→285195
    # 规模扫描实测 (docs/06 §3): 128→512 宽 + 预算/数据上调: k_rel −59%, 但边际递减明显
    n_layers: int = 2     # 骨干隐藏层数


@dataclass
class TrainConfig:
    """训练与损失 (优化旋钮集合)"""
    seed: int = 0             # 【教学】固定种子 = 可复现: 数据/初始化/批次顺序全由它派生, 这是实验间可比的前提
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
    # 【教学】这不是"喜欢谁给谁加"—— 梯度预算竞争 (docs/04 §2.3): sincos 让 φ 变成 2 个监督值,
    # 把 k 的损失份额从 1/4 挤到 1/5; 给 k/φ 权重×2 把份额调回 1/4, k 误差随之恢复 (0.199→0.159)
    lambda_recon_sq: float = 0.0   # 方波重建辅助损失权重 (0 = 关闭, 行为与历史完全一致)
    # 只作用于方波样本: 用"软方波"(可微 tanh 版, 见 wavegen.render_soft_square)把预测参数
    # 与真值参数各渲染一遍比 MSE —— 给 k/phi/duty 提供"波形级"梯度, 直击边沿敏感特性
    recon_sharp: float = 12.0      # 软方波边沿锐度 (tanh(sharp*sin(pi*u)); 越大越接近硬方波, 梯度越集中)
    square_weight: float = 1.0     # 方波样本的回归损失权重 (样本级重加权, 1.0 = 关闭)
    # 与 reg_weights 的区别: reg_weights 调"参数维度"间的梯度分配, 本开关调"类别"间的分配
    device: str = "cpu"            # 训练设备: "cpu" | "cuda" | "auto" (auto=有GPU用GPU)
    # 默认 cpu: 小模型 CPU 已够 (历史结果可比); 全量档(300ep)训练可切 "cuda" —— 实测 RTX 4060 ~3.5x
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
