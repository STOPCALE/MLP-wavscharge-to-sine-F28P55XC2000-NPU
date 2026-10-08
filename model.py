# -*- coding: utf-8 -*-
"""
多任务网络: 共享骨干 + 分类头 + 回归头 (hard parameter sharing)

结构 (实际尺寸由 config 驱动, 下面以默认值示意):
    输入 [B, 32]
      -> 共享骨干: Linear(32->128) + ReLU + Linear(128->128) + ReLU   # 特征提取
      -> 分类头:   Linear(128->5)  -> logits      (softmax 交给交叉熵损失)
      -> 回归头:   Linear(128->5或6) -> [0,1] 监督目标
                   raw    : (A, k, φ, offset, duty)
                   sincos : (A, k, cosφ, sinφ, offset, duty)   # B 线实验开关, 见 config

为什么共享骨干:
    分类要的"周期数/幅值/形状"特征, 回归也全都要用 —— 共享让两个任务的梯度
    共同塑造同一组特征, 等效于互相正则化. 副作用是可能互相拖累 (negative transfer),
    所以模板保留 M2 对照实验 (两个任务各训一个独立模型).

NPU 部署兼容性 (目标 F28P55x TINPU, 对齐 ti-npu 工具链的实测经验):
    * 只用 Linear+ReLU => 导出时按 1x1 卷积重写 (Linear ≡ Conv2d(k=1), 权重 reshape 直接复用),
      走 FCONV/PWCONV 路径上 NPU; 不要加花哨激活/归一化层 (否则回落 CPU)
    * batch=1 单窗口推理, 满足 TINPU 限制
    * 通道数须为 4 的倍数: n_points=32, hidden=128 均满足 (PWCONV 要求输入通道 4 的倍数)
    * FC 8bit 权重要求输入特征 >= 16: 32/128 满足; 4bit 要求 >= 8, 也满足
    * softmax / 低置信度门控 / 参数反归一化: 不进计算图, 在 C 端做
      (模型只输出原始 logits + 归一化参数)
    * 双头多输出: 编译支持性待验证; 若不支持 => 导出时 concat 成 10 维单输出, C 里再切开
    * 规模: 22026 参数 ≈ 21.5KB(int8); 延迟参照 saw2sin (4.8k MACs -> 638us) 粗估 1~3ms, 以板测为准

扩展点:
    M6 实验: 把 self.trunk 换成 1D-CNN (nn.Conv1d 系列), 两个头不用动
    (换结构后需重新验证 NPU 映射, 见上).
"""

import torch
import torch.nn as nn


class MultiTaskNet(nn.Module):
    def __init__(self, n_points=32, hidden=64, n_layers=2, n_classes=5, n_params=5):
        super().__init__()
        layers = []
        in_dim = n_points
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        self.trunk = nn.Sequential(*layers)          # 共享骨干
        self.cls_head = nn.Linear(hidden, n_classes) # 分类头
        self.reg_head = nn.Linear(hidden, n_params)  # 回归头

    def forward(self, x):
        feat = self.trunk(x)
        logits = self.cls_head(feat)     # [B,5] 未归一化分类分数
        params = self.reg_head(feat)     # [B,5或6] 归一化目标 (布局随 phase_repr)
        return logits, params


def count_parameters(model):
    """可训练参数量 (手算卡练习: 自己按层核一遍再对答案)"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 自检: 前向传一个假 batch, 看输出形状和参数量
    net = MultiTaskNet()
    logits, params = net(torch.randn(8, 32))
    print("logits:", tuple(logits.shape), "| params:", tuple(params.shape))
    print("参数量:", count_parameters(net))
    # 手算核对: 32*64+64=2112, 64*64+64=4160, 5*64+5=325 x2 个头 => 6922
