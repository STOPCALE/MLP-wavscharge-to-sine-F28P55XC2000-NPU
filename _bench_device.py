# -*- coding: utf-8 -*-
"""CPU vs CUDA 小模型训练速度对照 (AI 维护)

背景: Charge_Model 全量档训练 (512宽/300ep/15万样本) 在 CPU 上约 40 分钟。
本脚本实测同一模型/同一 batch 在 CPU 与 RTX 4060 上的 "前向+反向+step" 单批耗时,
用数据回答"值不值得切 GPU"。
"""
import time

import torch
import torch.nn as nn

from model import MultiTaskNet

torch.manual_seed(0)
BATCH = 256
ITERS = 300


def bench(dev):
    m = MultiTaskNet(n_points=32, hidden=512, n_layers=2, n_classes=5, n_params=6).to(dev)
    x = torch.randn(BATCH, 32, device=dev)
    y1 = torch.randn(BATCH, 5, device=dev)
    y2 = torch.randn(BATCH, 6, device=dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    lossf = nn.MSELoss()

    for _ in range(20):                      # warmup
        lo, pa = m(x)
        loss = lossf(lo, y1) + lossf(pa, y2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if dev == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(ITERS):
        lo, pa = m(x)
        loss = lossf(lo, y1) + lossf(pa, y2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / ITERS * 1000   # ms/batch
    est = dt * 586 * 300 / 1000 / 60         # 586 batch/epoch x 300 ep -> 分钟
    print(f"{dev:>4s}: {dt:6.2f} ms/batch | 300ep x 15万样本 预计 {est:5.1f} 分钟")
    return dt


d_cpu = bench("cpu")
if torch.cuda.is_available():
    d_gpu = bench("cuda")
    print(f"加速比: {d_cpu / d_gpu:.1f}x  (GPU: {torch.cuda.get_device_name(0)})")
