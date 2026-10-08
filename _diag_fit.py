# -*- coding: utf-8 -*-
"""
诊断工具 (AI 维护): 回归精度上不去, 是"训练不足"还是"结构问题"?

思路: 取一小批样本 (2048) 做全批量过拟合, 把训练集误差压到底, 看能到多低.
    - 若误差能压得很低  => 数据/结构没有问题; 主训练 (50k 样本, 5880 步) 只是预算不够
    - 若压不下去        => 结构 (骨干容量/头设计) 或损失需要改进

三种配置对照:
    A: 当前结构 + 当前数据 (含测量噪声)
    B: 当前结构 + 无测量噪声   (隔离"噪声"因素)
    C: 大骨干 (256x3) + 无噪声 (隔离"容量"因素)

运行: E:\\anaconda\\envs\\torch\\python.exe _diag_fit.py
"""

import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

sys.stdout.reconfigure(encoding="utf-8")

from config import CFG, PARAM_NAMES
from model import MultiTaskNet
from wavegen import denormalize_params, normalize_params, render, sample_dataset

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diag_out.txt")


def run_case(name, hidden, n_layers, snr_range, steps=6000, n=2048, lr=3e-3, seed=0, cosine=False, eta_min=5e-5):
    gen = torch.Generator().manual_seed(seed)

    old_snr = CFG.wave.snr_db_range
    CFG.wave.snr_db_range = snr_range
    d = sample_dataset(n, cfg=CFG, generator=gen)
    CFG.wave.snr_db_range = old_snr

    pn = normalize_params(d["params_raw"])
    model = MultiTaskNet(CFG.wave.n_points, hidden, n_layers)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=eta_min) if cosine else None

    def report(step, logits, pred):
        with torch.no_grad():
            pr = denormalize_params(pred)
            mae = {}
            for j, nm in enumerate(PARAM_NAMES):
                m = d["mask"][:, j] > 0
                mae[nm] = (pr[m, j] - d["params_raw"][m, j]).abs().mean().item()
            periodic = d["cls"] != 4
            x_re = render(d["cls"][periodic], pr[periodic], d["x"].shape[1])
            recon = (x_re - d["x"][periodic]).pow(2).mean().sqrt().item()
        acc = (logits.argmax(1) == d["cls"]).float().mean().item()
        line = (f"[{name}] step {step:5d}: acc {acc:.3f} recon {recon:.4f} | "
                + " ".join(f"MAE_{nm} {mae[nm]:.4f}" for nm in PARAM_NAMES))
        print(line)
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    print(f"[{name}] (overfit 全批量, n={n}, steps={steps})")
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(f"=== {name} (overfit, n={n}, steps={steps}) ===\n")
    for step in range(1, steps + 1):
        logits, pred = model(d["x"])
        loss = F.cross_entropy(logits, d["cls"]) + ((pred - pn) ** 2 * d["mask"]).sum() / d["mask"].sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        if step % 1000 == 0:
            report(step, logits, pred)


if __name__ == "__main__":
    # 第二轮结论: B/D 无噪 15k 常量 lr => 所有参数 MAE 齐刷刷停在 ~4% (归一化), 不再降
    # 本轮: E 加余弦衰减 (验证"优化抖动限制精度"假设); F 含噪+衰减 (真实配置的可达精度)
    CFG.phase_repr = "raw"        # 诊断脚本按历史 5 维目标布局 (不跟随实验开关)
    with open(LOG_PATH, "w", encoding="utf-8") as fh:
        fh.write("")
    run_case("E 无噪/cosine 15k", CFG.model.hidden, CFG.model.n_layers, None, steps=15000, cosine=True)
    run_case("F 含噪/cosine 15k", CFG.model.hidden, CFG.model.n_layers, CFG.wave.snr_db_range, steps=15000, cosine=True)
