# -*- coding: utf-8 -*-
"""
诊断工具 (第 4 轮): 误差切片 + 假设检验

背景: 前几轮发现过拟合条件下 k/φ 误差停在 ~0.12 / ~0.25, 含噪与否几乎无差别.
本轮回答: 误差到底"卡"在哪些样本上?

    P0  渲染/归一化往返自检 (排除管线 bug, 应该差 ≈ 0 或 ≈ 测量噪声)
    P1  全量 (k∈0.5~4) 过拟合 -> 按类别 / k 分箱 / A 分箱切开看误差分布
    P2  过滤 k>=1 (窗口内至少 1 个完整周期) 重训 -> 验证"欠周期窗口"是否罪魁
    P3  只留 正弦+k∈[1,3] 最干净子任务 -> 检验该参数化+该网络的可达精度上限

运行: conda run --no-capture-output -n torch python _diag_slice.py
"""

import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

sys.stdout.reconfigure(encoding="utf-8")

from config import CFG
from model import MultiTaskNet
from wavegen import denormalize_params, normalize_params, render, sample_dataset

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diag_out.txt")


def log(line):
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def overfit(d, tag, steps=6000, lr=3e-3, cosine=True, seed=0):
    torch.manual_seed(seed)
    pn = normalize_params(d["params_raw"])
    model = MultiTaskNet(d["x"].shape[1], CFG.model.hidden, CFG.model.n_layers)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=5e-5) if cosine else None
    for step in range(1, steps + 1):
        logits, pred = model(d["x"])
        loss = F.cross_entropy(logits, d["cls"]) + ((pred - pn) ** 2 * d["mask"]).sum() / d["mask"].sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        if step % 2000 == 0:
            log(f"[{tag}] step {step}: loss {loss.item():.5f}")
    return model


def report_slices(tag, d, model):
    """按类别 / k 分箱 / A 分箱报告 MAE 与逐样本重建 RMSE"""
    with torch.no_grad():
        logits, pred = model(d["x"])
        pr = denormalize_params(pred)
        acc = (logits.argmax(1) == d["cls"]).float().mean().item()
        err_k = (pr[:, 1] - d["params_raw"][:, 1]).abs()
        err_phi = (pr[:, 2] - d["params_raw"][:, 2]).abs()
        periodic = d["cls"] != 4
        x_re = render(d["cls"][periodic], pr[periodic], d["x"].shape[1])
        row_recon = (x_re - d["x"][periodic]).pow(2).mean(dim=1).sqrt()
        recon = torch.full((d["x"].shape[0],), float("nan"))
        recon[periodic] = row_recon

    log(f"== {tag}: acc {acc:.4f} | MAE_k {err_k[periodic].mean():.4f} "
        f"MAE_phi {err_phi[periodic].mean():.4f} | recon mean {row_recon.mean():.4f} p90 {row_recon.quantile(0.9):.4f}")

    for cid, cname in enumerate(["sine", "square", "triangle", "saw", "noise"]):
        m = d["cls"] == cid
        if cid == 4:
            log(f"   [{cname:8s}] n={int(m.sum())}")
        elif m.any():
            log(f"   [{cname:8s}] n={int(m.sum()):4d} MAE_k {err_k[m].mean():.4f} "
                f"MAE_phi {err_phi[m].mean():.4f} recon {recon[m].mean():.4f}")

    kv = d["params_raw"][:, 1]
    for lo, hi in [(0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.01)]:
        m = periodic & (kv >= lo) & (kv < hi)
        if m.any():
            log(f"   [k {lo:.1f}-{hi:.2f}] n={int(m.sum()):4d} MAE_k {err_k[m].mean():.4f} "
                f"MAE_phi {err_phi[m].mean():.4f} recon {recon[m].mean():.4f}")

    av = d["params_raw"][:, 0]
    for lo, hi in [(0.2, 0.5), (0.5, 1.01)]:
        m = periodic & (av >= lo) & (av < hi)
        if m.any():
            log(f"   [A {lo:.1f}-{hi:.2f}] n={int(m.sum()):4d} MAE_k {err_k[m].mean():.4f} "
                f"MAE_phi {err_phi[m].mean():.4f} recon {recon[m].mean():.4f}")


if __name__ == "__main__":
    CFG.phase_repr = "raw"        # 诊断脚本按历史 5 维目标布局 (不跟随实验开关)
    with open(LOG_PATH, "w", encoding="utf-8") as fh:
        fh.write("")

    # ---------- P0: 渲染往返自检 ----------
    old_snr = CFG.wave.snr_db_range
    CFG.wave.snr_db_range = None
    d0 = sample_dataset(512, generator=torch.Generator().manual_seed(9))
    CFG.wave.snr_db_range = old_snr
    per0 = d0["cls"] != 4
    x_clean = render(d0["cls"], d0["params_raw"], d0["x"].shape[1])
    rt0 = (x_clean[per0] - d0["x"][per0]).pow(2).mean().sqrt().item()
    d0n = sample_dataset(512, generator=torch.Generator().manual_seed(9))
    rt1 = (x_clean[per0] - d0n["x"][per0]).pow(2).mean().sqrt().item()
    log(f"[P0] 无噪往返 RMSE {rt0:.2e} (应 ~1e-7) | 有噪往返 RMSE {rt1:.5f} (应≈噪声水平)")

    # ---------- P1: 全量切片 ----------
    d1 = sample_dataset(2048, generator=torch.Generator().manual_seed(0))
    m1 = overfit(d1, "P1 全量 k∈[0.5,4]", steps=6000)
    report_slices("P1 全量", d1, m1)

    # ---------- P2: 过滤 k>=1 ----------
    d2 = sample_dataset(4096, generator=torch.Generator().manual_seed(1))
    sel = d2["params_raw"][:, 1] >= 1.0
    d2 = {k: v[sel] for k, v in d2.items()}
    log(f"[P2] 过滤 k>=1 后样本数 {d2['x'].shape[0]}")
    m2 = overfit(d2, "P2 k>=1", steps=6000, seed=1)
    report_slices("P2 k>=1", d2, m2)

    # ---------- P3: 正弦-only, k∈[1,3] ----------
    d3 = sample_dataset(4096, generator=torch.Generator().manual_seed(2))
    sel3 = (d3["cls"] == 0) & (d3["params_raw"][:, 1] >= 1.0) & (d3["params_raw"][:, 1] <= 3.0)
    d3 = {k: v[sel3] for k, v in d3.items()}
    log(f"[P3] 正弦-only 且 1<=k<=3 样本数 {d3['x'].shape[0]}")
    m3 = overfit(d3, "P3 sine-only", steps=4000, seed=2)
    report_slices("P3 sine k∈[1,3]", d3, m3)
