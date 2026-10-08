# -*- coding: utf-8 -*-
"""方波测试时精化 (TTR): 推理端两级网格搜索, 无需训练 (AI 维护)

思路: 网络给出初始 (k, φ, duty) 后, 用观测窗做局部搜索最小化【硬渲染残差】:
    网格两级: 粗 (k ±6/64, φ ±0.3, duty ±0.06) -> 细 (步长 /5)
    搜索维度只含 (k, φ, duty) (A/offset 估计已准, 固定)
优点: 不改训练、不伤其他类、与"拒识 v2 的残差质量分"天然联动 (精化后残差即质量分)
风险: 用带噪观测拟合 -> 可能过拟合噪声 (看 vs 真值的参数误差变化)

用法: conda run --no-capture-output -n torch python _ttr_square.py <run_name> [hidden]
产物: _ttr_out.txt + runs/<run_name>/ttr_square_report.json
"""

import json
import math
import os
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8")

from config import CFG, CLASS_NAMES
from model import MultiTaskNet
from wavegen import denormalize_params, sample_dataset

SEED = 0
NOISE = CLASS_NAMES.index("noise")
SQUARE = CLASS_NAMES.index("square")
ROOT = os.path.dirname(os.path.abspath(__file__))


def circ_err(a, b):
    return (torch.remainder(a - b + math.pi, 2 * math.pi) - math.pi).abs()


def render_sq(k, phi, duty, A, off, N):
    """批量硬方波渲染: k,phi,duty: [n,m] -> [n,m,N]"""
    t = torch.arange(N, dtype=torch.float32) / N
    theta = 2 * math.pi * k.unsqueeze(2) * t + phi.unsqueeze(2)      # [n,m,N]
    frac = torch.remainder(theta, 2 * math.pi) / (2 * math.pi)
    sq = torch.where(frac < duty.unsqueeze(2), torch.ones_like(frac), -torch.ones_like(frac))
    return A.unsqueeze(2) * sq + off.unsqueeze(2)


def obs_resid(params, x, N):
    """与观测窗的 RMSE (逐样本): params [n,5], x [n,N]"""
    xr = render_sq(params[:, 1:2], params[:, 2:3], params[:, 4:5], params[:, 0:1], params[:, 3:4], N)
    return ((xr.squeeze(1) - x) ** 2).mean(dim=1).sqrt()


def refine_round(params, x, dk, dphi, dd, N):
    """一轮网格搜索: 返回 (新参数, 本轮最优残差)"""
    dk = torch.tensor(dk, dtype=torch.float32)
    dphi = torch.tensor(dphi, dtype=torch.float32)
    dd = torch.tensor(dd, dtype=torch.float32)
    n = params.shape[0]
    k0, p0, d0 = params[:, 1:2], params[:, 2:3], params[:, 4:5]
    A0, off0 = params[:, 0:1], params[:, 3:4]
    nk, np_, nd = len(dk), len(dphi), len(dd)

    kg = (k0 + dk.view(1, -1)).clamp(0.5, 4.0)                    # [n,nk]
    pg = p0 + dphi.view(1, -1)                                    # [n,np] (wrap 由 remainder 处理)
    dg = (d0 + dd.view(1, -1)).clamp(0.2, 0.8)                    # [n,nd]

    K = kg.unsqueeze(2).unsqueeze(3).expand(n, nk, np_, nd).reshape(n, -1)
    P = pg.unsqueeze(1).unsqueeze(3).expand(n, nk, np_, nd).reshape(n, -1)
    D = dg.unsqueeze(1).unsqueeze(2).expand(n, nk, np_, nd).reshape(n, -1)

    xr = render_sq(K, P, D, A0, off0, N)                          # [n,m,N]
    se = (xr - x.unsqueeze(1)).pow(2).mean(dim=2)                 # [n,m]
    best = se.argmin(dim=1)

    out = params.clone()
    ar = torch.arange(n)
    out[:, 1] = K[ar, best]
    out[:, 2] = P[ar, best]
    out[:, 4] = D[ar, best]
    return out, se.min(dim=1).values.sqrt()


def main():
    run_name = sys.argv[1]
    if len(sys.argv) > 2:
        CFG.model.hidden = int(sys.argv[2])
    N = CFG.wave.n_points
    out_lines = []

    def log(s=""):
        out_lines.append(s)
        print(s)

    ckpt = torch.load(os.path.join(ROOT, "runs", run_name, "best.pt"),
                      map_location="cpu", weights_only=False)
    d = sample_dataset(CFG.train.n_val, generator=torch.Generator().manual_seed(SEED + 1))
    model = MultiTaskNet(n_points=N, hidden=CFG.model.hidden,
                         n_layers=CFG.model.n_layers, n_params=6)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        logits, pred = model(d["x"])
        pred_cls = logits.argmax(dim=1)
        pred_raw = denormalize_params(pred, CFG.phase_repr)

    # 分析口径: 真值方波 且 预测方波 (TTR 实际会应用的样本)
    m = (d["cls"] == SQUARE) & (pred_cls == SQUARE)
    x = d["x"][m]
    p0 = pred_raw[m]
    pt = d["params_raw"][m]
    n = int(m.sum())
    log(f"== 方波 TTR: {run_name} (best epoch {ckpt['epoch']}) == 样本 {n}")
    log("搜索: 粗网格 k±6/64 φ±0.3 duty±0.06 -> 细网格 步长/5; 目标=观测窗硬渲染残差")

    # 参考: 噪声地板 (真值参数的观测残差)
    floor = obs_resid(pt, x, N)

    # 两级精化
    p, _ = refine_round(p0, x, [-6, -4, -2, 0, 2, 4, 6], [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3],
                        [-0.06, -0.03, 0, 0.03, 0.06], N)
    p, _ = refine_round(p, x, [-0.4/32, -0.2/32, 0, 0.2/32, 0.4/32],
                        [-0.02, -0.01, 0, 0.01, 0.02],
                        [-0.006, -0.003, 0, 0.003, 0.006], N)

    def q(t, v):
        return torch.quantile(t, v).item()

    def stats(tag, prm):
        krt = (prm[:, 1] - pt[:, 1]).abs() / pt[:, 1]
        phe = circ_err(prm[:, 2], pt[:, 2])
        dte = (prm[:, 4] - pt[:, 4]).abs()
        rec = obs_resid(prm, x, N)
        ge1 = pt[:, 1] >= 1.0
        log(f"[{tag}] k_rel: p50 {q(krt,0.5)*100:5.2f}% p90 {q(krt,0.9)*100:5.2f}% "
            f"(k>=1: {q(krt[ge1],0.5)*100:5.2f}%/{q(krt[ge1],0.9)*100:5.2f}%) | "
            f"phi: p50 {q(phe,0.5):.3f} p90 {q(phe,0.9):.3f} | "
            f"duty: p50 {q(dte,0.5):.3f} p90 {q(dte,0.9):.3f} | obsres: p50 {q(rec,0.5):.3f}")
        return dict(krl50=q(krt,0.5), krl90=q(krt,0.9), phi50=q(phe,0.5), phi90=q(phe,0.9),
                    duty50=q(dte,0.5), duty90=q(dte,0.9), res50=q(rec,0.5))

    log(f"(噪声地板 obsres p50: {q(floor,0.5):.3f})")
    s0 = stats("精化前", p0)
    s1 = stats("精化后", p)

    # 收益/损失计数
    krt0 = ((p0[:, 1] - pt[:, 1]).abs() / pt[:, 1])
    krt1 = ((p[:, 1] - pt[:, 1]).abs() / pt[:, 1])
    better = (krt1 < krt0).float().mean().item()
    log(f"\nk_rel 改善样本占比: {better*100:.1f}% | "
        f"精化前后均方改善: {q(krt0,0.5)*100:.2f}% -> {q(krt1,0.5)*100:.2f}% (p50)")

    rep = dict(run_name=run_name, epoch=ckpt["epoch"], n=n, before=s0, after=s1,
               k_rel_improved_frac=better)
    with open(os.path.join(ROOT, "runs", run_name, "ttr_square_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    with open(os.path.join(ROOT, "_ttr_out.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\n已保存: _ttr_out.txt + runs/{run_name}/ttr_square_report.json")


if __name__ == "__main__":
    main()
