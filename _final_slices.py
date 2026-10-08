# -*- coding: utf-8 -*-
"""
最终误差分布分析 (AI 工具): 回答 "参数能测得多准" —— 按类别 / 区间 / 分位数切片

用法: conda run --no-capture-output -n torch python _final_slices.py <run_name> <hidden>
产物: 控制台 + _slices_out.txt + runs/<run_name>/slice_report.json

分位数说明: p50 = 一半样本比它好; p90 = 90% 样本比它好 (看尾部有多差)
"""

import json
import math
import os
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8")

from config import CFG, CLASS_NAMES
from model import MultiTaskNet
from wavegen import denormalize_params, render, sample_dataset

SEED = CFG.train.seed
NOISE = CLASS_NAMES.index("noise")
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_slices_out.txt")


def log(line):
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def q(t, p):
    return t.quantile(p).item()


def main():
    run_name = sys.argv[1]
    hidden = int(sys.argv[2])
    root = os.path.dirname(os.path.abspath(__file__))
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("")

    ckpt = torch.load(os.path.join(root, "runs", run_name, "best.pt"), map_location="cpu", weights_only=False)
    d = sample_dataset(CFG.train.n_val, generator=torch.Generator().manual_seed(SEED + 1))

    n_params = 6 if CFG.phase_repr == "sincos" else 5
    model = MultiTaskNet(n_points=CFG.wave.n_points, hidden=hidden,
                         n_layers=CFG.model.n_layers, n_params=n_params)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        logits, pred = model(d["x"])
        pred_raw = denormalize_params(pred, CFG.phase_repr)
        acc = (logits.argmax(1) == d["cls"]).float().mean().item()

    cls_true = d["cls"]
    periodic = cls_true != NOISE
    k_true = d["params_raw"][:, 1]
    A_true = d["params_raw"][:, 0]
    k_rel = (pred_raw[:, 1] - k_true).abs() / k_true
    dphi = torch.remainder(pred_raw[:, 2] - d["params_raw"][:, 2] + math.pi, 2 * math.pi) - math.pi
    phi_err = dphi.abs()

    x_re = render(cls_true[periodic], pred_raw[periodic], CFG.wave.n_points)
    row_rmse = ((x_re - d["x"][periodic]) ** 2).mean(dim=1).sqrt()
    recon = torch.full((len(cls_true),), float("nan"))
    recon[periodic] = row_rmse

    slices = {}

    def report(mask, name):
        m = mask & periodic
        if not m.any():
            return
        rec = dict(n=int(m.sum()),
                   k_rel_p50=q(k_rel[m], 0.5), k_rel_p90=q(k_rel[m], 0.9),
                   phi_p50=q(phi_err[m], 0.5), phi_p90=q(phi_err[m], 0.9),
                   recon_p50=q(recon[m], 0.5), recon_p90=q(recon[m], 0.9))
        slices[name] = rec
        log(f"  {name:14s} n={rec['n']:4d} | k_rel: p50 {rec['k_rel_p50']:.3f} p90 {rec['k_rel_p90']:.3f} "
            f"| phi(rad): p50 {rec['phi_p50']:.3f} p90 {rec['phi_p90']:.3f} "
            f"| recon: p50 {rec['recon_p50']:.3f} p90 {rec['recon_p90']:.3f}")

    log(f"== {run_name} (best epoch {ckpt['epoch']}) | acc {acc:.4f} ==")
    log("[全周期样本]")
    report(periodic, "all")
    log("[分波类]")
    for cid, cname in enumerate(CLASS_NAMES[:-1]):
        report(cls_true == cid, cname)
    log("[按 k 区间]")
    for lo, hi in [(0.5, 1), (1, 2), (2, 3), (3, 4.01)]:
        report((k_true >= lo) & (k_true < hi), f"k {lo}-{hi}")
    log("[按幅值]")
    for lo, hi in [(0.2, 0.5), (0.5, 1.01)]:
        report((A_true >= lo) & (A_true < hi), f"A {lo}-{hi}")
    log("[推荐核心区: k>=1 且 A>=0.5]")
    report((k_true >= 1) & (A_true >= 0.5), "core")

    with open(os.path.join(root, "runs", run_name, "slice_report.json"), "w", encoding="utf-8") as f:
        json.dump(dict(run=run_name, epoch=ckpt["epoch"], acc=acc, slices=slices),
                  f, ensure_ascii=False, indent=2)
    log("已保存: slice_report.json")


if __name__ == "__main__":
    main()
