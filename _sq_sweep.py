# -*- coding: utf-8 -*-
"""
方波专项快档扫描 (AI 维护): 验证"软方波重建辅助损失"的方向与量级

背景: 方波是四类周期波的最大短板 (push6: 方波 recon 0.377 / k_rel p90 18%).
      硬方波 where/sign 不可微 -> 常规"波形重建损失"无法训练; 本实验用
      wavegen.render_soft_square (tanh 软化边沿, 梯度集中在边沿) 做辅助损失 (train.py),
      只作用于方波样本, 目标是"预测参数 vs 真值参数"的软方波 MSE.

快档协议 (3 分钟内出方向): 20ep / 20k 样本 / 128 宽 / reg_weights(1,2,2,1,1).
变体: lambda_recon_sq = 0 (对照) / 3 / 10 —— 一次只改一个变量 (每个变体前重置种子,
      保证初始化/数据/批次顺序完全一致).

每变体训练后评估: 方波 recon(池化) / k 相对误差 / phi 圆周 / duty | 总体 acc / recon.
产物: _sq_sweep.json + _sq_sweep_out.txt (终端输出也落盘, 防截断)

用法: conda run --no-capture-output -n torch python _sq_sweep.py
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
import train as train_mod

ROOT = os.path.dirname(os.path.abspath(__file__))
SEED = 0
NOISE = CLASS_NAMES.index("noise")
SQUARE = CLASS_NAMES.index("square")

VARIANTS = [
    ("sq7_lam03_sw2", dict(lambda_recon_sq=0.3, square_weight=2.0)),
    ("sq8_lam03_sw4", dict(lambda_recon_sq=0.3, square_weight=4.0)),
]
BASE_SWITCH = dict(lambda_recon_sq=0.0, square_weight=1.0)  # 每变体前复位, 保证单变量


def eval_one(run_name, hidden):
    """方波专项评估 (验证集与 train.main 内一致: 种子 SEED+1)
    同时输出方波指标与其他类指标 —— 防"方波专精, 其他类退化"的偏科"""
    ckpt = torch.load(os.path.join(ROOT, "runs", run_name, "best.pt"),
                      map_location="cpu", weights_only=False)
    model = MultiTaskNet(n_points=CFG.wave.n_points, hidden=hidden,
                         n_layers=CFG.model.n_layers, n_params=6)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    d = sample_dataset(CFG.train.n_val, generator=torch.Generator().manual_seed(SEED + 1))
    with torch.no_grad():
        logits, pred = model(d["x"])
    pred_raw = denormalize_params(pred, "sincos")
    acc = (logits.argmax(1) == d["cls"]).float().mean().item()

    sq = d["cls"] == SQUARE
    periodic = d["cls"] != NOISE
    nonsq = periodic & ~sq

    def recon_of(mask):
        x_re = render(d["cls"][mask], pred_raw[mask], CFG.wave.n_points)
        return ((x_re - d["x"][mask]) ** 2).mean().sqrt().item()

    recon_sq, recon_nonsq, recon_all = recon_of(sq), recon_of(nonsq), recon_of(periodic)
    k_rel_sq = ((pred_raw[sq, 1] - d["params_raw"][sq, 1]).abs() / d["params_raw"][sq, 1]).mean().item()
    k_rel_all = ((pred_raw[periodic, 1] - d["params_raw"][periodic, 1]).abs()
                 / d["params_raw"][periodic, 1]).mean().item()
    dphi = torch.remainder(pred_raw[:, 2] - d["params_raw"][:, 2] + math.pi, 2 * math.pi) - math.pi
    phi_sq, phi_all = dphi[sq].abs().mean().item(), dphi[periodic].abs().mean().item()
    duty = (pred_raw[sq, 4] - d["params_raw"][sq, 4]).abs().mean().item()
    return dict(epoch=ckpt["epoch"], acc=acc, recon_all=recon_all, recon_nonsq=recon_nonsq,
                recon_sq=recon_sq, k_rel_sq=k_rel_sq, k_rel_all=k_rel_all,
                phi_sq=phi_sq, phi_all=phi_all, duty_sq=duty)


def main():
    out = []

    def log(s=""):
        out.append(s)
        print(s)

    # ---- 快档协议 ----
    CFG.train.epochs = 20
    CFG.train.n_train = 20000
    CFG.train.n_val = 2000
    CFG.model.hidden = 128

    results = []
    for name, kw in VARIANTS:
        CFG.train.run_name = name
        for k, v in BASE_SWITCH.items():       # 先复位所有开关
            setattr(CFG.train, k, v)
        for k, v in kw.items():                # 再应用本变体 (单变量)
            setattr(CFG.train, k, v)
        torch.manual_seed(SEED)          # 每变体: 同初始化
        train_mod.gen.manual_seed(SEED)  # 每变体: 同数据 + 同批序
        log(f"\n########## {name} ({kw}) ##########")
        train_mod.main()
        r = eval_one(name, 128)
        r.update(name=name, **kw)
        results.append(r)
        log(f"[专项评估] {name}: acc {r['acc']:.4f} | sq_recon {r['recon_sq']:.4f} "
            f"sq_k {r['k_rel_sq']:.4f} sq_phi {r['phi_sq']:.4f} duty {r['duty_sq']:.4f} | "
            f"非方波recon {r['recon_nonsq']:.4f} | 总体k {r['k_rel_all']:.4f} phi {r['phi_all']:.4f}")

    log("\n===== 汇总 (方波专项 第三轮: 组合) =====")
    log("  变体 | sq_recon | sq_k_rel | sq_phi | duty | 非方波recon | 总体k_rel | 总体phi | acc")
    for r in results:
        log(f"  {r['name']:<14s} | {r['recon_sq']:.4f} | {r['k_rel_sq']:.4f} | {r['phi_sq']:.4f} | "
            f"{r['duty_sq']:.4f} | {r['recon_nonsq']:.4f} | {r['k_rel_all']:.4f} | "
            f"{r['phi_all']:.4f} | {r['acc']:.4f}")

    with open(os.path.join(ROOT, "_sq_sweep3.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(ROOT, "_sq_sweep3_out.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n已保存: _sq_sweep3.json / _sq_sweep3_out.txt")


if __name__ == "__main__":
    main()
