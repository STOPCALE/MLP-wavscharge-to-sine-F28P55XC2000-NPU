# -*- coding: utf-8 -*-
"""
独立评估工具 (AI 维护): 用 runs/<run_name>/best.pt 做完整评估报告

回答四件事:
  1) 总体指标: acc / 各参数 MAE / 重建 RMSE
  2) 分类别指标: 每一类波形的召回率 + 参数误差 (哪个波形难)
  3) 端到端重建: 用【预测类别】的公式重建 (含分类错误的真实管线口径)
  4) 置信度与门控: max-softmax 作置信度 -> "覆盖率 vs 错误" 表
     —— 直接回答需求 "低置信度时拒绝输出" 是否有效

用法:
    conda run --no-capture-output -n torch python evaluate.py [run_name]
    (run_name 默认取 config; 要求 config.py 与该 run 的训练配置一致: phase_repr/规模)

产物:
    runs/<run_name>/eval_report.json
"""

import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")

from config import CFG, CLASS_NAMES, PARAM_NAMES
from model import MultiTaskNet
from wavegen import denormalize_params, normalize_params, render, sample_dataset

SEED = CFG.train.seed
NOISE = CLASS_NAMES.index("noise")


def circ_diff(a, b):
    return torch.remainder(a - b + math.pi, 2 * math.pi) - math.pi


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else CFG.train.run_name
    root = os.path.dirname(os.path.abspath(__file__))
    ckpt = torch.load(os.path.join(root, "runs", run_name, "best.pt"), map_location="cpu", weights_only=False)
    print(f"== 评估 {run_name} (best epoch {ckpt['epoch']}) ==")
    print(f"phase_repr={CFG.phase_repr} | hidden={CFG.model.hidden}x{CFG.model.n_layers} | n_val={CFG.train.n_val}")

    # 与训练时完全一致的验证集 (同种子, 保证指标可比)
    d = sample_dataset(CFG.train.n_val, generator=torch.Generator().manual_seed(SEED + 1))

    n_params = 6 if CFG.phase_repr == "sincos" else 5
    model = MultiTaskNet(n_points=CFG.wave.n_points, hidden=CFG.model.hidden,
                         n_layers=CFG.model.n_layers, n_params=n_params)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        logits, pred = model(d["x"])
        conf, pred_cls = F.softmax(logits, dim=1).max(dim=1)
        pred_raw = denormalize_params(pred, CFG.phase_repr)

    cls_true = d["cls"]
    periodic = cls_true != NOISE
    acc = (pred_cls == cls_true).float().mean().item()

    # ---- 总体指标 ----
    mask_cols = [0, 1, 2, 3, 4] if CFG.phase_repr == "raw" else [0, 1, 2, 4, 5]
    mae = {}
    for j, name in enumerate(PARAM_NAMES):
        m = d["mask"][:, mask_cols[j]] > 0
        if j == 2:
            err = circ_diff(pred_raw[:, 2], d["params_raw"][:, 2])
        else:
            err = (pred_raw[:, j] - d["params_raw"][:, j]).abs()
        mae[name] = err[m].abs().mean().item() if m.any() else float("nan")

    # 频率相对误差 (验收口径): |Δk| / k_true
    k_true = d["params_raw"][periodic, 1]
    k_err = (pred_raw[periodic, 1] - k_true).abs()
    k_rel_all = (k_err / k_true).mean().item()
    ge1 = k_true >= 1.0
    k_rel_ge1 = (k_err[ge1] / k_true[ge1]).mean().item() if ge1.any() else float("nan")

    # 重建 (真实类别公式口径). 口径说明:
    #   recon = 池化 RMSE (所有元素先平方求均值再开方) —— 与 train.py / log.json 完全一致, 用于历史对比
    #   逐样本口径仅用于分组统计 (先按样本开方再平均)
    x_re = render(cls_true[periodic], pred_raw[periodic], CFG.wave.n_points)
    se = (x_re - d["x"][periodic]) ** 2
    recon = se.mean().sqrt().item()
    recon_mean_persample = se.mean(dim=1).sqrt().mean().item()
    row_recon = se.mean(dim=1).sqrt()
    recon_full = torch.full((len(cls_true),), float("nan"))
    recon_full[periodic] = row_recon
    se_full = torch.full((len(cls_true),), float("nan"))
    se_full[periodic] = se.mean(dim=1)          # 逐样本 MSE, 便于池化口径分组统计

    print(f"\n[总体] acc {acc:.4f} | k {mae['k']:.4f} | phi_circ {mae['phi']:.4f} | "
          f"A {mae['A']:.4f} | offset {mae['offset']:.4f} | duty {mae['duty']:.4f} | "
          f"recon {recon:.4f} (逐样本均值口径 {recon_mean_persample:.4f})")
    print(f"[频率] k 相对误差: 全体均值 {k_rel_all:.4f} | k>=1 均值 {k_rel_ge1:.4f}")

    # ---- 分类别指标 ----
    print("\n[分类别] 类别 | n | 召回率 | k MAE | phi_circ | recon | 低置信(<0.9)占比")
    per_class = {}
    for cid, cname in enumerate(CLASS_NAMES):
        m = cls_true == cid
        rec = (pred_cls[m] == cid).float().mean().item()
        entry = dict(n=int(m.sum()), recall=rec, low_conf_rate=(conf[m] < 0.9).float().mean().item())
        line = f"  {cname:8s} | {int(m.sum()):4d} | {rec:.4f}"
        if cid != NOISE:
            k_ = (pred_raw[m, 1] - d["params_raw"][m, 1]).abs().mean().item()
            phi_ = circ_diff(pred_raw[m, 2], d["params_raw"][m, 2]).abs().mean().item()
            recon_cls = se_full[m].mean().sqrt().item()
            entry.update(k=k_, phi_circ=phi_, recon=recon_cls)
            line += f" | {k_:.4f} | {phi_:.4f} | {recon_cls:.4f}"
        else:
            line += " |  —     |  —     |  —    "
        line += f" | {entry['low_conf_rate']:.3f}"
        print(line)
        per_class[cname] = entry

    # ---- 混淆矩阵 (行=真实, 列=预测) ----
    cm = torch.bincount(cls_true * len(CLASS_NAMES) + pred_cls,
                        minlength=len(CLASS_NAMES) ** 2).reshape(len(CLASS_NAMES), len(CLASS_NAMES))
    print("\n[混淆矩阵] 行=真实, 列=预测")
    print("          " + " ".join(f"{c:>8s}" for c in CLASS_NAMES))
    for i, cname in enumerate(CLASS_NAMES):
        print(f"  {cname:8s}" + " ".join(f"{int(cm[i, j]):8d}" for j in range(len(CLASS_NAMES))))

    # ---- 噪声拒识 / 误接受 ----
    false_accept = (pred_cls[~periodic] != NOISE).float().mean().item()
    false_reject = (pred_cls[periodic] == NOISE).float().mean().item()
    print(f"\n[拒识原始] 噪声被误当作波形 (false accept): {false_accept:.4f} | "
          f"周期类被误拒 (false reject): {false_reject:.4f}")

    # ---- 端到端重建 (预测类别公式) ----
    m_e2e = periodic & (pred_cls != NOISE)
    x_e2e = render(pred_cls[m_e2e], pred_raw[m_e2e], CFG.wave.n_points)
    recon_e2e = (x_e2e - d["x"][m_e2e]).pow(2).mean().sqrt().item()
    print(f"[端到端] 用预测类别重建: recon {recon_e2e:.4f} (可用样本 {int(m_e2e.sum())})")

    # ---- 门控曲线: 置信度阈值 tau ----
    print("\n[门控] tau | 周期类保留率 | 保留样本 acc | 保留样本 recon | 噪声拒识率")
    gating = []
    for tau in (0.5, 0.7, 0.9, 0.95, 0.99):
        keep = conf >= tau
        keep_p = keep & periodic
        cov = keep_p.float().sum().item() / max(int(periodic.sum()), 1)
        acc_k = (pred_cls[keep] == cls_true[keep]).float().mean().item() if keep.any() else float("nan")
        rec_k = se_full[keep_p].mean().sqrt().item() if keep_p.any() else float("nan")
        noise_rej = (conf[~periodic] < tau).float().mean().item()
        gating.append(dict(tau=tau, periodic_keep=cov, acc_kept=acc_k, recon_kept=rec_k, noise_reject=noise_rej))
        print(f"  {tau:.2f} | {cov:.4f} | {acc_k:.4f} | {rec_k:.4f} | {noise_rej:.4f}")

    # ---- 归档 ----
    report = dict(run_name=run_name, epoch=ckpt["epoch"], acc=acc, mae=mae, recon=recon,
                  recon_mean_persample=recon_mean_persample,
                  k_rel_all=k_rel_all, k_rel_ge1=k_rel_ge1,
                  per_class=per_class, cm=cm.tolist(),
                  false_accept=false_accept, false_reject=false_reject,
                  recon_e2e=recon_e2e, gating=gating)
    out = os.path.join(root, "runs", run_name, "eval_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: runs/{run_name}/eval_report.json")


if __name__ == "__main__":
    main()
