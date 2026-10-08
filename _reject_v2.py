# -*- coding: utf-8 -*-
"""
拒识 v2 分析 (AI 维护): "重建残差质量分" — 能否用波形重建残差筛掉"参数不可信"的样本?

背景: 拒识 v1 = 分类置信度 (max-softmax). 已知结论 (docs/04 §2.4): 置信度与回归质量
      脱钩——低置信样本集中在回归最好的类; 方波回归最差却最"自信" → 置信度只能滤噪声.
v2 思路: 网络输出参数后, 用【预测类别+预测参数】重建波形, 与输入窗比残差.
      残差 ≈ 噪声地板 + 参数偏差贡献 → 残差大 ⇒ 参数可疑.
      (推理时即可计算, C 端只是多一次 cheap 重建, 不需要新网络)

本脚本在验证集上回答三个问题:
  1) 残差与参数误差的相关性 (并对照: 分类置信度 vs 参数误差)
  2) 按残差拒识最差 X% 后, 保留样本的参数误差分布如何变化 (拒识曲线)
  3) 代价核算: 误拒 (好样本被拒) / 漏拒 (坏样本留下)

用法: conda run --no-capture-output -n torch python _reject_v2.py <run_name> [hidden]
产物: _reject_out.txt + runs/<run_name>/reject_v2_report.json
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
from wavegen import denormalize_params, render, sample_dataset

SEED = CFG.train.seed
NOISE = CLASS_NAMES.index("noise")
ROOT = os.path.dirname(os.path.abspath(__file__))


def spearman(a, b):
    """Spearman 秩相关 (手算, 不依赖 scipy): 排名后算 Pearson"""
    ra = torch.argsort(torch.argsort(a)).float()
    rb = torch.argsort(torch.argsort(b)).float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return (ra * rb).sum().item() / max(ra.norm().item() * rb.norm().item(), 1e-12)


def q(t, p):
    return torch.quantile(t, p).item()


def main():
    run_name = sys.argv[1]
    if len(sys.argv) > 2:
        CFG.model.hidden = int(sys.argv[2])
    out = []

    def log(s=""):
        out.append(s)
        print(s)

    ckpt = torch.load(os.path.join(ROOT, "runs", run_name, "best.pt"),
                      map_location="cpu", weights_only=False)
    log(f"== 拒识 v2 分析: {run_name} (best epoch {ckpt['epoch']}) ==")
    log(f"phase_repr={CFG.phase_repr} | hidden={CFG.model.hidden} | n_val={CFG.train.n_val}")

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
    N = CFG.wave.n_points

    # ---- 分析口径: 真周期类 且 预测非噪声 (被分类头误拒的样本另行计数) ----
    m = periodic & (pred_cls != NOISE)
    n_rej_by_cls = int((periodic & (pred_cls == NOISE)).sum())
    cls_ok = pred_cls[m] == cls_true[m]
    n_cls_err = int((~cls_ok).sum())
    log(f"样本: 真周期 {int(periodic.sum())} | 其中预测为非噪声 {int(m.sum())} "
        f"(分类误拒 {n_rej_by_cls}) | 分类错误 {n_cls_err}")

    # ---- 逐样本重建残差 (推理口径: 预测类别 + 预测参数) ----
    x_re = render(pred_cls[m], pred_raw[m], N)
    resid = ((x_re - d["x"][m]) ** 2).mean(dim=1).sqrt()             # [n] 推理可用

    # ---- oracle 参考: 噪声地板 (真类别+真参数的重建残差, 推理时不可得, 仅用于分析) ----
    x_true = render(cls_true[m], d["params_raw"][m], N)
    floor = ((x_true - d["x"][m]) ** 2).mean(dim=1).sqrt()
    resid_excess = (resid.pow(2) - floor.pow(2)).clamp(min=0).sqrt()  # 超出噪声地板的部分

    # ---- 逐样本参数误差 ----
    k_true = d["params_raw"][m, 1]
    k_err = (pred_raw[m, 1] - k_true).abs()
    k_rel = k_err / k_true
    phi_err = (torch.remainder(pred_raw[m, 2] - d["params_raw"][m, 2] + math.pi, 2 * math.pi) - math.pi).abs()
    a_err = (pred_raw[m, 0] - d["params_raw"][m, 0]).abs()

    conf_m = conf[m]

    # ---- [1] 相关性 ----
    log("\n[1] Spearman 秩相关 (vs 样本参数误差; 越接近 1 = 该信号越能预测误差)")
    log(f"  resid        vs k_rel : {spearman(resid, k_rel):+.3f}")
    log(f"  resid-excess vs k_rel : {spearman(resid_excess, k_rel):+.3f}")
    log(f"  conf (v1)    vs k_rel : {spearman(conf_m, k_rel):+.3f}   <- 对照")
    log(f"  resid        vs phi   : {spearman(resid, phi_err):+.3f}")
    log(f"  conf (v1)    vs phi   : {spearman(conf_m, phi_err):+.3f}   <- 对照")
    log(f"  resid        vs A     : {spearman(resid, a_err):+.3f}")
    ge1 = k_true >= 1.0
    log(f"  (k>=1 子集) resid vs k_rel : {spearman(resid[ge1], k_rel[ge1]):+.3f} "
        f"| conf vs k_rel : {spearman(conf_m[ge1], k_rel[ge1]):+.3f}")

    # ---- [2] 残差分档 (decile) ----
    log("\n[2] 残差分档 (按 resid 从大到小: D10 = 残差最大的 10%)")
    log("  档位 |    n |  resid中位 | k_rel p50 | k_rel p90 | phi p50 | phi p90 | 分类错误")
    order = torch.argsort(resid, descending=True)
    n = int(m.sum())
    dec = max(n // 10, 1)
    for j in range(10):
        idx = order[j * dec:(j + 1) * dec]
        if len(idx) == 0:
            continue
        tag = f"  D{10 - j:>2d}"
        log(f"  {tag} | {len(idx):4d} | {q(resid[idx], 0.5):9.3f} | {q(k_rel[idx], 0.5)*100:8.2f}% | "
            f"{q(k_rel[idx], 0.9)*100:8.2f}% | {q(phi_err[idx], 0.5):7.3f} | {q(phi_err[idx], 0.9):7.3f} | "
            f"{int((~cls_ok[idx]).sum())}")

    # ---- [3] 拒识曲线 (按 resid 拒) ----
    log("\n[3] 拒识曲线 (按 resid 从大到小拒; 保留 = resid <= 分位阈值)")
    log("  拒识率 | 保留率 | k_rel p50 | k_rel p90 | k_rel_ge1 p90 | phi p90 | A p90")
    curve_resid = []
    for r in (0.0, 0.05, 0.10, 0.20, 0.30, 0.50):
        keep = resid <= torch.quantile(resid, 1.0 - r)
        kr = k_rel[keep]
        kr1 = k_rel[keep & ge1]
        ent = dict(reject=r, keep=keep.float().mean().item(),
                   k_rel_p50=q(kr, 0.5), k_rel_p90=q(kr, 0.9),
                   k_rel_ge1_p90=q(kr1, 0.9) if len(kr1) else float("nan"),
                   phi_p90=q(phi_err[keep], 0.9), a_p90=q(a_err[keep], 0.9))
        curve_resid.append(ent)
        log(f"  {r*100:5.0f}% | {keep.float().mean()*100:5.1f}% | {ent['k_rel_p50']*100:8.2f}% | "
            f"{ent['k_rel_p90']*100:8.2f}% | {ent['k_rel_ge1_p90']*100:8.2f}% | "
            f"{ent['phi_p90']:7.3f} | {ent['a_p90']:6.3f}")

    # ---- [4] 对照: 按分类置信度拒 (v1 机制, 低置信拒) ----
    log("\n[4] 对照: 按分类置信度 conf 从小拒 (v1 机制)")
    log("  拒识率 | 保留率 | k_rel p50 | k_rel p90 | k_rel_ge1 p90 | phi p90 | A p90")
    curve_conf = []
    for r in (0.0, 0.05, 0.10, 0.20, 0.30, 0.50):
        keep = conf_m >= torch.quantile(conf_m, r)
        kr = k_rel[keep]
        kr1 = k_rel[keep & ge1]
        ent = dict(reject=r, keep=keep.float().mean().item(),
                   k_rel_p50=q(kr, 0.5), k_rel_p90=q(kr, 0.9),
                   k_rel_ge1_p90=q(kr1, 0.9) if len(kr1) else float("nan"),
                   phi_p90=q(phi_err[keep], 0.9), a_p90=q(a_err[keep], 0.9))
        curve_conf.append(ent)
        log(f"  {r*100:5.0f}% | {keep.float().mean()*100:5.1f}% | {ent['k_rel_p50']*100:8.2f}% | "
            f"{ent['k_rel_p90']*100:8.2f}% | {ent['k_rel_ge1_p90']*100:8.2f}% | "
            f"{ent['phi_p90']:7.3f} | {ent['a_p90']:6.3f}")

    # ---- [5] 代价核算 (以"好样本"= k_rel<5% 且 phi<0.1 为标准) ----
    good = (k_rel < 0.05) & (phi_err < 0.1)
    log(f"\n[5] 代价核算 (好样本: k_rel<5% 且 phi<0.1; 共 {int(good.sum())} 个)")
    for r in (0.10, 0.20, 0.30):
        rej = resid > torch.quantile(resid, 1.0 - r)
        fr = (good & rej).float().sum().item() / max(good.float().sum().item(), 1)
        bad = (~good) & (k_rel > 0.10)  # "明显差"的样本 (k 相对误差 > 10%)
        missed = (bad & ~rej).float().sum().item() / max(bad.float().sum().item(), 1)
        bad_ge1, bad_lt1 = bad & ge1, bad & ~ge1
        m_ge1 = (bad_ge1 & ~rej).float().sum().item() / max(bad_ge1.float().sum().item(), 1)
        m_lt1 = (bad_lt1 & ~rej).float().sum().item() / max(bad_lt1.float().sum().item(), 1)
        log(f"  拒 {r*100:.0f}%: 误拒好样本 {fr*100:.1f}% | 漏拒差样本 总 {missed*100:.1f}%"
            f" (k>=1: {m_ge1*100:.1f}% / k<1: {m_lt1*100:.1f}%)")

    # ---- 归档 ----
    report = dict(run_name=run_name, epoch=ckpt["epoch"], n=int(m.sum()),
                  n_cls_err=n_cls_err, n_rej_by_cls=n_rej_by_cls,
                  spearman=dict(resid_k_rel=spearman(resid, k_rel),
                                resid_excess_k_rel=spearman(resid_excess, k_rel),
                                conf_k_rel=spearman(conf_m, k_rel),
                                resid_phi=spearman(resid, phi_err),
                                conf_phi=spearman(conf_m, phi_err)),
                  curve_resid=curve_resid, curve_conf=curve_conf)
    with open(os.path.join(ROOT, "runs", run_name, "reject_v2_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(ROOT, "_reject_out.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\n已保存: _reject_out.txt + runs/{run_name}/reject_v2_report.json")


if __name__ == "__main__":
    main()
