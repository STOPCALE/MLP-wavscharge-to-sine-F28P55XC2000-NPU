# -*- coding: utf-8 -*-
"""
训练入口: 分类 + 回归 多任务联合训练 (M1 基线)

运行 (二选一):
    1) E:\\anaconda\\envs\\torch\\python.exe train.py
    2) conda run --no-capture-output -n torch python train.py

产物:
    runs/<run_name>/best.pt    验证总损失最低的权重
    runs/<run_name>/log.json   每个 epoch 的指标记录 (后续画图 / 对比实验用)

教案: 对应 README 路线图 M1 —— 先跑通, 观察三类指标的数量级:
    分类准确率 / 各参数 MAE (物理单位) / 重建 RMSE

实验开关 (只改 config.py):
    CFG.phase_repr        : "raw" | "sincos"  相位监督表示 (B 线主角)
    CFG.train.reg_weights : (A,k,phi,offset,duty) 各参数回归权重
    CFG.train.epochs / CFG.model.hidden / lambda_* : 训练预算与容量
"""

import json
import math
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from config import CFG, CLASS_NAMES, PARAM_NAMES
from model import MultiTaskNet, count_parameters
from wavegen import denormalize_params, normalize_params, render, sample_dataset

# ----------------- 0. 种子 (可复现) -----------------
SEED = CFG.train.seed
torch.manual_seed(SEED)
gen = torch.Generator().manual_seed(SEED)


def build_data():
    """生成固定的训练/验证集 (一次性生成; 固定种子 => 实验之间可比)"""
    train = sample_dataset(CFG.train.n_train, generator=gen)
    val = sample_dataset(CFG.train.n_val, generator=torch.Generator().manual_seed(SEED + 1))
    for d in (train, val):
        d["params_norm"] = normalize_params(d["params_raw"], CFG.phase_repr)   # 布局随 phase_repr
    return train, val


# ----------------- 1. 多任务损失 -----------------
def _loss_weights():
    """把 5 个物理参数权重 (A,k,φ,offset,duty) 展开到监督目标空间

    phase_repr="sincos" 时目标 6 维: φ 的权重同时作用于 cos/sin 两项.
    默认全 1 => 与不带权重的旧行为完全一致 (config.reg_weights 是实验开关).
    """
    w5 = torch.tensor(CFG.train.reg_weights, dtype=torch.float32)
    if CFG.phase_repr == "raw":
        return w5
    return torch.stack([w5[0], w5[1], w5[2], w5[2], w5[3], w5[4]])


def multi_task_loss(logits, pred, cls, tgt_norm, mask):
    """L = λ_cls·CE + λ_reg·masked&weighted MSE

    masked MSE 的两处关键 (易错点):
      * 只对 mask=1 的目标监督: 噪声类没有回归标签; duty 只对方波有定义
      * 分母用 (mask*w).sum() 而不是元素总数 —— 否则被掩掉的 0 会把损失稀释;
        没有方波的 batch 里 duty 自动不贡献
    """
    loss_cls = F.cross_entropy(logits, cls)
    w = _loss_weights()
    se = ((pred - tgt_norm) ** 2) * w
    loss_reg = (se * mask).sum() / (mask * w).sum().clamp(min=1.0)
    total = CFG.train.lambda_cls * loss_cls + CFG.train.lambda_reg * loss_reg
    return total, loss_cls.detach(), loss_reg.detach()


# ----------------- 2. 评估 -----------------
@torch.no_grad()
def evaluate(model, d):
    """全量验证集评估: 分类 acc / 各参数 MAE(物理单位) / 重建 RMSE"""
    model.eval()
    logits, pred_norm = model(d["x"])
    total, lc, lr = multi_task_loss(logits, pred_norm, d["cls"], d["params_norm"], d["mask"])

    acc = (logits.argmax(dim=1) == d["cls"]).float().mean().item()

    # 各参数 MAE: 反归一化回物理单位, 只统计 mask=1 的元素
    # (mask 列布局随 phase_repr: raw=[A,k,φ,off,duty], sincos=[A,k,cos,sin,off,duty])
    pred_raw = denormalize_params(pred_norm, CFG.phase_repr)
    mask_cols = [0, 1, 2, 3, 4] if CFG.phase_repr == "raw" else [0, 1, 2, 4, 5]
    mae = {}
    for j, name in enumerate(PARAM_NAMES):
        m = d["mask"][:, mask_cols[j]] > 0
        mae[name] = (pred_raw[m, j] - d["params_raw"][m, j]).abs().mean().item() if m.any() else float("nan")

    # φ 的圆周误差: wrap 到 [-π, π] 再取绝对值 —— 跨 0/2π 的翻转不算大误差
    # (与普通 MAE 对比, 差值就是边界冲突贡献的量)
    dphi = pred_raw[:, 2] - d["params_raw"][:, 2]
    dphi = torch.remainder(dphi + math.pi, 2 * math.pi) - math.pi
    m2 = d["mask"][:, 2] > 0
    mae["phi_circ"] = dphi[m2].abs().mean().item() if m2.any() else float("nan")

    # 重建误差: 用【预测参数】按【真实波形公式】重建, 衡量参数估得准不准.
    # 用真实类 = 暂时排除分类错误的干扰, 先隔离评估回归本身;
    # 端到端版 (用预测的类) 和门控曲线留给 evaluate.py.
    periodic = d["cls"] != CLASS_NAMES.index("noise")
    x_re = render(d["cls"][periodic], pred_raw[periodic], CFG.wave.n_points)
    recon_rmse = (x_re - d["x"][periodic]).pow(2).mean().sqrt().item()

    return dict(total=total.item(), cls=lc.item(), reg=lr.item(),
                acc=acc, mae=mae, recon_rmse=recon_rmse)


# ----------------- 3. 训练循环 -----------------
def main():
    print(f"== 多任务训练 == run_name = {CFG.train.run_name}")
    print(f"窗口 {CFG.wave.n_points} 点 | 类别 {CLASS_NAMES}")
    print(f"phase_repr={CFG.phase_repr} | reg_weights={CFG.train.reg_weights} | "
          f"lambda_cls={CFG.train.lambda_cls} lambda_reg={CFG.train.lambda_reg}")
    print(f"lr={CFG.train.lr} batch={CFG.train.batch} epochs={CFG.train.epochs}")

    train, val = build_data()
    print(f"训练集 {CFG.train.n_train} | 验证集 {CFG.train.n_val}")

    n_params = 6 if CFG.phase_repr == "sincos" else 5
    model = MultiTaskNet(n_points=CFG.wave.n_points,
                         hidden=CFG.model.hidden, n_layers=CFG.model.n_layers,
                         n_params=n_params)
    # 手算练习: 按前面学的 参数量=权重+偏置 自己核一遍, 方法见 model.py 底部
    print(f"模型参数量: {count_parameters(model)}")
    optimizer = optim.Adam(model.parameters(), lr=CFG.train.lr)
    scheduler = (optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.train.epochs,
                                                      eta_min=CFG.train.eta_min)
                 if CFG.train.cosine else None)

    n = CFG.train.n_train
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", CFG.train.run_name)
    os.makedirs(run_dir, exist_ok=True)

    best_val, best_epoch = float("inf"), None
    log = []
    for epoch in range(1, CFG.train.epochs + 1):
        # ---- 训练 ----
        model.train()
        perm = torch.randperm(n, generator=gen)
        run_tot = run_cls = run_reg = 0.0
        nb = 0
        for i in range(0, n, CFG.train.batch):
            idx = perm[i:i + CFG.train.batch]
            logits, pred = model(train["x"][idx])
            total, lc, lr_ = multi_task_loss(logits, pred, train["cls"][idx],
                                             train["params_norm"][idx], train["mask"][idx])
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            run_tot += total.item()
            run_cls += lc.item()
            run_reg += lr_.item()
            nb += 1

        if scheduler is not None:
            scheduler.step()

        # ---- 验证 ----
        m = evaluate(model, val)
        rec = dict(epoch=epoch,
                   train_total=run_tot / nb, train_cls=run_cls / nb, train_reg=run_reg / nb,
                   val_total=m["total"], val_cls=m["cls"], val_reg=m["reg"],
                   val_acc=m["acc"], val_recon_rmse=m["recon_rmse"],
                   **{f"mae_{k}": v for k, v in m["mae"].items()})
        log.append(rec)

        mae_str = " ".join(f"{k}={m['mae'][k]:.3g}" for k in PARAM_NAMES if not math.isnan(m["mae"][k]))
        if not math.isnan(m["mae"]["phi_circ"]):
            mae_str += f" phi_circ={m['mae']['phi_circ']:.3g}"
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:3d}/{CFG.train.epochs} lr {lr_now:.1e} | "
              f"train total {rec['train_total']:.4f} (cls {rec['train_cls']:.4f} reg {rec['train_reg']:.4f}) | "
              f"val {rec['val_total']:.4f} | acc {rec['val_acc']:.4f} | "
              f"recon {rec['val_recon_rmse']:.4f} | {mae_str}")

        # ---- 保存最优 (按验证总损失; MTL 的"最优"本身可讨论, 见 README) ----
        if rec["val_total"] < best_val:
            best_val, best_epoch = rec["val_total"], epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val": m},
                       os.path.join(run_dir, "best.pt"))

    with open(os.path.join(run_dir, "log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n完成: 最佳 epoch {best_epoch} (val_total {best_val:.4f})")
    print(f"产物: runs/{CFG.train.run_name}/best.pt 和 log.json")
    print("下一步: evaluate.py (混淆矩阵 / 门控曲线) 与 visualize.py (重建回放), 见 README 路线图")


if __name__ == "__main__":
    main()
