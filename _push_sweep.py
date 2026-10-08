# -*- coding: utf-8 -*-
"""
冲刺扫描 (AI 内部工具): 依次训练多个 配置变体, 每个训练完自动评估

基线固定 = 当前最优配置: sincos + reg_weights(1,2,2,1,1) + 50k/5000 + batch 256
变体只动 hidden / epochs (容量与预算两个杠杆).

用法: conda run --no-capture-output -n torch python _push_sweep.py
产物: runs/<run_name>/{best.pt, log.json, eval_report.json} + _push_summary.json
"""

import json
import os
import sys
import time

import torch

sys.stdout.reconfigure(encoding="utf-8")

import evaluate
import train
from config import CFG

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---- 基线 (当前最优) ----
CFG.phase_repr = "sincos"
CFG.train.reg_weights = (1.0, 2.0, 2.0, 1.0, 1.0)
CFG.train.n_train = 50000
CFG.train.n_val = 5000
CFG.train.batch = 256

VARIANTS = [
    dict(run_name="push6_h512e300_d150k", hidden=512, epochs=300, n_train=150000),
]

results = []
for v in VARIANTS:
    # 每个变体都从同样的种子开始 (数据/初始化一致, 保证可比)
    torch.manual_seed(train.SEED)
    train.gen = torch.Generator().manual_seed(train.SEED)
    CFG.model.hidden = v["hidden"]
    CFG.train.epochs = v["epochs"]
    CFG.train.n_train = v.get("n_train", 50000)
    CFG.train.run_name = v["run_name"]

    t0 = time.time()
    print(f"\n########## {v['run_name']} (hidden={v['hidden']}, epochs={v['epochs']}, "
          f"n_train={CFG.train.n_train}) ##########", flush=True)
    try:
        train.main()
    except Exception as e:      # 单个变体失败不影响后续
        print("训练异常:", repr(e), flush=True)
        continue

    sys.argv = ["evaluate.py", v["run_name"]]
    try:
        evaluate.main()
    except Exception as e:
        print("评估异常:", repr(e), flush=True)

    rp = os.path.join(ROOT, "runs", v["run_name"], "eval_report.json")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            r = json.load(f)
        results.append(dict(run=v["run_name"], hidden=v["hidden"], epochs=v["epochs"],
                            acc=r["acc"], k_rel_ge1=r["k_rel_ge1"], phi_circ=r["mae"]["phi"],
                            recon=r["recon"], recon_square=r["per_class"]["square"]["recon"],
                            sec=round(time.time() - t0)))
        print(f"-- {v['run_name']} 用时 {time.time() - t0:.0f}s", flush=True)

print("\n===== 扫描汇总 =====")
for r in results:
    print(f"{r['run']}: acc {r['acc']:.4f} | k_rel(k>=1) {r['k_rel_ge1']:.4f} | phi_circ {r['phi_circ']:.4f} "
          f"| recon {r['recon']:.4f} | square_recon {r['recon_square']:.4f} | {r['sec']}s", flush=True)

summary_path = os.path.join(ROOT, "_push_summary.json")
all_results = []
if os.path.exists(summary_path):
    with open(summary_path, encoding="utf-8") as f:
        all_results = json.load(f)
all_results.extend(results)
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print("汇总已保存: _push_summary.json (含历史波次)", flush=True)
