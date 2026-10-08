# -*- coding: utf-8 -*-
"""
验收测试 (AI 维护): 对照验收标准检查某个实验的评估报告

用法:
    conda run --no-capture-output -n torch python acceptance.py [run_name]
    (默认 run_name = m1_sincos_w; 读取 runs/<run_name>/eval_report.json)

验收标准 v1 (2026-10-08 定, 按需求推导):
    分类 acc              >= 0.995        (后处理策略依赖分类可靠)
    噪声误接受率           <= 0.005        (不把噪声当波形)
    频率相对误差 (k>=1)    <= 0.01         (频率误差 <1%)
    相位 phi_circ          <= 0.087 rad    (5°)
    幅值 A MAE             <= 0.02
    偏置 offset MAE        <= 0.02
    占空比 duty MAE        <= 0.03
    方波重建 recon         <= 0.20
    总体重建 recon         <= 0.15
"""

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

CRITERIA = [
    ("分类 acc",            "acc",                    ">=", 0.995),
    ("噪声误接受率",         "false_accept",           "<=", 0.005),
    ("频率相对误差 (k>=1)",  "k_rel_ge1",              "<=", 0.01),
    ("相位 phi_circ (rad)", "mae.phi",                "<=", 0.087),
    ("幅值 A MAE",          "mae.A",                  "<=", 0.02),
    ("偏置 offset MAE",     "mae.offset",             "<=", 0.02),
    ("占空比 duty MAE",     "mae.duty",               "<=", 0.03),
    ("方波重建 recon",       "per_class.square.recon", "<=", 0.20),
    ("总体重建 recon",       "recon",                  "<=", 0.15),
]


def get(d, path):
    for k in path.split("."):
        d = d[k]
    return d


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "m1_sincos_w"
    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, "runs", run_name, "eval_report.json"), encoding="utf-8") as f:
        r = json.load(f)

    print(f"== 验收测试: {run_name} (best epoch {r['epoch']}) ==")
    print(f"{'指标':<20} {'目标':<10} {'实测':<10} 判定")
    n_pass = 0
    for name, path, op, thr in CRITERIA:
        val = get(r, path)
        ok = (val >= thr) if op == ">=" else (val <= thr)
        n_pass += int(ok)
        print(f"{name:<20} {op} {thr:<7} {val:<10.4f} {'PASS' if ok else 'FAIL'}")
    print(f"\n通过 {n_pass}/{len(CRITERIA)}")


if __name__ == "__main__":
    main()
