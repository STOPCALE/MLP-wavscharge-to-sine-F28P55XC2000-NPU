# -*- coding: utf-8 -*-
"""
方波专项全量档训练 (AI 维护): "软方波重建辅助损失 + 方波样本加权" 在 push6 规模上的验证

配置 (对照 push6_h512e300_d150k, 只加方波专项两个开关 —— 组合实验):
    hidden=512 / epochs=300 / n_train=150k / n_val=5000
    lambda_recon_sq=0.3  (软方波重建辅助损失, 快档第二轮甜点: 更大 λ 会主导训练)
    square_weight=2.0    (方波样本回归加权; 快档第三轮: sw=2 比 sw=4 更平衡)

依据: 快档三连 (见 _sq_sweep*.json):
    λ=3/10 -> 重建损失主导, acc/phi 崩 (反例记录)
    λ=0.3  -> 方波 recon -43%, k_rel -24% (甜点)
    λ=0.3+sw2 -> 方波 phi 再改善, 非方波类保持最好

流程: 训练 -> evaluate (自动出 eval_report.json)
后续手动: acceptance.py / _final_slices.py (见验收流程)

设备: GPU (RTX 4060 实测 ~3.5x 加速, 见 _bench_device.py; 训练~7min + 评估)

用法 (后台运行): conda run --no-capture-output -n torch python _sq_full.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from config import CFG

RUN_NAME = "sq9_full_lam03_sw2"


def main():
    CFG.train.device = "cuda"
    CFG.model.hidden = 512
    CFG.train.epochs = 300
    CFG.train.n_train = 150000
    CFG.train.n_val = 5000
    CFG.train.lambda_recon_sq = 0.3
    CFG.train.square_weight = 2.0
    CFG.train.run_name = RUN_NAME

    import train as train_mod
    train_mod.main()

    # 训练完自动评估 (同进程: CFG 已就位, hidden=512)
    import evaluate as evaluate_mod
    sys.argv = ["evaluate", RUN_NAME]
    evaluate_mod.main()


if __name__ == "__main__":
    main()
