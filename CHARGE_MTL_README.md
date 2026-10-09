# charge_mtl 板端验证工程说明

> 2026-10-08 集成完成。原 saw2sin 集成版本已完整备份到
> `E:\desk\CCS\_backup_npu_and_charge_saw2sin_20261008\`（如需恢复, 把备份里的
> `artifacts/` 与 `empty_driverlib_main.c` 拷回即可）。

## 工程内容

| 文件 | 说明 |
|---|---|
| `artifacts/mod.a` + `artifacts/tvmgen_default.h` | charge_mtl 模型 NPU 库（TI QAT 15ep, int8; PWCONV ×2 上 NPU; ≈16.9 KB） |
| `empty_driverlib_main.c` | 板端验证主程序（4 类波形轮换 → NPU 推理 → C 端后处理 → VOFA 输出） |
| `lnk.cmd` | 已含 NPU 段配置：`.rodata.tvm → FLASH_BANK0`, `.bss.noinit.tvm → RAMGS3` |
| `hand_infer.c/h`、`quant_params_mcu.h` 等 | saw2sin 遗留文件（仍参与编译, 新主程序不调用；如要精简可手动从工程排除） |

## 使用方法

1. CCS 打开工程 → **Build** → **下载（烧录）**
2. XDS110 虚拟串口（115200-8N1）接 VOFA+（JustFloat 协议）
3. 开机先跑 golden 自检，然后主循环持续输出（~35 帧/秒）

## VOFA+ 通道表（82 通道）

| 通道 | 内容 |
|---|---|
| CH0..31 | 输入窗（32 点，板内生成：4 类波形轮换） |
| CH32..36 | 分类 logits（5, 部署版值 = 已 ÷128） |
| CH37..42 | 参数原始输出：A', k', cosφ', sinφ', offset', duty'（[0,1] 归一化） |
| CH43 | 分类结果（0=sine 1=square 2=triangle 3=saw 4=noise） |
| CH44 | 置信度（softmax） |
| CH45 | 推理耗时 [us] |
| CH46..77 | **重建正弦 32 点**（C 端用参数合成 = "任意波形 → 参数 → 等效正弦"落地演示） |
| CH78 | golden 自检最大偏差（期望 < 0.05；若 ≈1 说明 NPU 未执行） || CH79..81 | 推理耗时统计：min / avg / max [us]（开机以来累计） |

> 耗时口径：CH45/79/80/81 = `tvmgen_default_run()` 调用 + 等待 NPU 完成的总微秒数
> （含图内 CPU 回退算子；不含 C 端后处理）。
## 板端自检怎么看

- **CH78（golden 偏差）**：应 < 0.05 量级（板端 NPU 与 PC int8 模拟一致 → 模型搬运正确）。
  若 ≈ 1.0 或更大 → 检查：链接段（`.rodata.tvm`/`.bss.noinit.tvm`）、`TI_NPU_init()` 调用、
  `TVMGEN_DEFAULT_TI_NPU` 宏（没有则编译成了 CPU 软库）。
- **CH46..77 vs CH0..31**：重建正弦应大致穿过输入波形（同相）；方波帧能看到"方波 → 正弦"的形态。
- **CH45（耗时）**：NPU 推理微秒数（与 PC 端无关, 以板测为准）。
- 全局变量 `g_golden_maxerr / g_cls_pred / g_conf / g_us_last / g_frame_count` 可在 CCS Expressions 里 Watch。

## 模型来源与重新生成

- 模型工程：`E:\desk\DeepStudy\Charge_Model`（验收报告 docs/06；部署专项 docs/07 §3）
- 部署工具链：`E:\desk\ti-npu\saw2sin_npu\examples\charge_mtl\`（adapter + job_charge_mtl.yaml, QAT）
- 重新生成：`E:\anaconda\envs\ti-npu\python.exe -m torch2tinpu convert/compile/deploy examples\charge_mtl\job_charge_mtl.yaml`
- golden 再生：`E:\anaconda\envs\ti-npu\python.exe _golden_charge_mtl.py`（在 examples\charge_mtl 下）
