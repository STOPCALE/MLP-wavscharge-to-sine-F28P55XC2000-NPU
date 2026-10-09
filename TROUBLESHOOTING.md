# TROUBLESHOOTING — 工程构建与识别问题排查档案

> 工程: `E:\desk\CCS\npu_and_charge`（charge_mtl 板端验证工程）
> 环境: CCS 21.0.1（Theia 版，无 eclipsec.exe）；编译器 ti-cgt-c2000_25.11.1.LTS；C2000Ware 26.1.0.00
> 归档: 2026-10-08（2026-10-09 增补问题 5）

---

## 问题 1：CCS 中"工程不能被识别"（导入失败 / 出现两个同名工程）

**现象**：在 CCS 里导入 `E:\desk\CCS\npu_and_charge` 时失败或困惑；工程列表中出现两个都叫 `uart_and_npu` 的工程。

**根因**：CCS 用 `.project` 文件里的 `<name>` 作为工程内部名（不是目录名）。
- 目录 `npu_and_charge` 的内部名原为 `uart_and_npu`
- 目录 `uart_and_npu`（旧 saw2sin 工程）的内部名也是 `uart_and_npu`
- → 两个同名工程无法导入到同一个 workspace（第二个被拒），且显示名与目录名不符造成混淆

**修复**：把 `npu_and_charge\.project` 的 `<name>uart_and_npu</name>` 改为 `<name>npu_and_charge</name>`（与目录一致）。

**经验**：以后复制工程目录做新版本时，记得同步改 `.project` 内部名，避免和旧工程同名。
（还不够——改名只是**前提**；复制完还必须把工程"导入工作区"，缺这步工程不会出现，见问题 5。）

---

## 问题 2：链接失败 — unresolved symbol `TI_NPU_init`（工程"不能编译"）

**现象**：

```
undefined   first referenced
 symbol         in file
 ---------   ----------------
 TI_NPU_init ./empty_driverlib_main.obj
error #10234-D: unresolved symbols remain
error #10010: errors encountered during linking; "uart_and_npu.out" not built
```

**排查过程**（用 `nm2000 -g` 对比符号表）：
| 库 | 是否含 `TI_NPU_init` |
|---|---|
| 新版 charge_mtl `artifacts/mod.a`（modelmaker 25.x 编译） | ❌ 无 |
| C2000Ware 官方 `arc_fault` 例程老版 `mod.a` | ✅ 有（`T TI_NPU_init`） |
| 旧 saw2sin 工程 `mod.a` | ❌ 无（且其主程序没调用它 → 构建成功） |

**根因**：旧教程/模板（基于老版工具链）在主程序里调用 `TI_NPU_init()` 初始化 NPU；
**新版 modelmaker 的运行库已在内部完成 NPU 初始化**，不再导出该符号
（`tvmgen_default.h` 里仍保留 `extern void TI_NPU_init();` 声明，属历史遗留，仅供手动实现时参考）。

**修复**：删除主程序 `main()` 里的 `TI_NPU_init();` 调用（已在原位留注释说明原因）。

**验证**：重新构建 → **0 errors**，`npu_and_charge.out` 成功生成。

---

## 问题 3：targetConfigs 缺少 ccxml（影响烧录/调试，不影响编译）

**现象**：工程 `targetConfigs\` 目录里只剩 `readme.txt`，缺
`TMS320F28P550SJ9.ccxml` 与 `TMS320F28P550SJ9_LaunchPad.ccxml`。
（排查发现旧工程曾把这两个文件误移到 `uart_and_npu\targetConfigs - 副本\` 目录。）

**修复**：从 `E:\desk\CCS\empty_driverlib_project\targetConfigs\`
（或 C2000Ware `device_support\f28p55x\common\targetConfigs\`）拷回两个 ccxml。

**说明**：ccxml 是调试器目标配置（XDS110 / 芯片选择），**编译不需要**，
但 CCS GUI 烧录/调试时必需。

---

## 问题 4：构建日志中的 advice / warning（2026-10-08 构建时）

**现象**：

```
"../empty_driverlib_main.c", line 102: advice #2614-D: Use --fp_mode=relaxed to enable TMU hardware support for FP division.
"../empty_driverlib_main.c", line 110: advice #2614-D: ... sinf.
"../empty_driverlib_main.c", line 168: advice #2614-D: ... atan2f.
"../empty_driverlib_main.c", line 234: warning #179-D: variable "k" was declared but never referenced
```

**处理**：
1. **启用 `--fp_mode=relaxed`**（4 个构建配置全部）：除法 / `sinf` / `atan2f` 改走 **TMU 硬件加速**。
   - 精度影响可忽略（TMU 三角函数精度远高于应用需求；golden 自检比对的是 NPU 输出，不经过 sinf）
   - 收益：C 端"参数 → 重建正弦"后处理提速
   - 设置方式：`projectModify -ccs.setCompilerOptions "--fp_mode=relaxed"`（见附录 A）
2. **删除未用变量 `k`**。
3. **排除死代码 `hand_infer.c` / `hand_infer.h`**（saw2sin 时代手写推理残件，当前主程序不引用；
   已在 `.cproject` 的 4 个配置中排除编译，文件本身保留在磁盘上备查）。

**另注**：构建日志末尾的 `echo "========= Build of the CLB simulation..."` 与三行 `if 0 == 1 ...`
是 TI 模板自带的 postbuild 脚本（与 CLB 工具相关），非错误，可忽略。

---

## 问题 5：复制的新工程"不被工作区识别"（Projects 视图里不出现）

**现象**（2026-10-09，`E:\desk\CCS\cnn_and_elect` / `cnn_and_number`）：
从 `npu_and_charge` 复制出两个新工程，按问题 1 改好 `.project` 内部名、重启 CCS 多次，
工程仍然不出现在工作区（Projects 视图）里。

**根因**：CCS 工作区**只显示"注册进工作区"的工程**，注册信息由后端（Eclipse 服务）维护在元数据目录：

```
C:\Users\HP\AppData\Local\Texas Instruments\CCS\.ccs-server\workspaces\<哈希>\.metadata\.plugins\org.eclipse.core.resources\.projects\
```

- **复制目录 ≠ 导入工程**；**重启 CCS 不会扫描目录**（重启只是让后端重读注册表）
- 问题 1 的"改 `.project` 内部名"只是**让导入能成功的前提条件**（避免同名冲突），不是识别机制本身
- 排查实测：主工作区注册表里当时只有 5 个旧工程（empty_driverlib_project / empty_project /
  npu_and_charge / uart_and_npu / uart_text），无任何 `cnn_*`；后端日志中也只有"复制→改名→重启"，
  无任何导入记录 → 结论：**这两个工程从未执行过"导入"**

**修复**：在 CCS 里对每个新工程手动执行一次"导入"：

1. 左侧活动栏 → **CCStudio** → **Projects** 视图
2. 视图标题栏点 **Import Project**（📁＋ 图标；命令面板搜 `CCStudio: Import Project` 亦可）
3. 弹出 "Select folder containing CCStudio projects" → 选 `E:\desk\CCS`
4. 弹出 "Select projects to import" 列表 → 勾选要导入的工程 → 确认
5. 工程出现在 Projects 视图后，即可正常 Build / Debug

**一句话流程**：复制工程做新版本 = **① 改 `.project` 内部名 → ② 导入工作区**（两个动作缺一不可）。

**配套排查技巧**（2026-10-09 实测有效）：

```powershell
# 列出每套后端工作区里"已注册"的工程名（判断某工程有没有被导入过）
$b='C:\Users\HP\AppData\Local\Texas Instruments\CCS\.ccs-server\workspaces'
Get-ChildItem $b -Directory | ForEach-Object { $p="$($_.FullName)\.metadata\.plugins\org.eclipse.core.resources\.projects"; if (Test-Path $p) { "### $($_.Name)"; Get-ChildItem $p | Select-Object -ExpandProperty Name } }
```

> 每套 Theia 工作区各占一个 `<哈希>` 子目录；**`.metadata` 最近被修改的那个 = 你正在用的工作区**。

```bat
:: 用命令行把工程导入【临时工作区】做体检（不碰 GUI 工作区；用完删掉 C:\temp\ccs_diag 即可）
cmd /c '"E:\ccs\ccs\eclipse\ccs-server-cli.bat" -workspace "C:\temp\ccs_diag" -application projectImport -ccs.location "E:\desk\CCS\cnn_and_elect"'
```

成功时输出（2026-10-09 实测通过）：

```
Project details:
    Name: cnn_and_elect
    Target: TMS320F28P550SJ9
    Location: E:/desk/CCS/cnn_and_elect
```

> ⚠️ 若要用命令行导入**正在使用的 GUI 工作区**，需先完全退出 CCS（工作区被运行中的后端锁定）。

---

## 附录 A：命令行无头构建（CCS 21 Theia 版）

> ⚠️ CCS 21.0.1 无 `eclipsec.exe`；使用 `E:\ccs\ccs\eclipse\ccs-server-cli.bat`。
> ⚠️ **必须用 `cmd /c '...'` 包裹调用**——PowerShell 直接调用 .bat 会丢参数、只打印 usage。

```bat
:: 导入工程（导入到工作区, 不复制文件）
cmd /c '"E:\ccs\ccs\eclipse\ccs-server-cli.bat" -workspace "C:\temp\ccs_ws" -application projectImport -ccs.location "E:\desk\CCS\npu_and_charge"'

:: 完整构建 + 列出问题
cmd /c '"E:\ccs\ccs\eclipse\ccs-server-cli.bat" -workspace "C:\temp\ccs_ws" -application projectBuild -ccs.projects npu_and_charge -ccs.configuration CPU1_LAUNCHXL_FLASH -ccs.buildType full -ccs.listProblems'

:: 修改工程编译选项（例: 启用 TMU relaxed 模式; 不带 @configurations 则应用到全部配置）
cmd /c '"E:\ccs\ccs\eclipse\ccs-server-cli.bat" -workspace "C:\temp\ccs_ws" -application projectModify -ccs.location "E:\desk\CCS\npu_and_charge" -ccs.setCompilerOptions "--fp_mode=relaxed"'
```

---

## 附录 B：构建配置与产物对照

| 配置 | 用途 | 链接脚本 | 调试目标配置(ccxml) | 产物 |
|---|---|---|---|---|
| `CPU1_LAUNCHXL_FLASH` | **LaunchPad 开发板**（XDS110, 推荐） | 28p55x_generic_flash_lnk.cmd | TMS320F28P550SJ9_LaunchPad.ccxml | `CPU1_LAUNCHXL_FLASH\npu_and_charge.out` |
| `CPU1_FLASH` | 独立板 / 量产固件 | 28p55x_generic_flash_lnk.cmd | TMS320F28P550SJ9.ccxml | `CPU1_FLASH\npu_and_charge.out` |
| `CPU1_LAUNCHXL_RAM` | RAM 调试（掉电即失） | 28p55x_generic_ram_lnk.cmd | TMS320F28P550SJ9_LaunchPad.ccxml | `CPU1_LAUNCHXL_RAM\npu_and_charge.out` |
| `CPU1_RAM` | RAM 调试 | 28p55x_generic_ram_lnk.cmd | TMS320F28P550SJ9.ccxml | `CPU1_RAM\npu_and_charge.out` |

**NPU 模型数据链接段**（两个 lnk.cmd 均已含；缺了板端 golden 自检会失败）：

```
.rodata.tvm      : > FLASH_BANK0, ALIGN(8)   /* 模型常量 -> Flash */
.bss.noinit.tvm  : > RAMGS3                  /* NPU 数据 -> 全局共享 RAM (必须) */
```

**mod.a 会被 CCS 自动发现并加入链接**（无需手工配置；构建命令行末尾可见 `../artifacts/mod.a`）。

---

## 附录 C：板端验证要点（烧录后看 VOFA+）

- **CH78** golden 自检最大偏差：**< 0.05** 为通过（QAT 模型期望 ~3e-2 量级）
- **CH45** 单次推理耗时 [us]；**CH79/80/81** = min / avg / max（开机以来累计）
- **CH43/44** 分类结果（0=sine 1=square 2=triangle 3=saw 4=noise）与置信度
- **CH46..77** C 端用参数重建的正弦（应与 CH0..31 输入窗 "同频同相"）
- 串口：SCIA @ 115200（XDS110 虚拟串口，VOFA+ JustFloat 帧，82 通道）
