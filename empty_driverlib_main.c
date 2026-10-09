/* ============================================================================
 *  empty_driverlib_main.c —— charge_mtl 模型 F28P55 NPU 板端验证程序
 *
 *  模型: 波形多任务网络
 *        分类 5 类: sine / square / triangle / saw / noise
 *        回归 5 参数: A(幅值) k(窗内周期数) φ(相位) offset(偏置) duty(占空比)
 *        输入 32 点窗 (1,32,1,1) -> 输出 11 维 (1,11,1,1) = [logits×5 | 参数×6]
 *        参数布局 (sincos 相位表示): A', k', cosφ', sinφ', offset', duty' (均为 [0,1] 归一化输出)
 *        注意: 部署版 logits 已整体 ÷128 (数值工程: 防无界 logits 夺占输出量化 scale,
 *              argmax 不受影响; C 端若用 softmax 置信度需 ×128 恢复 -> 见 softmax_conf)
 *        NPU 库: artifacts/mod.a (TI QAT 15 epoch, int8; PC 实测 k_rel 0.098 vs float 0.097)
 *
 *  功能 (对应板端验证):
 *   1. 板内生成 32 点测试窗 (滑窗模式: 每类保持 ~7s, 帧间窗滑动 1/32 窗, VOFA 上呈连续波形)
 *   2. NPU 推理 + CPUTimer1 (150MHz 递减) 计时
 *   3. C 端后处理演示: argmax 分类 / softmax 置信度 / sincos 反归一化 / 参数合成正弦
 *   4. 开机 golden 自检: 4 个 PC int8 推理样本逐值对比 -> g_golden_maxerr
 *
 *  VOFA+ JustFloat 帧 (82 通道, 帧尾 00 00 80 7F):
 *     CH0..31   输入窗 (32 点, 长信号滑窗: 每帧整体滑动 1 个样本点)
 *     CH32..36  logits (5, 部署版值)
 *     CH37..42  参数原始输出: A',k',cosφ',sinφ',offset',duty'
 *     CH43      分类结果 (0=sine 1=square 2=triangle 3=saw 4=noise)
 *     CH44      置信度 (softmax)
 *     CH45      本次推理耗时 [us]
 *     CH46..77  重建正弦 32 点 (C 端用参数合成 = "参数->正弦" 落地演示)
 *     CH78      golden 自检最大偏差 (QAT 后期望 ~3e-2 量级)
 *     CH79..81  推理耗时统计: min / avg / max [us] (开机以来累计)
 *
 *  耗时口径说明: CH45/79/80/81 = tvmgen_default_run() 调用 + 等待 NPU 完成的总微秒数
 *  (含图内 CPU 回退算子; 不含本文件的 C 端后处理) —— NPU 子图硬耗时看 avg/min 即可
 * ==========================================================================*/
#include "board.h"
#include "device.h"
#include "driverlib.h"
#include <stdint.h>
#include <math.h>

#include "artifacts/tvmgen_default.h"      /* NPU 模型接口 (mod.a 提供) */

#define N_WIN       32
#define N_OUT       11
#define SYSCLK_MHZ  150.0f

/* 参数范围 (与 PC 端 config.py 一致; C 端反归一化) */
#define A_LO        0.2f
#define A_SPAN      0.8f
#define K_LO        0.5f
#define K_SPAN      3.5f
#define OFF_LO      (-0.3f)
#define OFF_SPAN    0.6f
#define DUTY_LO     0.2f
#define DUTY_SPAN   0.6f
#define TWO_PI_F    6.28318530718f
#define PI_F        3.14159265359f

/* ---------------------------------------------------------------------------
 * golden 数组: 4 个标准样本的 PC int8 推理结果 (由 _golden_charge_mtl.py 生成)
 * 用于开机自检 "板端 NPU 输出 vs PC int8 模拟输出" 的一致性
 * -------------------------------------------------------------------------*/
static const float GOLD_IN[4][N_WIN] = {
  {+0.773177f, +0.887346f, +0.881649f, +0.756953f, +0.532242f, +0.241726f, -0.070367f, -0.356522f, -0.573177f, -0.687346f, -0.681649f, -0.556953f, -0.332242f, -0.041726f, +0.270367f, +0.556522f, +0.773177f, +0.887346f, +0.881649f, +0.756953f, +0.532242f, +0.241726f, -0.070367f, -0.356522f, -0.573177f, -0.687346f, -0.681649f, -0.556953f, -0.332242f, -0.041726f, +0.270367f, +0.556522f},
  {+0.700000f, +0.700000f, +0.700000f, -0.700000f, -0.700000f, -0.700000f, -0.700000f, -0.700000f, -0.700000f, -0.700000f, -0.700000f, -0.700000f, +0.700000f, +0.700000f, +0.700000f, +0.700000f, -0.700000f, -0.700000f, -0.700000f, -0.700000f, -0.700000f, -0.700000f, -0.700000f, -0.700000f, -0.700000f, +0.700000f, +0.700000f, +0.700000f, +0.700000f, -0.700000f, -0.700000f, -0.700000f},
  {+0.604084f, +0.435334f, +0.266584f, +0.097834f, -0.070916f, -0.239666f, -0.408416f, -0.577166f, -0.745916f, -0.914666f, -0.816584f, -0.647834f, -0.479084f, -0.310334f, -0.141584f, +0.027166f, +0.195916f, +0.364666f, +0.533416f, +0.702166f, +0.829084f, +0.660334f, +0.491584f, +0.322834f, +0.154084f, -0.014666f, -0.183416f, -0.352166f, -0.520916f, -0.689666f, -0.858416f, -0.872834f},
  {-0.297211f, -0.184711f, -0.072211f, +0.040289f, +0.152789f, +0.265289f, +0.377789f, +0.490289f, +0.602789f, +0.715289f, -0.372211f, -0.259711f, -0.147211f, -0.034711f, +0.077789f, +0.190289f, +0.302789f, +0.415289f, +0.527789f, +0.640289f, -0.447211f, -0.334711f, -0.222211f, -0.109711f, +0.002789f, +0.115289f, +0.227789f, +0.340289f, +0.452789f, +0.565289f, +0.677789f, -0.409711f},
};
static const float GOLD_OUT[4][N_OUT] = {
  {+0.156250f, -0.093750f, +0.062500f, -0.531250f, -0.687500f, +0.718750f, +0.406250f, +0.781250f, +1.015625f, +0.671875f, +0.593750f},
  {-0.843750f, +0.296875f, -0.046875f, +0.093750f, -0.203125f, +0.578125f, +0.500000f, +1.093750f, +0.796875f, +0.343750f, +0.250000f},
  {+0.015625f, -0.062500f, +0.062500f, -0.328125f, -0.453125f, +0.859375f, +0.250000f, +0.234375f, +1.046875f, +0.484375f, +0.296875f},
  {-0.671875f, +0.140625f, -0.031250f, +0.234375f, +0.093750f, +0.578125f, +0.781250f, +0.984375f, +0.781250f, +0.593750f, +0.562500f},
};

/* 4 组测试参数 (与 _golden_charge_mtl.py 的 CASES 完全一致) */
/*         cid 0=sine   1=square  2=triangle  3=saw      */
static const float CASE_A[4]    = {0.8f,  0.7f,  0.9f,  0.6f };
static const float CASE_K[4]    = {2.0f,  2.5f,  1.5f,  3.0f };
static const float CASE_PHI[4]  = {1.0f,  0.5f,  2.0f,  0.8f };
static const float CASE_OFF[4]  = {0.10f, 0.00f, -0.05f, 0.15f};
static const float CASE_DUTY[4] = {0.35f, 0.30f, 0.35f, 0.35f};

#define CLASS_HOLD_FRAMES  256u    /* 滑窗模式: 每类波形保持帧数 (~7s @35帧/s) */

/* ------------------- 全局变量 (CCS Expressions 窗口可 Watch) ------------------- */
static float g_in[N_WIN];                   /* 推理输入缓冲 (32 点)   */
static float g_out[N_OUT];                  /* 推理输出缓冲 (11 维)   */
static float g_recon[N_WIN];                /* C 端重建正弦 (32 点)   */
volatile uint32_t g_frame_count = 0;        /* 已推理帧数             */
volatile float    g_us_last     = 0.0f;     /* 最近一次推理耗时 us     */
volatile float    g_us_min      = 1.0e9f;   /* 推理耗时统计 (us)       */
volatile float    g_us_max      = 0.0f;
volatile float    g_us_avg      = 0.0f;
static   float    g_us_sum      = 0.0f;     /* 累计和 (avg = sum/n)    */
volatile float    g_cls_pred    = 0.0f;     /* 最近分类结果 (0~4)     */
volatile float    g_conf        = 0.0f;     /* 最近 softmax 置信度     */
volatile float    g_golden_maxerr = -1.0f;  /* golden 自检最大偏差     */

static struct tvmgen_default_inputs  g_npu_in;    /* 模型输入结构 (含指针) */
static struct tvmgen_default_outputs g_npu_out;   /* 模型输出结构 (含指针) */

/* ------------------- 板端波形生成 (与 PC 端 wavegen.render 同公式) -------------------
 * 直接以相位 th 求值; 滑窗模式下 th = theta + 2π·k·n/32 (theta 逐帧推进) */
static float shape_at(int cid, float A, float off, float duty, float th)
{
    float fra = th / TWO_PI_F;
    float w;

    fra = fra - floorf(fra);                       /* frac(θ/2π) ∈ [0,1) */
    switch (cid)
    {
        case 0:  w = sinf(th); break;                              /* sine     */
        case 1:  w = (fra < duty) ? 1.0f : -1.0f; break;           /* square   */
        case 2:  w = (2.0f / PI_F) * asinf(sinf(th)); break;       /* triangle */
        default: w = 2.0f * fra - 1.0f; break;                     /* saw      */
    }
    return A * w + off;
}

/* NPU 推理一次; 返回耗时 (us) */
static float npu_infer(void)
{
    uint32_t t0 = CPUTimer_getTimerCount(CPUTIMER1_BASE);
    (void)tvmgen_default_run(&g_npu_in, &g_npu_out);
    while (!tvmgen_default_finished)
    {
        /* 轮询等待 NPU 子图完成 (完成标志由 NPU 中断服务程序置位) */
    }
    uint32_t t1 = CPUTimer_getTimerCount(CPUTIMER1_BASE);
    return (float)(t0 - t1) / SYSCLK_MHZ;          /* 递减计数: 差值 = 经过周期数 */
}

/* ------------------- C 端后处理 ------------------- */
static int argmax5(const float *v)
{
    int i, bi = 0;
    for (i = 1; i < 5; i++)
    {
        if (v[i] > v[bi]) { bi = i; }
    }
    return bi;
}

static float softmax_conf(const float *logits)
{
    float e[5], sum = 0.0f, m = logits[0];
    int i, bi;
    for (i = 1; i < 5; i++)
    {
        if (logits[i] > m) { m = logits[i]; }
    }
    for (i = 0; i < 5; i++)
    {
        e[i] = expf((logits[i] - m) * 128.0f);     /* ×128: 恢复真实 logits 的 softmax */
        sum += e[i];
    }
    bi = argmax5(logits);
    return e[bi] / sum;
}

/* sincos 参数反归一化: [0,1] 输出 -> 物理量 (与 wavegen.denormalize_params 一致) */
static void params_to_phys(const float *p, float *A, float *k, float *phi,
                           float *off, float *duty)
{
    float c, s, ph;
    *A    = p[0] * A_SPAN + A_LO;
    *k    = p[1] * K_SPAN + K_LO;
    c     = p[2] * 2.0f - 1.0f;                    /* cosφ' -> [-1,1] */
    s     = p[3] * 2.0f - 1.0f;                    /* sinφ' -> [-1,1] */
    ph    = atan2f(s, c);
    if (ph < 0.0f) { ph += TWO_PI_F; }
    *phi  = ph;
    *off  = p[4] * OFF_SPAN + OFF_LO;
    *duty = p[5] * DUTY_SPAN + DUTY_LO;
}

/* 用参数合成正弦 (C 端落地演示: 任意波形 -> 参数 -> 等效正弦) */
static float synth_sine(float A, float k, float phi, float off, int n)
{
    float t = (float)n / (float)N_WIN;
    return A * sinf(TWO_PI_F * k * t + phi) + off;
}

/* ------------------- golden 自检: 板端 NPU vs PC int8 ------------------- */
static float golden_check(void)
{
    float maxe = 0.0f;
    int s, i;
    for (s = 0; s < 4; s++)
    {
        for (i = 0; i < N_WIN; i++) { g_in[i] = GOLD_IN[s][i]; }
        npu_infer();
        for (i = 0; i < N_OUT; i++)
        {
            float d = fabsf(g_out[i] - GOLD_OUT[s][i]);
            if (d > maxe) { maxe = d; }
        }
    }
    return maxe;
}

/* ------------------- 计时器 / 串口 ------------------- */
/* CPUTimer1: 最大周期、/1 预分频、递减计数; 读取 t0-t1 = 经过周期数 */
static void CPUTimer1_init(void)
{
    CPUTimer_stopTimer(CPUTIMER1_BASE);
    CPUTimer_setPeriod(CPUTIMER1_BASE, 0xFFFFFFFFu);
    CPUTimer_setPreScaler(CPUTIMER1_BASE, 0u);
    CPUTimer_reloadTimerCounter(CPUTIMER1_BASE);
    CPUTimer_startTimer(CPUTIMER1_BASE);
}

/* 发送一个 float: 拆 4 字节小端 (VOFA JustFloat 基本单元) */
static void SCI_sendFloat(float v)
{
    union { float f; uint32_t u; } conv;
    conv.f = v;
    SCI_writeCharBlockingFIFO(mySCI0_BASE, (uint16_t)( conv.u        & 0xFFu));
    SCI_writeCharBlockingFIFO(mySCI0_BASE, (uint16_t)((conv.u >> 8)  & 0xFFu));
    SCI_writeCharBlockingFIFO(mySCI0_BASE, (uint16_t)((conv.u >> 16) & 0xFFu));
    SCI_writeCharBlockingFIFO(mySCI0_BASE, (uint16_t)((conv.u >> 24) & 0xFFu));
}

/* VOFA+ JustFloat 帧尾: 0x00 0x00 0x80 0x7F */
static void SCI_sendFrameTail(void)
{
    SCI_writeCharBlockingFIFO(mySCI0_BASE, 0x00u);
    SCI_writeCharBlockingFIFO(mySCI0_BASE, 0x00u);
    SCI_writeCharBlockingFIFO(mySCI0_BASE, 0x80u);
    SCI_writeCharBlockingFIFO(mySCI0_BASE, 0x7Fu);
}

/* ------------------- 主程序 ------------------- */
void main(void)
{
    int n, cid, cid_prev = -1;
    int cls_pred;
    float Ap, kp, phip, offp, dutyp, conf;
    float theta = 0.0f;                /* 长信号相位 (滑窗用), 每帧推进 */

    /* 1. 系统初始化 (与工程原有流程一致; device.c 已含 NPU 时钟使能) */
    Device_init();
    Interrupt_initModule();
    Interrupt_initVectorTable();
    Board_init();                       /* SysConfig 生成 (含 SCI_init: SCIA 115200-8N1) */
    EINT;
    ERTM;

    /* 2. 计时器 (上电一次) */
    CPUTimer1_init();
    /* 注意: 这里不要调用 TI_NPU_init()!
       新版 modelmaker 的 mod.a 运行库已在内部完成 NPU 初始化;
       头文件 tvmgen_default.h 里虽有 extern 声明, 但库中没有该符号的实现,
       显式调用会导致链接错误: error #10234-D unresolved symbol TI_NPU_init */

    /* 3. 绑定模型输入输出指针 */
    g_npu_in.input   = (void *)g_in;
    g_npu_out.output = (void *)g_out;

    /* 4. 留 0.5s 给 XDS110 虚拟串口枚举, 避免首批帧丢失 */
    DEVICE_DELAY_US(500000);

    /* 5. 开机 golden 自检 (结果看 g_golden_maxerr; VOFA CH78 也会发出) */
    g_golden_maxerr = golden_check();

    /* 6. 主循环: 生成窗口 -> NPU 推理 -> C 端后处理 -> 发送 VOFA 帧
     *   滑窗模式: 窗 = 长信号的连续片段, 每帧整体滑动 1 个样本点
     *   -> VOFA 时间轴上每个通道都呈现连续波形 (不再逐帧跳变) */
    while (1)
    {
        cid = (int)((g_frame_count / CLASS_HOLD_FRAMES) & 3u);  /* 4 类慢速轮换 */

        /* (a) 生成当前 32 点窗 (从连续相位 theta 起, 每帧滑动 1/32 窗) */
        {
            float dth = TWO_PI_F * CASE_K[cid] / (float)N_WIN;
            if (cid != cid_prev) { theta = CASE_PHI[cid]; cid_prev = cid; }
            for (n = 0; n < N_WIN; n++)
            {
                g_in[n] = shape_at(cid, CASE_A[cid], CASE_OFF[cid], CASE_DUTY[cid],
                                   theta + dth * (float)n);
            }
            theta += dth;
            if (theta >= TWO_PI_F) { theta -= TWO_PI_F; }   /* 限幅保浮点精度 */
        }

        /* (b) NPU 推理 + 计时 (含图内 CPU 回退算子; 不含 C 端后处理) */
        g_us_last = npu_infer();
        if (g_us_last < g_us_min) { g_us_min = g_us_last; }
        if (g_us_last > g_us_max) { g_us_max = g_us_last; }
        g_us_sum += g_us_last;
        g_us_avg  = g_us_sum / (float)(g_frame_count + 1u);

        /* (c) C 端后处理: 分类 / 置信度 / 反归一化 / 重建正弦 */
        cls_pred = argmax5(g_out);
        conf     = softmax_conf(g_out);
        /* 注意: g_out 布局 = [logits×5 | 参数×6], 参数从 g_out[5] 开始!
           (曾误传 g_out 导致 A/k/φ/off/duty 读到的是 logits -> 重建正弦错误) */
        params_to_phys(g_out + 5, &Ap, &kp, &phip, &offp, &dutyp);
        for (n = 0; n < N_WIN; n++)
        {
            g_recon[n] = synth_sine(Ap, kp, phip, offp, n);
        }
        g_cls_pred = (float)cls_pred;
        g_conf     = conf;

        /* (d) 发送一帧 (VOFA+ JustFloat, 82 通道; 铁律: 数据在帧尾之前) */
        for (n = 0; n < N_WIN; n++) { SCI_sendFloat(g_in[n]);  }   /* CH0..31  输入窗   */
        for (n = 0; n < 5;      n++) { SCI_sendFloat(g_out[n]); }   /* CH32..36 logits   */
        for (n = 5; n < N_OUT;  n++) { SCI_sendFloat(g_out[n]); }   /* CH37..42 参数原始 */
        SCI_sendFloat((float)cls_pred);                             /* CH43 分类结果     */
        SCI_sendFloat(conf);                                        /* CH44 置信度       */
        SCI_sendFloat(g_us_last);                                   /* CH45 耗时 us      */
        for (n = 0; n < N_WIN; n++) { SCI_sendFloat(g_recon[n]); }  /* CH46..77 重建正弦 */
        SCI_sendFloat(g_golden_maxerr);                             /* CH78 golden 偏差  */
        SCI_sendFloat(g_us_min);                                    /* CH79 耗时 min us  */
        SCI_sendFloat(g_us_avg);                                    /* CH80 耗时 avg us  */
        SCI_sendFloat(g_us_max);                                    /* CH81 耗时 max us  */
        SCI_sendFrameTail();

        g_frame_count++;
    }
}
