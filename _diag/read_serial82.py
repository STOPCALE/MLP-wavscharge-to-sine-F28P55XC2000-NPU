# -*- coding: utf-8 -*-
"""
read_serial82.py -- AI 诊断工具：直接读取 charge_mtl 板端 82ch JustFloat 帧
================================================================================
用途: 不经过 VOFA, AI 直接自己读串口验证板端数据（波形/耗时/golden 自检）。
帧格式: 82 个 float32 (小端) + 帧尾 00 00 80 7F (共 332 字节/帧)

通道布局 (charge_mtl npu_and_charge 固件):
  CH0..31   输入窗 (32 点, 4 类波形轮换: sine/square/triangle/saw)
  CH32..36  logits (5, 部署版已 ÷128)
  CH37..42  参数原始输出: A',k',cosφ',sinφ',offset',duty'
  CH43      分类结果 (0..4)      CH44 置信度
  CH45      本次推理耗时 us       CH46..77 重建正弦 (32 点)
  CH78      golden 自检最大偏差   CH79..81 耗时 min/avg/max

[WARN] 串口独占：运行前请关闭 VOFA+ 等串口工具!
用法:
  E:\anaconda\envs\torch\python.exe read_serial82.py
  E:\anaconda\envs\torch\python.exe read_serial82.py --seconds 6 --port COM6
"""
import argparse
import os
import struct
import sys
import time

import numpy as np

TAIL = b"\x00\x00\x80\x7f"
CLASS_NAMES = ["sine", "square", "triangle", "saw"]
NCH_EXPECT = 82


def parse_frames(buf, nch):
    """用等差数列滤掉误检帧尾, 切出所有完整帧 (复刻 read_vofa.py 的稳健逻辑)"""
    fl = nch * 4 + 4
    tails = []
    i = buf.find(TAIL)
    while i != -1:
        tails.append(i)
        i = buf.find(TAIL, i + 1)
    if not tails:
        return [], fl
    best, run = [], [tails[0]]
    for t in tails[1:]:
        d = t - run[-1]
        if d == fl:
            run.append(t)
        elif d > fl:
            if len(run) > len(best):
                best = run
            run = [t]
    if len(run) > len(best):
        best = run
    frames = []
    for t in best:
        if t - nch * 4 < 0:
            continue
        frames.append(struct.unpack_from("<%df" % nch, buf, t - nch * 4))
    return frames, fl


def try_detect_nch(buf):
    """按候选通道数尝试解析, 返回 (nch, nframes) —— 用于判断板上跑的哪版固件"""
    cands = [82, 36, 21, 33, 80]
    out = []
    for nch in cands:
        fr, _ = parse_frames(buf, nch)
        out.append((nch, len(fr)))
    return out


def main():
    ap = argparse.ArgumentParser(description="charge_mtl 82ch 串口诊断")
    ap.add_argument("--port", default="COM6")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args()

    import serial
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except Exception as e:
        print("[FAIL] 打不开 %s: %s" % (args.port, e))
        print("       -> VOFA+/其他串口工具可能正占用该口, 请先关闭。")
        sys.exit(2)

    print("[..] 打开 %s @ %d, 读取 %.1f 秒 ..." % (args.port, args.baud, args.seconds))
    t0 = time.time()
    buf = bytearray()
    while time.time() - t0 < args.seconds:
        buf += ser.read(4096)
    ser.close()
    buf = bytes(buf)
    print("[OK] 共收到 %d 字节 (~%.1f KB/s)" % (len(buf), len(buf) / args.seconds / 1024.0))

    frames, fl = parse_frames(buf, NCH_EXPECT)
    if len(frames) < 4:
        print("[WARN] 按 82ch 只解析出 %d 帧; 尝试其他通道数 ..." % len(frames))
        cand = try_detect_nch(buf)
        print("       候选 (nch, nframes):", cand)
        print("       前 64 字节 hex:", buf[:64].hex(" "))
        print("       -> 若确实无帧, 检查板端程序是否在跑 / 波特率是否匹配。")
        sys.exit(3)

    F = np.array(frames, dtype=np.float64)   # (nframes, 82)
    nf = F.shape[0]
    fps = nf / args.seconds
    print("[OK] 解析出 %d 帧完整数据 (%.1f 帧/s, 帧长 %d 字节)" % (nf, fps, fl))

    # ---- 关键通道统计 ----
    cls = F[:, 43]
    conf = F[:, 44]
    us = F[:, 45]
    gold = F[:, 78]
    umin, uavg, umax = F[-1, 79], F[-1, 80], F[-1, 81]
    print("---- 关键通道 -------------------------------")
    print("CH43 分类分布: " + ", ".join(
        "%s:%d" % (CLASS_NAMES[i], int(np.sum(np.round(cls) == i))) for i in range(4))
        + ", noise:%d" % int(np.sum(np.round(cls) == 4)))
    print("CH44 conf   : mean=%.4f min=%.4f" % (conf.mean(), conf.min()))
    print("CH45 us/次  : mean=%.2f min=%.2f max=%.2f" % (us.mean(), us.min(), us.max()))
    print("CH78 golden : %.5f  (期望 <0.05; -1=未执行)" % gold[-1])
    print("CH79-81     : min=%.2f avg=%.2f max=%.2f us" % (umin, uavg, umax))
    p = F[-1, 37:43]
    print("CH37-42 参数: A'=%.3f k'=%.3f cos=%.3f sin=%.3f off=%.3f duty'=%.3f" % tuple(p))
    print("CH32-36 logits:", np.round(F[-1, 32:37], 3))

    # ---- 先把原始数据落盘 (绘图之前, 保证数据安全) ----
    fc = os.path.join(args.outdir, "diag_frames.csv")
    np.savetxt(fc, F, delimiter=",", fmt="%.6g")
    np.savez(os.path.join(args.outdir, "diag_frames.npz"), F=F)
    print("[OK] 全帧数据 ->", fc, " shape=", F.shape)

    # ---- 逐类别取帧 (统计数据) ----
    stats = []
    for c in range(4):
        idx = np.where(np.round(cls) == c)[0]
        stats.append((CLASS_NAMES[c], F[idx[len(idx) // 2]] if len(idx) else None))

    print("---- 波形形状速览 (相邻点平均差分: 大=跳变/尖峰, 小=平滑) ----")
    for name, f in stats:
        if f is None:
            print("  %-8s: (无该类别帧)" % name)
            continue
        win, rec = f[0:32], f[46:78]
        print("  %-8s: 窗 mean|d|=%.3f range[%.2f,%.2f] | 重建 mean|d|=%.3f range[%.2f,%.2f]"
              % (name, np.abs(np.diff(win)).mean(), win.min(), win.max(),
                 np.abs(np.diff(rec)).mean(), rec.min(), rec.max()))

    # ---- 绘图提示 (绘图交给 plot_report.py; torch 环境的 matplotlib 在本机有原生崩溃) ----
    print("[i] 需要出图请用 ti-npu 环境运行同目录 plot_report.py")


if __name__ == "__main__":
    main()
