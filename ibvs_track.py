#!/usr/bin/env python3
"""手部追踪：腕部相机 + 图像视觉伺服（IBVS）。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

ROOT = HERE
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import follow_hand_v2 as fv  # noqa: E402  带入 SDK 路径 + Arm / Follow / HandReader

MOUNT = os.path.join(HERE, "mount.json")


def clamp_norm(v: np.ndarray, m: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n <= m or n < 1e-9 else v * (m / n)


def frame_jac(arm, q6: np.ndarray) -> np.ndarray:
    """末端平移雅可比（世界系），3×6。"""
    import pinocchio as pin

    arm.q_full[:6] = q6
    pin.forwardKinematics(arm.model, arm.data, arm.q_full)
    pin.updateFramePlacements(arm.model, arm.data)
    return pin.computeFrameJacobian(arm.model, arm.data, arm.q_full, arm.fid,
                                    pin.LOCAL_WORLD_ALIGNED)[:3, :6].copy()


def frame_rot(arm, q6: np.ndarray) -> np.ndarray:
    import pinocchio as pin

    arm.q_full[:6] = q6
    pin.forwardKinematics(arm.model, arm.data, arm.q_full)
    pin.updateFramePlacements(arm.model, arm.data)
    return arm.data.oMf[arm.fid].rotation.copy()


def feature_of(d: dict, cx: float, cy: float, k: float) -> np.ndarray:
    """(u, v, k·log size) 三特征；size 太小说明手没了。"""
    s = float(d.get("size") or 0.0)
    if s <= 1e-4:
        return np.array([np.nan, np.nan, np.nan])
    return np.array([float(d["u"]) - cx, float(d["v"]) - cy, k * float(np.log(s))])


#   位置-only、阻尼最小二乘、按 Gram 行列式自适应阻尼、每步限关节速度、
#   而且每拍都从"当前真实关节角"起步 —— 这就是它姿态不扭的原因。
NOMINAL_REACH = 0.56


def _pos_jac(arm, q6):
    arm.q_full[:6] = q6
    fv.pin.forwardKinematics(arm.model, arm.data, arm.q_full)
    fv.pin.updateFramePlacements(arm.model, arm.data)
    J = fv.pin.computeFrameJacobian(arm.model, arm.data, arm.q_full,
                                    arm.fid, fv.pin.LOCAL_WORLD_ALIGNED)
    return J[:3, :6].copy()


def clamp_workspace(p):
    p = np.array(p, dtype=float)
    p[2] = np.clip(p[2], 0.02, NOMINAL_REACH)
    r = float(np.hypot(p[0], p[1]))
    lim = max(0.03, NOMINAL_REACH * np.sqrt(max(0.0, 1.0 - (p[2] / NOMINAL_REACH) ** 2)))
    if r > lim and r > 1e-9:
        p[:2] *= lim / r
    return p


def official_ik(arm, p_tgt, q_seed, dt, iters=20, max_speed=2.8):
    q = np.clip(q_seed, arm.lo, arm.hi).copy()
    scale = float(np.clip(12.0 * dt, 0.08, 0.65))
    err = 0.0
    for _ in range(iters):
        p = arm.fk(q)
        e = p_tgt - p
        err = float(np.linalg.norm(e))
        if err < 1.5e-3:
            break
        J = _pos_jac(arm, q)
        G = J @ J.T
        sc = max(float(np.trace(G)) / 3.0, 1e-9)
        nd = max(0.0, float(np.linalg.det(G))) / max(sc ** 3, 1e-12)
        sing = float(np.clip(1.0 - nd / 0.08, 0.0, 1.0))
        damp = 0.018 + (0.075 - 0.018) * sing * sing
        try:
            dq = J.T @ np.linalg.solve(G + (damp ** 2) * np.eye(3), e * scale)
        except np.linalg.LinAlgError:
            break
        dq = np.clip(dq, -max_speed * dt, max_speed * dt)
        q = np.clip(q + dq, arm.lo, arm.hi)
    return q, err


class Grip:
    """手张开→夹爪张开，手握拳→夹爪闭合；只允许限速变化，不会突然夹。"""

    def __init__(self, node, ns: str, args) -> None:
        self.pub = node.create_publisher(
            fv.JointMitCmd, f"{ns}/gripper/cmd/mit", 1)
        self.pos = None
        self.args = args
        self.mn = None
        self.mx = None
        self.pinch = float('nan')

    def step(self, pinch: float, dt: float) -> None:
        a = self.args
        if self.pos is None:
            self.pos = a.grip_close          # 第一次就按闭合值起步，绝不会突然动
        # 自适应阈值：每个人"握拳/张开"时 pinch 的绝对值不同，用实测范围定，
        # 固定阈值会出现"只能打开、关不上"（握拳也降不到阈值以下）。
        if self.mn is None:
            self.mn = self.mx = pinch
        else:
            self.mn = min(self.mn, pinch)
            self.mx = max(self.mx, pinch)
            self.mn += (pinch - self.mn) * 0.0005      # 极慢回弹，手掌姿态变了也能跟上
            self.mx -= (self.mx - pinch) * 0.0005
        span = self.mx - self.mn
        self.pinch = pinch
        self.t = getattr(self, "t", 0.0) + dt
        if span < a.grip_min_span:
            # 还没学到开合幅度：头 1.5 秒按一个保守的绝对阈值走，
            # 免得"手一开始就是张开的"时候夹爪一直闭着不动。
            if self.t < 1.5 and pinch > 0.55:
                tgt = a.grip_open
            elif self.t < 1.5 and pinch < 0.30:
                tgt = a.grip_close
            else:
                tgt = self.pos
        else:
            hi = self.mn + 0.70 * span
            lo = self.mn + 0.30 * span
            if pinch > hi:
                tgt = a.grip_open
            elif pinch < lo:
                tgt = a.grip_close
            else:
                tgt = self.pos                        # 迟滞区保持
        dmax = a.grip_rate * dt
        self.pos = float(np.clip(tgt, self.pos - dmax, self.pos + dmax))
        m = fv.JointMitCmd()
        m.pos = self.pos
        m.vel = max(a.grip_rate, 0.2)
        self.pub.publish(m)


class Probe:
    """末端沿基座 x/y/z 各走 step 米，量特征变化 → Ĵ = Δs/Δp（3×3）。"""

    def __init__(self, arm, node, reader, k: float, step: float = 0.01, settle: float = 0.9):
        self.arm, self.node, self.reader = arm, node, reader
        self.k, self.step, self.settle = k, step, settle

    def wait_q(self, timeout: float = 2.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            fv.rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.node.q is not None:
                return self.node.q.copy()
        raise RuntimeError("收不到 joint_states —— 控制器起了吗？")

    def hold(self, q: np.ndarray, sec: float) -> None:
        t0 = time.time()
        while time.time() - t0 < sec:
            fv.rclpy.spin_once(self.node, timeout_sec=0.0)
            self.node.send(q, 0.3)
            time.sleep(1 / 60.0)

    def sample(self, cx, cy, timeout: float = 2.0) -> np.ndarray:
        t0 = time.time()
        while time.time() - t0 < timeout:
            fv.rclpy.spin_once(self.node, timeout_sec=0.0)
            d, ts = self.reader.get()
            if d and time.time() - ts < 0.3:
                f = feature_of(d, cx, cy, self.k)
                if np.isfinite(f).all():
                    return f
            time.sleep(0.02)
        raise RuntimeError("看不到手 —— 把手放到腕部相机前面再试")

    def sample_stable(self, cx, cy, n: int = 5, tol: float = 3.0, timeout: float = 4.0) -> np.ndarray:
        """等特征稳定下来再取（中位数）—— 手在动的时候采样会把手的运动算进雅可比里。"""
        t0 = time.time()
        buf: list[np.ndarray] = []
        while time.time() - t0 < timeout:
            fv.rclpy.spin_once(self.node, timeout_sec=0.0)
            d, ts = self.reader.get()
            if d and time.time() - ts < 0.3:
                f = feature_of(d, cx, cy, self.k)
                if np.isfinite(f).all():
                    if float(d.get("size", 0.0)) > 0.55:
                        raise RuntimeError(
                            f"手离相机太近（占画面 {d['size']*100:.0f}%），整个手都出画了 —— "
                            f"往后退到 0.6–0.9 m，让手占画面 25–40%")
                    buf.append(f)
                    buf = buf[-n:]
                    if len(buf) == n:
                        arr = np.array(buf)
                        if np.ptp(arr[:, :2], axis=0).max() <= tol:
                            return np.median(arr, axis=0)
            time.sleep(0.02)
        raise RuntimeError("看不到手（或手一直在动）—— 把手放稳在腕部相机前再试")

    def run(self, cx: float, cy: float) -> np.ndarray:
        q0 = self.wait_q()
        self.hold(q0, 0.4)
        self.sample_stable(cx, cy)
        J = np.zeros((3, 3))
        for k in range(3):
            qk, err = self.arm.ik(self.arm.fk(q0) + np.eye(3)[k] * self.step, q0, iters=60)
            if err > 2e-3:
                raise RuntimeError(f"探测第 {k} 轴时 IK 解不出来（误差 {err*1000:.0f} mm）")
            s0 = self.sample_stable(cx, cy)
            self.hold(qk, self.settle)
            s1 = self.sample_stable(cx, cy)
            self.hold(q0, self.settle)
            J[:, k] = (s1 - s0) / self.step  # px/m 与 1/m
            print(f"  探测 {('xyz'[k])}: Δu={s1[0]-s0[0]:+.1f}px  "
                  f"Δv={s1[1]-s0[1]:+.1f}px  Δ(k·log size)={s1[2]-s0[2]:+.2f}", flush=True)
        check_probe(J)
        return J


def check_probe(J: np.ndarray, min_angle: float = 20.0,
                lo: float = 8.0, hi: float = 8000.0) -> None:
    """探测结果自检：三列方向必须真的不一样，幅值也要像「相机装了 1cm」的样子。"""
    norms = np.linalg.norm(J, axis=0)
    ang = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        c = J[:, a] @ J[:, b] / max(norms[a] * norms[b], 1e-9)
        ang.append(float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))
    print(f"  探测自检：列长 {np.round(norms, 0)} px/m，列间夹角 {np.round(ang, 1)}°", flush=True)
    bad = []
    if min(ang) < min_angle:
        bad.append(f"三列几乎同向（最小夹角 {min(ang):.1f}° < {min_angle}°）")
    if norms.min() < lo or norms.max() > hi:
        bad.append(f"幅值异常（{norms.min():.0f}–{norms.max():.0f} px/m，正常 8–8000）")
    if bad:
        raise RuntimeError(
            "雅可比探测不可用：" + "；".join(bad) + "\n"
            "多半是探测那 3 秒里手动了。把手平放在夹爪前（离 0.3–0.5 m）别动，重跑一次。")


def prepare_jac(J: np.ndarray, min_ratio: float = 0.25) -> tuple[np.ndarray, bool]:
    """检查实测雅可比；「手掌大小」这一维不可观测就自动降成 2 维（只跟 u、v）。"""
    J = np.array(J, float)
    r = np.linalg.norm(J, axis=1)
    use_size = r[2] >= min_ratio * max(min(r[0], r[1]), 1e-9)
    J2 = J.copy()
    if not use_size:
        J2[2, :] = 0.0
    sv = np.linalg.svd(J2[:2, :], compute_uv=False)
    cond = sv[0] / max(sv[1], 1e-12)
    print(f"Ĵ 行范数 [u {r[0]:.0f}  v {r[1]:.0f}  size {r[2]:.0f}] px/m  "
          f"u-v 条件数 {cond:.1f}  → "
          f"{'3 维跟随（含手掌大小）' if use_size else '自动降为 2 维（只看 u、v）'}")
    if cond > 200 or sv[1] < 1e-3:
        raise RuntimeError(
            "雅可比不可用：手臂几乎没动，或手不在画面里。\n"
            "检查：腕部相机是不是对着手、手臂有没有使能、控制器在不在。")
    return J2, use_size


class Controller:
    def __init__(self, arm, gain: float, vmax: float, qdmax: float,
                 dt: float, jac_damp: float, adaptive: float):
        self.arm = arm
        self.gain = gain
        self.vmax = vmax
        self.qdmax = qdmax
        self.dt = dt
        self.jac_damp = jac_damp
        self.adaptive = adaptive
        self.J: np.ndarray | None = None
        self.q_ref: np.ndarray | None = None   # 参考姿态：零空间里往这儿拉
        self.k_null = 0.0     # 默认不介入；--posture 才打开
        self.es_dead = 0.0                     # 尺度误差死区（保持距离用）
        self._last_p: np.ndarray | None = None
        self._last_s: np.ndarray | None = None
        self.ignore_size = False   # True 时只跟 u、v（尺度维不可靠）

    def set_jac(self, J: np.ndarray) -> None:
        self.J = J.copy()

    def step(self, q: np.ndarray, s: np.ndarray, s_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (qd, v)。qd 单位 rad/s。"""
        e = s - s_star
        if self.ignore_size:
            e = e.copy()
            e[2] = 0.0
        if self.J is None:
            raise RuntimeError("还没有雅可比")
        # Broyden 在线修正：用上一拍的实测变化修 Ĵ
        if self.adaptive > 0 and self._last_p is not None:
            dp = self.arm.fk(q) - self._last_p
            dz = s - self._last_s
            n = float(dp @ dp)
            # 只在「臂确实动了一两厘米、而且特征也确实变了」时修正；
            # 否则（比如臂被卡住、特征没反应）Broyden 会把 Ĵ 修到 0，跟随就会悄悄停死。
            if 1e-8 < n < 1e-4 and float(np.linalg.norm(dz)) > 2.0:
                Jn = self.J + self.adaptive * np.outer(dz - self.J @ dp, dp) / n
                r0 = np.linalg.norm(self.J, axis=1)
                r1 = np.linalg.norm(Jn, axis=1)
                scale = np.clip(r1 / np.maximum(r0, 1e-9), 0.5, 2.0)
                self.J = Jn * scale[:, None]
        self._last_p = self.arm.fk(q)
        self._last_s = s.copy()

        # 尺度维加死区：手在「合适距离带」内就不参与控制，只剩 u、v 两维；
        # 出了带才把第三行放进来 —— 于是近了会退、远了会靠近。
        v2 = -self.gain * np.linalg.pinv(self.J[:2], rcond=1e-3) @ e[:2]
        if self.ignore_size or abs(e[2]) < self.es_dead:
            v = v2
        else:
            # 距离这一维只给 40% 话语权：2D 跟随始终占主导，
            # 否则一个大的尺度误差会把速度全占满，上下就"不跟了"。
            e3 = e.copy()
            e3[2] = float(np.clip(e3[2], -4 * self.es_dead, 4 * self.es_dead))
            v3 = -self.gain * np.linalg.pinv(self.J, rcond=1e-3) @ e3
            v = v2 + 0.4 * (v3 - v2)
        v = clamp_norm(v, self.vmax)
        Ja = frame_jac(self.arm, q)
        Ja_pinv = Ja.T @ np.linalg.inv(Ja @ Ja.T + self.jac_damp * np.eye(3))
        qd = Ja_pinv @ v
        # ★ 零空间姿态保持：6 关节只用了 3 个自由度，剩下 3 维没人管，
        #   关节积分会越走越歪（"姿态越来越怪异"、手一抬先往后退再抬起来）。
        #   把零空间分量拉回参考姿态，形状就稳住了。
        if self.q_ref is not None:
            # 只在"飘出参考姿态 0.15 rad 以上"时才往回拉；
            # 全程硬拉会把正常的跟随动作也吃掉（上下就不动了）。
            dref = self.q_ref - q
            excess = np.sign(dref) * np.maximum(np.abs(dref) - 0.15, 0.0)
            N = np.eye(len(q)) - Ja_pinv @ Ja
            qd = qd + N @ (self.k_null * excess)
        return np.clip(qd, -self.qdmax, self.qdmax), v


def run_live(args) -> int:
    fv.rclpy.init()
    arm = fv.Arm()
    node = fv.Follow(args.ns)
    reader = fv.HandReader(args.port, "hand")
    reader.start()
    W, H = args.width, args.height
    cx, cy = W / 2.0, H / 2.0
    real_wh = [None]

    print(f"腕部相机 {W}x{H}，控制 {args.rate:.0f} Hz，增益 {args.gain}")
    print("等机械臂使能…")
    if not fv.wait_enabled(node, timeout=60.0):
        print("60 秒还没等到使能，先退出。检查控制器起没起、命名空间对不对。")
        return 2

    # 雅可比：已有就复用，没有就探一次（末端只走 1 cm）
    J = None
    Jz = None
    if os.path.isfile(MOUNT) and not args.recal:
        try:
            J0 = np.array(json.load(open(MOUNT))["J"], dtype=float)
            check_probe(J0)
            Jz = np.array(J0[2], dtype=float)      # 手掌尺度对基座 xyz 的梯度（原始测量）
            J, use_size = prepare_jac(J0)
            print(f"复用 {MOUNT}")
        except RuntimeError as e:
            print(f"旧的 {MOUNT} 不合格：{e}\n→ 重新探测", flush=True)
            J = None
    if J is None:
        for attempt in range(1, 4):
            print(f"测特征雅可比（第 {attempt} 次）：末端沿 x/y/z 各走 1 cm，"
                  f"手放在夹爪前 0.3–0.5 m 别动…", flush=True)
            try:
                Jraw = Probe(arm, node, reader, args.foc, step=args.probe_step,
                             settle=args.settle).run(cx, cy)
                Jz = np.array(Jraw[2], dtype=float)
                J, use_size = prepare_jac(Jraw)
                os.makedirs(os.path.dirname(MOUNT), exist_ok=True)
                json.dump({"J": J.tolist(), "stamp": time.time(), "w": W, "h": H},
                          open(MOUNT, "w"))
                print(f"已存 {MOUNT}")
                break
            except RuntimeError as e:
                print(f"  这次不成：{e}", flush=True)
                J = None
        if J is None:
            print("探测连续三次失败，退出。")
            return 2

    # 期望特征：手在画面中心、大小回到开始时的值（或 --size）
    d, ts = reader.get()
    s_star = feature_of(d, cx, cy, args.foc) if d else None
    if s_star is None or not np.isfinite(s_star).all():
        print("看不到手")
        return 2
    s_star[:2] = 0.0
    if args.size > 0:
        s_star[2] = float(np.log(args.size))
    print(f"目标特征：画面中心，手掌尺度 {np.exp(s_star[2] / args.foc):.3f}")

    ctrl = Controller(arm, args.gain, args.vmax, args.qdmax, 1.0 / args.rate,
                      args.jac_damp, args.adaptive)
    grip = Grip(node, args.ns, args) if args.grip else None
    if grip:
        print(f"夹爪：手张开→{args.grip_open:.1f}，握拳→{args.grip_close:.1f}，"
              f"最大 {args.grip_rate:.1f} rad/s")
    sat_since = None
    ctrl.set_jac(J)
    ctrl.es_dead = args.foc * abs(np.log(1.0 + args.size_band)) if args.size_band > 0 else 0.0
    if not use_size or not args.three_d:
        ctrl.ignore_size = True   # 默认 2D：标记大小那一路最容易把臂带跑

    dt = 1.0 / args.rate
    q = node.q.copy() if node.q is not None else np.zeros(6)
    if args.posture > 0:
        ctrl.q_ref = q.copy()
        ctrl.k_null = args.posture
    if args.size_band > 0 and not args.three_d:
        print(f"保持距离：手掌尺度偏离 {args.size_band*100:.0f}% 以内不动，超出才靠近/后退")
    hold_zone = [False]
    guard_skip = None
    p_goal = None
    gap = None
    es = 0.0
    logf = open(args.log, "w", buffering=1) if args.log else None
    if logf:
        logf.write("t,u,v,size,eu,ev,es,vx,vy,vz,qdmax,lead,fresh\n")
    last_good = time.time()
    last_show = 0.0
    print("开始跟随（Ctrl-C 停）")
    print("  ↑ 腕部相机只看得见夹爪正前方约 ±30° 的锥体；手要放在夹爪前 0.3–0.6 m")
    print("    若一直打印 [等待] 没收到包，就是手不在画面里 —— 把手移到夹爪正前方")

    while True:
        t_next = time.time() + dt
        fv.rclpy.spin_once(node, timeout_sec=0.0)
        d, ts = reader.get()
        now = time.time()
        fresh = d is not None and (now - ts) < args.lost
        if fresh and d.get("w") and real_wh[0] != (d["w"], d["h"]):
            real_wh[0] = (d["w"], d["h"])
            if (d["w"], d["h"]) != (W, H):
                W, H = d["w"], d["h"]
                cx, cy = W / 2.0, H / 2.0
                print(f"相机实际分辨率 {W}x{H}，画面中心改为 ({cx:.0f},{cy:.0f})")
                s_star[0] = s_star[1] = 0.0
        s = feature_of(d, cx, cy, args.foc) if fresh else np.array([np.nan] * 3)

        # 夹爪：每拍都按限速更新（手丢了就保持，不乱动）
        if grip and d is not None:
            pin0 = d.get("pinch", float("nan"))
            if np.isfinite(pin0):
                grip.step(float(pin0), dt)

        # 姿态保护：快贴到限位、或者一直追不上（积分饱和）就不发了
        # 起点本身就贴着限位的关节（home 就在 joint2/3 的下限上）不算"快撞限位"，
        # 否则一启动就误判、整个跟随都不发指令。
        q_now = node.q
        limit_hit = None
        if q_now is not None:
            if guard_skip is None:
                gs = [set(), set()]
                for j in range(6):
                    if q_now[j] - arm.lo[j] < 0.15:
                        gs[0].add(j)
                    if arm.hi[j] - q_now[j] < 0.15:
                        gs[1].add(j)
                guard_skip = gs
            best = (None, 1e9)
            for j in range(6):
                cand = []
                if j not in guard_skip[0]:
                    cand.append(q_now[j] - arm.lo[j])
                if j not in guard_skip[1]:
                    cand.append(arm.hi[j] - q_now[j])
                if not cand:
                    continue
                m = min(cand)
                if m < best[1]:
                    best = (j + 1, float(m))
            if best[0] is not None and best[1] < args.limit_margin:
                limit_hit = best

        paused = args.pinch_pause > 0 and (d or {}).get("pinch", 1.0) < args.pinch_pause
        if fresh and np.isfinite(s).all() and not paused:
            err = s[:2] - s_star[:2]
            near = float(np.hypot(*err)) < (args.deadband if not hold_zone[0] else args.deadband * 1.6)
            hold_zone[0] = near
            if near:
                qd = np.zeros(6); v = np.zeros(3)
                q = node.q.copy() if node.q is not None else q
                p_goal = None
            else:
                qd, v = ctrl.step(q, s, s_star)
            # ★ 目标位置要自己累积。每帧都用 node.q 重开 → 永远只领先实测 0.002 rad，
            #   控制器吃掉了这点余量，机械臂看起来一动也不动。
            if args.ik == "official" and node.q is not None:
                # 图像误差 → 关节步 → 等效笛卡尔位移；笛卡尔目标自己累积，
                # 但每一拍都从"当前实测关节角"重新解 IK（官方 tcp-ik 的做法）。
                q_now = node.q.copy()
                p_now = arm.fk(q_now)
                if p_goal is None:
                    p_goal = p_now.copy()
                dp = _pos_jac(arm, q_now) @ (qd * dt)
                dp = np.clip(dp, -args.max_speed * dt, args.max_speed * dt)
                p_goal = p_goal + dp
                # 保持距离：只用"手掌大小"这一个慢速外环，沿实测的尺度梯度方向挪一点点。
                # 不进速度求解、不抢主环，所以不会像以前那样一路往前顶。
                es = 0.0
                if args.size_hold > 0 and Jz is not None and np.isfinite(s[2]):
                    es = float(s_star[2] - s[2])
                    if abs(es) > args.size_deadband:
                        g = Jz
                        dpz = g * (es / max(float(g @ g), 1e-9)) * args.size_gain
                        nn = float(np.linalg.norm(dpz))
                        cap = args.size_speed * dt
                        if nn > cap:
                            dpz *= cap / nn
                        p_goal = p_goal + dpz
                delta = p_goal - p_now
                gap = float(np.linalg.norm(delta))
                if gap > args.max_lead_m:          # 离实测太远就拉回来，防积分饱和
                    p_goal = p_now + delta * (args.max_lead_m / gap)
                p_goal = clamp_workspace(p_goal)
                q, ik_err = official_ik(arm, p_goal, q_now, dt, max_speed=args.ik_speed)
                if ik_err > 0.05:
                    q = q_now                      # 解不出来就别发，保持现状
            else:
                q = np.clip(q + qd * dt, arm.lo, arm.hi)
                if node.q is not None:
                    lead = q - node.q
                    if np.abs(lead).max() > args.max_lead:
                        q = node.q + np.clip(lead, -args.max_lead, args.max_lead)
                    if np.abs(node.q - q).max() > 0.5:
                        q = node.q.copy()
            # 积分饱和检测：目标一直顶在 max-lead 上 = 追不上
            if node.q is not None and np.abs(q - node.q).max() >= 0.95 * args.max_lead:
                sat_since = sat_since or now
            else:
                sat_since = None
            windup = sat_since is not None and (now - sat_since) > args.windup

            if limit_hit is not None:
                if now - last_show > 1.0:
                    last_show = now
                    print(f"[保护] 关节 {limit_hit[0]} 离限位只剩 {limit_hit[1]:.2f} rad，暂停跟随", flush=True)
                q = node.q.copy() if node.q is not None else q
                if not args.dry_run:
                    node.send(q, args.vlim)
            elif windup:
                if now - last_show > 1.0:
                    last_show = now
                    print(f"[保护] 目标连续 {args.windup:.1f}s 追不上（姿态太扭），暂停等手回来", flush=True)
                q = node.q.copy() if node.q is not None else q
                if not args.dry_run:
                    node.send(q, args.vlim)
            else:
                if not args.dry_run:
                    node.send(q, args.vlim)
            if logf:
                lev = float(np.abs(q - node.q).max()) if node.q is not None else -1.0
                logf.write("%.3f,%.1f,%.1f,%.3f,%.1f,%.1f,%.1f,%.4f,%.4f,%.4f,%.3f,%.3f,%d\n" % (
                    now, d["u"], d["v"], d.get("size", 0.0),
                    s[0]-s_star[0], s[1]-s_star[1], s[2]-s_star[2],
                    v[0], v[1], v[2], np.abs(qd).max(), lev, 1))
            last_good = now
            if now - last_show > 0.5:
                last_show = now
                print(f"[跟随] 手 ({d['u']:.0f},{d['v']:.0f}) size {d.get('size',0):.3f} "
                      f"pinch {d.get('pinch',float('nan')):.2f} "
                      f"误差 ({s[0]-s_star[0]:+.0f},{s[1]-s_star[1]:+.0f},{s[2]-s_star[2]:+.0f}) "
                      f"速度 {np.round(v,3)} |qd|max {np.abs(qd).max():.2f} "
                      f"领先 {np.abs(q - node.q).max() if node.q is not None else -1:.3f}rad "
                      f"目标离实测 {gap*1000 if gap is not None else -1:.0f}mm "
                      f"尺寸偏差 {es:+.0f}{'  [保持距离]' if abs(es) > args.size_deadband and args.size_hold > 0 else ''}"
                      f"{'  [dry-run 不发]' if args.dry_run else ''}", flush=True)
        else:
            if now - last_good > args.lost:
                if not args.dry_run:
                    node.send(q, args.vlim)  # 原地保持
            if now - last_show > 1.0:
                last_show = now
                why = "没收到包" if d is None else (
                    f"包太旧 {(now-ts)*1000:.0f}ms" if not fresh else
                    f"捏合暂停 pinch={d.get('pinch',1):.2f}")
                print(f"[等待] {why}", flush=True)

        sleep = t_next - time.time()
        if sleep > 0:
            time.sleep(sleep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="/rebot_arm_rs")
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--foc", type=float, default=600.0, help="焦距（像素），只用来给尺度特征定量纲")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--vmax", type=float, default=0.12, help="末端最大速度 m/s")
    ap.add_argument("--qdmax", type=float, default=0.8, help="关节最大速度 rad/s")
    ap.add_argument("--vlim", type=float, default=0.4)
    ap.add_argument("--size", type=float, default=0.0, help="期望的手掌尺度（0=保持开始时的大小）")
    ap.add_argument("--lost", type=float, default=0.30)
    ap.add_argument("--size-hold", type=int, default=1, help="1=保持和手的距离（慢速外环）")
    ap.add_argument("--size-deadband", type=float, default=90.0,
                    help="手掌尺度偏差死区（foc·ln 单位，600 时 90 ≈ 16%%）")
    ap.add_argument("--size-speed", type=float, default=0.04, help="保持距离的最大速度 m/s")
    ap.add_argument("--size-gain", type=float, default=1.0)
    ap.add_argument("--max-lead-m", type=float, default=0.05,
                    help="笛卡尔目标最多领先实测多少米")
    ap.add_argument("--max-lead", type=float, default=0.25, help="目标最多领先实测多少 rad")
    ap.add_argument("--pinch-pause", type=float, default=0.0,
                    help="捏合比例低于它暂停（0=不暂停；手正对相机时比例本来就小）")
    ap.add_argument("--jac-damp", type=float, default=0.02)
    ap.add_argument("--adaptive", type=float, default=0.0, help="Broyden 修正强度（0=关，默认关）")
    ap.add_argument("--3d", dest="three_d", action="store_true", help="连手掌大小一起跟（默认只跟 u、v）")
    ap.add_argument("--log", default="/tmp/ibvs.csv", help="CSV 日志")
    ap.add_argument("--probe-step", type=float, default=0.01, help="探测步长 m")
    ap.add_argument("--deadband", type=float, default=15.0, help="像素死区：手在中心附近就不动")
    ap.add_argument("--posture", type=float, default=0.0, help="零空间姿态保持强度（0=关，默认关）")
    ap.add_argument("--size-band", type=float, default=0.0, help="距离保持带（手掌尺度偏差比例），0=关掉（默认关）")
    ap.add_argument("--settle", type=float, default=0.9)
    ap.add_argument("--recal", action="store_true", help="重新探测雅可比")
    ap.add_argument("--dry-run", action="store_true", help="只算不发（不动机械臂）")
    ap.add_argument("--selftest", action="store_true", help="离线自检（不用机器人）")
    # ---- 夹爪：手张开=张开，握拳=闭合，全程限速
    ap.add_argument("--grip", type=int, default=1, help="1=接管夹爪（默认），0=不管夹爪")
    ap.add_argument("--grip-open", type=float, default=0.05, help="张开位置")
    ap.add_argument("--grip-close", type=float, default=0.0, help="闭合位置")
    ap.add_argument("--grip-rate", type=float, default=0.05,
                    help="夹爪最大速度 rad/s；行程 0.05 rad，0.05 时约 1 秒走完")
    ap.add_argument("--grip-min-span", type=float, default=0.10,
                    help="pinch 的开合幅度小于它就判定为没在比划，夹爪不动")
    ap.add_argument("--grip-open-pinch", type=float, default=0.55)
    ap.add_argument("--grip-close-pinch", type=float, default=0.35)
    # ---- 姿态失控保护
    ap.add_argument("--limit-margin", type=float, default=0.12,
                    help="任一关节离限位小于这个值就停止跟随（rad）")
    ap.add_argument("--ik", choices=["official", "integrate"], default="official",
                    help="official=照 RS 官方 tcp-ik.js 的位置 IK（每拍从当前姿态重解）")
    ap.add_argument("--ik-speed", type=float, default=2.8, help="IK 每次迭代的关节速度上限 rad/s")
    ap.add_argument("--max-speed", type=float, default=0.15, help="末端最大速度 m/s")
    ap.add_argument("--windup", type=float, default=1.5,
                    help="目标持续被 max-lead 卡住超过这么多秒就暂停（秒）")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a)
    try:
        return run_live(a)
    except KeyboardInterrupt:
        print("\n停。")
        return 0
    except Exception as ex:
        import traceback

        traceback.print_exc()
        print(f"✗ {ex}")
        return 1


def selftest(a) -> int:  # noqa: C901
    """合成一个腕部相机：手固定在基座系一点，相机装在末端 z 轴朝前，"""
    arm = fv.Arm()
    W, H, f = 640.0, 480.0, 600.0
    cx, cy = W / 2, H / 2
    palm = 0.09                              # 手掌尺度
    q0 = np.array([0.0, 0.9, 0.6, 0.9, 0.0, 0.0])   # 离开关节限位
    # 手放在起点相机前方偏右上的位置，保证一开始就在画面里
    hand = arm.fk(q0) + frame_rot(arm, q0) @ np.array([0.07, 0.05, 0.40])

    def observe(q):
        p = arm.fk(q)
        R = frame_rot(arm, q)                # 末端系 = 相机系（x右 y下 z前）
        pc = R.T @ (hand - p)
        if pc[2] <= 0.02:
            return None
        u = cx + f * pc[0] / pc[2]
        v = cy + f * pc[1] / pc[2]
        size = f * palm / pc[2] / H
        return np.array([u - cx, v - cy, f * float(np.log(size))])

    q = q0
    s0 = observe(q)
    if s0 is None:
        print("自检起点手在相机后面，调整 hand 位置")
        return 1
    print(f"起点特征 {np.round(s0, 2)}")

    # 用解析雅可比代替探测（自检里没有真机，探测做不了）
    J = np.zeros((3, 3))
    for k in range(3):
        p0 = arm.fk(q)
        qk, _ = arm.ik(p0 + np.eye(3)[k] * 0.01, q, iters=80)
        sk = observe(qk)
        J[:, k] = (sk - s0) / 0.01  # px/m，1/m
    J, use_size = prepare_jac(J)

    star = np.array([0.0, 0.0, s0[2] - 0.25 * f])  # 希望手再大一点（靠近 25%）
    ctrl = Controller(arm, a.gain, a.vmax, a.qdmax, 0.02, a.jac_damp, 0.0)
    ctrl.set_jac(J)
    if not use_size or not a.three_d:
        ctrl.ignore_size = True
    for n in range(400):
        s = observe(q)
        if s is None:
            print("手跑出画面")
            return 1
        qd, v = ctrl.step(q, s, star)
        q = np.clip(q + qd * 0.02, arm.lo, arm.hi)
        if n % 80 == 0:
            print(f"  t={n*0.02:5.1f}s e={np.round(s-star, 3)} v={np.round(v, 3)}")
    s = observe(q)
    e = s - star
    print(f"收敛后误差 u={e[0]:+.1f}px v={e[1]:+.1f}px logsize={e[2]:+.3f}")
    ok = abs(e[0]) < 25 and abs(e[1]) < 25 and abs(e[2]) < 0.15 * f
    print("✓ 通过" if ok else "✗ 没收敛")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())