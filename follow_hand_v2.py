#!/usr/bin/env python3
"""手部 1:1 跟随（纸面版）：俯视相机看到的掌心 -> A4 纸面坐标 -> 基座 XY -> 60Hz 关节流。"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))

for _p in (
    os.path.join(os.environ.get("REBOT_WS", os.path.expanduser("~/ReBot_Arm_DigitalTwin_RS")), "rebotarm_ros2_RS", "third_party"),
    os.path.expanduser("~/reBotArm_control_py"),
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from rebotarm_msgs.msg import ArmStatus, JointMitCmd
from reBotArm_control_py.kinematics.robot_model import load_robot_model, get_end_effector_frame_id

NS = "/rebot_arm_rs"
ARM = [f"joint{i}" for i in range(1, 7)]
CALIB = os.path.join(HERE, "calib_paper.json")


class HandReader(threading.Thread):
    """收 vision_track 的 UDP 包。"""

    def __init__(self, port: int, target: str = "hand") -> None:
        super().__init__(daemon=True)
        self.target = target
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.settimeout(0.05)
        self.lock = threading.Lock()
        self.data: dict | None = None
        self.stamp = 0.0
        self._run = True

    def run(self) -> None:
        while self._run:
            try:
                raw, _ = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            try:
                p = json.loads(raw)
            except Exception:
                continue
            if self.target == "hand":
                d = p.get("hand")
                if not d:
                    continue
                # 分辨率在包的最外层，手部子字典里没有 —— 拷贝一份并把 w/h 带上，
                # 否则下游只能拿到 u,v，永远按默认 640x480 算中心。
                d = dict(d)
                d["w"] = p.get("w")
                d["h"] = p.get("h")
            else:
                a = p.get("aruco")
                if not a:
                    continue
                d = {"u": a["u"], "v": a["v"], "pinch": 1.0, "size": 0.0}
            with self.lock:
                self.data = d
                self.stamp = time.time()

    def get(self):
        with self.lock:
            return self.data, self.stamp

    def stop(self) -> None:
        self._run = False


class AB:
    """alpha-beta 滤波 + 匀速预测，压抖动、补延迟。"""

    def __init__(self, alpha: float, beta: float) -> None:
        self.alpha, self.beta = alpha, beta
        self.x: np.ndarray | None = None
        self.v = None
        self.t = 0.0

    def reset(self, x: np.ndarray, t: float) -> None:
        self.x, self.v, self.t = np.array(x, float).copy(), np.zeros_like(x), t

    def update(self, z: np.ndarray, t: float) -> np.ndarray:
        if self.x is None:
            self.reset(z, t)
            return self.x.copy()
        if self.v is None:
            self.v = np.zeros_like(self.x)
        dt = max(t - self.t, 1e-3)
        pred = self.x + self.v * dt
        r = z - pred
        self.x = pred + self.alpha * r
        self.v = self.v + (self.beta / dt) * r
        sp = np.linalg.norm(self.v)
        if sp > 3.0:                                # 手速上限，防滤波发散
            self.v *= 3.0 / sp
        self.t = t
        return self.x.copy()

    def predict(self, horizon: float) -> np.ndarray:
        return self.x + self.v * horizon


URDF_CANDIDATES = (
    os.path.expanduser("~/reBotArm_control_py/urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf"),
    os.path.expanduser(
        os.path.join(os.environ.get("REBOT_WS", os.path.expanduser("~/ReBot_Arm_DigitalTwin_RS")), "rebotarm_ros2_RS", "install", "rebotarm_bringup", "share") + "/"
        "rebotarm_bringup/description/urdf/ReBot_Arm_RS.urdf"),
)
END_FRAME = "gripper_end"
HOME = np.array([0.3017, 0.0, 0.2177])


def load_arm_model():
    """必须加载 RS 版本的 URDF。"""
    for path in URDF_CANDIDATES:
        if os.path.isfile(path):
            return load_robot_model(path)
    return load_robot_model()


class Arm:
    def __init__(self) -> None:
        self.model = load_arm_model()
        self.data = self.model.createData()
        self.fid = self.model.getFrameId(END_FRAME)
        if self.fid >= self.model.nframes:
            raise RuntimeError(f"URDF 里没有 {END_FRAME} 这个 frame")
        self.q_full = np.zeros(self.model.nq)
        self.lo = self.model.lowerPositionLimit[:6].copy()
        self.hi = self.model.upperPositionLimit[:6].copy()
        p0 = self.fk(np.zeros(6))
        if np.linalg.norm(p0 - HOME) > 0.03:
            raise RuntimeError(
                f"加载到的模型不对：零位末端在 {np.round(p0, 3)}，RS 应该接近 {HOME}。"
                "检查 URDF 是不是 DM 那份。")

    def fk(self, q6: np.ndarray) -> np.ndarray:
        self.q_full[:6] = q6
        pin.forwardKinematics(self.model, self.data, self.q_full)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.fid].translation.copy()

    def ik(self, p_target: np.ndarray, q_seed: np.ndarray, iters: int = 24,
           damp: float = 3e-3, step: float = 0.6, tol: float = 2e-4):
        q = q_seed.copy()
        err = np.full(3, 1e9)
        for _ in range(iters):
            self.q_full[:6] = q
            pin.forwardKinematics(self.model, self.data, self.q_full)
            pin.updateFramePlacements(self.model, self.data)
            err = p_target - self.data.oMf[self.fid].translation
            if np.linalg.norm(err) < tol:
                break
            J = pin.computeFrameJacobian(self.model, self.data, self.q_full, self.fid,
                                         pin.LOCAL_WORLD_ALIGNED)[:3, :6]
            dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(3), err)
            q = np.clip(q + step * dq, self.lo, self.hi)
        return q, float(np.linalg.norm(err))


class Follow(Node):
    def __init__(self, ns: str = NS) -> None:
        super().__init__("hand_follow")
        self.ns = ns
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        # arm_status 是「只在事件时发一次 + latched（TRANSIENT_LOCAL）」，
        # 用默认 VOLATILE 订阅会永远收不到启动前那一条 —— 必须同样要 latched。
        qos_status = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.q: np.ndarray | None = None
        self.enabled = False
        self.n_q = 0
        self.n_status = 0
        self.create_subscription(JointState, f"{ns}/joint_states", self._on_joint, qos)
        self.create_subscription(ArmStatus, f"{ns}/arm_status", self._on_status, qos_status)
        self.pubs = [self.create_publisher(JointMitCmd, f"{ns}/joints/{j}/cmd/mit", 1) for j in ARM]

    def _on_joint(self, msg: JointState) -> None:
        self.n_q += 1
        d = dict(zip(msg.name, msg.position))
        if all(n in d for n in ARM):
            self.q = np.array([d[n] for n in ARM], dtype=float)

    def _on_status(self, msg: ArmStatus) -> None:
        self.n_status += 1
        self.enabled = bool(msg.enabled)

    def send(self, q6: np.ndarray, vlim: float) -> None:
        for pub, v in zip(self.pubs, q6):
            m = JointMitCmd()
            m.pos = float(v)
            m.vel = float(vlim)
            pub.publish(m)


def wait_enabled(node, timeout: float = 60.0) -> bool:
    """等机械臂使能，同时每 2 秒报告收到多少条消息 —— 别再无声地傻等。"""
    t0 = time.time()
    last = 0.0
    while time.time() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.enabled:
            return True
        el = time.time() - t0
        if el - last >= 2.0:
            last = el
            print(f"  等使能 {el:4.1f}s：joint_states {node.n_q} 条 / arm_status {node.n_status} 条",
                  flush=True)
            if node.n_status == 0 and el > 3.0:
                print("  ↑ 一条 arm_status 都没收到：控制器没起（ros2 node list 里应有 "
                      "reBotArmController），或命名空间不对（真机 /rebot_arm_rs，仿真 /rebot_arm_rs_sim）",
                      flush=True)
    return False


def _pause(msg: str = "按 Enter 关闭…") -> None:
    try:
        if sys.stdin and sys.stdin.isatty():
            input(msg)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=CALIB)
    ap.add_argument("--ns", default=NS)
    ap.add_argument("--cx", type=float, default=0.32, help="纸面中心对应的基座 X（米）")
    ap.add_argument("--cy", type=float, default=0.0, help="纸面中心对应的基座 Y（米）")
    ap.add_argument("--z", type=float, default=0.12, help="跟随高度（基座系，米）")
    ap.add_argument("--gain", type=float, default=1.0, help="1.0 = 手挪多少末端挪多少")
    ap.add_argument("--rate", type=float, default=60.0)
    ap.add_argument("--horizon", type=float, default=0.10, help="延迟补偿预测（秒）")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=0.15)
    ap.add_argument("--tau", type=float, default=0.06)
    ap.add_argument("--max-speed", type=float, default=0.35)
    ap.add_argument("--vlim", type=float, default=1.0)
    ap.add_argument("--pinch-pause", type=float, default=0.30)
    ap.add_argument("--lost-hold", type=float, default=0.30)
    ap.add_argument("--box", type=float, nargs=4, default=[0.21, 0.45, -0.14, 0.14],
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    ap.add_argument("--swap-xy", action="store_true", help="左右/前后整体对调")
    ap.add_argument("--flip-x", action="store_true", help="前后取反")
    ap.add_argument("--flip-y", action="store_true", help="左右取反")
    ap.add_argument("--probe", action="store_true", help="只打印手的坐标，不发指令、不按 Enter")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    path = os.path.expanduser(args.calib)
    if not os.path.isfile(path):
        print(f"\n找不到标定文件：{path}\n先跑一次：\n"
              f"  ~/hand_vision_venv/bin/python paper_map.py")
        _pause()
        return 2
    calib = json.load(open(path, encoding="utf-8"))
    H = np.array(calib["H"])
    W, Hh = calib.get("paper_mm", [210.0, 297.0])
    print(f"标定：纸面 {W:.0f}x{Hh:.0f} mm，角点残差 {calib.get('residual_mm', float('nan')):.2f} mm，"
          f"分辨率 {calib.get('mm_per_px', float('nan')):.2f} mm/px", flush=True)

    def paper_of(u: float, v: float):
        p = H @ np.array([u, v, 1.0])
        if abs(p[2]) < 1e-9:
            return None
        return p[0] / p[2], p[1] / p[2]

    def target_of(mx: float, my: float) -> np.ndarray:
        ex = (mx - W / 2.0) / 1000.0        # 你的右手方向
        ey = (my - Hh / 2.0) / 1000.0       # 朝你
        dx, dy = -ey, ex                    # 基座：手往你这边 -> -x；手往右 -> +y
        if args.swap_xy:
            dx, dy = dy, dx
        if args.flip_x:
            dx = -dx
        if args.flip_y:
            dy = -dy
        return np.array([args.cx + args.gain * dx, args.cy + args.gain * dy])

    reader = HandReader(8899)
    reader.start()

    if args.probe:
        print("probe：把手放纸上移动，看纸面坐标（mm）。Ctrl-C 退出。", flush=True)
        last = 0.0
        try:
            while True:
                h, st = reader.get()
                if h and time.time() - last > 0.4:
                    p = paper_of(h["u"], h["v"])
                    if p:
                        t = target_of(*p)
                        print(f"像素({h['u']:.0f},{h['v']:.0f}) -> 纸面({p[0]:6.1f},{p[1]:6.1f}) mm "
                              f"-> 末端({t[0]:.3f},{t[1]:.3f})", flush=True)
                        last = time.time()
        except KeyboardInterrupt:
            return 0

    arm = Arm()
    rclpy.init()
    node = Follow(args.ns)
    t0 = time.time()
    while node.q is None and time.time() - t0 < 5.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.q is None:
        print("没收到 joint_states —— 控制器没在跑？", flush=True)
        return 1
    print(f"当前末端 {np.round(arm.fk(node.q), 3)}", flush=True)

    if args.dry_run:
        print("dry-run：只算不发，Ctrl-C 退出", flush=True)
    else:
        if not node.enabled:
            print("★ 机械臂未使能，请先在网页点「使能」。脚本不会替你使能。", flush=True)
        if "sim" in args.ns or not (sys.stdin and sys.stdin.isatty()):
            print("（仿真/非交互：直接开始）", flush=True)
        else:
            input("确认周围安全，按 Enter 开始跟随（Ctrl-C 停止）…")

    xmin, xmax, ymin, ymax = args.box
    csv = None
    if args.log:
        csv = open(os.path.expanduser(args.log), "w", encoding="utf-8")
        csv.write("t,tx,ty,tz,ax,ay,az,err_mm,u,v,mx,my,fresh\n")

    ab = AB(args.alpha, args.beta)
    q_cmd = node.q.copy()
    p_cmd = arm.fk(q_cmd)
    period = 1.0 / args.rate
    paused = False
    last_print = 0.0
    last_t = time.time()
    print("跟随中…（双指捏合=暂停，Ctrl-C=停止发送）", flush=True)
    try:
        while rclpy.ok():
            loop_t = time.time()
            rclpy.spin_once(node, timeout_sec=0.0)
            hand, hstamp = reader.get()
            now = time.time()
            fresh = hand is not None and (now - hstamp) < args.lost_hold
            tar = None
            mx = my = float("nan")

            if fresh and not paused:
                p = paper_of(hand["u"], hand["v"])
                if p is not None and not (-30.0 <= p[0] <= W + 30.0 and -30.0 <= p[1] <= Hh + 30.0):
                    p = None              # 手跑出纸面就别外推了，外推会飞出天际
                if p is not None:
                    mx, my = p
                    xy = target_of(mx, my)
                    if ab.x is None or hstamp > ab.t + 1e-9:
                        ab.update(xy, hstamp)
                    horizon = args.horizon + max(0.0, now - hstamp)
                    tar = ab.predict(horizon)
                    tar[0] = float(np.clip(tar[0], xmin, xmax))
                    tar[1] = float(np.clip(tar[1], ymin, ymax))

                    dt = max(now - last_t, 1e-3)
                    d = tar - p_cmd[:2]
                    dist = float(np.linalg.norm(d))
                    if dist > 0:
                        step_len = min(dist * (dt / args.tau), args.max_speed * dt)
                        p_cmd[:2] += d / dist * step_len
                    p_cmd[2] = args.z

                    q_new, err = arm.ik(p_cmd, q_cmd)
                    if err < 0.005:
                        q_cmd = q_cmd + np.clip(q_new - q_cmd, -0.06, 0.06)
                        q_cmd = np.clip(q_cmd, arm.lo, arm.hi)
                    if not args.dry_run:
                        node.send(q_cmd, args.vlim)
            else:
                p_cmd[2] = args.z
                if not args.dry_run and not paused:
                    node.send(q_cmd, args.vlim)
                if not fresh:
                    ab.x = None

            if fresh and hand is not None:
                want = hand.get("pinch", 1.0) < args.pinch_pause
                if want != paused:
                    paused = want
                    print("\n[暂停] 捏合" if paused else "\n[继续]", flush=True)

            if csv and now - last_print > 0.05:
                ax = arm.fk(q_cmd)
                csv.write(f"{now:.4f},{p_cmd[0]:.4f},{p_cmd[1]:.4f},{p_cmd[2]:.4f},"
                          f"{ax[0]:.4f},{ax[1]:.4f},{ax[2]:.4f},"
                          f"{np.linalg.norm(p_cmd - ax) * 1000:.2f},"
                          f"{hand['u'] if hand else -1:.1f},{hand['v'] if hand else -1:.1f},"
                          f"{mx:.1f},{my:.1f},{int(fresh)}\n")
                last_print = now
            last_t = now
            dt = period - (time.time() - loop_t)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        print("\n停止发送指令（机械臂保持当前位置，不回零、不失能）", flush=True)
    finally:
        if csv:
            csv.close()
        reader.stop()
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())