#!/usr/bin/env python3
"""俯视相机视觉进程：MediaPipe 手部关键点 + ArUco 标记检测，UDP JSON 输出。"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import os

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL = os.path.join(HERE, "models", "hand_landmarker.task")

# MediaPipe 手部关键点索引
WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 0, 5, 9, 13, 17
THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20


def open_camera(index: int, width: int, height: int, fps: int, focus: int | None) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"打不开 /dev/video{index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if focus is not None:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, focus)
    return cap


def find_blob(frame, thresh: float, invert: bool, min_area: float, max_area: float,
              min_circ: float):
    """找画面里最像圆的一个亮（或暗）斑，返回 (u, v, r) 像素。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if invert:
        bw = cv2.compare(gray, thresh, cv2.CMP_LT)
    else:
        bw = cv2.compare(gray, thresh, cv2.CMP_GT)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    best = None
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area or a > max_area:
            continue
        p = cv2.arcLength(c, True)
        if p <= 0:
            continue
        circ = 4.0 * np.pi * a / (p * p)
        if circ < min_circ:
            continue
        x, y, bw_, bh_ = cv2.boundingRect(c)
        if x <= 2 or y <= 2 or x + bw_ >= w - 2 or y + bh_ >= h - 2:
            continue
        if max(bw_, bh_) > max_aspect * min(bw_, bh_):
            continue
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        # 按面积挑最大的：靶标跑了会忽大忽小，但永远是画面里最显眼的那块彩色
        cand = (a, float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"]),
                float(np.sqrt(a / np.pi)))
        if best is None or cand[0] > best[0]:
            best = cand
    if best is None:
        return None
    return best[1], best[2], best[3]


def find_color(frame, hue_deg: float, hue_tol: float, sat_min: float, val_min: float,
               min_area: float, max_area: float, min_circ: float, max_aspect: float = 2.5):
    """找画面里最像圆的一块高饱和彩色斑（默认红色），返回 (u, v, r)。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int16) * 2.0          # OpenCV H 是 0-179，换成 0-358 度
    d = np.abs(h - (hue_deg % 360.0))
    d = np.minimum(d, 360.0 - d)
    mask = ((d <= hue_tol) & (hsv[:, :, 1] >= sat_min) & (hsv[:, :, 2] >= val_min))
    bw = (mask.astype(np.uint8)) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hh, ww = bw.shape[:2]
    best = None
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area or a > max_area:
            continue
        p = cv2.arcLength(c, True)
        if p <= 0:
            continue
        circ = 4.0 * np.pi * a / (p * p)
        if circ < min_circ:
            continue
        x, y, bw_, bh_ = cv2.boundingRect(c)
        if x <= 2 or y <= 2 or x + bw_ >= ww - 2 or y + bh_ >= hh - 2:
            continue
        if abs(bw_ - bh_) > 0.35 * max(bw_, bh_):
            continue
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cand = (circ, float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"]),
                float(np.sqrt(a / np.pi)))
        if best is None or cand[0] > best[0]:
            best = cand
    if best is None:
        return None
    return best[1], best[2], best[3]


def hand_metrics(lms, w: int, h: int) -> dict:
    """由 21 个归一化关键点算手部指标，全部换算到像素。"""
    px = np.array([[lm.x * w, lm.y * h] for lm in lms])  # (21,2)
    palm = px[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]].mean(axis=0)
    d = lambda a, b: float(np.linalg.norm(px[a] - px[b]))
    wrist_size = d(PINKY_MCP, WRIST)  # 手掌宽度，用于估距离
    pinch = d(THUMB_TIP, INDEX_TIP) / max(wrist_size, 1e-6)
    tips = [d(INDEX_TIP, WRIST), d(MIDDLE_TIP, WRIST), d(RING_TIP, WRIST), d(PINKY_TIP, WRIST)]
    spread = float(np.mean(sorted(tips)[:2])) / max(wrist_size, 1e-6)
    return {
        "u": float(palm[0]),
        "v": float(palm[1]),
        "pinch": float(pinch),
        "size": float(wrist_size) / float(h),
        "spread": float(spread),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="0",
                    help="相机编号；auto = 自动挑（插两个时挑编号大的那个）")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--focus", type=int, default=108, help="手动对焦值，-1 表示交给自动对焦")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--udp", default="127.0.0.1:8899")
    ap.add_argument("--udp2", default="127.0.0.1:8898",
                    help="第二个收包端口（标定用），空字符串表示不发")
    ap.add_argument("--aruco-id", type=int, default=0)
    ap.add_argument("--target", choices=("hand", "aruco", "blob", "color", "both"), default="both",
                    help="标定靶标：打印的 ArUco、夹爪上贴的白圆点，或两者都试")
    ap.add_argument("--blob-thresh", type=float, default=200.0, help="白点亮度阈值 0-255")
    ap.add_argument("--blob-invert", action="store_true", help="靶标是暗点、背景亮时加这个")
    ap.add_argument("--blob-min-area", type=float, default=150.0)
    ap.add_argument("--blob-max-area", type=float, default=40000.0)
    ap.add_argument("--blob-circ", type=float, default=0.60)
    ap.add_argument("--color-circ", type=float, default=0.30, help="彩色靶标的圆度下限（方块取 0.2）")
    ap.add_argument("--color-aspect", type=float, default=2.5, help="彩色靶标最长/最短边上限")
    ap.add_argument("--det-conf", type=float, default=0.30, help="手检测置信度门限")
    ap.add_argument("--track-conf", type=float, default=0.30, help="手跟踪置信度门限")
    ap.add_argument("--hold-ms", type=float, default=250.0,
                    help="短暂丢失靶标时沿用上一帧位置多少毫秒（0 表示不沿用）")
    ap.add_argument("--hue", type=float, default=0.0,
                    help="色相，单位是度(0-358)：红0 橙30 黄60 绿120 青180 蓝224 紫270。"
                         "不是 OpenCV 的 0-179（OpenCV H × 2 = 这里）")
    ap.add_argument("--hue-tol", type=float, default=18.0)
    ap.add_argument("--sat-min", type=float, default=120.0)
    ap.add_argument("--val-min", type=float, default=60.0)
    ap.add_argument("--preview", action="store_true", help="显示本地窗口（需要桌面）")
    ap.add_argument("--stream", default=None, help="MJPEG 预览端口，例如 8770")
    args = ap.parse_args()

    host, port = args.udp.rsplit(":", 1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dst = (host, int(port))
    dsts = [dst]
    if args.udp2:
        h2, _, p2 = args.udp2.rpartition(":")
        dsts.append((h2 or "127.0.0.1", int(p2)))

    focus = None if args.focus < 0 else args.focus
    if str(args.camera).lower() == "auto":
        cap = None
        for cand in (6, 4, 2, 0, 1, 3, 5, 7):
            try:
                c = open_camera(cand, args.width, args.height, args.fps, focus)
            except SystemExit:
                continue
            ok, frame = c.read()
            if ok and frame is not None:
                cap = c
                print(f"自动选了 /dev/video{cand}", flush=True)
                break
            c.release()
        if cap is None:
            raise SystemExit("没找到可用相机")
    else:
        cap = open_camera(int(args.camera), args.width, args.height, args.fps, focus)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base = mp_python.BaseOptions(model_asset_path=args.model)
    opts = vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=args.det_conf,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=args.track_conf,
    )
    detector = vision.HandLandmarker.create_from_options(opts)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_det = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    t_start = time.time()
    last_val, last_ok = None, 0.0
    frames = 0
    lost = 0
    print(f"相机 {width}x{height}，UDP -> {host}:{port}，Ctrl-C 退出", flush=True)
    print(f"target={args.target}" + (f"，色相 {args.hue:.0f}°（OpenCV H={args.hue/2:.0f}）"
          f"，S≥{args.sat_min:.0f} V≥{args.val_min:.0f}" if args.target == "color" else ""),
          flush=True)

    last_log = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            lost += 1
            if lost > 30:
                print("连续 30 帧读不到画面，退出", file=sys.stderr, flush=True)
                break
            continue
        lost = 0
        frames += 1
        t = time.time()

        packet = {"t": t, "w": width, "h": height, "hand": None, "aruco": None}
        if args.target in ("hand", "both"):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                int((t - t_start) * 1000),
            )
            if res.hand_landmarks:
                packet["hand"] = hand_metrics(res.hand_landmarks[0], width, height)

        if args.target == "color":
            c = find_color(frame, args.hue, args.hue_tol, args.sat_min, args.val_min,
                           args.blob_min_area, args.blob_max_area,
                           args.color_circ, args.color_aspect)
            if c is not None:
                packet["aruco"] = {"u": c[0], "v": c[1], "r": c[2], "type": "color"}

        if packet["aruco"] is None and args.target in ("blob", "both"):
            b = find_blob(frame, args.blob_thresh, args.blob_invert,
                          args.blob_min_area, args.blob_max_area, args.blob_circ)
            if b is not None:
                packet["aruco"] = {"u": b[0], "v": b[1], "r": b[2], "type": "blob"}

        if packet["aruco"] is None and args.target in ("aruco", "both"):
            corners, ids, _ = aruco_det.detectMarkers(frame)
            if ids is not None and len(ids):
                i = int(np.argmin(np.abs(ids.ravel() - args.aruco_id)))
                if int(ids.ravel()[i]) == args.aruco_id:
                    packet["aruco"] = {
                        "u": float(corners[i][0][:, 0].mean()),
                        "v": float(corners[i][0][:, 1].mean()),
                        "type": "aruco",
                    }

        if packet["aruco"] is not None:
            last_val, last_ok = dict(packet["aruco"]), t
        elif args.hold_ms > 0 and last_val is not None and (t - last_ok) * 1000.0 <= args.hold_ms:
            packet["aruco"] = dict(last_val, hold=True)

        raw = json.dumps(packet).encode()
        for d in dsts:
            sock.sendto(raw, d)

        if t - last_log >= 1.0:
            last_log = t
            h = packet["hand"]
            if h:
                print(f"[视觉] 手 ({h['u']:.0f},{h['v']:.0f}) size {h['size']:.3f} pinch {h['pinch']:.2f}",
                      flush=True)
            else:
                print("[视觉] 没看到手", flush=True)

        if args.preview or args.stream:
            vis = frame.copy()
            if packet["hand"]:
                h = packet["hand"]
                cv2.circle(vis, (int(h["u"]), int(h["v"])), 8, (0, 255, 0), -1)
                cv2.putText(vis, f"pinch {h['pinch']:.2f} size {h['size']:.3f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if packet["aruco"]:
                a = packet["aruco"]
                r = int(a.get("r", 12))
                cv2.circle(vis, (int(a["u"]), int(a["v"])), max(r, 6), (0, 0, 255), 2)
                cv2.drawMarker(vis, (int(a["u"]), int(a["v"])), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
                cv2.putText(vis, f'{a.get("type","?")} r={r}px', (int(a["u"]) + 14, int(a["v"]) - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(vis, f"target: {args.target}", (20, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if args.preview:
                cv2.imshow("vision_track", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    dt = time.time() - t_start
    print(f"结束：{frames} 帧 / {dt:.1f}s = {frames/dt:.1f} fps", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())