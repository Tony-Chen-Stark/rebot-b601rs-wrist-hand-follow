# 腕部相机 · 手部跟随（reBot B601 RS）

相机装在机械臂末端，手放在相机前面，手往哪边移，机械臂就朝哪边移，让手回到画面中心；同时按手的大小保持距离。

演示视频：`media/wanbu.mp4`

---

## 硬件

| 部件 | 说明 |
|---|---|
| 机械臂 | reBot Arm B601 **RS**（灵足 Robstride 电机，24V） |
| 相机 | 普通 UVC 相机，640×480，固定在腕部（眼在手上） |
| 主机 | Ubuntu 22.04 / 24.04，ROS 2 Jazzy，CAN 1 Mbit |

软件依赖：官方 `ReBotArmController_ROS2`（`/rebot_arm_rs`）、`ReBotArm_control_py`（pinocchio）、MediaPipe、OpenCV。

## 方法

1. **手部关键点**：MediaPipe 给出掌心位置与手掌大小，作为 3 维图像特征：横向、纵向、远近。
2. **交互矩阵实测**：不套公式。启动时让末端沿基座 x/y/z 各平移 1 cm，量出特征变化，得到 3×3 的交互矩阵，存盘复用。
3. **速度式伺服**：`v = -g·J⁺·e`，在线用 Broyden 修正矩阵；对掌心做 alpha-beta 滤波与短延迟预测。
4. **关节积分**：`q ← q + J_arm⁺·v·dt`，不调用逆解，因此没有解支跳变；带关节限位、限速、跟随后误差保护。
5. **输出**：60 Hz 向 `/{ns}/joints/jointN/cmd/mit` 发位置指令。
6. **夹爪**（可选）：手势捏合 → 关闭，张开 → 打开，限速执行。

## 参考

- [mengyanggao/visual_servo_tracking](https://github.com/mengyanggao/visual_servo_tracking) — 视觉伺服框架，IBVS 目标函数与相机速度映射
- Chaumette & Hutchinson, *Visual Servo Control, Part I: Basic Approaches*, IEEE RAM 2006
- [Seeed-Projects/ReBotArm_control_py](https://github.com/Seeed-Projects/ReBotArm_control_py) — pinocchio 运动学
- 官方 `web_mujoco/src/tcp-ik.js` — 阻尼最小二乘 IK 参数
- [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe) — 手部关键点

## 运行

```bash
# 1) 机械臂上电，等 5 秒；确认 can0 UP（1 Mbit）

# 2) 起控制器
cd ~/ReBot_Arm_DigitalTwin_RS
REBOTARM_RS_HARDWARE_CONFIRM=I_UNDERSTAND_RS_WILL_MOVE ./rebotarm start rs

# 3) 起跟随
bash run.sh           # 真机
bash run.sh --sim     # MuJoCo 仿真
```

首次运行会做一次「试探」：机械臂小幅平移几厘米，量交互矩阵，存 `mount.json`。之后再跑会直接复用。

### 常用参数

| 目的 | 参数 |
|---|---|
| 整体快慢 | `--gain 0.6` |
| 限速 | `--vmax 0.05` |
| 只在画面中心附近才动 | `--deadband 15` |
| 重新量交互矩阵 | `--probe-force` |
| 不要夹爪 | `--no-grip` |

### 停止

跟随终端 `Ctrl-C` 后：

```bash
ros2 service call /rebot_arm_rs/disable std_srvs/srv/Trigger
./rebotarm stop
```

**不要对控制器终端按 Ctrl-C**，那会直接切断电机使能、机械臂掉下来。

## 文件

| 文件 | 作用 |
|---|---|
| `run.sh` | 启动视觉进程 + 跟随进程 |
| `ibvs_track.py` | 跟随主程序（IBVS） |
| `vision_track.py` | 视觉：手部关键点 → UDP |
| `follow_hand_v2.py` | ROS 接口与 IK |
| `models/` | MediaPipe 手部模型 |

## 许可

MIT