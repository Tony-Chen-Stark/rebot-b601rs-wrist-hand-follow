#!/usr/bin/env bash
# 腕部相机手部跟随：视觉进程 + 跟随进程
#   bash run.sh            真机
#   bash run.sh --sim      仿真
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VPY="${HAND_VISION_PY:-$HOME/hand_vision_venv/bin/python}"
WS="${REBOT_WS:-$HOME/ReBot_Arm_DigitalTwin_RS}"
NS=/rebot_arm_rs; PORT=8901; ARGS=()
for a in "$@"; do
  case "$a" in
    --sim) NS=/rebot_arm_rs_sim ;;
    *) ARGS+=("$a") ;;
  esac
done
die(){ echo; echo "✗ $*"; [ -t 0 ] && read -r -p "按回车关闭…" _; exit 1; }
[ -f "$HERE/ibvs_track.py" ] || die "缺 ibvs_track.py"

pkill -f "vision_track.py .*127.0.0.1:${PORT}" 2>/dev/null && sleep 0.4
"$VPY" "$HERE/vision_track.py" --camera auto --target hand \
    --udp "127.0.0.1:${PORT}" --preview >"$HERE/vision.log" 2>&1 &
VPID=$!; trap 'kill $VPID 2>/dev/null' EXIT INT TERM
for i in $(seq 60); do grep -q "UDP ->" "$HERE/vision.log" 2>/dev/null && break; sleep 0.2; done
grep -q "UDP ->" "$HERE/vision.log" || die "视觉没起来，看 $HERE/vision.log"

set +u
source "$WS/scripts/rs_env.sh" >/dev/null 2>&1 || die "rs_env.sh 加载失败"
set -u
python3 "$HERE/ibvs_track.py" --ns "$NS" --port "$PORT" ${ARGS[@]+"${ARGS[@]}"} &
FPID=$!
trap 'kill "$FPID" "$VPID" 2>/dev/null' EXIT INT TERM
wait "$FPID"
