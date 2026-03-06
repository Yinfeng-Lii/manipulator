#!/usr/bin/env bash
set -euo pipefail

# One-click bringup for real hardware mode:
# 1) move_group + ros2_control + controllers
# 2) arm_worker (serial hardware path)
# 3) optional commander CLI

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="${WS_DIR:-$ROOT_DIR}"
SETUP_FILE="${SETUP_FILE:-$WS_DIR/install/setup.bash}"
LOG_DIR="${LOG_DIR:-$WS_DIR/.logs}"
WITH_COMMANDER="${WITH_COMMANDER:-0}"
MOVE_GROUP_DELAY_SEC="${MOVE_GROUP_DELAY_SEC:-4}"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --with-commander      Start arm_pick_place commander after worker is up.
  --ws-dir <path>       ROS2 workspace directory (default: repo root).
  --setup <path>        setup.bash path (default: <ws-dir>/install/setup.bash).
  --delay <sec>         Delay before starting arm_worker (default: 4).
  -h, --help            Show this help.

Environment overrides:
  WS_DIR, SETUP_FILE, LOG_DIR, WITH_COMMANDER, MOVE_GROUP_DELAY_SEC
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-commander)
      WITH_COMMANDER=1
      shift
      ;;
    --ws-dir)
      WS_DIR="$2"
      shift 2
      ;;
    --setup)
      SETUP_FILE="$2"
      shift 2
      ;;
    --delay)
      MOVE_GROUP_DELAY_SEC="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
 done

if [[ ! -f "$SETUP_FILE" ]]; then
  echo "[ERROR] setup file not found: $SETUP_FILE" >&2
  echo "        Please run colcon build first, or pass --setup <path>." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
MOVE_GROUP_LOG="$LOG_DIR/move_group_real.log"
WORKER_LOG="$LOG_DIR/arm_worker_real.log"
COMMANDER_LOG="$LOG_DIR/commander.log"

# shellcheck disable=SC1090
source "$SETUP_FILE"

echo "[INFO] WS_DIR=$WS_DIR"
echo "[INFO] SETUP_FILE=$SETUP_FILE"
echo "[INFO] ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<unset>}"
echo "[INFO] ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-<unset>}"
echo "[INFO] Logs: $LOG_DIR"

cleanup() {
  set +e
  echo "\n[INFO] Stopping launched processes..."
  [[ -n "${CMD_PID:-}" ]] && kill "$CMD_PID" 2>/dev/null
  [[ -n "${WORKER_PID:-}" ]] && kill "$WORKER_PID" 2>/dev/null
  [[ -n "${MOVEIT_PID:-}" ]] && kill "$MOVEIT_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "[INFO] Starting move_group.launch.py ..."
ros2 launch mycobot_moveit_config move_group.launch.py >"$MOVE_GROUP_LOG" 2>&1 &
MOVEIT_PID=$!

sleep "$MOVE_GROUP_DELAY_SEC"

echo "[INFO] Starting arm_worker ..."
ros2 run arm_pick_place arm_worker >"$WORKER_LOG" 2>&1 &
WORKER_PID=$!

if [[ "$WITH_COMMANDER" == "1" ]]; then
  echo "[INFO] Starting commander ..."
  ros2 run arm_pick_place commander >"$COMMANDER_LOG" 2>&1 &
  CMD_PID=$!
fi

echo "[OK] Real-hardware bringup started."
echo "     move_group log : $MOVE_GROUP_LOG"
echo "     arm_worker log : $WORKER_LOG"
if [[ "$WITH_COMMANDER" == "1" ]]; then
  echo "     commander log  : $COMMANDER_LOG"
fi

echo "[INFO] Press Ctrl+C to stop all." 
wait "$WORKER_PID"
