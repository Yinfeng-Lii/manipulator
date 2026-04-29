#!/usr/bin/env python3
import os
import sys
import time
import math
import re
from enum import Enum, auto
from dataclasses import dataclass

pymycobot_path = os.environ.get("PYMYCOBOT_PATH", os.path.expanduser("~/arm_ws/src/pymycobot"))
if os.path.isdir(pymycobot_path):
    sys.path.insert(0, pymycobot_path)

from pymycobot import MyCobot280  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from std_msgs.msg import String, Bool    # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402

from moveit_msgs.srv import GetMotionPlan  # noqa: E402
from moveit_msgs.srv import GetPositionFK  # noqa: E402
from moveit_msgs.msg import (  # noqa: E402
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    MoveItErrorCodes,
    RobotState,
)
from shape_msgs.msg import SolidPrimitive  # noqa: E402
from geometry_msgs.msg import PoseStamped, Quaternion  # noqa: E402

from trajectory_msgs.msg import JointTrajectory  # noqa: E402

def quat_from_rpy(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def moveit_error_to_str(val: int) -> str:
    m = {
        MoveItErrorCodes.SUCCESS: "SUCCESS",
        MoveItErrorCodes.FAILURE: "FAILURE",
        MoveItErrorCodes.PLANNING_FAILED: "PLANNING_FAILED",
        MoveItErrorCodes.INVALID_MOTION_PLAN: "INVALID_MOTION_PLAN",
        MoveItErrorCodes.TIMED_OUT: "TIMED_OUT",
        MoveItErrorCodes.START_STATE_IN_COLLISION: "START_STATE_IN_COLLISION",
        MoveItErrorCodes.GOAL_IN_COLLISION: "GOAL_IN_COLLISION",
        MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED: "GOAL_CONSTRAINTS_VIOLATED",
        MoveItErrorCodes.FRAME_TRANSFORM_FAILURE: "FRAME_TRANSFORM_FAILURE",
        MoveItErrorCodes.NO_IK_SOLUTION: "NO_IK_SOLUTION",
    }
    return m.get(val, f"UNKNOWN({val})")


def _extract_joint_index(name: str):
    nums = re.findall(r"\d+", name)
    if not nums:
        return None
    return int(nums[0])

class State(Enum):
    IDLE = auto()
    PLAN_EXEC = auto()
    WAIT_PRE = auto()
    HOMING = auto()



@dataclass
class Task:
    mode: str
    x: float
    y: float
    z: float
    retries_left: int = 1  # 抓取失败：从 home 重试一次（默认 1 次）
    # ================= 移植：新增目标旋转角属性 =================
    target_rz: float = None 
    # ============================================================
    # ================= 0311修改：增加正方体大小属性 =================
    size: str = "big" 
    # ================================================================
    # 当夹爪闭合后读数落在该区间[min,max]时，触发松开+旋转45°再抓取
    regrasp_trigger_value: tuple = None



class ArmWorker(Node):
    def __init__(self):
        super().__init__("arm_worker_node")
        # ---------------- HW ----------------
        self.port = "/dev/ttyAMA0"
        self.baud = 1000000
        self.mc = None

        # ---------------- MoveIt----------------
        self.group_name = "arm"
        self.ee_link = "tcp"
        self.base_frame = "base_link"  # REP-103: x左 y前 z上
        self.allowed_planning_time = 8.0
        self.num_planning_attempts = 20
        self.max_vel_scale = 0.15
        self.max_acc_scale = 0.15

        # OMPL vs LIN position box
        # Position constraint box
        self.pos_box_m = 0.03
        self.pos_box_m_lin = 0.0001
        
        # pick 任务为了避免横向漂移，收紧 pregrasp 与下抓目标容差
        self.pick_pregrasp_box_m = 0.0001
        self.pick_down_box_m = 0.00001
        self.place_goal_box_m = 0.0003
        self.place_down_box_m = 0.0002
        self.lin_goal_radius_m = 0.0002
        self.lin_max_vel_scale = 0.08
        self.lin_max_acc_scale = 0.03

        # ---------------- 末端约束：tcp +Y 对齐 base -Z（保持第一版） ----------------
        self.ee_roll_deg = -90.0
        self.ee_pitch_deg = 0.0
        self.ee_yaw_deg = 0.0

        # 允许绕“竖直轴”(tcp 的 Y) 自由转动：放开 about Y
        self.tilt_tol_rad = math.radians(3.0)
        self.free_about_y_rad = math.pi
        self.ori_weight = 1.0

        # ---------------- Pilz LIN（保证垂直直线下抓） ----------------
        self.pilz_pipeline_id = "pilz_industrial_motion_planner"
        self.pilz_lin_planner_id = "LIN"
        
        # ---------------- Behavior ----------------
        self.pregrasp_offset_mm = 30.0
        self.pregrasp_wait_sec = 1
        self.place_settle_sec = 0.25

        # ---------------- Home ----------------
        self.home_angles_deg = [0, 0, 0, 0, 0, 0]
        self.home_speed = 25
        self.home_timeout_sec = 10.0
        self.home_open_gripper = True

        # ---------------- Smooth HW exec ----------------
        self.hw_rate_hz = 190.0
        self.hw_time_scale = 0.2
        self.hw_send_speed = 30
        self.hw_exec_settle_timeout_sec = 1.2

        # ---------------- Gripper ----------------
        self.gripper_cmd_speed = 50
        self.gripper_open_value = 100
        self.gripper_close_value = 5
        self.gripper_verify_tol = 20 
        self.gripper_retries = 3


        # 二次抓取前：要求夹爪更充分张开再旋转，避免带着物体一起转
        self.regrasp_open_full_tol = 5
        self.regrasp_open_confirm_wait_sec = 0.25

        # pick big/small 对应“二次抓取触发区间”（闭合后读数落在区间内时触发）
        self.regrasp_trigger_big_range = (70, 82)
        self.regrasp_trigger_small_range = (35, 45)

        # 大/小正方体：边长抓取与对角抓取时的典型夹爪读数（用于动态映射旋转角）
        self.grasp_profile = {
            "big": {"edge": 50, "diag": 78},
            "small": {"edge": 29, "diag": 42},
        }
        self.regrasp_max_rotate_deg = 45.0
        self.regrasp_loop_max_attempts = 3
        # 闭合值接近“边长抓取”时不旋转的容差
        self.regrasp_edge_no_rotate_tol = 15
        # 兼容旧字段名（历史版本曾使用 *_value 单点触发）
        self.regrasp_trigger_big_value = 79
        self.regrasp_trigger_small_value = 41

        # 抓取即时判定 + 重试节奏
        self.grasp_check_delay_sec = 0.35
        self.grasp_success_min_value = 5
        self.grasp_retry_pause_sec = 0.4

        # ---------------- 掉落检测（保留第二版逻辑） ----------------
        self.gripper_monitor_period = 1 #0306修改：掉落检测频率为每秒一次
        self.gripper_drop_threshold = 10
        self._gripper_monitor_timer = None

        # ================= 移植：0225新增：定义机械臂 XYZ 坐标的物理极限范围 (单位: mm) =================
        self.limit_x_min, self.limit_x_max = -281.45, 281.45
        self.limit_y_min, self.limit_y_max = -281.45, 281.45
        self.limit_z_min, self.limit_z_max = -70.0, 412.67
        # ====================================================================

        # ---------------- ROS ----------------
        self.plan_cli = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        self.fk_cli = self.create_client(GetPositionFK, "/compute_fk")
        self.sub = self.create_subscription(String, "arm_command", self.command_callback, 10)
        
        # ================= 移植：0225新增：创建发布者，用于向发送指令的节点发送报错/超限信息 =================
        self.feedback_pub = self.create_publisher(String, "arm_feedback", 10)
        # ====================================================================
        self.holding_state_pub = self.create_publisher(Bool, "arm_holding_state", 10)
        self.holding_state_timer = self.create_timer(0.1, self._publish_holding_state)

        self.js_pub = self.create_publisher(JointState, "/joint_states", 10)

        # joint mapping (MoveIt -> HW)
        self.moveit_joint_names = None
        self._moveit_name_to_hw_idx = None
        self._last_joint_rad_by_moveit = None

        # ---------------- FSM ----------------
        self.state = State.IDLE
        self.has_object = False
        self._queue = []
        self._task: Task | None = None
        self._steps = []
        self._step_idx = 0
        self._token = 0
        self._wait_timer = None
        self._place_step1_fallback_used = False
        # ---------------- Connect HW ----------------
        try:
            self.mc = MyCobot280(self.port, self.baud)
            self.mc.power_on()
            time.sleep(0.5)
            self.get_logger().info(f"Robotic arm connection successful:")
            self._init_gripper()
        except Exception as e:
            self.mc = None
            self.get_logger().warn(f"Not connected to real device（{e}）.")


        if self.mc:
            self.timer_hw = self.create_timer(0.05, self._poll_hw_and_publish_joint_states)
            self._gripper_monitor_timer = self.create_timer(self.gripper_monitor_period, self._monitor_gripper)

        self.get_logger().info("Waiting for /plan_kinematic_path ...")
        ok = self.plan_cli.wait_for_service(timeout_sec=30.0)
        if not ok:
            self.get_logger().error("plan_kinematic_path not available. Is move_group running?")

        self.timer_tick = self.create_timer(0.02, self._tick)
        self.get_logger().info("ArmWorker ready.")

    def _publish_pick_pose_error(self, x_mm: float, y_mm: float, z_mm: float):
        if (not self.mc) or (self.moveit_joint_names is None) or (self._last_joint_rad_by_moveit is None):
            return
        if not self.fk_cli.service_is_ready():
            return

        req = GetPositionFK.Request()
        req.header.frame_id = self.base_frame
        req.fk_link_names = [self.ee_link]

        rs = RobotState()
        js = JointState()
        js.name = self.moveit_joint_names
        js.position = self._last_joint_rad_by_moveit
        js.header.stamp = self.get_clock().now().to_msg()
        rs.joint_state = js
        req.robot_state = rs

        fut = self.fk_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=1.0)
        if not fut.done():
            return
        try:
            resp = fut.result()
        except Exception:
            return
        if (resp is None) or (len(resp.pose_stamped) == 0):
            return

        p = resp.pose_stamped[0].pose.position
        ax_mm = float(p.x) * 1000.0
        ay_mm = float(p.y) * 1000.0
        az_mm = float(p.z) * 1000.0
        ex = ax_mm - float(x_mm)-3
        ey = ay_mm - float(y_mm)+7
        ez = az_mm - float(z_mm)-24.0
        en = math.sqrt(ex * ex + ey * ey + ez * ez)

        msg = String()
        msg.data = (
            f"pick error(mm): target=({x_mm:.2f},{y_mm:.2f},{z_mm:.2f}), "
            f"actual=({ax_mm:.2f},{ay_mm:.2f},{az_mm:.2f}), "
            f"error=({ex:.2f},{ey:.2f},{ez:.2f}), norm={en:.2f}"
        )
        self.feedback_pub.publish(msg)
        self.get_logger().warn(msg.data)

    # -----------------------------
    # busy 判断（掉落检测用）
    # -----------------------------
    def _is_busy(self) -> bool:
        return self.state in (State.PLAN_EXEC, State.WAIT_PRE, State.HOMING)

    def _publish_holding_state(self):
        msg = Bool()
        # self.has_object 标志位在抓取验证、掉落检测中会实时更新
        msg.data = bool(self.has_object)
        self.holding_state_pub.publish(msg)

    # ====================================================================
    # ================= 移植：✅ 0225新增：坐标范围拦截与反馈器函数 =================
    # ====================================================================
    def _check_limits_and_feedback(self, x, y, z) -> bool:
        """
        检查目标坐标是否超出机械臂物理极限，如果超限则发布消息说明具体超出的轴和数值。
        返回 True 表示安全，返回 False 表示超限拦截。
        """
        errors = []
        
        # 检查 X 轴
        if x < self.limit_x_min:
            errors.append(f"The X-axis is out of range {self.limit_x_min - x:.2f} mm")
        elif x > self.limit_x_max:
            errors.append(f"The X-axis is out of range {x - self.limit_x_max:.2f} mm")

        # 检查 Y 轴
        if y < self.limit_y_min:
            errors.append(f"The Y-axis is out of range {self.limit_y_min - y:.2f} mm")
        elif y > self.limit_y_max:
            errors.append(f"The Y-axis is out of range {y - self.limit_y_max:.2f} mm")
            
        # 检查 Z 轴
        if z < self.limit_z_min:
            errors.append(f"The Z-axis is out of range {self.limit_z_min - z:.2f} mm")
        elif z > self.limit_z_max:
            errors.append(f"The Z-axis is out of range {z - self.limit_z_max:.2f} mm")
            
        if errors:
            # 拼接完整的错误信息
            error_msg = f"coordinate({x:.1f}, {y:.1f}, {z:.1f}) is invalid: " + "; ".join(errors)
            
            # 在本地终端打印红字报警
            self.get_logger().error(error_msg)
            
            # 通过 ROS 话题把报警信息发送给调用者
            msg = String()
            msg.data = error_msg
            self.feedback_pub.publish(msg)
            
            return False # 拦截，不安全
            
        return True # 安全
    # ====================================================================

    # =========================================================
    # 掉落检测（后台）
    # =========================================================
    def _monitor_gripper(self):
        if (not self.mc) or (not self.has_object):
            return
        if self._is_busy():
            return
        try:
            # =======================================================
            # ✅ 0306 新增修改：主动保压机制 (Active Clamping)
            # 每隔 0.5s，用很低的速度(如 15) 再次下发闭合指令。
            # 如果物体掉了，这个指令会立刻让空夹爪完全闭合。
            self.mc.set_gripper_value(self.gripper_close_value, 15)
            # 给出 0.4 秒的物理响应时间让夹爪动一动
            time.sleep(0.4) 
            # =======================================================
            v = self.mc.get_gripper_value()
            if v is None:
                return
            if v < self.gripper_drop_threshold:
                self.get_logger().warn(f"Drop! Grip value ={v} < {self.gripper_drop_threshold}")
                self.has_object = False
                # ================= ✅ 0306新增修改：掉落后自动恢复 =================
                self.get_logger().warn("The object fell, return to Home...")
                
                # 步骤 A：立刻强行张开夹爪，防止夹爪钩拽住半掉落的物体划伤桌面
                self._gripper_open()
                # 步骤 B：清空可能还在排队等待的其它指令，防止发生动作逻辑混乱
                self._queue.clear()
                # 步骤 C：调用回零函数，让机械臂安全退回全部为 0 度的初始姿态
                self._enter_home()
                
                self.get_logger().warn("has_object=False Waiting for the next instruction (or to pick again).")
        except Exception as e:
            self.get_logger().error(f"Fall detection failed to read: {e}")

    # =========================================================
    # 夹爪：设置并验证 + 重试
    # =========================================================
    def _set_gripper_and_verify(
        self,
        target_value: int,
        speed: int,
        *,
        expect_open: bool,
        retries: int = 3,
        wait_each: float = 0.5,
        tol: int = 20,
    ) -> bool:
        if not self.mc:
            return True

        for k in range(max(1, int(retries))):
            try:
                try:
                    self.mc.set_gripper_value(int(target_value), int(speed))
                except AttributeError:
                    state = 0 if expect_open else 1
                    self.mc.set_gripper_state(int(state), int(speed))

                time.sleep(float(wait_each))

                try:
                    v = self.mc.get_gripper_value()
                except Exception:
                    v = None

                if v is None:
                    time.sleep(0.2)
                    continue

                if expect_open:
                    if v >= (int(target_value) - int(tol)):
                        return True
                else:
                    if v <= (int(target_value) + int(tol)):
                        return True
            except Exception:
                pass
            time.sleep(0.2)

        return False

    def _init_gripper(self):
        if not self.mc:
            return
        ok = self._set_gripper_and_verify(
            self.gripper_open_value,
            self.gripper_cmd_speed,
            expect_open=True,
            retries=2,
            wait_each=1.0,
            tol=self.gripper_verify_tol,
        )
        if not ok:
            self.get_logger().warn("Failed to initialize gripper open state")
        self.has_object = False
    
    def _gripper_open(self) -> bool:       
        if not self.mc:
            self.get_logger().error("Real device connection failed")
            return False



        return self._set_gripper_and_verify(
            self.gripper_open_value,
            self.gripper_cmd_speed,
            expect_open=True,
            retries=self.gripper_retries,
            wait_each=0.9,
            tol=self.gripper_verify_tol,
        )

    def _gripper_close(self) -> bool:
        if not self.mc:
            self.get_logger().error("Real device connection failed")
            return False


        return self._set_gripper_and_verify(
            self.gripper_close_value,
            self.gripper_cmd_speed,
            expect_open=False,
            retries=self.gripper_retries,
            wait_each=1.0,
            tol=self.gripper_verify_tol,
        )

    # =========================================================
    # 抓取即时判定：夹爪闭合后立即读值判断
    #=====================================================
    def _verify_grasp_now(self) -> bool:
        if not self.mc:
            self.get_logger().error("Real device connection failed")
            return False


        time.sleep(float(self.grasp_check_delay_sec))

        try:
            v = self.mc.get_gripper_value()
        except Exception as e:
            self.get_logger().warn(f"Grasp verification: Failed to read gripper value ({e}) -> Handle as failure")
            return False

        if v is None:
            self.get_logger().warn("Grasp verification: Gripper value is None -> Handle as failure")
            return False

        if v <= int(self.grasp_success_min_value):
            self.get_logger().warn(f"Grasp verification: Gripper value ={v} <= {self.grasp_success_min_value} -> Handle as failure")
            return False
        return True

    # =========================================================
    # HW -> joint_states
    # =========================================================
    def _poll_hw_and_publish_joint_states(self):
        try:
            deg = self.mc.get_angles()
            if not deg or len(deg) < 6:
                return
        except Exception:
            return

        rad_hw = [math.radians(x) for x in deg[:6]]
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()

        if self.moveit_joint_names is None:
            js.name = [f"joint{i}" for i in range(1, 7)]
            js.position = rad_hw
            self.js_pub.publish(js)
            return

        js.name = self.moveit_joint_names
        pos = [0.0] * 6
        for mi, hw_i in enumerate(self._moveit_name_to_hw_idx):
            pos[mi] = rad_hw[hw_i]
        js.position = pos
        self.js_pub.publish(js)
        self._last_joint_rad_by_moveit = pos

    def _learn_moveit_joint_mapping(self, traj_joint_names):
        names = list(traj_joint_names[:6])
        indices = []
        for n in names:
            idx = _extract_joint_index(n)
            if idx is None or idx < 1 or idx > 6:
                return False
            indices.append(idx - 1)
        self.moveit_joint_names = names
        self._moveit_name_to_hw_idx = indices
        return True

    # Command
    def command_callback(self, msg: String):
        s = msg.data.strip()
        cmd = s.lower()
        self.get_logger().info(f"Received command: {s}")

        if cmd == "home":
            self._queue.clear()
            self._queue.append(("home",))
            self._enter_home()
            return

        parts = cmd.split()
        if len(parts) == 0:
            return

        mode = parts[0]
        
        # ================= 移植：1. 模式合法性统一拦截 =================
        if mode not in ("pick", "place"):
            self.get_logger().error("Mode must be pick or place")
            return

        if mode == "pick":
            # ================= 0311修改：增加正方体大小：big/small =================
            if len(parts) != 5:
                self.get_logger().error("Format error! pick requires 4 parameters: size x y z")
                return
            
            size_str = parts[1].lower()
            if size_str not in ("big", "small"):
                self.get_logger().error("Cube size must be big or small")
                return

            try:
                cx = float(parts[2])
                cy = float(parts[3])
                z = float(parts[4])
            # =======================================================================
            except ValueError:
                self.get_logger().error("Coordinates must be numbers")
                return

            OFFSET_X = 0.0  
            OFFSET_Y = 0 
            OFFSET_Z = 0   
            

            cx += OFFSET_X
            cy += OFFSET_Y
            z += OFFSET_Z

            # ✅ 0225新增：在抓取前，检查经过 OFFSET 补偿后的质心是否超限
            if not self._check_limits_and_feedback(cx, cy, z):
                return # 如果超限，函数 _check_limits_and_feedback 已经发布了报警消息，这里直接终止任务
            
            # ================= 0311修改：只需要抓取正方体方块，不旋转 =================
            j6_angle = None 
            
            regrasp_trigger_value = (
                self.regrasp_trigger_big_range if size_str == "big" else self.regrasp_trigger_small_range
            )

            # ================= 0311修改：任务增加 size 属性 =================
            self._queue.append(
                Task(
                    "pick", cx, cy, z,
                    retries_left=1,
                    target_rz=j6_angle,
                    size=size_str,
                    regrasp_trigger_value=regrasp_trigger_value,
                )
            )
            # ================================================================
            return

        # ================= 移植：新增：独立的 place 逻辑(目前不补偿相机偏置，为机械臂坐标系) =================
        elif mode == "place":
            if len(parts) != 4:
                self.get_logger().error("Format error! place requires 3 parameters: x y z")
                return
                
            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
            except ValueError:
                self.get_logger().error("Coordinates must be numbers")
                return
            
            # ✅ 0225新增：放置指令同样需要检查是否超出机械臂物理界限
            if not self._check_limits_and_feedback(x, y, z):
                return

            self._queue.append(Task("place", x, y, z, retries_left=1, target_rz=None))
        return 

    # =========================================================
    # Tick / FSM
    # =========================================================
    def _tick(self):
        if self.state != State.IDLE:
            return
        if not self._queue:
            return

        item = self._queue.pop(0)
        if isinstance(item, tuple) and item[0] == "home":
            self._enter_home()
            return

        assert isinstance(item, Task)
        t = item

        if t.mode == "pick" and self.has_object:
            self.get_logger().warn("object already exists, ignore the pick function")
            return
        if t.mode == "place" and (not self.has_object):
            self.get_logger().warn("nothing in manipulator, so ignore place.")
            return

        pre_z = t.z + self.pregrasp_offset_mm
        self._task = t
        self._steps = [(t.x, t.y, pre_z), (t.x, t.y, t.z)]
        self._step_idx = 0
        self._place_step1_fallback_used = False
        self._start_step()

    def _start_step(self):
        if self._task is None:
            self.state = State.IDLE
            return

        if self._step_idx >= len(self._steps):
            self._enter_home()
            return
        if self._task.mode != "place" or self._step_idx != 1:
            self._place_step1_fallback_used = False

        self.state = State.PLAN_EXEC
        self._token += 1
        token = self._token
        x, y, z = self._steps[self._step_idx]

        # ✅ 保证直线下抓：pick 的 step1 使用 Pilz LIN（保持第一版）
        pipeline_id = None
        planner_id = None
        goal_box_m = None
        if self._task.mode == "pick":
            # pick 过程移动时保持夹爪最大张开，直到到达抓取位才闭合
            if self._step_idx in (0, 1):
                self._gripper_open()

            # step0 先把 x/y 收紧到目标点正上方，减少 step1 横向修正
            if self._step_idx == 0:
                goal_box_m = self.pick_pregrasp_box_m
            # step1 使用 Pilz LIN 下抓，并进一步收紧目标容差，尽量保证纯竖直下移
            elif self._step_idx == 1:
                pipeline_id = self.pilz_pipeline_id
                planner_id = self.pilz_lin_planner_id
                goal_box_m = self.pick_down_box_m
        elif self._task.mode == "place":
            if self._step_idx == 1 and (not self._place_step1_fallback_used):
                pipeline_id = self.pilz_pipeline_id
                planner_id = self.pilz_lin_planner_id
                goal_box_m = self.place_down_box_m
            else:
                goal_box_m = self.place_goal_box_m

        if planner_id == self.pilz_lin_planner_id:
            req = self._build_lin_plan_request_with_start_state(x, y, z)
        else:
            req = self._build_plan_request_with_start_state(
                x, y, z,
                pipeline_id=pipeline_id,
                planner_id=planner_id,
                goal_box_m=goal_box_m,
            )

        self.plan_cli.call_async(req).add_done_callback(lambda f: self._on_plan_done(f, token))

    def _on_plan_done(self, fut, token: int):
        if token != self._token:
            return
        try:
            result = fut.result()
        except Exception as e:
            self._handle_runtime_error(f"planning exception: {e}")
            return

        res = result.motion_plan_response
        code = res.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            if self._task and self._task.mode == "place" and self._step_idx == 1 and (not self._place_step1_fallback_used):
                self._place_step1_fallback_used = True
                self.state = State.IDLE
                self._start_step()
                return

            self._handle_runtime_error(f"planning failed: {moveit_error_to_str(code)} ({code})")
            return


        traj = res.trajectory.joint_trajectory
        if (not traj.joint_names) or (len(traj.points) == 0):
            self._handle_runtime_error("planning returned empty trajectory")
            return

        if self.mc and self.moveit_joint_names is None:
            if not self._learn_moveit_joint_mapping(traj.joint_names):
                self.get_logger().error(f"无法解析 joint_names: {traj.joint_names}")
                self._reset()
                return
        ok = self._exec_hw_smooth_interpolated(traj)
        if not ok:
            self._handle_runtime_error("hardware trajectory execution failed")

            return
        self._on_step_finished()

    def _on_step_finished(self):
        idx = self._step_idx

        # pick：到预抓取点后等待，再下抓
        if self._task and self._task.mode == "pick" and idx == 0:
            self.state = State.WAIT_PRE
            self._token += 1
            token = self._token
            self._cancel_wait_timer()
            self._wait_timer = self.create_timer(self.pregrasp_wait_sec, lambda: self._on_wait_done(token))
            return

        # step1 到目标点：pick / place 的末端动作
        if self._task and idx == 1:
            if self._task.mode == "pick":
                self._gripper_close()

                # 抓取后始终进行夹爪读数检测：仅当读数约等于边长时不旋转，否则按映射角重抓
                if self.mc:
                    try:
                        for i in range(int(self.regrasp_loop_max_attempts)):
                            v = self.mc.get_gripper_value()
                            if v is None:
                                break
                            vi = int(v)
                            rotate_deg = self._compute_regrasp_rotate_deg(self._task.size, vi)
                            if rotate_deg <= 0.0:
                                break
                            # 先最大张开（确认到位）再旋转
                            opened = self._gripper_open_fully_before_regrasp()
                            if not opened:
                                continue

                            curr_angles = self.mc.get_angles()
                            if curr_angles and len(curr_angles) >= 6:
                                self.mc.send_angle(6, curr_angles[5] + rotate_deg, 50)
                                time.sleep(0.5)
                            self._gripper_close()
                    except Exception as e:

                        self.get_logger().error(f"Second fetch failed: {e}")

                ok = self._verify_grasp_now()
                self.has_object = bool(ok)
                self._publish_pick_pose_error(self._task.x, self._task.y, self._task.z)

                if ok:
                    self._enter_home()
                    return

                # 失败：从 home 重试一次（保留第二版 E）
                self.get_logger().warn("Fetch failed, re-fetch from Home.")

                # 失败时务必开爪（保证下一轮从 home 开始夹爪张开）
                self._gripper_open()
                self.has_object = False

                if self._task.retries_left > 0:
                    retry_task = Task(
                        mode="pick",
                        x=self._task.x,
                        y=self._task.y,
                        z=self._task.z,
                        retries_left=self._task.retries_left - 1,
                        # ================= 移植：重试时保留旋转角 =================
                        target_rz=self._task.target_rz,
                        # ==========================================================
                        # ================= 0311修改：重试时保留正方体大小 =================
                        size=self._task.size,
                        # ==================================================================
                        regrasp_trigger_value=self._task.regrasp_trigger_value,
                    )

                    time.sleep(float(self.grasp_retry_pause_sec))
                    self._enter_home()
                    time.sleep(float(self.grasp_retry_pause_sec))

                    # 双保险：回 home 后再开一次
                    try:
                        self._gripper_open()
                    except Exception:
                        pass

                    # 插队重试：从 home 开始再次跑 step0/step1（同一点）
                    self._queue.insert(0, retry_task)
                    return

                self.get_logger().warn("Fetch failed and no retries left, returning to Home to wait for the next command")
                self._enter_home()
                return

            else:
                if self.mc:
                    time.sleep(float(self.place_settle_sec))
                ok = self._gripper_open()
                if self.mc:
                    try:
                        v = self.mc.get_gripper_value()
                    except Exception:
                        pass
                self.has_object = False
                self._enter_home()
                return

        # 其它情况：推进下一 step
        self._step_idx += 1
        self.state = State.IDLE
        self._start_step()

    def _on_wait_done(self, token: int):
        self._cancel_wait_timer()
        if token != self._token:
            return
        if self.state != State.WAIT_PRE:
            return
        self.state = State.IDLE
        self._step_idx = 1
        self._start_step()

    def _cancel_wait_timer(self):
        if self._wait_timer is not None:
            try:
                self._wait_timer.cancel()
            except Exception:
                pass
            self._wait_timer = None
    
    def _gripper_open_fully_before_regrasp(self) -> bool:
        """二次抓取专用：要求夹爪尽量张开到最大值附近再允许旋转。"""
        ok = self._gripper_open()
        if not self.mc:
            return False

        target_min = int(self.gripper_open_value) - int(self.regrasp_open_full_tol)
        for _ in range(max(1, int(self.gripper_retries))):
            try:
                time.sleep(float(self.regrasp_open_confirm_wait_sec))
                v = self.mc.get_gripper_value()
                if v is not None and int(v) >= target_min:
                    return True
                # 再次补发开夹指令
                self._gripper_open()
            except Exception:
                pass
        return False

    def _compute_regrasp_rotate_deg(self, size: str, gripper_value: int) -> float:
        """根据物体大小与夹爪闭合读数动态计算二次抓取旋转角。
        仅当读数明显偏离边长抓取值时旋转；越接近对角线值，旋转角越大。
        """
        profile = self.grasp_profile.get(size)
        if not profile:
            return 0.0

        edge = float(profile["edge"])
        diag = float(profile["diag"])
        if diag <= edge:
            return 0.0

        v = float(gripper_value)
        if abs(v - edge) <= float(self.regrasp_edge_no_rotate_tol):
            return 0.0

        ratio = (v - edge) / (diag - edge)
        ratio = max(0.0, min(1.0, ratio))
        return float(self.regrasp_max_rotate_deg) * ratio


    def _is_near_gripper_min(self, gripper_value: int) -> bool:
        return abs(int(gripper_value) - int(self.gripper_close_value)) <= int(self.regrasp_near_min_tol)



    # =========================================================
    # HOME
    # =========================================================
    def _enter_home(self):
        self.state = State.HOMING
        self._token += 1
        self._cancel_wait_timer()

        # 清理当前任务/步骤（保持第一版风格）
        self._task = None
        self._steps = []
        self._step_idx = 0

        if not self.mc:
            self.get_logger().error("Real device connection failed")
            self.state = State.IDLE
            return
        self.get_logger().info("back Home ...")
        try:
            self.mc.send_angles(self.home_angles_deg, self.home_speed)
        except Exception as e:
            self.get_logger().error(f"home send failed: {e}")

        t0 = time.time()
        while time.time() - t0 < self.home_timeout_sec:
            try:
                if hasattr(self.mc, "is_moving") and (not self.mc.is_moving()):
                    break
            except Exception:
                pass
            time.sleep(0.1)

        if self.home_open_gripper and (not self.has_object):
            try:
                self._gripper_open()
            except Exception:
                pass

        self.state = State.IDLE
    
    def _handle_runtime_error(self, reason: str):
        # 先保留并打印具体错误原因，再执行安全回 Home
        self.get_logger().error(f"Runtime error reason: {reason}")
        self.get_logger().warn("Runtime error detected: returning to Home and waiting for the next pick/place command")
        self._queue.clear()
        self._enter_home()

    def _reset(self):
        self._cancel_wait_timer()
        self._token += 1
        self._task = None
        self._steps = []
        self._step_idx = 0
        self.state = State.IDLE

    # =========================================================
    # Build plan request（带 start_state + 末端约束 + LIN 时更小 box）
    # =========================================================
    def _apply_common_plan_request_fields(
        self,
        mpr: MotionPlanRequest,
        *,
        pipeline_id=None,
        planner_id=None,
        max_vel_scale=None,
        max_acc_scale=None,        
        ) -> None:
        mpr.group_name = self.group_name
        mpr.allowed_planning_time = self.allowed_planning_time
        mpr.num_planning_attempts = self.num_planning_attempts
        mpr.max_velocity_scaling_factor = (
            self.max_vel_scale if max_vel_scale is None else float(max_vel_scale)
        )
        mpr.max_acceleration_scaling_factor = (
            self.max_acc_scale if max_acc_scale is None else float(max_acc_scale)
        )


        if pipeline_id:
            mpr.pipeline_id = str(pipeline_id)
        if planner_id:
            mpr.planner_id = str(planner_id)

        # start_state：真机且已掌握 MoveIt joint 顺序时，使用当前关节角作为规划起点
        if self.mc and self.moveit_joint_names is not None and self._last_joint_rad_by_moveit is not None:
            rs = RobotState()
            js = JointState()
            js.name = self.moveit_joint_names
            js.position = self._last_joint_rad_by_moveit
            # Pilz LIN 明确要求 start_state 速度为 0；这里显式填零，
            # 避免某些硬件/驱动链路把缺省值解释成非静止起点。
            js.velocity = [0.0] * len(self.moveit_joint_names)
            js.header.stamp = self.get_clock().now().to_msg()
            rs.joint_state = js
            mpr.start_state = rs
    def _build_target_pose(self, x_mm, y_mm, z_mm) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.pose.position.x = x_mm / 1000.0
        pose.pose.position.y = y_mm / 1000.0
        pose.pose.position.z = z_mm / 1000.0

        roll = math.radians(self.ee_roll_deg)
        pitch = math.radians(self.ee_pitch_deg)
        yaw = math.radians(self.ee_yaw_deg)
        pose.pose.orientation = quat_from_rpy(roll, pitch, yaw)
        return pose

    def _build_lin_plan_request_with_start_state(self, x_mm, y_mm, z_mm) -> GetMotionPlan.Request:
        pose = self._build_target_pose(x_mm, y_mm, z_mm)

        req = GetMotionPlan.Request()
        mpr = MotionPlanRequest()
        self._apply_common_plan_request_fields(
            mpr,
            pipeline_id=self.pilz_pipeline_id,
            planner_id=self.pilz_lin_planner_id,
            max_vel_scale=self.lin_max_vel_scale,
            max_acc_scale=self.lin_max_acc_scale,
        )

        c = Constraints()

        # Pilz LIN 对“Pose 风格”的目标更稳定；这里改用球形位置容差，
        # 避免把 OMPL 风格的 box 目标约束直接喂给 LIN 导致系统性失败。
        pc = PositionConstraint()
        pc.header.frame_id = self.base_frame
        pc.link_name = self.ee_link

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [float(self.lin_goal_radius_m)]

        bv = BoundingVolume()
        bv.primitives.append(sphere)
        bv.primitive_poses.append(pose.pose)
        pc.constraint_region = bv
        pc.weight = 1.0

        oc = OrientationConstraint()
        oc.header.frame_id = self.base_frame
        oc.link_name = self.ee_link
        oc.orientation = pose.pose.orientation
        oc.absolute_x_axis_tolerance = float(self.tilt_tol_rad)
        oc.absolute_y_axis_tolerance = float(self.free_about_y_rad)
        oc.absolute_z_axis_tolerance = float(self.tilt_tol_rad)
        oc.weight = float(self.ori_weight)

        c.position_constraints.append(pc)
        c.orientation_constraints.append(oc)
        mpr.goal_constraints.append(c)

        req.motion_plan_request = mpr
        return req

    def _build_plan_request_with_start_state(
        self,
        x_mm,
        y_mm,
        z_mm,
        *,
        pipeline_id=None,
        planner_id=None,
        goal_box_m=None,
    ) -> GetMotionPlan.Request:
        pose = self._build_target_pose(x_mm, y_mm, z_mm)

        req = GetMotionPlan.Request()
        mpr = MotionPlanRequest()
        self._apply_common_plan_request_fields(
            mpr,
            pipeline_id=pipeline_id,
            planner_id=planner_id,
        )

        c = Constraints()

        # Position constraint (box)
        pc = PositionConstraint()
        pc.header.frame_id = self.base_frame
        pc.link_name = self.ee_link

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        if goal_box_m is None:
            box_size = self.pos_box_m_lin if planner_id == "LIN" else self.pos_box_m
        else:
            box_size = float(goal_box_m)

        box.dimensions = [box_size, box_size, box_size]

        bv = BoundingVolume()
        bv.primitives.append(box)
        bv.primitive_poses.append(pose.pose)
        pc.constraint_region = bv
        pc.weight = 1.0

        # Orientation constraint（tcp +Y 向下；允许绕 Y 自由转）
        oc = OrientationConstraint()
        oc.header.frame_id = self.base_frame
        oc.link_name = self.ee_link
        oc.orientation = pose.pose.orientation

        oc.absolute_x_axis_tolerance = float(self.tilt_tol_rad)
        oc.absolute_y_axis_tolerance = float(self.free_about_y_rad)
        oc.absolute_z_axis_tolerance = float(self.tilt_tol_rad)
        oc.weight = float(self.ori_weight)

        c.position_constraints.append(pc)
        c.orientation_constraints.append(oc)
        mpr.goal_constraints.append(c)

        req.motion_plan_request = mpr
        return req

    # =========================================================
    # HW exec：插值发送更丝滑
    # =========================================================
    def _exec_hw_smooth_interpolated(self, traj: JointTrajectory) -> bool:
        if self.moveit_joint_names is None:
            self.get_logger().error("moveit_joint_names Uninitialized, cannot be executed")
            return False

        name_to_idx = {n: i for i, n in enumerate(traj.joint_names)}
        for n in self.moveit_joint_names:
            if n not in name_to_idx:
                self.get_logger().error(f"trajectory missing joint: {n}")
                return False

        waypoints = []
        for pt in traj.points:
            t = float(pt.time_from_start.sec) + float(pt.time_from_start.nanosec) * 1e-9
            moveit_rad = [pt.positions[name_to_idx[n]] for n in self.moveit_joint_names]
            hw_rad = [0.0] * 6
            for mi, hw_i in enumerate(self._moveit_name_to_hw_idx):
                hw_rad[hw_i] = moveit_rad[mi]
            waypoints.append((t, hw_rad))

        if len(waypoints) < 2:
            return False

        rate = max(10.0, float(self.hw_rate_hz))
        dt = 1.0 / rate
        total_t = waypoints[-1][0] * float(self.hw_time_scale)

        start_real = time.time()
        k = 0
        try:
            while True:
                elapsed = time.time() - start_real
                if elapsed >= total_t:
                    break

                t_ref = elapsed / float(self.hw_time_scale)
                while k < len(waypoints) - 2 and t_ref > waypoints[k + 1][0]:
                    k += 1

                t0, q0 = waypoints[k]
                t1, q1 = waypoints[k + 1]
                alpha = 1.0 if (t1 <= t0) else max(0.0, min(1.0, (t_ref - t0) / (t1 - t0)))
                q = [q0[i] + alpha * (q1[i] - q0[i]) for i in range(6)]
                deg = [math.degrees(x) for x in q]
                self.mc.send_angles(deg, self.hw_send_speed)
                time.sleep(dt)

            last = waypoints[-1][1]
            deg = [math.degrees(x) for x in last]
            self.mc.send_angles(deg, self.hw_send_speed)

            # 排空/等待：降低“紧接着夹爪指令被吞”的概率
            time.sleep(0.6)
            try:
                if hasattr(self.mc, "is_moving"):
                    t0 = time.time()
                    while time.time() - t0 < float(self.hw_exec_settle_timeout_sec):
                        if not self.mc.is_moving():
                            break
                        time.sleep(0.05)
            except Exception:
                pass

            return True
        except Exception as e:
            self.get_logger().error(f"hardware smooth exec failed: {e}")
            return False


def main(args=None):
    try:
        rclpy.init(args=args)
        node = ArmWorker()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
