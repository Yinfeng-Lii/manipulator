#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler, OpaqueFunction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    package_name_moveit_config = "mycobot_moveit_config"
    package_name_description = "mycobot_description"

    # Launch args
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_rviz = LaunchConfiguration("use_rviz")
    rviz_config_file = LaunchConfiguration("rviz_config_file")
    rviz_config_package = LaunchConfiguration("rviz_config_package")

    pkg_share_moveit_config_temp = FindPackageShare(package=package_name_moveit_config)
    pkg_share_description_temp = FindPackageShare(package=package_name_description)

    # -----------------------------
    # Declare arguments
    # -----------------------------
    declare_robot_name_cmd = DeclareLaunchArgument(
        name="robot_name",
        default_value="mycobot_280",
        description="Name of the robot to use",
    )

    # 你现在不跑 Gazebo：默认 false
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        name="use_sim_time",
        default_value="false",
        description="Use simulation clock if true",
    )

    declare_use_rviz_cmd = DeclareLaunchArgument(
        name="use_rviz",
        default_value="true",
        description="Whether to start RViz",
    )

    declare_rviz_config_file_cmd = DeclareLaunchArgument(
        name="rviz_config_file",
        default_value="move_group.rviz",
        description="RViz configuration file",
    )

    declare_rviz_config_package_cmd = DeclareLaunchArgument(
        name="rviz_config_package",
        default_value=package_name_moveit_config,
        description="Package containing the RViz configuration file",
    )

    # -----------------------------
    # Configure nodes
    # -----------------------------
    def configure_setup(context):
        robot_name_str = LaunchConfiguration("robot_name").perform(context)

        pkg_share_moveit_config = pkg_share_moveit_config_temp.find(package_name_moveit_config)
        pkg_share_description = pkg_share_description_temp.find(package_name_description)

        config_path = os.path.join(pkg_share_moveit_config, "config", robot_name_str)

        initial_positions_file_path = os.path.join(config_path, "initial_positions.yaml")
        joint_limits_file_path = os.path.join(config_path, "joint_limits.yaml")
        srdf_model_path = os.path.join(config_path, f"{robot_name_str}.srdf")
        moveit_controllers_file_path = os.path.join(config_path, "moveit_controllers.yaml")
        pilz_cartesian_limits_file_path = os.path.join(config_path, "pilz_cartesian_limits.yaml")
        ros2_controllers_file_path = os.path.join(config_path, "ros2_controllers.yaml")
        # URDF/XACRO from mycobot_description
        urdf_xacro_path = os.path.join(
            pkg_share_description, "urdf", "robots", "mycobot_280.urdf.xacro"
        )

        # xacro mappings (无 Gazebo + mock)
        xacro_mappings = {
            "robot_name": robot_name_str,
            "use_gazebo": "false",
            "use_camera": "false",
            "use_gripper": "true",
            "add_world": "true",
            "prefix": "",
        }

        # Build MoveIt config
        moveit_config = (
            MoveItConfigsBuilder(robot_name_str, package_name=package_name_moveit_config)
            .robot_description(file_path=urdf_xacro_path, mappings=xacro_mappings)
            .robot_description_semantic(file_path=srdf_model_path)
            .robot_description_kinematics(file_path=f"config/{robot_name_str}/kinematics.yaml")
            .joint_limits(file_path=joint_limits_file_path)
            .trajectory_execution(file_path=moveit_controllers_file_path)
            .planning_pipelines(
                pipelines=["ompl", "pilz_industrial_motion_planner", "stomp"],
                default_planning_pipeline="ompl",
            )
            .planning_scene_monitor(
                publish_robot_description=False,
                publish_robot_description_semantic=True,
                publish_planning_scene=True,
            )
            .pilz_cartesian_limits(file_path=pilz_cartesian_limits_file_path)
            .to_moveit_configs()
        )

        # 关键：robot_state_publisher（给 TF 链路用）
        # MoveIt 官方示例/工具就是把 moveit_config.robot_description 传给 robot_state_publisher
        rsp_node = Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[
                moveit_config.robot_description,
                {"use_sim_time": use_sim_time},
            ],
        )


          # ros2_control + controllers（仿真/Mock 执行与夹爪动作可视化都依赖它）
        control_node = Node(
            package="controller_manager",
            executable="ros2_control_node",
            output="screen",
            parameters=[
                moveit_config.robot_description,
                ros2_controllers_file_path,
                {"use_sim_time": use_sim_time},
            ],
        )

        joint_state_broadcaster_spawner = Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager", "--controller-manager-timeout", "120"],
            output="screen",
        )

        arm_controller_spawner = Node(
            package="controller_manager",
            executable="spawner",
            arguments=["arm_controller", "--controller-manager", "/controller_manager", "--controller-manager-timeout", "120"],
            output="screen",
        )

        gripper_controller_spawner = Node(
            package="controller_manager",
            executable="spawner",
            arguments=["gripper_action_controller", "--controller-manager", "/controller_manager", "--controller-manager-timeout", "120"],
            output="screen",
        )

        # MoveIt capabilities（保持你原来的）
        move_group_capabilities = {"capabilities": "move_group/ExecuteTaskSolutionCapability"}

        # ✅ 最关键修复：把 robot_description_kinematics 明确塞进 move_group 参数
        start_move_group_node_cmd = Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[
                # Builder 生成的一揽子参数（planning_pipeline、controllers、limits 等）
                moveit_config.to_dict(),

                # ✅ 强制注入 IK 参数（否则你就是 Parameter not set）
                moveit_config.robot_description_kinematics,

                # （可选但稳）也显式注入 URDF/SRDF，防止某些情况下 dict 覆盖/丢失
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,

                {"use_sim_time": use_sim_time},
                {"start_state": {"content": initial_positions_file_path}},
                move_group_capabilities,
            ],
        )

        # RViz
        start_rviz_node_cmd = Node(
            condition=IfCondition(use_rviz),
            package="rviz2",
            executable="rviz2",
            arguments=["-d", [FindPackageShare(rviz_config_package), "/rviz/", rviz_config_file]],
            output="screen",
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                moveit_config.planning_pipelines,
                moveit_config.joint_limits,
                {"use_sim_time": use_sim_time},
            ],
        )

        exit_event_handler = RegisterEventHandler(
            condition=IfCondition(use_rviz),
            event_handler=OnProcessExit(
                target_action=start_rviz_node_cmd,
                on_exit=EmitEvent(event=Shutdown(reason="rviz exited")),
            ),
        )

        return [
            rsp_node,
            control_node,
            joint_state_broadcaster_spawner,
            arm_controller_spawner,
            gripper_controller_spawner,
            start_move_group_node_cmd,
            start_rviz_node_cmd,
            exit_event_handler,
        ]

    # LaunchDescription
    ld = LaunchDescription()
    ld.add_action(declare_robot_name_cmd)
    ld.add_action(declare_rviz_config_file_cmd)
    ld.add_action(declare_rviz_config_package_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_use_rviz_cmd)
    ld.add_action(OpaqueFunction(function=configure_setup))
    return ld