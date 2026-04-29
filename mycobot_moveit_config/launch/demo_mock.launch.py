import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command

def generate_launch_description():
    desc_share = get_package_share_directory("mycobot_description")
    xacro_path = os.path.join(desc_share, "urdf", "robots", "mycobot_280.urdf.xacro")

    robot_description = {"robot_description": Command(["xacro ", xacro_path])}

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        output="screen",
        parameters=[
            robot_description,
            os.path.join(get_package_share_directory("mycobot_moveit_config"),
                         "config", "mycobot_280", "ros2_controllers.yaml"),
        ],
    )

    jsb = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    arm = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    gripper = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_action_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    moveit = Node(
        package="mycobot_moveit_config",
        executable="move_group.launch.py",  # 如果你仓库里是 launch 文件，需要 IncludeLaunchDescription；这里先不展开
    )

    # 更稳的做法：直接 ros2 launch mycobot_moveit_config move_group.launch.py
    # 所以这里 demo_mock.launch.py 只负责 rsp + control + spawners
    return LaunchDescription([rsp, control_node, jsb, arm, gripper])
