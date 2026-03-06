#!/usr/bin/env python3
import argparse
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def main(args=None):
    parser = argparse.ArgumentParser(description="MyCobot280 command publisher")
    parser.add_argument("--topic", default="/arm_command", help="ROS topic for arm commands")
    cli_args, ros_args = parser.parse_known_args(args=args)

    rclpy.init(args=ros_args)
    node = Node('arm_commander')
    pub = node.create_publisher(String, cli_args.topic, 10)

    print("========================================")
    print("   MyCobot280 MoveIt 指挥台 (Pick & Place)")
    print("========================================")
    print("格式: [模式] [x] [y] [z]   单位: mm")
    print("模式: pick / place / home")
    print("例:   pick 150 0 160")
    print("      place 200 -80 160")
    print("      home")
    print("输入 q 退出")
    print(f"topic: {cli_args.topic}")
    print("----------------------------------------")
    print(f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '<unset>')}")
    print(f"ROS_LOCALHOST_ONLY={os.environ.get('ROS_LOCALHOST_ONLY', '<unset>')}")

    try:
        while rclpy.ok():
            s = input("\n请输入指令 >>> ").strip()
            if s.lower() == "q":
                break

            msg = String()
            msg.data = s
            pub.publish(msg)
            print(f"📡 已发送: {s}")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
