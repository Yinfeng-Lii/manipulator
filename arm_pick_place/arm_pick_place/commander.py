#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def main(args=None):
    rclpy.init(args=args)
    node = Node('arm_commander')
    pub = node.create_publisher(String, 'arm_command', 10)

    print("========================================")
    print("   MyCobot280 MoveIt 指挥台 (Pick & Place)")
    print("========================================")
    print("格式: [模式] [x] [y] [z]   单位: mm")
    print("模式: pick / place / home")
    
    # ================= 0311修改：更新抓取指令的界面提示 =================
    print("1. 质心抓取 (大小 + 质心XYZ):")
    print("   格式: pick size x y z (size: big/small)")
    print("   示例: pick big 30.0 200.0 -20")
    print("")
    # ====================================================================
    
    print("2. 定点放置 (目标XYZ):")
    print("   格式: place x y z")
    print("   示例: place 200 -80 160")
    print("")
    print("3. 机械臂回零位:")
    print("   格式: home")
    print("")
    # ====================================================================
    # ✅ 0311修改：在终端菜单里加上快捷指令的提示 (使用新计算的质心坐标和大小)
    # ====================================================================
    print("4. 快捷测试指令:")
    print("   输入 a: 自动发送 pick big 30.0 200.0 -20")
    print("   输入 b: 自动发送 place 150 150 0")
    print("")
    # ====================================================================
    print("输入 q 退出")
    print("----------------------------------------")

    try:
        while rclpy.ok():
            s = input("\n请输入指令 >>> ").strip()
            if s.lower() == "q":
                break

            # ====================================================================
            # ✅ 0311修改：拦截输入并替换为带物体大小和质心坐标的新版长指令
            # ====================================================================
            if s.lower() == "a":
                s = "pick big 30.0 200.0 -20"
            elif s.lower() == "b":
                s = "place 150 150 0"
            # ====================================================================

            # 为了防止直接回车导致发送空指令，可以顺手加个非空判断（可选保护）
            if not s:
                continue

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