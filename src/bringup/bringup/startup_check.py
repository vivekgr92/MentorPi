#!/usr/bin/env python3
# encoding: utf-8

import rclpy
import time
from ros_robot_controller_msgs.msg import BuzzerState
from rclpy.node import Node

class StartupCheckNode(Node):
    def __init__(self):
        super().__init__('startup_check_node')
        self.buzzer_pub = self.create_publisher(BuzzerState, '/ros_robot_controller/set_buzzer', 1)
        self.timer = self.create_timer(5.0, self.publish_buzzer)
        self.get_logger().info('StartupCheckNode initialized') 

    def publish_buzzer(self):
        msg = BuzzerState()
        msg.freq = 1900
        msg.on_time = 0.2
        msg.off_time = 0.01
        msg.repeat = 1
        self.buzzer_pub.publish(msg)
        self.get_logger().info('Buzzer state published')
        self.get_logger().info(f'Buzzer state published: {msg}')
        rclpy.shutdown()

def main():
    rclpy.init()
    node = StartupCheckNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()


