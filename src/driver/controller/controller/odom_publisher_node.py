#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import math
import time
import yaml
import rclpy
import signal
import threading
from rclpy.node import Node
from std_srvs.srv import Trigger
from nav_msgs.msg import Odometry
from controller import ackermann, mecanum
from ros_robot_controller_msgs.msg import MotorsState, SetPWMServoState, PWMServoState
from geometry_msgs.msg import Pose2D, Pose, Twist, PoseWithCovarianceStamped, TransformStamped


ODOM_POSE_COVARIANCE = list(map(float, 
                        [1e-3, 0, 0, 0, 0, 0, 
                        0, 1e-3, 0, 0, 0, 0,
                        0, 0, 1e6, 0, 0, 0,
                        0, 0, 0, 1e6, 0, 0,
                        0, 0, 0, 0, 1e6, 0,
                        0, 0, 0, 0, 0, 1e3]))

ODOM_POSE_COVARIANCE_STOP = list(map(float, 
                            [1e-9, 0, 0, 0, 0, 0, 
                             0, 1e-3, 1e-9, 0, 0, 0,
                             0, 0, 1e6, 0, 0, 0,
                             0, 0, 0, 1e6, 0, 0,
                             0, 0, 0, 0, 1e6, 0,
                             0, 0, 0, 0, 0, 1e-9]))

ODOM_TWIST_COVARIANCE = list(map(float, 
                        [1e-3, 0, 0, 0, 0, 0, 
                         0, 1e-3, 0, 0, 0, 0,
                         0, 0, 1e6, 0, 0, 0,
                         0, 0, 0, 1e6, 0, 0,
                         0, 0, 0, 0, 1e6, 0,
                         0, 0, 0, 0, 0, 1e3]))

ODOM_TWIST_COVARIANCE_STOP = list(map(float, 
                            [1e-9, 0, 0, 0, 0, 0, 
                              0, 1e-3, 1e-9, 0, 0, 0,
                              0, 0, 1e6, 0, 0, 0,
                              0, 0, 0, 1e6, 0, 0,
                              0, 0, 0, 0, 1e6, 0,
                              0, 0, 0, 0, 0, 1e-9]))

def rpy2qua(roll, pitch, yaw):
    cy = math.cos(yaw*0.5)
    sy = math.sin(yaw*0.5)
    cp = math.cos(pitch*0.5)
    sp = math.sin(pitch*0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    q = Pose()
    q.orientation.w = cy * cp * cr + sy * sp * sr
    q.orientation.x = cy * cp * sr - sy * sp * cr
    q.orientation.y = sy * cp * sr + cy * sp * cr
    q.orientation.z = sy * cp * cr - cy * sp * sr
    return q.orientation

def qua2rpy(x, y, z, w):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
  
    return roll, pitch, yaw

class Controller(Node):
    
    def __init__(self, name):
        rclpy.init()
        super().__init__(name)

        self.x = 0.0
        self.y = 0.0
        self.linear_x = 0.0
        self.linear_y = 0.0
        self.angular_z = 0.0
        self.pose_yaw = 0
        self.last_time = None
        self.current_time = None
        signal.signal(signal.SIGINT, self.shutdown)

        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.mecanum = mecanum.MecanumChassis(wheelbase=0.1368, track_width=0.1446, wheel_diameter=0.075)
        # Declare parameters
        self.declare_parameter('pub_odom_topic', True)
        self.declare_parameter('base_frame_id', 'base_footprint')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('linear_correction_factor', 1.00)
        self.declare_parameter('linear_correction_factor_tank', 0.52)
        self.declare_parameter('angular_correction_factor', 1.00)
        self.declare_parameter('machine_type', os.environ['MACHINE_TYPE'])
        
        self.pub_odom_topic = self.get_parameter('pub_odom_topic').value
        self.base_frame_id = self.get_parameter('base_frame_id').value
        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        
        #self.machine_type = os.environ.get('MACHINE_TYPE', 'MentorPi_Mecanum')
        
        if self.machine_type == 'MentorPi_Tank':
            self.linear_factor = self.get_parameter('linear_correction_factor_tank').value
        else:
            self.linear_factor = self.get_parameter('linear_correction_factor').value
        self.angular_factor = self.get_parameter('angular_correction_factor').value
        
        self.correction_factors = self.load_correction_factors()
        self.left_correction = self.correction_factors['left_correction_factor']
        self.right_correction = self.correction_factors['right_correction_factor']

        self.clock = self.get_clock() 
        if self.pub_odom_topic:
            self.odom = Odometry()
            self.odom.header.frame_id = self.odom_frame_id
            self.odom.child_frame_id = self.base_frame_id
            
            self.odom.pose.covariance = ODOM_POSE_COVARIANCE
            self.odom.twist.covariance = ODOM_TWIST_COVARIANCE
            
            self.odom_pub = self.create_publisher(Odometry, 'odom_raw', 1)
            self.dt = 1.0/50.0

            threading.Thread(target=self.cal_odom_fun, daemon=True).start()
        self.get_logger().info('\033[1;32m%f %f\033[0m' % (self.linear_factor, self.angular_factor))
        self.motor_pub = self.create_publisher(MotorsState, 'ros_robot_controller/set_motor', 1)
        self.servo_state_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'set_pose', 1)
        self.create_subscription(Pose2D, 'set_odom', self.set_odom, 1)
        self.create_subscription(Twist, 'controller/cmd_vel', self.cmd_vel_callback, 1)
        self.create_subscription(Twist, '/app/cmd_vel', self.acker_cmd_vel_callback, 1)
        self.create_subscription(Twist, 'cmd_vel', self.app_cmd_vel_callback, 1)
        self.create_service(Trigger, 'controller/load_calibrate_param', self.load_calibrate_param)
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def get_node_state(self, request, response):
        response.success = True
        return response

    def shutdown(self, signum, frame):
        self.get_logger().info('\033[1;32m%s\033[0m' % 'shutdown')
        rclpy.shutdown()

    def load_calibrate_param(self, request, response):
        if self.machine_type == 'JetRover_Tank':
            self.linear_factor = self.get_parameter('~linear_correction_factor_tank').value or 0.52
        else:
            self.linear_factor = self.get_parameter('~linear_correction_factor').value or 1.00
        self.angular_factor = self.get_parameter('~angular_correction_factor').value or 1.00
        self.get_logger().info('\033[1;32m%s\033[0m' % 'load_calibrate_param')

        response.success = True
        return response

    def set_odom(self, msg):
        self.odom = Odometry()
        self.odom.header.frame_id = self.odom_frame_id
        self.odom.child_frame_id = self.base_frame_id
        
        self.odom.pose.covariance = ODOM_POSE_COVARIANCE
        self.odom.twist.covariance = ODOM_TWIST_COVARIANCE
        self.odom.pose.pose.position.x = msg.x
        self.odom.pose.pose.position.y = msg.y
        self.pose_yaw = msg.theta
        self.odom.pose.pose.orientation = rpy2qua(0, 0, self.pose_yaw)
        
        self.linear_x = 0
        self.linear_y = 0
        self.angular_z = 0
        
        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = self.odom_frame_id
        pose.header.stamp = self.clock().now().to_msg()
        pose.pose.pose = self.odom.pose.pose
        pose.pose.covariance = ODOM_POSE_COVARIANCE
        self.pose_pub.publish(pose)

    def app_cmd_vel_callback(self, msg):
        if msg.linear.x > 0.2:
            msg.linear.x = 0.2
        if msg.linear.x < -0.2:
            msg.linear.x = -0.2
        if msg.linear.y > 0.2:
            msg.linear.y = 0.2
        if msg.linear.y < -0.2:
            msg.linear.y = -0.2
        if msg.angular.z > 0.5:
            msg.angular.z = 0.5
        if msg.angular.z < -0.5:
            msg.angular.z = -0.5
        self.cmd_vel_callback(msg)
    
    def acker_cmd_vel_callback(self, msg):
        if msg.linear.x == 0:
            msg.linear.z = 1.0
            servo_state = PWMServoState()
            servo_state.id = [3]
            servo_state.position = [1500 + int(math.degrees(msg.angular.z)/180*2000)]
            data = SetPWMServoState()
            data.state = [servo_state]
            data.duration = 0.02
            self.servo_state_pub.publish(data)
        else:
            if msg.angular.z != 0:
                r = 0.145/math.tan(msg.angular.z)
                msg.angular.z = msg.linear.x / r
        self.cmd_vel_callback(msg)


    def cmd_vel_callback(self, msg):
        if self.machine_type == 'MentorPi_Tank':
            self.linear_x = msg.linear.x
            self.linear_y = 0.0
            self.angular_z = msg.angular.z
            if self.linear_x >= 0.0 and self.angular_z == 0.0:
                factor = 5.50
                correction_diff = self.right_correction - self.left_correction
                if self.right_correction != 1.0 or self.left_correction != 1.0:
                    self.angular_z = self.linear_x * correction_diff * factor
                else:
                    self.angular_z = 0.0
            #     self.get_logger().info(f'self.right_correction: {self.right_correction}')
            #     self.get_logger().info(f'self.left_correction: {self.left_correction}')
            # self.get_logger().info(f'self.angular_z: {self.angular_z}')
            speeds = self.mecanum.set_velocity(self.linear_x, self.linear_y, self.angular_z)
            self.motor_pub.publish(speeds)


    def cal_odom_fun(self):
        while True:
            self.current_time = time.time()
            if self.last_time is None:
                self.dt = 0.0
            else:
                self.dt = self.current_time - self.last_time

            self.odom.header.stamp = self.clock.now().to_msg()
            
            delta_x = self.linear_x * self.dt * math.cos(self.pose_yaw)
            delta_y = self.linear_x * self.dt * math.sin(self.pose_yaw)
            delta_yaw = self.angular_z * self.dt

            self.x += delta_x
            self.y += delta_y
            self.pose_yaw += delta_yaw

            self.odom.pose.pose.position.x = self.linear_factor * self.x
            self.odom.pose.pose.position.y = self.linear_factor * self.y
            self.odom.pose.pose.orientation = rpy2qua(0.0, 0.0, self.pose_yaw)
            self.odom.twist.twist.linear.x = self.linear_x
            self.odom.twist.twist.linear.y = self.linear_y
            self.odom.twist.twist.angular.z = self.angular_z

            if self.linear_x == 0 and self.linear_y == 0 and self.angular_z == 0:
                self.odom.pose.covariance = ODOM_POSE_COVARIANCE_STOP
                self.odom.twist.covariance = ODOM_TWIST_COVARIANCE_STOP
            else:
                self.odom.pose.covariance = ODOM_POSE_COVARIANCE
                self.odom.twist.covariance = ODOM_TWIST_COVARIANCE

            self.odom_pub.publish(self.odom)
            self.last_time = self.current_time
            time.sleep(0.02)


    def load_correction_factors(self):
        """从YAML文件加载校正因子"""
        default_factors = {
            'left_correction_factor': 1.0,
            'right_correction_factor': 1.0
        }
        
        try:
            yaml_path = os.path.join('/home/ubuntu/software/chassis_adjustment/robot_correction_factors.yaml')
            
            with open(yaml_path, 'r') as f:
                factors = yaml.safe_load(f)
                
            if factors is None:
                self.get_logger().warn("YAML file is empty, using default factors")
                return default_factors
                
            if not all(key in factors for key in default_factors.keys()):
                self.get_logger().warn("Missing some correction factors in YAML, using defaults")
                return {**default_factors, **factors}
                
            self.get_logger().info(f"Loaded correction factors: {factors}")
            return factors
            
        except FileNotFoundError:
            self.get_logger().warn("Correction factors file not found, using defaults")
            return default_factors
        except yaml.YAMLError as e:
            self.get_logger().error(f"YAML parsing error: {e}, using defaults")
            return default_factors
        except Exception as e:
            self.get_logger().error(f"Error loading correction factors: {e}, using defaults")
            return default_factors


def main():
    node = Controller('odom_publisher')
    rclpy.spin(node)  # 循环等待ROS2退出(loop waiting for ROS2 to exit)
    rclpy.spin(node)  
if __name__ == "__main__":
    main()

