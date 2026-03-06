#!/usr/bin/env python3
# encoding: utf-8
# 颜色跟踪(color tracking)
import os
import cv2
import math
import queue
import rclpy
import threading
import numpy as np
import sdk.pid as pid
import sdk.common as common
import sdk.yaml_handle as yaml_handle
from rclpy.node import Node
from app.common import Heart
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from app.common import ColorPicker
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool, Trigger
from interfaces.srv import SetPoint, SetFloat64
from large_models_msgs.srv import SetString
from ros_robot_controller_msgs.msg import MotorsState, SetPWMServoState, PWMServoState, RGBState, RGBStates
import time
from std_msgs.msg import Bool


class ObjectTracker:
    def __init__(self, color, node, set_color=None, set_status=False):
        self.node = node
        self.machine_type = os.environ['MACHINE_TYPE']
        self.camera_type = node.camera_type

        self.pid_yaw = pid.PID(0.006, 0.0, 0.0)
        self.pid_dist = pid.PID(0.002, 0.0, 0.00)
        self.last_color_circle = None
        self.lost_target_count = 0  # 初始化丢失目标计数器
        self.lost_threshold = 5  # 连续丢失目标多少帧后才认为目标丢失
        self.weight_sum = 1.0
        self.x_stop = 320

        try:
            self.lab_data = yaml_handle.get_yaml_data(yaml_handle.lab_file_path)
            self.node.get_logger().info("成功加载 yaml_handle.lab_file_path 文件")

            try:
                temp = self.lab_data['lab']
                self.node.get_logger().info("成功读取 self.lab_data['lab']")
            except KeyError as e:
                self.node.get_logger().error(f"读取 self.lab_data['lab'] 失败: {e}")
                raise  # 重新抛出异常，确保错误被捕获

        except Exception as e:
            self.node.get_logger().error(f"加载或解析 YAML 文件失败: {e}")
            raise  # 重新抛出异常，确保错误被捕获

        self.set_status = set_status
        self.set_color = set_color
        if color is not None:
            self.target_lab, self.target_rgb = color

        self.y_stop = 300
        self.pro_size = (320, 180)


        self.range_rgb = {
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
        }

    def __call__(self, image, result_image, threshold):
        twist = Twist()
        h, w = image.shape[:2]
        image = cv2.resize(image, self.pro_size)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  # RGB转LAB空间(convert RGB to LAB space)
        image = cv2.GaussianBlur(image, (5, 5), 5)

        if self.set_status == False:
            min_color = [int(self.target_lab[0] - 50 * threshold * 2),
                         int(self.target_lab[1] - 50 * threshold),
                         int(self.target_lab[2] - 50 * threshold)]
            max_color = [int(self.target_lab[0] + 50 * threshold * 2),
                         int(self.target_lab[1] + 50 * threshold),
                         int(self.target_lab[2] + 50 * threshold)]
            target_color = self.target_lab, min_color, max_color
        else:
            try:
                # 优先使用指定的相机类型，如果不存在则使用默认相机类型
                if self.camera_type == 'aurora' or self.camera_type == 'ascamera':
                    camera_type = 'Stereo'
                else:
                    camera_type == 'Mono'
                if 'lab' in self.lab_data and camera_type in self.lab_data['lab']:
                    if self.set_color in self.lab_data['lab'][camera_type]:
                        color_data = self.lab_data['lab'][camera_type][self.set_color]
                        min_color = color_data['min']
                        max_color = color_data['max']
                    else:
                        self.node.get_logger().error(
                            f"Color '{self.set_color}' not found in lab_data.yaml for camera type '{camera_type}'")
                        return result_image, twist  # 颜色未找到，返回
                else:
                    self.node.get_logger().error(
                        "Invalid lab_data.yaml structure: 'lab' or camera type key not found.")
                    return result_image, twist

            except KeyError as e:
                self.node.get_logger().error(f"KeyError: {e} while accessing lab_data.yaml: {e}")
                return result_image, twist

            target_color = 0, min_color, max_color

        mask = cv2.inRange(image, tuple(target_color[1]), tuple(target_color[2]))  # 二值化(binarization)
        # cv2.imshow('mask', cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        # cv2.waitKey(1)
        eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # 腐蚀(erode)
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # 膨胀(dilate)
        contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # 找出轮廓(find contours)
        contour_area = map(lambda c: (c, math.fabs(cv2.contourArea(c))), contours)  # 计算各个轮廓的面积(calculate the area of each contour)
        contour_area = list(filter(lambda c: c[1] > 40, contour_area))  # 剔除>面积过小的轮廓(Exclude contours with area too small)
        circle = None
        if len(contour_area) > 0:
            if self.last_color_circle is None:
                contour, area = max(contour_area, key=lambda c_a: c_a[1])
                circle = cv2.minEnclosingCircle(contour)
            else:
                (last_x, last_y), last_r = self.last_color_circle
                circles = map(lambda c: cv2.minEnclosingCircle(c[0]), contour_area)
                circle_dist = list(map(lambda c: (c, math.sqrt(((c[0][0] - last_x) ** 2) + ((c[0][1] - last_y) ** 2))),
                                       circles))
                circle, dist = min(circle_dist, key=lambda c: c[1])
                if dist < 100:
                    circle = circle
        if circle is not None:
            self.lost_target_count = 0  # 重置丢失计数器，因为找到了目标
            (x, y), r = circle
            x = x / self.pro_size[0] * w
            y = y / self.pro_size[1] * h
            r = r / self.pro_size[0] * w

            cv2.circle(result_image, (self.x_stop, self.y_stop), 5, (255, 255, 0), -1)

            if self.set_status == False:
                result_image = cv2.circle(result_image, (int(x), int(y)), int(r), (self.target_rgb[0],
                                                                                   self.target_rgb[1],
                                                                                   self.target_rgb[2]), 2)
            else:
                result_image = cv2.circle(result_image, (int(x), int(y)), int(r), self.range_rgb[self.set_color], 2)

            vx = 0
            vw = 0
            if abs(y - self.y_stop) > 20:
                self.pid_dist.update(y - self.y_stop)
                twist.linear.x = common.set_range(self.pid_dist.output, -0.45, 0.45)
            else:
                self.pid_dist.clear()
            if abs(x - self.x_stop) > 20:
                self.pid_yaw.update(x - self.x_stop)
                if self.machine_type == 'MentorPi_Acker':
                    steering_angle = common.set_range(-self.pid_yaw.output, -math.radians(40),
                                                       math.radians(40))
                    if steering_angle != 0:
                        R = 0.145 / math.tan(steering_angle)
                        twist.angular.z = -twist.linear.x / R
                else:
                    twist.angular.z = common.set_range(self.pid_yaw.output, -2, 2)
            else:
                self.pid_yaw.clear()
            return result_image, twist  # Target Found return twist

        else:  # No circle was found
            self.lost_target_count += 1
            return result_image, Twist()  # Target lost return zero twist


class OjbectTrackingNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.set_callback = False
        self.color_picker = None
        self.tracker = None
        self.is_running = False
        self.threshold = 0.1
        self.dist_threshold = 0.3
        self.lock = threading.RLock()
        self.image_sub = None
        self.result_image = None
        self.image_height = None
        self.image_width = None
        self.bridge = CvBridge()
        self.image_queue = queue.Queue(2)
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)
        self.enter_srv = self.create_service(Trigger, '~/enter', self.enter_srv_callback)
        self.exit_srv = self.create_service(Trigger, '~/exit', self.exit_srv_callback)
        self.set_running_srv = self.create_service(SetBool, '~/set_running', self.set_running_srv_callback)
        self.set_target_color_srv = self.create_service(SetPoint, '~/set_target_color',
                                                          self.set_target_color_srv_callback)
        self.get_target_color_srv = self.create_service(Trigger, '~/get_target_color',
                                                          self.get_target_color_srv_callback)
        self.set_threshold_srv = self.create_service(SetFloat64, '~/set_threshold',
                                                        self.set_threshold_srv_callback)
        self.set_large_model_target_color_srv = self.create_service(SetString, '~/set_large_model_target_color',
                                                                      self.set_large_model_target_color_srv_callback)  # 创建服务

        self.servo_state_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 1)
        Heart(self, self.name + '/heartbeat', 5,
              lambda _: self.exit_srv_callback(request=Trigger.Request(), response=Trigger.Response()))  # 心跳包(heartbeat package)
        self.debug = self.get_parameter('debug').value
        if self.debug:
            threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)

        self.wakeup_sub = self.create_subscription(Bool, '/vocal_detect/wakeup', self.wakeup_callback, 1)

        self.get_logger().info(f"yaml_handle.lab_file_path: {yaml_handle.lab_file_path}")
        self.target_lost = False  # 目标是否丢失的标志
        self.large_model_tracking = False  # 标志是否由大模型启动
        self.target_lost_timer = None  # 定时器

        # 获取相机类型
        self.camera_type = os.environ['DEPTH_CAMERA_TYPE']

        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def get_node_state(self, request, response):
        response.success = True
        return response

    def main(self):
        while True:
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("result", result)
            if self.debug and not self.set_callback:
                self.set_callback = True
                # 设置鼠标点击事件的回调函数(Set a callback function for mouse click event)
                cv2.setMouseCallback("result", self.mouse_callback)
            k = cv2.waitKey(1)
            if k != -1:
                break
        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.get_logger().info("x:{} y{}".format(x, y))
            msg = SetPoint.Request()
            if self.image_height is not None and self.image_width is not None:
                msg.data.x = x / self.image_width
                msg.data.y = y / self.image_height
                self.set_target_color_srv_callback(msg, SetPoint.Response())

    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % 'object tracking enter')
        self.get_logger().info(f"lab_data.yaml 文件路径: {yaml_handle.lab_file_path}")
        with self.lock:
            self.is_running = False
            self.threshold = 0.5
            self.tracker = None
            self.color_picker = None
            self.dist_threshold = 0.3
            self.large_model_tracking = False  # 进入时重置大模型跟踪标志
            if self.image_sub is None:
                self.image_sub = self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image',
                                                            self.image_callback, 1)  # 摄像头订阅(subscribe to the camera)
            self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "enter"
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % 'object tracking exit')
        try:
            if self.image_sub is not None:
                self.destroy_subscription(self.image_sub)
                self.image_sub = None
        except Exception as e:
            self.get_logger().error(str(e))
        with self.lock:
            self.is_running = False
            self.color_picker = None
            self.tracker = None
            self.threshold = 0.5
            self.dist_threshold = 0.3
            self.mecanum_pub.publish(Twist())
            self.large_model_tracking = False  # 关闭大模型跟踪标志
            if self.target_lost_timer is not None:
                self.target_lost_timer.cancel()  # 取消定时器
                self.target_lost_timer = None
            self.target_lost = False  # 确保退出时重置目标丢失标志
        response.success = True
        response.message = "exit"
        return response

    def set_target_color_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % 'set_target_color')
        with self.lock:
            x, y = request.data.x, request.data.y
            if x == -1 and y == -1:
                self.color_picker = None
                self.tracker = None
            else:
                self.color_picker = ColorPicker(request.data, 10)
            self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "set_target_color"
        return response

    def get_target_color_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % 'get_target_color')
        response.success = False
        response.message = "get_target_color"
        with self.lock:
            if self.tracker is not None and hasattr(self.tracker, 'target_rgb'):
                response.success = True
                rgb = self.tracker.target_rgb
                response.message = "{},{},{}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            else:
                response.message = "Tracker not initialized or target_rgb not available."
        return response

    def set_running_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % 'set_running')
        with self.lock:
            self.is_running = request.data
            if not self.is_running:
                self.mecanum_pub.publish(Twist())
                if self.target_lost_timer is not None:
                    self.target_lost_timer.cancel()  # 取消定时器
                    self.target_lost_timer = None
                self.target_lost = False  # 确保停止时重置目标丢失标志
        response.success = True
        response.message = "set_running"
        return response

    def set_threshold_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % 'threshold')
        with self.lock:
            self.threshold = request.data
            response.success = True
            response.message = "set_threshold"
        return response

    def set_large_model_target_color_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "set_large_model_target_color")
        with self.lock:
            color_name = request.data
            self.get_logger().info(f"请求的颜色名称: {color_name}")

            self.tracker = None  # Reset the tracker
            self.large_model_tracking = True  # 启用大模型跟踪标志
            self.target_lost = False  # 确保启动时重置目标丢失标志
            if self.target_lost_timer is not None:
                self.target_lost_timer.cancel()
                self.target_lost_timer = None

            try:
                # 确定使用哪个相机类型：如果当前相机类型是 'ascamera'，则使用 'Stereo'，否则使用 'Mono'
                camera_type = 'Stereo' if self.camera_type == 'ascamera' else 'Mono'
                # 将相机类型传递给 ObjectTracker 类
                self.tracker = ObjectTracker(None, self, color_name, True)
                try:
                    temp = self.tracker.lab_data['lab'][camera_type][color_name]
                    self.get_logger().info(f"成功读取 颜色配置: self.lab_data['lab'][{camera_type}][{color_name}]")
                except KeyError as e:
                    self.get_logger().error(f"读取 颜色配置失败: {e}")
                    raise
                self.mecanum_pub.publish(Twist())  # 发布零速度指令，防止意外移动
                response.success = True
                response.message = "set_large_model_target_color"
            except Exception as e:
                response.success = False
                response.message = str(e)
                self.get_logger().error(f"设置目标颜色时发生错误: {e}")
            return response

    def start_stop_timer(self):
        with self.lock:
            if self.target_lost_timer is None and self.large_model_tracking:  # 确保只启动一次定时器, 并且是由大模型启动的
                self.target_lost_timer = threading.Timer(5.0, self.stop_after_lose)
                self.target_lost_timer.start()
                self.target_lost = True
                self.get_logger().warn("启动停止定时器... (Starting stop timer...)")

    def stop_after_lose(self):
        """Stops the robot after the target is lost for a specified duration."""
        with self.lock:
            if self.large_model_tracking and self.target_lost:  # 再次确认是大模型启动
                self.get_logger().warn("目标丢失超过5秒，停止移动 (Target lost for more than 5 seconds, stopping movement)")
                self.mecanum_pub.publish(Twist())  # 发布零速度指令停止机器人
                self.is_running = False  # 停止运行
                self.target_lost = False
                self.target_lost_timer = None  # 清除定时器

    def image_callback(self, ros_image):
        # 将ros格式(rgb)转为opencv的rgb格式(convert RGB format of ROS to that of OpenCV)
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        self.image_height, self.image_width = rgb_image.shape[:2]

        result_image = np.copy(rgb_image)  # 显示结果用的画面(the image used for display the result)
        with self.lock:
            # 颜色拾取器和识别追踪互斥, 如果拾取器存在
            # 颜色拾取器和识别追踪互斥, 如果拾取器存在就开始拾取(color picker and object tracking are mutually exclusive. If the color picker exists, start picking colors)
            if self.color_picker is not None:  # 拾取器存在(color pick exists)
                target_color, result_image = self.color_picker(rgb_image, result_image)
                if target_color is not None:
                    self.color_picker = None
                    self.tracker = ObjectTracker(target_color, self)

                    self.get_logger().info("target color: {}".format(target_color))
            else:
                if self.tracker is not None:
                    try:
                        result_image, twist = self.tracker(rgb_image, result_image, self.threshold)

                        if self.is_running:
                            self.mecanum_pub.publish(twist)

                             # 目标重新出现，取消定时器
                            if self.large_model_tracking and self.target_lost and self.target_lost_timer is not None and self.tracker.lost_target_count <= self.tracker.lost_threshold:  #tracker.lost_target_count <= self.tracker.lost_threshold 确保在判断为追踪到之前取消
                                self.target_lost_timer.cancel()
                                self.target_lost_timer = None
                                self.target_lost = False
                                self.get_logger().info("重新找到目标，停止定时器 (Target reacquired, stopping timer)")

                        else:
                            self.tracker.pid_dist.clear()
                            self.tracker.pid_yaw.clear()
                    except Exception as e:
                        self.get_logger().error(str(e))

                    # 启动/保持停止定时器
                    if self.large_model_tracking and self.is_running and self.tracker.lost_target_count > self.tracker.lost_threshold and not self.target_lost:  #保证连续丢失才启动
                        self.start_stop_timer() #在image_callback里面调用
                        self.get_logger().warn("目标丢失，启动停止定时器... (Target lost, starting stop timer...)")

        if self.debug:
            if self.image_queue.full():
                # 如果队列已满，丢弃最旧的图像(if the queue is full, discard the oldest image)
                self.image_queue.get()
                # 将图像放入队列(put the image into the queue)
            self.image_queue.put(result_image)
        else:
            # 将opencv的格式(bgr)转为ros的rgb格式(convert BGR format of OpenCV to RGB format of ROS)
            self.result_publisher.publish(self.bridge.cv2_to_imgmsg(result_image, "rgb8"))

    def wakeup_callback(self, msg):
        """Callback function for the /vocal_detect/wakeup topic."""
        if msg.data:
            self.get_logger().info("Wake-up detected, exiting large model tracking.")
            with self.lock:
                if self.large_model_tracking:
                    self.large_model_tracking = False
                    self.is_running = False
                    self.mecanum_pub.publish(Twist())  # Stop the robot
                    if self.target_lost_timer is not None:
                        self.target_lost_timer.cancel()
                        self.target_lost_timer = None
                    self.tracker = None  # 清除tracker
                    self.get_logger().info("Large model tracking stopped.")


def main():
    node = OjbectTrackingNode('object_tracking')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
