#!/usr/bin/env python3
# encoding: utf-8
# 巡线(line following)
import os
import cv2
import math
import rclpy
import queue
import threading
import numpy as np
import sdk.pid as pid
import sdk.common as common
import sdk.yaml_handle as yaml_handle
from rclpy.node import Node
from app.common import Heart
from cv_bridge import CvBridge
from app.common import ColorPicker
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool, Trigger
from sensor_msgs.msg import Image, LaserScan
from interfaces.srv import SetPoint, SetFloat64
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from ros_robot_controller_msgs.msg import MotorsState, SetPWMServoState, PWMServoState
from large_models_msgs.srv import SetString  #
from std_msgs.msg import Bool  #

MAX_SCAN_ANGLE = 240  # 激光的扫描角度,去掉总是被遮挡的部分degree(the scanning angle of lidar. The covered part is always eliminated)


class LineFollower:
    def __init__(self, color, node, set_color=None):  #
        self.node = node
        self.set_color = set_color  # Save the color name(保存颜色名称)
        self.target_lab = None
        self.target_rgb = None
        self.camera_type = os.environ['DEPTH_CAMERA_TYPE']  # Acquire the camera type(获取相机类型)

        if color is not None:
            self.target_lab, self.target_rgb = color

        self.lab_data = yaml_handle.get_yaml_data(yaml_handle.lab_file_path)
        if os.environ['DEPTH_CAMERA_TYPE'] == 'ascamera' or os.environ['DEPTH_CAMERA_TYPE'] == 'aurora':
            self.rois = ((0.9, 0.95, 0, 1, 0.7), (0.8, 0.85, 0, 1, 0.2), (0.7, 0.75, 0, 1, 0.1))
        else:
            self.rois = ((0.81, 0.83, 0, 1, 0.7), (0.69, 0.71, 0, 1, 0.2), (0.57, 0.59, 0, 1, 0.1))
        self.weight_sum = 1.0
        self.lost_target_count = 0  # Add lost target counter(添加丢失目标计数器)

    @staticmethod
    def get_area_max_contour(contours, threshold=100):
        '''
        获取最大面积对应的轮廓(get the contour of the largest area)
        :param contours:
        :param threshold:
        :return:
        '''
        contour_area = zip(contours, tuple(map(lambda c: math.fabs(cv2.contourArea(c)), contours)))
        contour_area = tuple(filter(lambda c_a: c_a[1] > threshold, contour_area))
        if len(contour_area) > 0:
            max_c_a = max(contour_area, key=lambda c_a: c_a[1])
            return max_c_a

    def __call__(self, image, result_image, threshold):
        centroid_sum = 0
        h, w = image.shape[:2]
        if os.environ['DEPTH_CAMERA_TYPE'] == 'ascamera':
            w = w + 200

        # Determine the color threshold based on whether set_color is set(根据set_color是否设置来决定颜色阈值)
        if self.set_color is None:  # Use color picker(使用颜色拾取)
            min_color = [int(self.target_lab[0] - 50 * threshold * 2),
                         int(self.target_lab[1] - 50 * threshold),
                         int(self.target_lab[2] - 50 * threshold)]
            max_color = [int(self.target_lab[0] + 50 * threshold * 2),
                         int(self.target_lab[1] + 50 * threshold),
                         int(self.target_lab[2] + 50 * threshold)]
        else:
            try:
                # Determine which camera type to use: if the current camera type is 'ascamera', use 'Stereo'; otherwise, use 'Mono'(确定使用哪个相机类型：如果当前相机类型是 'ascamera'，则使用 'Stereo'，否则使用 'Mono')
                if self.camera_type == 'aurora' or self.camera_type == 'ascamera':
                    camera_type = 'Stereo'
                else:
                    camera_type = 'Mono'
                if 'lab' in self.lab_data and camera_type in self.lab_data['lab']:
                    if self.set_color in self.lab_data['lab'][camera_type]:
                        color_data = self.lab_data['lab'][camera_type][self.set_color]
                        min_color = color_data['min']
                        max_color = color_data['max']
                    else:
                        self.node.get_logger().error(
                            f"Color '{self.set_color}' not found in lab_data.yaml for camera type '{camera_type}'")
                        return result_image, None  # Color not found, return (颜色未找到，返回)
                else:
                    self.node.get_logger().error(
                        "Invalid lab_data.yaml structure: 'lab' or camera type key not found.")
                    return result_image, None

            except KeyError as e:
                self.node.get_logger().error(f"KeyError: {e} while accessing lab_data.yaml: {e}")
                return result_image, None

        target_color = 0, min_color, max_color

        for roi in self.rois:
            blob = image[int(roi[0] * h):int(roi[1] * h), int(roi[2] * w):int(roi[3] * w)]  # 截取roi(intercept roi)
            img_lab = cv2.cvtColor(blob, cv2.COLOR_RGB2LAB)  # rgb转lab(convert rgb into lab)
            img_blur = cv2.GaussianBlur(img_lab, (3, 3), 3)  # 高斯模糊去噪(perform Gaussian filtering to reduce noise)
            mask = cv2.inRange(img_blur, tuple(target_color[1]), tuple(target_color[2]))  # 二值化(image binarization)
            eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # 腐蚀(corrode)
            dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # 膨胀(dilate)
            # cv2.imshow('section:{}:{}'.format(roi[0], roi[1]), cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR))
            contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[
                       -2]  # 找轮廓(find the contour)
            max_contour_area = self.get_area_max_contour(contours,
                                                          30)  # 获取最大面积对应轮廓(get the contour corresponding to the largest contour)
            if max_contour_area is not None:
                rect = cv2.minAreaRect(max_contour_area[0])  # 最小外接矩形(minimum circumscribed rectangle)
                box = np.intp(cv2.boxPoints(rect))  # 四个角(four corners)
                for j in range(4):
                    box[j, 1] = box[j, 1] + int(roi[0] * h)
                cv2.drawContours(result_image, [box], -1, (0, 255, 255), 2)  # 画出四个点组成的矩形(draw the rectangle composed of four points)

                # 获取矩形对角点(acquire the diagonal points of the rectangle)
                pt1_x, pt1_y = box[0, 0], box[0, 1]
                pt3_x, pt3_y = box[2, 0], box[2, 1]
                # 线的中心点(center point of the line)
                line_center_x, line_center_y = (pt1_x + pt3_x) / 2, (pt1_y + pt3_y) / 2

                cv2.circle(result_image, (int(line_center_x), int(line_center_y)), 5, (0, 0, 255),
                           -1)  # 画出中心点(draw the center point)
                centroid_sum += line_center_x * roi[-1]

        # Add a condition: if the line-following target is not found, increment the counter(添加判断，如果没有找到巡线目标，则增加计数器)
        if centroid_sum == 0:
            self.lost_target_count += 1
            return result_image, None
        else:
            self.lost_target_count = 0  # Reset the counter(重置计数器)
        center_pos = centroid_sum / self.weight_sum  # 按比重计算中心点(calculate the center point according to the ratio)
        deflection_angle = -math.atan((center_pos - (w / 2.0)) / (h / 2.0))  # 计算线角度(calculate the line angle)
        return result_image, deflection_angle


class LineFollowingNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        self.name = name
        self.set_callback = False
        self.is_running = False
        self.color_picker = None
        self.follower = None
        self.scan_angle = math.radians(40)
        self.pid = pid.PID(0.005, 0.001, 0.0)
        self.empty = 0
        self.count = 0
        self.stop = False
        self.threshold = 0.5
        self.stop_threshold = 0.4
        self.lock = threading.RLock()
        self.image_sub = None
        self.lidar_sub = None
        self.image_height = None
        self.image_width = None
        self.bridge = CvBridge()
        self.image_queue = queue.Queue(2)
        self.lab_data = yaml_handle.get_yaml_data(yaml_handle.lab_file_path)  # 加载LAB数据
        self.lidar_type = os.environ.get('LIDAR_TYPE')
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.pwm_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 10)
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)  # 底盘控制(chassis control)
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)  # 图像处理结果发布(publish the image processing result)
        self.create_service(Trigger, '~/enter', self.enter_srv_callback)  # 进入玩法(enter the game)
        self.create_service(Trigger, '~/exit', self.exit_srv_callback)  # 退出玩法(exit the game)
        self.create_service(SetBool, '~/set_running', self.set_running_srv_callback)  # 开启玩法(start the game)
        self.create_service(SetPoint, '~/set_target_color', self.set_target_color_srv_callback)  # 设置颜色(set the color)
        self.create_service(Trigger, '~/get_target_color', self.get_target_color_srv_callback)  # 获取颜色(get the color)
        self.create_service(SetFloat64, '~/set_threshold', self.set_threshold_srv_callback)  # 设置阈值(set the threshold)
        self.create_service(SetString, '~/set_large_model_target_color',
                            self.set_large_model_target_color_srv_callback)  # Create service(创建服务)
        Heart(self, self.name + '/heartbeat', 5,
              lambda _: self.exit_srv_callback(request=Trigger.Request(), response=Trigger.Response()))  # 心跳包(heartbeat package)
        self.debug = self.get_parameter('debug').value
        if self.debug:
            threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

        self.large_model_tracking = False  # Add a flag to indicate whether it was triggered by the large model(添加标志，指示是否由大模型启动)
        self.target_lost_timer = None  # Add a timer(添加计时器)
        self.target_lost = False  # Add target_lost flag(添加target_lost)

        self.wakeup_sub = self.create_subscription(Bool, '/vocal_detect/wakeup', self.wakeup_callback, 1)
        # Get the camera type(获取相机类型)
        self.camera_type = os.environ['DEPTH_CAMERA_TYPE']

    def pwm_controller(self, position_data):
        pwm_list = []
        msg = SetPWMServoState()
        msg.duration = 0.2
        for i in range(len(position_data)):
            pos = PWMServoState()
            pos.id = [i + 1]
            pos.position = [int(position_data[i])]
            pwm_list.append(pos)
        msg.state = pwm_list
        self.pwm_pub.publish(msg)

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
                # 设置鼠标点击事件的回调函数(set a callback function for mouse click event)
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
        self.get_logger().info('\033[1;32m%s\033[0m' % "line following enter")
        if os.environ['DEPTH_CAMERA_TYPE'] == 'usb_cam':
            self.pwm_controller([1850, 1500])
        with self.lock:
            self.stop = False
            self.is_running = False
            self.color_picker = None
            self.pid = pid.PID(1.1, 0.0, 0.0)
            self.follower = None
            self.threshold = 0.5
            self.empty = 0
            self.large_model_tracking = False  # Reset the flag(重置标志)
            if self.image_sub is None:
                self.image_sub = self.create_subscription(Image, 'ascamera/camera_publisher/rgb0/image',
                                                            self.image_callback, 1)  # 摄像头订阅(subscribe to the camera)
            if self.lidar_sub is None:
                qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
                self.lidar_sub = self.create_subscription(LaserScan, '/scan_raw', self.lidar_callback,
                                                            qos)  # 订阅雷达数据(subscribe to Lidar data)
            self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "enter"
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "line following exit")
        if os.environ['DEPTH_CAMERA_TYPE'] == 'usb_cam':
            self.pwm_controller([1500, 1500])
        try:
            if self.image_sub is not None:
                self.destroy_subscription(self.image_sub)
                self.image_sub = None
            if self.lidar_sub is not None:
                self.destroy_subscription(self.lidar_sub)
                self.lidar_sub = None
        except Exception as e:
            self.get_logger().error(str(e))
        with self.lock:
            self.is_running = False
            self.color_picker = None
            self.pid = pid.PID(0.00, 0.001, 0.0)
            self.follower = None
            self.threshold = 0.5
            self.mecanum_pub.publish(Twist())
            self.large_model_tracking = False  
            if self.target_lost_timer is not None:  
                self.target_lost_timer.cancel()
                self.target_lost_timer = None
        response.success = True
        response.message = "exit"
        return response

    def set_target_color_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "set_target_color")
        with self.lock:
            x, y = request.data.x, request.data.y
            self.follower = None
            if x == -1 and y == -1:
                self.color_picker = None
            else:
                self.color_picker = ColorPicker(request.data, 5)
                self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "set_target_color"
        return response

    def get_target_color_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "get_target_color")
        response.success = False
        response.message = "get_target_color"
        with self.lock:
            if self.follower is not None:
                response.success = True
                rgb = self.follower.target_rgb
                response.message = "{},{},{}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        return response

    def set_running_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "set_running")
        with self.lock:
            self.is_running = request.data
            self.empty = 0
            if not self.is_running:
                self.mecanum_pub.publish(Twist())
                if self.target_lost_timer is not None:  
                    self.target_lost_timer.cancel()
                    self.target_lost_timer = None
        response.success = True
        response.message = "set_running"
        return response

    def set_threshold_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "set threshold")
        with self.lock:
            self.threshold = request.data
            response.success = True
            response.message = "set_threshold"
            return response

    # Add a service callback function to set preset color thresholds(添加设置预设颜色阈值的服务回调函数)
    def set_large_model_target_color_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "set_large_model_target_color")
        with self.lock:
            color_name = request.data  
            self.follower = LineFollower(None, self, color_name)  
            self.mecanum_pub.publish(Twist())  
            self.large_model_tracking = True 
            self.is_running = True
            self.get_logger().info("由大模型启动巡线 (Line following started by large model)")
        response.success = True
        response.message = "set_large_model_target_color"
        return response

    def lidar_callback(self, lidar_data):
        # 数据大小 = 扫描角度/每扫描一次增加的角度(data size= scanning angle/ the increased angle per scan)
        if self.lidar_type != 'G4':
            min_index = int(math.radians(MAX_SCAN_ANGLE / 2.0) / lidar_data.angle_increment)
            max_index = int(math.radians(MAX_SCAN_ANGLE / 2.0) / lidar_data.angle_increment)
            left_ranges = lidar_data.ranges[:max_index]  # 左半边数据(left data)
            right_ranges = lidar_data.ranges[::-1][:max_index]  # 右半边数据(right data)
        elif self.lidar_type == 'G4':
            min_index = int(math.radians((360 - MAX_SCAN_ANGLE) / 2.0) / lidar_data.angle_increment)
            max_index = int(math.radians(180) / lidar_data.angle_increment)
            left_ranges = lidar_data.ranges[min_index:max_index][::-1]  # 左半边数据 (the left data)
            right_ranges = lidar_data.ranges[::-1][min_index:max_index][::-1]  # 右半边数据 (the right data)

        # 根据设定取数据(Get data according to settings)
        angle = self.scan_angle / 2
        angle_index = int(angle / lidar_data.angle_increment + 0.50)
        left_range, right_range = np.array(left_ranges[:angle_index]), np.array(right_ranges[:angle_index])

        left_nonzero = left_range.nonzero()
        right_nonzero = right_range.nonzero()
        left_nonan = np.isfinite(left_range[left_nonzero])
        right_nonan = np.isfinite(right_range[right_nonzero])
        # 取左右最近的距离(Take the nearest distance left and right)
        min_dist_left_ = left_range[left_nonzero][left_nonan]
        min_dist_right_ = right_range[right_nonzero][right_nonan]
        if len(min_dist_left_) > 1 and len(min_dist_right_) > 1:
            min_dist_left = min_dist_left_.min()
            min_dist_right = min_dist_right_.min()
            if min_dist_left < self.stop_threshold or min_dist_right < self.stop_threshold:
                self.stop = True
            else:
                self.count += 1
                if self.count > 5:
                    self.count = 0
                    self.stop = False

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        self.image_height, self.image_width = rgb_image.shape[:2]
        result_image = np.copy(rgb_image)
        
        with self.lock:
            if self.color_picker is not None:
                try:
                    target_color, result_image = self.color_picker(rgb_image, result_image)
                    if target_color is not None:
                        self.color_picker = None
                        self.follower = LineFollower(target_color, self)
                        self.get_logger().info("目标颜色设置成功: {}".format(target_color))
                except Exception as e:
                    self.get_logger().error(str(e))
            else:
                twist = Twist()
                twist.linear.x = 0.15
                
                if self.follower is not None:
                    try:
                        # Execute the line following detection (执行巡线检测)
                        result_image, deflection_angle = self.follower(rgb_image, result_image, self.threshold)
                        
                        # Target detection status handling(目标检测状态处理)
                        if deflection_angle is None:
                            # Target not detected, increment the lost target counter(未检测到目标，增加丢失计数器)
                            self.follower.lost_target_count += 1
                            
                            # If the game was started by the large model and the target is not detected for 5 consecutive frames(如果是大模型启动的玩法且连续5帧未检测到目标)
                            if (self.large_model_tracking and 
                                self.follower.lost_target_count >= 5 and 
                                not self.target_lost):
                                
                                self.get_logger().warn("目标丢失，启动5秒计时器...")
                                self.start_stop_timer()  # Start a 5-second timer(启动5秒计时器)
                                
                        else:
                            # Target detected, reset the lost target counter(检测到目标，重置丢失计数器)
                            self.follower.lost_target_count = 0
                            
                            # If the timer is running and the game was started by the large model(果计时器正在运行且是大模型启动的玩法)
                            if (self.large_model_tracking and 
                                self.target_lost_timer is not None and 
                                self.target_lost_timer.is_alive()):
                                
                                self.get_logger().info("重新检测到目标，取消停止计时")
                                self.target_lost_timer.cancel()
                                self.target_lost_timer = None
                                self.target_lost = False
                        
                        # Control logic(控制逻辑)
                        if (deflection_angle is not None and 
                            self.is_running and 
                            not self.stop and 
                            not self.target_lost):
                            
                            self.pid.update(deflection_angle)
                            if self.machine_type == 'MentorPi_Acker':
                                steering_angle = common.set_range(-self.pid.output, 
                                                                -math.radians(40), 
                                                                math.radians(40))
                                if steering_angle != 0:
                                    R = 0.145 / math.tan(steering_angle)
                                    twist.angular.z = twist.linear.x / R
                            else:
                                twist.angular.z = common.set_range(-self.pid.output, -1.0, 1.0)
                            self.mecanum_pub.publish(twist)
                            
                        elif self.stop or self.target_lost:
                            self.mecanum_pub.publish(Twist())
                        else:
                            self.pid.clear()
                            
                    except Exception as e:
                        self.get_logger().error(str(e))
        
        # 发布处理结果
        if self.debug:
            if self.image_queue.full():
                # 如果队列已满，丢弃最旧的图像(if the queue is full, remove the oldest image)
                self.image_queue.get()
                # 将图像放入队列(put the image into the queue)
                self.image_queue.get()
            self.image_queue.put(result_image)
        else:
            self.result_publisher.publish(self.bridge.cv2_to_imgmsg(result_image, "rgb8"))

    def start_stop_timer(self):
        """启动5秒停止计时器"""
        if self.target_lost_timer is None:
            self.target_lost_timer = threading.Timer(5.0, self.stop_after_lose)
            self.target_lost_timer.start()
            self.target_lost = True
            self.get_logger().warn("5秒计时器已启动")

    def stop_after_lose(self):
        """Handling after target is lost for more than 5 seconds(目标丢失超过5秒后的处理)s"""
        with self.lock:
            if self.large_model_tracking:
                self.get_logger().warn("目标丢失超过5秒，停止移动")
                self.mecanum_pub.publish(Twist())
                self.is_running = False
                self.target_lost_timer = None
                self.target_lost = False


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
                    self.follower = None  
                    self.get_logger().info("Large model tracking stopped.")
def main():
    node = LineFollowingNode('line_following')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
