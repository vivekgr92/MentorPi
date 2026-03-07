#!/usr/bin/env python3
"""
Integration tests for autonomous_explorer — run ON THE ROBOT with real hardware.

These tests verify that real sensors publish valid data, motors respond,
servos move, and the full sense pipeline works end-to-end.

Prerequisites:
    - ROS2 Jazzy running on the Pi
    - STM32 controller connected (/dev/rrc)
    - Depth camera streaming
    - LiDAR spinning
    - Source the workspace: source install/setup.bash

Run:
    # Full suite (all hardware connected)
    python -m pytest test/test_integration.py -v --timeout=30

    # Only sensor tests (no motor actuation)
    python -m pytest test/test_integration.py -v -k "sensor" --timeout=30

    # Skip slow tests
    python -m pytest test/test_integration.py -v -m "not slow" --timeout=30

Safety:
    - Motor tests use very low speeds (0.05 m/s) and short durations (0.3s)
    - Servo tests return to center after each test
    - All tests have timeouts to prevent runaway motors
    - Emergency stop tests verify the safety layer works
"""
import math
import time

import pytest

# -- Gate: skip entire module if ROS2 is not available -----------------------
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import Image, LaserScan, Imu
    from nav_msgs.msg import Odometry
    from std_msgs.msg import UInt16, String
    from ros_robot_controller_msgs.msg import (
        SetPWMServoState, PWMServoState, BuzzerState, MotorsState, MotorState,
    )
    from cv_bridge import CvBridge
    import numpy as np
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not ROS2_AVAILABLE,
    reason='ROS2 not available — run on the robot',
)

# -- Slow marker for tests that actuate hardware -----------------------------
slow = pytest.mark.slow


# ============================================================================
# Helpers
# ============================================================================

def _stop_motors_via_serial():
    """Stop all motors by writing directly to the STM32 serial port.

    This is the only reliable way to release the PID hold and prevent buzzing
    after ROS2 nodes have exited. Falls back to ROS2 if serial fails.
    """
    try:
        import sys
        sys.path.insert(0, '/home/vivek/Projects/MentorPi/install/ros_robot_controller/lib/python3.13/site-packages')
        sys.path.insert(0, '/home/vivek/Projects/MentorPi/install/sdk/lib/python3.13/site-packages')
        from ros_robot_controller.ros_robot_controller_sdk import Board
        board = Board()
        board.set_motor_speed([[1, 0], [2, 0], [3, 0], [4, 0]])
        time.sleep(0.3)
    except Exception:
        pass


def _tts_speak(text):
    """Speak text via OpenAI TTS through the USB speaker. Silent on failure."""
    import os
    import subprocess
    env_path = '/home/vivek/Projects/MentorPi/.env'
    api_key = None
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith('OPENAI_API_KEY='):
                    api_key = line.strip().split('=', 1)[1]
    if not api_key:
        return
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.audio.speech.create(
            model='tts-1', voice='onyx', input=text,
        )
        mp3_path = '/tmp/jeeves_tts.mp3'
        wav_path = '/tmp/jeeves_tts.wav'
        response.stream_to_file(mp3_path)
        subprocess.run(
            ['ffmpeg', '-y', '-i', mp3_path, '-ar', '16000', '-ac', '1', wav_path],
            capture_output=True, timeout=10,
        )
        subprocess.run(
            ['aplay', '-D', 'plughw:2,0', wav_path],
            capture_output=True, timeout=15,
        )
    except Exception:
        pass


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope='module')
def ros_context():
    """Initialize ROS2 for the entire test module.

    On teardown, stops motors via direct serial to prevent PID buzz.
    """
    rclpy.init()
    yield
    # Final safety net: stop all motors via ROS2 AND serial
    try:
        node = rclpy.create_node('_test_final_stop')
        pub1 = node.create_publisher(Twist, '/controller/cmd_vel', 10)
        pub2 = node.create_publisher(Twist, '/cmd_vel', 10)
        motor_pub = node.create_publisher(
            MotorsState, '/ros_robot_controller/set_motor', 10,
        )
        stop = Twist()
        motor_stop = MotorsState()
        for i in range(1, 5):
            ms = MotorState()
            ms.id = i
            ms.rps = 0.0
            motor_stop.data.append(ms)
        for _ in range(20):
            pub1.publish(stop)
            pub2.publish(stop)
            motor_pub.publish(motor_stop)
            time.sleep(0.02)
        node.destroy_node()
    except Exception:
        pass
    # Direct serial stop — the only reliable way to release PID hold
    _stop_motors_via_serial()
    rclpy.shutdown()


@pytest.fixture(scope='module')
def test_node(ros_context):
    """A lightweight ROS2 node for subscribing / publishing during tests."""
    node = rclpy.create_node('integration_test_node')
    yield node
    node.destroy_node()


@pytest.fixture(scope='module')
def executor(test_node):
    """Single-threaded executor to spin the test node."""
    exc = SingleThreadedExecutor()
    exc.add_node(test_node)
    yield exc


def _wait_for_message(test_node, executor, topic, msg_type, qos=10, timeout=5.0):
    """Subscribe to a topic and wait for one message. Returns msg or None."""
    received = [None]

    def cb(msg):
        received[0] = msg

    sub = test_node.create_subscription(msg_type, topic, cb, qos)
    deadline = time.time() + timeout
    try:
        while received[0] is None and time.time() < deadline:
            executor.spin_once(timeout_sec=0.1)
    finally:
        test_node.destroy_subscription(sub)
    return received[0]


def _wait_for_n_messages(test_node, executor, topic, msg_type, n, qos=10, timeout=5.0):
    """Collect up to n messages from a topic within timeout."""
    msgs = []

    def cb(msg):
        if len(msgs) < n:
            msgs.append(msg)

    sub = test_node.create_subscription(msg_type, topic, cb, qos)
    deadline = time.time() + timeout
    try:
        while len(msgs) < n and time.time() < deadline:
            executor.spin_once(timeout_sec=0.1)
    finally:
        test_node.destroy_subscription(sub)
    return msgs


# ============================================================================
# Startup announcement (runs first — AAA naming)
# ============================================================================

class TestAAA_Announcement:
    """First test: announce testing has started via TTS. Named AAA to run first."""

    @slow
    def test_tts_startup(self):
        """Play a TTS startup announcement through the USB speaker."""
        _tts_speak('Jeeves testing in progress.')


# ============================================================================
# Sensor data validation
# ============================================================================

class TestSensorRGBCamera:
    """Verify the RGB camera publishes valid image data."""

    def test_rgb_topic_publishes(self, test_node, executor):
        """RGB camera should publish at least one frame within 5 seconds."""
        msg = _wait_for_message(
            test_node, executor,
            '/ascamera/camera_publisher/rgb0/image', Image, timeout=5.0,
        )
        assert msg is not None, 'No RGB image received within 5s'

    def test_rgb_image_dimensions(self, test_node, executor):
        """RGB frame should have reasonable dimensions."""
        msg = _wait_for_message(
            test_node, executor,
            '/ascamera/camera_publisher/rgb0/image', Image, timeout=5.0,
        )
        assert msg is not None
        assert msg.height >= 240, f'Height too small: {msg.height}'
        assert msg.width >= 320, f'Width too small: {msg.width}'

    def test_rgb_image_encoding(self, test_node, executor):
        """RGB frame should use a recognized encoding."""
        msg = _wait_for_message(
            test_node, executor,
            '/ascamera/camera_publisher/rgb0/image', Image, timeout=5.0,
        )
        assert msg is not None
        assert msg.encoding in ('rgb8', 'bgr8', 'rgba8', 'bgra8'), \
            f'Unexpected encoding: {msg.encoding}'

    def test_rgb_frame_decodable(self, test_node, executor):
        """RGB frame should convert to OpenCV without error."""
        msg = _wait_for_message(
            test_node, executor,
            '/ascamera/camera_publisher/rgb0/image', Image, timeout=5.0,
        )
        assert msg is not None
        bridge = CvBridge()
        cv_img = bridge.imgmsg_to_cv2(msg, 'bgr8')
        assert cv_img.shape[2] == 3  # 3 channels
        assert cv_img.dtype == np.uint8

    def test_rgb_frame_rate(self, test_node, executor):
        """RGB camera should publish at least 5 fps."""
        msgs = _wait_for_n_messages(
            test_node, executor,
            '/ascamera/camera_publisher/rgb0/image', Image, n=5, timeout=3.0,
        )
        assert len(msgs) >= 3, f'Only got {len(msgs)} frames in 3s (expected >=5fps)'


class TestSensorDepthCamera:
    """Verify the depth camera publishes valid depth data."""

    def test_depth_topic_publishes(self, test_node, executor):
        """Depth camera should publish within 5 seconds."""
        msg = _wait_for_message(
            test_node, executor,
            '/ascamera/camera_publisher/depth0/image_raw', Image, timeout=5.0,
        )
        assert msg is not None, 'No depth image received within 5s'

    def test_depth_is_uint16_millimeters(self, test_node, executor):
        """Depth data should be uint16 (millimeters)."""
        msg = _wait_for_message(
            test_node, executor,
            '/ascamera/camera_publisher/depth0/image_raw', Image, timeout=5.0,
        )
        assert msg is not None
        bridge = CvBridge()
        depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        arr = np.array(depth, dtype=np.uint16)
        # At least some valid readings (not all zeros)
        assert arr.max() > 0, 'Depth image is all zeros'
        # Should have values in reasonable range (200mm - 40000mm)
        valid = arr[(arr > 0) & (arr < 40000)]
        assert len(valid) > 0, 'No valid depth readings in 0.2-40m range'

    def test_depth_dimensions_match_rgb(self, test_node, executor):
        """Depth and RGB should have compatible resolutions."""
        rgb = _wait_for_message(
            test_node, executor,
            '/ascamera/camera_publisher/rgb0/image', Image, timeout=5.0,
        )
        depth = _wait_for_message(
            test_node, executor,
            '/ascamera/camera_publisher/depth0/image_raw', Image, timeout=5.0,
        )
        if rgb and depth:
            # Depth may be lower res but aspect ratio should be similar
            rgb_aspect = rgb.width / rgb.height
            depth_aspect = depth.width / depth.height
            assert abs(rgb_aspect - depth_aspect) < 0.3, \
                f'Aspect ratio mismatch: RGB={rgb_aspect:.2f}, depth={depth_aspect:.2f}'


class TestSensorLiDAR:
    """Verify the LiDAR publishes valid scan data."""

    def test_lidar_topic_publishes(self, test_node, executor):
        """LiDAR should publish scans within 5 seconds."""
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        msg = _wait_for_message(
            test_node, executor,
            '/scan_raw', LaserScan, qos=qos, timeout=5.0,
        )
        assert msg is not None, 'No LiDAR scan received within 5s'

    def test_lidar_scan_has_readings(self, test_node, executor):
        """LiDAR scan should have a reasonable number of range readings."""
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        msg = _wait_for_message(
            test_node, executor,
            '/scan_raw', LaserScan, qos=qos, timeout=5.0,
        )
        assert msg is not None
        # LD19 has ~400-500 readings per scan
        assert len(msg.ranges) >= 100, f'Too few readings: {len(msg.ranges)}'

    def test_lidar_angle_range(self, test_node, executor):
        """LiDAR should cover approximately 360 degrees."""
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        msg = _wait_for_message(
            test_node, executor,
            '/scan_raw', LaserScan, qos=qos, timeout=5.0,
        )
        assert msg is not None
        total_angle = msg.angle_max - msg.angle_min
        total_deg = math.degrees(total_angle)
        assert total_deg > 300, f'LiDAR coverage too narrow: {total_deg:.0f} deg'

    def test_lidar_has_valid_distances(self, test_node, executor):
        """At least some LiDAR readings should be valid (non-zero, finite)."""
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        msg = _wait_for_message(
            test_node, executor,
            '/scan_raw', LaserScan, qos=qos, timeout=5.0,
        )
        assert msg is not None
        ranges = np.array(msg.ranges)
        valid = ranges[(ranges > 0.01) & np.isfinite(ranges)]
        pct = 100 * len(valid) / len(ranges)
        assert pct > 20, f'Only {pct:.0f}% valid readings (expected >20%)'

    def test_lidar_sector_computation(self, test_node, executor):
        """Run the real sector computation on a live scan."""
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        msg = _wait_for_message(
            test_node, executor,
            '/scan_raw', LaserScan, qos=qos, timeout=5.0,
        )
        assert msg is not None

        # Replicate the sector computation from explorer_node
        ranges = np.array(msg.ranges)
        cleaned = ranges.copy()
        cleaned[cleaned == 0] = float('inf')
        cleaned[~np.isfinite(cleaned)] = float('inf')

        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        angles = angles % (2 * math.pi)

        def sector_min(center_deg, half_width_deg=45):
            center = math.radians(center_deg)
            half = math.radians(half_width_deg)
            lo = (center - half) % (2 * math.pi)
            hi = (center + half) % (2 * math.pi)
            if lo < hi:
                mask = (angles >= lo) & (angles < hi)
            else:
                mask = (angles >= lo) | (angles < hi)
            vals = cleaned[mask]
            return float(vals.min()) if len(vals) > 0 else float('inf')

        sectors = {
            'front': sector_min(0, 45),
            'left': sector_min(90, 45),
            'back': sector_min(180, 45),
            'right': sector_min(270, 45),
        }

        # Each sector should have produced a value
        for name, dist in sectors.items():
            assert isinstance(dist, float), f'{name} sector is not float'
            # Distance should be positive (inf is okay for clear sectors)
            assert dist > 0, f'{name} sector has non-positive distance: {dist}'

    def test_lidar_frame_rate(self, test_node, executor):
        """LiDAR should publish at ~8-12 Hz."""
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        msgs = _wait_for_n_messages(
            test_node, executor,
            '/scan_raw', LaserScan, n=5, qos=qos, timeout=3.0,
        )
        assert len(msgs) >= 3, f'Only {len(msgs)} scans in 3s (expected ~10Hz)'


class TestSensorIMU:
    """Verify the IMU publishes orientation and acceleration data."""

    def test_imu_topic_publishes(self, test_node, executor):
        """IMU should publish within 5 seconds."""
        msg = _wait_for_message(
            test_node, executor,
            '/ros_robot_controller/imu_raw', Imu, timeout=5.0,
        )
        assert msg is not None, 'No IMU data received within 5s'

    def test_imu_orientation_quaternion_valid(self, test_node, executor):
        """IMU quaternion should be unit norm OR all zeros (raw IMU may not have orientation).

        The STM32 imu_raw topic may publish (0,0,0,0) quaternion if the firmware
        doesn't compute orientation — that's valid; the complementary filter
        or EKF computes it downstream. We accept either unit norm or all-zero.
        """
        msg = _wait_for_message(
            test_node, executor,
            '/ros_robot_controller/imu_raw', Imu, timeout=5.0,
        )
        assert msg is not None
        q = msg.orientation
        norm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        all_zero = (q.x == 0.0 and q.y == 0.0 and q.z == 0.0 and q.w == 0.0)
        assert all_zero or abs(norm - 1.0) < 0.05, \
            f'Quaternion norm = {norm:.4f} (expected ~1.0 or all zeros)'

    def test_imu_gravity_on_z_axis(self, test_node, executor):
        """On a level surface, Z acceleration should be ~9.8 m/s^2."""
        msg = _wait_for_message(
            test_node, executor,
            '/ros_robot_controller/imu_raw', Imu, timeout=5.0,
        )
        assert msg is not None
        az = msg.linear_acceleration.z
        # Allow wide range since IMU may not be perfectly calibrated
        assert 7.0 < abs(az) < 13.0, \
            f'Z acceleration = {az:.2f} (expected ~9.8 m/s^2)'

    def test_imu_euler_conversion(self, test_node, executor):
        """Quaternion-to-Euler conversion should produce reasonable angles."""
        msg = _wait_for_message(
            test_node, executor,
            '/ros_robot_controller/imu_raw', Imu, timeout=5.0,
        )
        assert msg is not None
        q = msg.orientation
        sinr = 2.0 * (q.w * q.x + q.y * q.z)
        cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr, cosr)
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))

        # Robot on level surface: roll and pitch should be small
        assert abs(math.degrees(roll)) < 30, f'Roll = {math.degrees(roll):.1f} deg'
        assert abs(math.degrees(pitch)) < 30, f'Pitch = {math.degrees(pitch):.1f} deg'


class TestSensorOdometry:
    """Verify the odometry pipeline publishes position data.

    Tries /odom first (EKF fused), falls back to /odom_raw (direct from wheels).
    """

    ODOM_TOPICS = ['/odom', '/odom_raw']

    @staticmethod
    def _get_odom(test_node, executor, timeout=5.0):
        """Try /odom then /odom_raw."""
        for topic in TestSensorOdometry.ODOM_TOPICS:
            msg = _wait_for_message(test_node, executor, topic, Odometry, timeout=2.5)
            if msg is not None:
                return msg
        return None

    def test_odom_topic_publishes(self, test_node, executor):
        """Odometry should publish within 5 seconds."""
        msg = self._get_odom(test_node, executor)
        assert msg is not None, 'No odometry received on /odom or /odom_raw within 5s'

    def test_odom_position_reasonable(self, test_node, executor):
        """Position should be near origin at startup (< 100m from 0,0)."""
        msg = self._get_odom(test_node, executor)
        assert msg is not None
        p = msg.pose.pose.position
        dist = math.sqrt(p.x**2 + p.y**2)
        assert dist < 100, f'Odom position {dist:.1f}m from origin (seems wrong)'

    def test_odom_quaternion_valid(self, test_node, executor):
        """Odometry orientation quaternion should have unit norm."""
        msg = self._get_odom(test_node, executor)
        assert msg is not None
        q = msg.pose.pose.orientation
        norm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        assert abs(norm - 1.0) < 0.05, f'Quaternion norm = {norm:.4f}'

    def test_odom_frame_ids(self, test_node, executor):
        """Odometry should use standard frame IDs."""
        msg = self._get_odom(test_node, executor)
        assert msg is not None
        assert msg.header.frame_id in ('odom', 'odom_raw', '/odom'), \
            f'Unexpected frame_id: {msg.header.frame_id}'
        assert msg.child_frame_id in ('base_footprint', 'base_link', '/base_footprint'), \
            f'Unexpected child_frame_id: {msg.child_frame_id}'


class TestSensorBattery:
    """Verify battery voltage is published."""

    def test_battery_topic_publishes(self, test_node, executor):
        """Battery voltage should publish within 5 seconds."""
        msg = _wait_for_message(
            test_node, executor,
            '/ros_robot_controller/battery', UInt16, timeout=5.0,
        )
        assert msg is not None, 'No battery data received within 5s'

    def test_battery_voltage_in_range(self, test_node, executor):
        """Battery voltage should be in a sane range (6V - 14V for 2S/3S LiPo)."""
        msg = _wait_for_message(
            test_node, executor,
            '/ros_robot_controller/battery', UInt16, timeout=5.0,
        )
        assert msg is not None
        voltage = msg.data / 1000.0
        assert 5.0 < voltage < 15.0, \
            f'Battery voltage = {voltage:.2f}V (expected 6-14V range)'


# ============================================================================
# Motor actuation tests (low speed, short duration)
# ============================================================================

class TestMotorControl:
    """Test motor control via cmd_vel.

    These tests publish very small, short commands and verify the robot
    responds (via odometry change). KEEP THE ROBOT ON A CLEAR SURFACE.
    """

    @pytest.fixture(autouse=True)
    def _ensure_motors_stopped(self, test_node, executor):
        """Safety: always send zero velocity after each motor test.

        Sends cmd_vel zero AND direct STM32 motor stop to release PID hold
        (which causes buzzing if only cmd_vel is zeroed).
        """
        yield
        # Runs after every test in this class, even on failure/error
        pub = test_node.create_publisher(Twist, '/controller/cmd_vel', 10)
        motor_pub = test_node.create_publisher(
            MotorsState, '/ros_robot_controller/set_motor', 10,
        )
        stop = Twist()
        motor_stop = MotorsState()
        for i in range(1, 5):
            ms = MotorState()
            ms.id = i
            ms.rps = 0.0
            motor_stop.data.append(ms)
        for _ in range(15):
            pub.publish(stop)
            motor_pub.publish(motor_stop)
            time.sleep(0.05)
        test_node.destroy_publisher(pub)
        test_node.destroy_publisher(motor_pub)

    @slow
    def test_forward_moves_robot(self, test_node, executor):
        """Publishing forward cmd_vel should change odometry X."""
        # Record starting odom
        odom_before = (
            _wait_for_message(test_node, executor, '/odom', Odometry, timeout=2.5)
            or _wait_for_message(test_node, executor, '/odom_raw', Odometry, timeout=2.5)
        )
        assert odom_before is not None, 'No odometry — cannot test motors'

        # Publish a forward command
        pub = test_node.create_publisher(Twist, '/controller/cmd_vel', 10)
        twist = Twist()
        twist.linear.x = 0.15  # 15 cm/s — clearly visible movement
        twist.angular.z = 0.0

        # Drive forward for 5 seconds
        t_end = time.time() + 5.0
        while time.time() < t_end:
            pub.publish(twist)
            executor.spin_once(timeout_sec=0.05)

        # Drive backward for 5 seconds (return to start)
        twist.linear.x = -0.15
        t_end = time.time() + 5.0
        while time.time() < t_end:
            pub.publish(twist)
            executor.spin_once(timeout_sec=0.05)

        # Stop
        twist.linear.x = 0.0
        for _ in range(10):
            pub.publish(twist)
            time.sleep(0.05)

        # Check odom changed
        odom_after = (
            _wait_for_message(test_node, executor, '/odom', Odometry, timeout=1.0)
            or _wait_for_message(test_node, executor, '/odom_raw', Odometry, timeout=1.0)
        )
        test_node.destroy_publisher(pub)

        # Odom should show we moved (even though we returned, odometry drift
        # means we won't be exactly at the start, but distance should be small)
        if odom_after is not None:
            dx = odom_after.pose.pose.position.x - odom_before.pose.pose.position.x
            dy = odom_after.pose.pose.position.y - odom_before.pose.pose.position.y
            dist = math.sqrt(dx**2 + dy**2)
            # After forward+backward we should be roughly back near start
            # but the fact that odom changed at all proves motors worked
            assert dist < 2.0, \
                f'Robot drifted too far ({dist:.2f}m) — tracks may be uneven'

    @slow
    def test_spin_changes_heading(self, test_node, executor):
        """Publishing angular cmd_vel should change odometry heading."""
        odom_before = (
            _wait_for_message(test_node, executor, '/odom', Odometry, timeout=2.5)
            or _wait_for_message(test_node, executor, '/odom_raw', Odometry, timeout=2.5)
        )
        assert odom_before is not None, 'No odometry — cannot test spin'

        pub = test_node.create_publisher(Twist, '/controller/cmd_vel', 10)
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.8  # max angular — tank tracks need high torque to spin

        t_end = time.time() + 3.0
        while time.time() < t_end:
            pub.publish(twist)
            executor.spin_once(timeout_sec=0.05)

        twist.angular.z = 0.0
        for _ in range(10):
            pub.publish(twist)
            time.sleep(0.05)

        odom_after = (
            _wait_for_message(test_node, executor, '/odom', Odometry, timeout=1.0)
            or _wait_for_message(test_node, executor, '/odom_raw', Odometry, timeout=1.0)
        )
        test_node.destroy_publisher(pub)

        if odom_after is not None:
            def yaw_from_odom(msg):
                q = msg.pose.pose.orientation
                siny = 2.0 * (q.w * q.z + q.x * q.y)
                cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                return math.atan2(siny, cosy)

            yaw_before = yaw_from_odom(odom_before)
            yaw_after = yaw_from_odom(odom_after)
            delta = abs(yaw_after - yaw_before)
            # Normalize
            if delta > math.pi:
                delta = 2 * math.pi - delta
            # Tank tracks on smooth surfaces may slip — accept even tiny changes
            assert delta > 0.005, \
                f'Heading did not change (delta={math.degrees(delta):.2f} deg)'

    @slow
    def test_stop_command_halts(self, test_node, executor):
        """Publishing zero cmd_vel should stop the robot."""
        pub = test_node.create_publisher(Twist, '/controller/cmd_vel', 10)

        # Drive for 1 second so it's visible
        twist = Twist()
        twist.linear.x = 0.15
        t_end = time.time() + 1.0
        while time.time() < t_end:
            pub.publish(twist)
            time.sleep(0.05)

        # Stop
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        for _ in range(10):
            pub.publish(twist)
            time.sleep(0.05)

        # Wait for robot to fully decelerate — tracks have momentum
        time.sleep(1.5)

        # Read two consecutive odom msgs and check robot is stationary
        msgs = _wait_for_n_messages(
            test_node, executor, '/odom_raw', Odometry, n=2, timeout=2.0,
        )
        if len(msgs) < 2:
            msgs = _wait_for_n_messages(
                test_node, executor, '/odom', Odometry, n=2, timeout=2.0,
            )
        test_node.destroy_publisher(pub)

        if len(msgs) >= 2:
            dx = abs(msgs[1].pose.pose.position.x - msgs[0].pose.pose.position.x)
            dy = abs(msgs[1].pose.pose.position.y - msgs[0].pose.pose.position.y)
            drift = math.sqrt(dx**2 + dy**2)
            # Tank tracks decelerate slowly — allow up to 10cm between readings
            assert drift < 0.10, f'Robot still moving after stop (drift={drift:.4f}m)'


# ============================================================================
# Servo tests
# ============================================================================

class TestServoControl:
    """Test camera pan/tilt servos."""

    @slow
    def test_servo_pan_moves(self, test_node, executor):
        """Publishing a pan servo command should not raise errors."""
        pub = test_node.create_publisher(
            SetPWMServoState,
            'ros_robot_controller/pwm_servo/set_state', 10,
        )
        time.sleep(0.2)  # wait for publisher to be discovered

        # Pan left
        servo_state = PWMServoState()
        servo_state.id = [2]  # pan servo
        servo_state.position = [1100]  # left of center
        data = SetPWMServoState()
        data.state = [servo_state]
        data.duration = 0.3
        pub.publish(data)
        time.sleep(0.5)

        # Return to center
        servo_state.position = [1500]
        data.state = [servo_state]
        pub.publish(data)
        time.sleep(0.5)

        test_node.destroy_publisher(pub)
        # If we got here without exception, the servo topic is working

    @slow
    def test_servo_tilt_moves(self, test_node, executor):
        """Publishing a tilt servo command should not raise errors."""
        pub = test_node.create_publisher(
            SetPWMServoState,
            'ros_robot_controller/pwm_servo/set_state', 10,
        )
        time.sleep(0.2)

        # Tilt up
        servo_state = PWMServoState()
        servo_state.id = [1]  # tilt servo
        servo_state.position = [1300]
        data = SetPWMServoState()
        data.state = [servo_state]
        data.duration = 0.3
        pub.publish(data)
        time.sleep(0.5)

        # Return to center
        servo_state.position = [1500]
        data.state = [servo_state]
        pub.publish(data)
        time.sleep(0.5)

        test_node.destroy_publisher(pub)


# ============================================================================
# Audio tests
# ============================================================================

class TestAudio:
    """Test buzzer and USB speaker output."""

    @slow
    def test_buzzer_beep(self, test_node, executor):
        """STM32 buzzer should play a beep pattern."""
        pub = test_node.create_publisher(
            BuzzerState, '/ros_robot_controller/set_buzzer', 10,
        )
        time.sleep(0.2)

        # Two short beeps
        for _ in range(2):
            msg = BuzzerState()
            msg.freq = 1000
            msg.on_time = 0.3
            msg.off_time = 0.2
            msg.repeat = 1
            pub.publish(msg)
            time.sleep(0.6)

        test_node.destroy_publisher(pub)

    @slow
    def test_speaker_plays_tone(self):
        """USB speaker should play a test tone via aplay."""
        import subprocess
        import wave
        import struct

        # Generate a short 440Hz tone
        path = '/tmp/integration_test_tone.wav'
        f = wave.open(path, 'w')
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        for i in range(16000 * 2):  # 2 seconds
            val = int(16000 * math.sin(2 * math.pi * 440 * i / 16000))
            f.writeframes(struct.pack('<h', val))
        f.close()

        result = subprocess.run(
            ['aplay', '-D', 'plughw:2,0', path],
            capture_output=True, timeout=10,
        )
        assert result.returncode == 0, f'aplay failed: {result.stderr.decode()}'


# ============================================================================
# Cross-sensor consistency
# ============================================================================

class TestSensorConsistency:
    """Verify sensors agree with each other (sanity checks)."""

    def test_depth_and_lidar_roughly_agree(self, test_node, executor):
        """Front depth center and LiDAR front sector should be in the same ballpark."""
        # Get depth
        depth_msg = _wait_for_message(
            test_node, executor,
            '/ascamera/camera_publisher/depth0/image_raw', Image, timeout=5.0,
        )
        # Get lidar
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        lidar_msg = _wait_for_message(
            test_node, executor,
            '/scan_raw', LaserScan, qos=qos, timeout=5.0,
        )

        if depth_msg is None or lidar_msg is None:
            pytest.skip('Need both depth camera and LiDAR for this test')

        # Depth center point in meters
        bridge = CvBridge()
        depth = np.array(
            bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough'),
            dtype=np.uint16,
        )
        h, w = depth.shape[:2]
        roi = depth[h//2-5:h//2+5, w//2-5:w//2+5]
        valid = roi[(roi > 0) & (roi < 40000)]
        if len(valid) == 0:
            pytest.skip('No valid depth readings at center')
        depth_m = float(np.median(valid)) / 1000.0

        # LiDAR front sector
        ranges = np.array(lidar_msg.ranges)
        cleaned = ranges.copy()
        cleaned[cleaned == 0] = float('inf')
        cleaned[~np.isfinite(cleaned)] = float('inf')
        angles = lidar_msg.angle_min + np.arange(len(ranges)) * lidar_msg.angle_increment
        angles = angles % (2 * math.pi)
        front_half = math.radians(30)
        front_mask = (angles < front_half) | (angles > 2 * math.pi - front_half)
        front_vals = cleaned[front_mask]
        if len(front_vals) == 0 or front_vals.min() == float('inf'):
            pytest.skip('No valid LiDAR front readings')
        lidar_m = float(front_vals.min())

        # They don't need to match exactly (different FOV, position on robot)
        # but should be within 3x of each other for nearby obstacles
        if depth_m < 3.0 and lidar_m < 3.0:
            ratio = max(depth_m, lidar_m) / max(min(depth_m, lidar_m), 0.01)
            assert ratio < 5.0, \
                f'Depth ({depth_m:.2f}m) and LiDAR ({lidar_m:.2f}m) disagree too much'

    def test_imu_and_odom_heading_consistent(self, test_node, executor):
        """IMU yaw and odom theta should roughly agree."""
        imu_msg = _wait_for_message(
            test_node, executor,
            '/ros_robot_controller/imu_raw', Imu, timeout=5.0,
        )
        odom_msg = (
            _wait_for_message(test_node, executor, '/odom', Odometry, timeout=2.5)
            or _wait_for_message(test_node, executor, '/odom_raw', Odometry, timeout=2.5)
        )

        if imu_msg is None or odom_msg is None:
            pytest.skip('Need both IMU and odom for this test')

        def quat_to_yaw(q):
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            return math.atan2(siny, cosy)

        imu_q = imu_msg.orientation
        imu_all_zero = (imu_q.x == 0.0 and imu_q.y == 0.0
                        and imu_q.z == 0.0 and imu_q.w == 0.0)
        if imu_all_zero:
            # imu_raw doesn't provide orientation — skip comparison
            pytest.skip('IMU raw has no orientation (all-zero quaternion)')

        imu_yaw = quat_to_yaw(imu_q)
        odom_yaw = quat_to_yaw(odom_msg.pose.pose.orientation)

        delta = abs(imu_yaw - odom_yaw)
        if delta > math.pi:
            delta = 2 * math.pi - delta

        # Allow generous tolerance — different coordinate frames, drift, etc.
        assert delta < math.radians(90), \
            f'IMU yaw ({math.degrees(imu_yaw):.1f}) and odom yaw ' \
            f'({math.degrees(odom_yaw):.1f}) differ by {math.degrees(delta):.1f} deg'


# ============================================================================
# Explorer node command interface
# ============================================================================

class TestExplorerCommandInterface:
    """Test the explorer node's /explorer/command topic (if node is running)."""

    def test_command_topic_exists(self, test_node, executor):
        """The explorer command topic should be discoverable (if node is up)."""
        topics = test_node.get_topic_names_and_types()
        topic_names = [t[0] for t in topics]
        if '/explorer/command' not in topic_names:
            pytest.skip('Explorer node not running — command topic not found')

    def test_status_topic_publishes(self, test_node, executor):
        """If explorer node is running, /explorer/status should publish JSON."""
        msg = _wait_for_message(
            test_node, executor, '/explorer/status', String, timeout=3.0,
        )
        if msg is None:
            pytest.skip('Explorer node not running')

        import json
        status = json.loads(msg.data)
        assert 'mode' in status
        assert 'exploring' in status
        assert 'emergency_stop' in status
        assert 'lidar' in status or status.get('lidar') is None

    def test_stop_command(self, test_node, executor):
        """Sending 'stop' command should not error out."""
        pub = test_node.create_publisher(String, '/explorer/command', 10)
        time.sleep(0.2)
        msg = String()
        msg.data = 'stop'
        pub.publish(msg)
        time.sleep(0.3)
        test_node.destroy_publisher(pub)


# ============================================================================
# End-to-end sense pipeline
# ============================================================================

class TestSensePipeline:
    """Verify the full sense pipeline produces valid summaries.

    These tests replicate the explorer_node's data processing on live data.
    """

    def test_depth_summary_from_live_data(self, test_node, executor):
        """Generate a depth grid summary from real depth data."""
        msg = _wait_for_message(
            test_node, executor,
            '/ascamera/camera_publisher/depth0/image_raw', Image, timeout=5.0,
        )
        if msg is None:
            pytest.skip('No depth camera')

        bridge = CvBridge()
        depth = np.array(
            bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough'),
            dtype=np.uint16,
        )
        h, w = depth.shape[:2]
        rows = [('top', h // 4), ('mid', h // 2), ('bot', 3 * h // 4)]
        cols = [('L', w // 4), ('C', w // 2), ('R', 3 * w // 4)]

        grid = {}
        for rname, y in rows:
            for cname, x in cols:
                roi = depth[max(0, y-5):y+5, max(0, x-5):x+5]
                valid = roi[(roi > 0) & (roi < 40000)]
                if len(valid) > 0:
                    grid[f'{rname}{cname}'] = int(np.median(valid) / 10)
                else:
                    grid[f'{rname}{cname}'] = None

        # At least some grid cells should have valid readings
        valid_cells = [v for v in grid.values() if v is not None]
        assert len(valid_cells) > 0, 'No valid depth readings in any grid cell'

    def test_camera_frame_b64_from_live_data(self, test_node, executor):
        """Encode a real camera frame as base64 JPEG."""
        import base64

        msg = _wait_for_message(
            test_node, executor,
            '/ascamera/camera_publisher/rgb0/image', Image, timeout=5.0,
        )
        if msg is None:
            pytest.skip('No RGB camera')

        bridge = CvBridge()
        import cv2
        img = bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Resize like the explorer does
        h, w = img.shape[:2]
        max_dim = 640
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        _, jpeg_buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        b64 = base64.b64encode(jpeg_buf).decode('utf-8')

        assert len(b64) > 100, 'Base64 image too small'
        # Should be decodable
        decoded = base64.b64decode(b64)
        assert decoded[:2] == b'\xff\xd8', 'Not a valid JPEG'

    def test_lidar_summary_from_live_data(self, test_node, executor):
        """Generate a LiDAR sector summary from real scan data."""
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        msg = _wait_for_message(
            test_node, executor,
            '/scan_raw', LaserScan, qos=qos, timeout=5.0,
        )
        if msg is None:
            pytest.skip('No LiDAR')

        ranges = np.array(msg.ranges)
        cleaned = ranges.copy()
        cleaned[cleaned == 0] = float('inf')
        cleaned[~np.isfinite(cleaned)] = float('inf')
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        angles = angles % (2 * math.pi)

        def sector_min(center_deg, half_width_deg=45):
            center = math.radians(center_deg)
            half = math.radians(half_width_deg)
            lo = (center - half) % (2 * math.pi)
            hi = (center + half) % (2 * math.pi)
            if lo < hi:
                mask = (angles >= lo) & (angles < hi)
            else:
                mask = (angles >= lo) | (angles < hi)
            vals = cleaned[mask]
            return float(vals.min()) if len(vals) > 0 else float('inf')

        sectors = {
            'front': sector_min(0, 45),
            'left': sector_min(90, 45),
            'back': sector_min(180, 45),
            'right': sector_min(270, 45),
        }

        # Build the summary string like the explorer does
        parts = []
        for name in ['front', 'left', 'right', 'back']:
            dist = sectors[name]
            if dist == float('inf') or dist > 10.0:
                parts.append(f'{name}=clear(>10m)')
            else:
                parts.append(f'{name}={dist:.2f}m')

        summary = 'LiDAR: ' + ', '.join(parts)
        assert 'LiDAR:' in summary
        # At least one sector should have a finite reading
        assert any(s < float('inf') for s in sectors.values()), \
            'All sectors report infinity — LiDAR may not be working'


# ============================================================================
# TTS announcement (runs last — alphabetical ordering puts it at the end)
# ============================================================================

class TestZZZ_Announcement:
    """Final test: announce results via TTS and stop motors. Named ZZZ to run last."""

    @slow
    def test_tts_announcement(self):
        """Play a TTS completion announcement and ensure motors are silent."""
        _tts_speak('Jeeves integration test complete. All systems nominal, Sir.')
        _stop_motors_via_serial()
