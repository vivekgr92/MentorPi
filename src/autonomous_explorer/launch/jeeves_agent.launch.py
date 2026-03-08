"""
Launch file for the Jeeves Agent: full hardware + SLAM + Nav2 + Agent.

Single-command launch for the complete Jeeves hackathon demo stack:
  1. Hardware drivers (STM32 controller, odom publisher, LiDAR, depth camera)
  2. IMU filter chain (calibration + complementary filter → /imu)
  3. Static TF transforms (base_footprint → lidar_frame, camera_link)
  4. robot_localization EKF (odom_raw + imu → /odom + TF)
  5. twist_mux (priority-based cmd_vel multiplexer)
  6. slam_toolbox async mode (builds occupancy grid from LiDAR)
  7. Nav2 navigation stack (DWB controller, NavFn planner)
  8. foxglove_bridge (WebSocket visualization on port 8765)
  9. rosbridge_websocket (JSON WebSocket bridge on port 9090 for web dashboards)
  10. Explorer node in agent mode (ROSA-style tool-calling)

Usage:
    # Full stack (all hardware + navigation + agent):
    ros2 launch autonomous_explorer jeeves_agent.launch.py

    # With OpenAI provider:
    ros2 launch autonomous_explorer jeeves_agent.launch.py llm_provider:=openai

    # Without hardware (launch drivers separately):
    ros2 launch autonomous_explorer jeeves_agent.launch.py hardware:=false

    # Without foxglove:
    ros2 launch autonomous_explorer jeeves_agent.launch.py foxglove:=false

    # Dry-run (no API keys needed):
    ros2 launch autonomous_explorer jeeves_agent.launch.py llm_provider:=dryrun

    # With ascamera instead of aurora:
    ros2 launch autonomous_explorer jeeves_agent.launch.py camera_type:=ascamera
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
    GroupAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('autonomous_explorer')
    peripherals_dir = get_package_share_directory('peripherals')

    # Config file paths
    explorer_params = os.path.join(pkg_dir, 'config', 'explorer_params.yaml')
    nav2_params = os.path.join(pkg_dir, 'config', 'nav2_explorer_params.yaml')
    slam_params = os.path.join(pkg_dir, 'config', 'slam_toolbox_params.yaml')
    ekf_params = os.path.join(pkg_dir, 'config', 'ekf.yaml')
    twist_mux_params = os.path.join(pkg_dir, 'config', 'twist_mux.yaml')

    # IMU calibration file from calibration package
    try:
        calib_dir = get_package_share_directory('calibration')
        imu_calib_file = os.path.join(calib_dir, 'config', 'imu_calib.yaml')
    except Exception:
        imu_calib_file = ''

    # ── Launch Arguments ──

    args = [
        DeclareLaunchArgument(
            'hardware',
            default_value='true',
            description='Launch hardware drivers (STM32, odom, LiDAR, camera)',
        ),
        DeclareLaunchArgument(
            'camera_type',
            default_value=os.environ.get('DEPTH_CAMERA_TYPE', 'aurora'),
            description='Depth camera type: aurora or ascamera',
        ),
        DeclareLaunchArgument(
            'llm_provider',
            default_value=os.environ.get('LLM_PROVIDER', 'claude'),
            description='LLM provider: claude, openai, or dryrun',
        ),
        DeclareLaunchArgument(
            'voice_enabled',
            default_value=os.environ.get('VOICE_ENABLED', 'true'),
            description='Enable voice I/O via WonderEcho Pro',
        ),
        DeclareLaunchArgument(
            'loop_interval',
            default_value='5.0',
            description='Seconds between agent turns (slower — Nav2 handles movement)',
        ),
        DeclareLaunchArgument(
            'foxglove',
            default_value='true',
            description='Launch foxglove_bridge for WebSocket visualization',
        ),
        DeclareLaunchArgument(
            'foxglove_port',
            default_value='8765',
            description='Foxglove bridge WebSocket port',
        ),
        DeclareLaunchArgument(
            'use_ekf',
            default_value='true',
            description='Launch robot_localization EKF for sensor fusion',
        ),
        DeclareLaunchArgument(
            'dashboard',
            default_value='true',
            description='Launch curses dashboard in a detached screen session (attach: screen -r jeeves_dash)',
        ),
    ]

    hw_condition = IfCondition(LaunchConfiguration('hardware'))

    # ── Hardware: STM32 Controller ──
    # Manages motors, servos, IMU, LEDs, buzzer, battery via /dev/rrc serial
    stm32_node = Node(
        package='ros_robot_controller',
        executable='ros_robot_controller',
        name='ros_robot_controller',
        parameters=[{'imu_frame': 'imu_link'}],
        output='screen',
        condition=hw_condition,
    )

    # ── Hardware: Odom Publisher ──
    # Converts Twist on /controller/cmd_vel → MotorsState, publishes odom_raw
    # When EKF is enabled: odom_publisher uses frame 'odom_raw' to avoid TF
    # conflict (EKF publishes odom→base_footprint). When EKF is disabled:
    # odom_publisher uses 'odom' directly since it's the sole TF provider.
    odom_node = Node(
        package='controller',
        executable='odom_publisher',
        name='odom_publisher',
        parameters=[{
            'base_frame_id': 'base_footprint',
            'odom_frame_id': PythonExpression([
                "'odom_raw' if '",
                LaunchConfiguration('use_ekf'),
                "' == 'true' else 'odom'",
            ]),
            'pub_odom_topic': True,
            'linear_correction_factor': 1.0,
            'angular_correction_factor': 1.0,
        }],
        output='screen',
        condition=hw_condition,
    )

    # ── Hardware: LD19 LiDAR ──
    # Publishes /scan_raw (LaserScan) from /dev/ldlidar at 230400 baud
    lidar_node = Node(
        package='ldlidar_stl_ros2',
        executable='ldlidar_stl_ros2_node',
        name='LD19',
        parameters=[{
            'topic_name': 'scan',
            'product_name': 'LDLiDAR_LD19',
            'port_baudrate': 230400,
            'port_name': '/dev/ldlidar',
            'frame_id': 'lidar_frame',
            'laser_scan_dir': True,
            'enable_angle_crop_func': False,
            'angle_crop_min': 135.0,
            'angle_crop_max': 225.0,
        }],
        remappings=[('scan', 'scan_raw')],
        output='screen',
        condition=hw_condition,
    )

    # ── Hardware: Depth Camera ──
    # Aurora 930 or AScamera — both remap to /ascamera/camera_publisher/* topics
    aurora_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_dir, 'launch', 'include', 'aurora930.launch.py')
        ),
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('hardware'), "' == 'true' and '",
            LaunchConfiguration('camera_type'), "' == 'aurora'",
        ])),
    )

    ascamera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_dir, 'launch', 'include', 'ascamera.launch.py')
        ),
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('hardware'), "' == 'true' and '",
            LaunchConfiguration('camera_type'), "' == 'ascamera'",
        ])),
    )

    # ── IMU Filter Chain ──
    # If imu_calib is installed: imu_raw → imu_calib → imu_corrected → complementary_filter → imu
    # If not: imu_raw → complementary_filter → imu (skip calibration)
    imu_calib_node = None
    imu_input_topic = '/ros_robot_controller/imu_raw'  # default: raw IMU directly
    try:
        get_package_share_directory('imu_calib')
        imu_calib_node = Node(
            package='imu_calib',
            executable='apply_calib',
            name='imu_calib',
            parameters=[{'calib_file': imu_calib_file}],
            remappings=[
                ('raw', '/ros_robot_controller/imu_raw'),
                ('corrected', 'imu_corrected'),
            ],
            output='log',
            condition=hw_condition,
        )
        imu_input_topic = 'imu_corrected'  # calibrated IMU
    except Exception:
        pass  # imu_calib not installed — feed raw IMU to filter

    imu_filter_node = Node(
        package='imu_complementary_filter',
        executable='complementary_filter_node',
        name='imu_filter',
        parameters=[{
            'use_mag': False,
            'do_bias_estimation': True,
            'do_adaptive_gain': True,
            'publish_debug_topics': False,
        }],
        remappings=[
            ('/imu/data_raw', imu_input_topic),
            ('imu/data', 'imu'),
        ],
        output='log',
        condition=hw_condition,
    )

    # ── Static TF Transforms ──
    # Spatial relationship between robot base and sensors (from CAD measurements)

    # LiDAR: mounted on top, ~10cm above base, centered, facing backward (π yaw)
    lidar_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lidar_tf',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.10',
            '--roll', '0', '--pitch', '0', '--yaw', '3.14159',
            '--frame-id', 'base_footprint',
            '--child-frame-id', 'lidar_frame',
        ],
        output='log',
    )

    # Camera: mounted front-center, ~8cm above base, ~5cm forward
    camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf',
        arguments=[
            '--x', '0.05', '--y', '0.0', '--z', '0.08',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'base_footprint',
            '--child-frame-id', 'camera_link',
        ],
        output='log',
    )

    # ── Robot Localization EKF ──
    # Fuses wheel odometry (odom_raw) + IMU (imu) → /odom + odom→base_footprint TF
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        parameters=[ekf_params],
        output='log',
        condition=IfCondition(LaunchConfiguration('use_ekf')),
    )

    # ── twist_mux ──
    # Priority: safety(0) > joystick(1) > nav2(2) > autonomous(3)
    # Output: /controller/cmd_vel
    twist_mux_node = Node(
        package='twist_mux',
        executable='twist_mux',
        name='twist_mux',
        parameters=[twist_mux_params],
        remappings=[('cmd_vel_out', '/controller/cmd_vel')],
        output='log',
    )

    # ── slam_toolbox (async online) ──
    # Builds occupancy grid map from LiDAR scans.
    # Publishes: /map (OccupancyGrid), map→odom TF
    # Note: slam_toolbox is a lifecycle node — needs a lifecycle manager to activate it.
    slam_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        parameters=[slam_params],
        output='screen',
    )

    # Lifecycle manager to auto-activate slam_toolbox
    # bond_timeout increased from default 4s — slam_toolbox takes 5-8s to init on Pi 5
    slam_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_slam',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'bond_timeout': 15.0,
            'node_names': ['slam_toolbox'],
        }],
        output='screen',
    )

    # ── Nav2 Navigation Stack ──
    # DWB local planner + NavFn global planner
    # Controller publishes to /cmd_vel_nav2_raw → velocity_smoother → /cmd_vel/nav2
    nav2_cmd_vel_out = '/cmd_vel/nav2'

    nav2_controller = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        parameters=[nav2_params],
        remappings=[('cmd_vel', '/cmd_vel_nav2_raw')],
        output='log',
    )

    nav2_planner = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        parameters=[nav2_params],
        output='log',
    )

    nav2_behaviors = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        parameters=[nav2_params],
        output='log',
    )

    nav2_bt = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        parameters=[nav2_params],
        output='log',
    )

    nav2_smoother = Node(
        package='nav2_smoother',
        executable='smoother_server',
        name='smoother_server',
        parameters=[nav2_params],
        output='log',
    )

    # Smooths controller output before sending to twist_mux
    nav2_velocity = Node(
        package='nav2_velocity_smoother',
        executable='velocity_smoother',
        name='velocity_smoother',
        parameters=[nav2_params],
        remappings=[
            ('cmd_vel', '/cmd_vel_nav2_raw'),
            ('cmd_vel_smoothed', nav2_cmd_vel_out),
        ],
        output='log',
    )

    # Lifecycle manager brings up all Nav2 nodes in order
    nav2_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'bond_timeout': 15.0,
            'node_names': [
                'controller_server',
                'smoother_server',
                'planner_server',
                'behavior_server',
                'bt_navigator',
                'velocity_smoother',
            ],
        }],
        output='screen',
    )

    # ── Foxglove Bridge ──
    # WebSocket bridge for browser-based visualization (Foxglove Studio)
    foxglove_node = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[{
            'port': LaunchConfiguration('foxglove_port'),
            'send_buffer_limit': 10000000,
        }],
        output='log',
        condition=IfCondition(LaunchConfiguration('foxglove')),
    )

    # ── rosbridge WebSocket (optional) ──
    # JSON-based WebSocket bridge for web dashboards and roslibjs clients
    # Skipped if rosbridge_server is not installed
    rosbridge_node = None
    try:
        get_package_share_directory('rosbridge_server')
        rosbridge_node = Node(
            package='rosbridge_server',
            executable='rosbridge_websocket_node',
            name='rosbridge_websocket',
            parameters=[{
                'port': 9090,
                'unregister_timeout': 10.0,
            }],
            output='log',
            condition=IfCondition(LaunchConfiguration('foxglove')),
        )
    except Exception:
        pass  # rosbridge_server not installed — skip

    # ── Semantic Map Publisher ──
    # Reads WorldKnowledge JSON, publishes MarkerArray for Foxglove visualization
    semantic_map_node = Node(
        package='autonomous_explorer',
        executable='semantic_map_publisher',
        name='semantic_map_publisher',
        output='log',
    )

    # ── Dashboard (curses TUI in detached screen session) ──
    # Launches in a GNU screen session so you can attach from any terminal:
    #   screen -r jeeves_dash
    # Detach with Ctrl-A D. Kill with: screen -S jeeves_dash -X quit
    dashboard_proc = ExecuteProcess(
        cmd=[
            'screen', '-dmS', 'jeeves_dash',
            'bash', '-c',
            'source /opt/ros/jazzy/setup.bash && '
            'source ' + os.path.join(os.environ.get('COLCON_PREFIX_PATH', pkg_dir + '/../../..'), 'setup.bash') + ' 2>/dev/null; '
            'ros2 run autonomous_explorer dashboard; '
            'exec bash',
        ],
        output='log',
        condition=IfCondition(LaunchConfiguration('dashboard')),
    )

    # ── Explorer Node (Agent Mode) ──
    explorer_node = Node(
        package='autonomous_explorer',
        executable='explorer_node',
        name='autonomous_explorer',
        parameters=[
            explorer_params,
            {
                'llm_provider': LaunchConfiguration('llm_provider'),
                'voice_enabled': LaunchConfiguration('voice_enabled'),
                'loop_interval': LaunchConfiguration('loop_interval'),
                'use_nav2': True,
                'agent_mode': True,
            },
        ],
        output='screen',
    )

    # ── Staged Launch Sequence ──
    # Stagger startup to allow dependencies to initialize:
    #   t=0s:  Hardware (STM32, odom, LiDAR, camera) + static TFs + twist_mux + foxglove
    #   t=3s:  IMU filter chain (needs imu_raw from STM32)
    #   t=5s:  EKF + slam_toolbox (need odom_raw + imu + /scan_raw)
    #   t=10s: Nav2 stack (needs /map from slam_toolbox)
    #   t=18s: Explorer agent (needs Nav2 action server)

    imu_delayed = TimerAction(
        period=3.0,
        actions=([imu_calib_node] if imu_calib_node else []) + [imu_filter_node],
    )

    # t=5s: EKF + slam_toolbox (no lifecycle manager yet — let slam init first)
    ekf_slam_delayed = TimerAction(
        period=5.0,
        actions=[ekf_node, slam_node],
    )

    # t=12s: slam lifecycle manager (slam_toolbox needs 5-8s to init on Pi 5)
    slam_lifecycle_delayed = TimerAction(
        period=12.0,
        actions=[slam_lifecycle],
    )

    # t=15s: Nav2 nodes + lifecycle manager
    nav2_delayed = TimerAction(
        period=15.0,
        actions=[
            nav2_controller,
            nav2_planner,
            nav2_behaviors,
            nav2_bt,
            nav2_smoother,
            nav2_velocity,
            nav2_lifecycle,
        ],
    )

    # t=25s: Explorer agent (needs Nav2 lifecycle to finish configuring)
    explorer_delayed = TimerAction(
        period=25.0,
        actions=[semantic_map_node, explorer_node, dashboard_proc],
    )

    # ── CycloneDDS Configuration ──
    # ManySocketsMode=false prevents participant index exhaustion when launching
    # 20+ nodes. This env var MUST be set via SetEnvironmentVariable so it
    # propagates to all child processes (C++ and Python) spawned by ros2 launch.
    cyclonedds_xml = os.path.expanduser('~/.cyclonedds.xml')
    cyclonedds_env = []
    if os.path.exists(cyclonedds_xml):
        cyclonedds_env = [
            SetEnvironmentVariable(
                'CYCLONEDDS_URI', f'file://{cyclonedds_xml}'),
        ]

    return LaunchDescription(
        args + cyclonedds_env + [
            # Immediate: hardware drivers + TF + support nodes
            stm32_node,
            odom_node,
            lidar_node,
            aurora_camera,
            ascamera,
            lidar_tf,
            camera_tf,
            twist_mux_node,
            foxglove_node,
            *([rosbridge_node] if rosbridge_node else []),
            # Delayed: IMU → EKF+SLAM → SLAM lifecycle → Nav2 → Explorer
            imu_delayed,
            ekf_slam_delayed,
            slam_lifecycle_delayed,
            nav2_delayed,
            explorer_delayed,
        ]
    )
