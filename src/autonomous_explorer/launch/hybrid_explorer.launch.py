"""
Launch file for hybrid explorer: SLAM + Nav2 + LLM Explorer.

Assumes hardware nodes (STM32, odom_publisher, LiDAR, camera) are
already running — launched by launch_explorer.sh before this file.

This launch file adds:
  1. Static TF transforms (base_footprint → lidar_frame, camera_link)
  2. RTAbMap SLAM (builds occupancy grid map)
  3. Nav2 navigation stack (path planning + obstacle avoidance)
  4. Explorer node in hybrid mode
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('autonomous_explorer')
    nav2_params = os.path.join(pkg_dir, 'config', 'nav2_explorer_params.yaml')

    # Launch arguments
    llm_provider_arg = DeclareLaunchArgument(
        'llm_provider', default_value=os.environ.get('LLM_PROVIDER', 'claude'),
    )
    voice_arg = DeclareLaunchArgument(
        'voice_enabled', default_value=os.environ.get('VOICE_ENABLED', 'true'),
    )
    loop_arg = DeclareLaunchArgument(
        'loop_interval', default_value='5.0',  # slower — Nav2 handles movement
    )

    # ── Static TF: base_footprint → lidar_frame ──
    # LiDAR is mounted on top, ~0.10m above base, centered
    lidar_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.10',
            '--roll', '0', '--pitch', '0', '--yaw', '3.14159',
            '--frame-id', 'base_footprint',
            '--child-frame-id', 'lidar_frame',
        ],
        output='screen',
    )

    # ── Static TF: base_footprint → camera_link ──
    # Camera is mounted front-center, ~0.08m above base, ~0.05m forward
    camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0.05', '--y', '0.0', '--z', '0.08',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'base_footprint',
            '--child-frame-id', 'camera_link',
        ],
        output='screen',
    )

    # ── RTAbMap SLAM ──
    # Builds occupancy grid from LiDAR + depth camera + odometry
    rtabmap_slam = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        output='screen',
        parameters=[{
            'frame_id': 'base_footprint',
            'use_sim_time': False,
            'subscribe_rgbd': True,
            'subscribe_scan': True,
            'use_action_for_goal': False,
            'qos_scan': 2,
            'qos_image': 2,
            'queue_size': 30,
            'Reg/Strategy': '1',
            'Reg/Force3DoF': 'true',
            'Grid/RangeMin': '0.2',
            'Grid/Sensor': 'true',
            'Optimizer/GravitySigma': '0',
            'RGBD/CreateOccupancyGrid': 'true',
            'Grid/FromDepth': 'true',
        }],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
            ('rgb/image', '/ascamera/camera_publisher/rgb0/image'),
            ('rgb/camera_info', '/ascamera/camera_publisher/rgb0/camera_info'),
            ('depth/image', '/ascamera/camera_publisher/depth0/image_raw'),
            ('odom', '/odom'),
            ('scan', '/scan_raw'),
        ],
        arguments=['-d'],  # delete previous database on start
    )

    # ── RTAbMap RGBD sync ──
    rgbd_sync = Node(
        package='rtabmap_sync',
        executable='rgbd_sync',
        output='screen',
        parameters=[{
            'approx_sync': True,
            'approx_sync_max_interval': 0.01,
            'use_sim_time': False,
            'qos': 2,
        }],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
            ('rgb/image', '/ascamera/camera_publisher/rgb0/image'),
            ('rgb/camera_info', '/ascamera/camera_publisher/rgb0/camera_info'),
            ('depth/image', '/ascamera/camera_publisher/depth0/image_raw'),
        ],
    )

    # ── Nav2 component container ──
    nav2_container = Node(
        package='rclcpp_components',
        executable='component_container_isolated',
        name='nav2_container',
        parameters=[nav2_params, {'autostart': True}],
        output='screen',
    )

    # ── Nav2 nodes (loaded into container after delay) ──
    # These are launched as regular nodes instead of composable for simplicity
    nav2_controller = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        parameters=[nav2_params],
        remappings=[('cmd_vel', '/controller/cmd_vel')],
        output='screen',
    )

    nav2_planner = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        parameters=[nav2_params],
        output='screen',
    )

    nav2_behaviors = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        parameters=[nav2_params],
        output='screen',
    )

    nav2_bt = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        parameters=[nav2_params],
        output='screen',
    )

    nav2_smoother = Node(
        package='nav2_smoother',
        executable='smoother_server',
        name='smoother_server',
        parameters=[nav2_params],
        output='screen',
    )

    nav2_velocity = Node(
        package='nav2_velocity_smoother',
        executable='velocity_smoother',
        name='velocity_smoother',
        parameters=[nav2_params],
        remappings=[
            ('cmd_vel', '/controller/cmd_vel'),
            ('cmd_vel_smoothed', '/controller/cmd_vel'),
        ],
        output='screen',
    )

    nav2_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
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

    # ── Explorer node (hybrid mode) ──
    explorer = Node(
        package='autonomous_explorer',
        executable='explorer_node',
        name='explorer_node',
        parameters=[{
            'llm_provider': LaunchConfiguration('llm_provider'),
            'voice_enabled': LaunchConfiguration('voice_enabled'),
            'loop_interval': LaunchConfiguration('loop_interval'),
            'use_nav2': True,
        }],
        output='screen',
    )

    # Delayed launches to allow TF and SLAM to initialize
    slam_delayed = TimerAction(
        period=3.0,
        actions=[rgbd_sync, rtabmap_slam],
    )

    nav2_delayed = TimerAction(
        period=8.0,
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

    explorer_delayed = TimerAction(
        period=15.0,
        actions=[explorer],
    )

    return LaunchDescription([
        llm_provider_arg,
        voice_arg,
        loop_arg,
        lidar_tf,
        camera_tf,
        slam_delayed,
        nav2_delayed,
        explorer_delayed,
    ])
