"""
Launch file for the Jeeves Agent: full hardware + SLAM + Nav2 + Agent.

Single-command launch for the complete Jeeves hackathon demo stack.
Handles EVERYTHING automatically — device symlinks, environment variables,
API keys, ROS2 middleware config — so you only need ONE command:

    ros2 launch autonomous_explorer jeeves_agent.launch.py

What it does before launching nodes:
  0. Loads API keys from workspace .env file
  1. Auto-detects USB devices (STM32, LiDAR, WonderEcho) and creates symlinks
  2. Sets MACHINE_TYPE, LIDAR_TYPE, need_compile, RMW_IMPLEMENTATION, etc.
  3. Validates hardware is present and warns about missing devices

Then launches (staged):
  t=0s:  Hardware drivers (STM32, odom, LiDAR, camera) + TFs + twist_mux + foxglove
  t=3s:  IMU filter chain (calibration + complementary filter → /imu)
  t=5s:  EKF + slam_toolbox (sensor fusion + SLAM)
  t=12s: SLAM lifecycle manager
  t=15s: Nav2 navigation stack (DWB controller, NavFn planner)
  t=25s: Semantic map + Explorer agent + dashboard

Usage:
    # Full stack — just this, nothing else needed:
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
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
    GroupAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-launch setup: runs BEFORE any nodes start
# ═══════════════════════════════════════════════════════════════════════════════


def _load_env_file():
    """Load API keys and env vars from workspace .env file."""
    # Walk up from this file to find the workspace root .env
    # Works from both source (src/autonomous_explorer/launch/) and
    # installed (install/autonomous_explorer/share/.../launch/) locations.
    launch_file = Path(__file__).resolve()
    candidates = [Path.home() / '.env']  # always check ~/.env as fallback
    # Walk up looking for .env (stops at filesystem root)
    d = launch_file.parent
    for _ in range(10):
        env_path = d / '.env'
        if env_path.is_file():
            candidates.insert(0, env_path)
            break
        if d == d.parent:
            break
        d = d.parent
    for candidate in candidates:
        if candidate.is_file():
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        key, _, value = line.partition('=')
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value
            print(f'[SETUP] Loaded env from {candidate}')
            return
    print('[SETUP] No .env file found (API keys must be set in environment)')


def _setup_environment():
    """Set all required environment variables."""
    defaults = {
        'MACHINE_TYPE': 'MentorPi_Tank',
        'DEPTH_CAMERA_TYPE': 'aurora',
        'LIDAR_TYPE': 'LD19',
        'need_compile': 'True',
        'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp',
    }
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value

    # LD_LIBRARY_PATH for CycloneDDS on Debian Trixie
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    ros_lib = '/opt/ros/jazzy/lib/aarch64-linux-gnu'
    if ros_lib not in ld_path:
        os.environ['LD_LIBRARY_PATH'] = f'{ros_lib}:{ld_path}' if ld_path else ros_lib


# Load .env and set defaults at MODULE level so os.environ has API keys
# BEFORE generate_launch_description() builds SetEnvironmentVariable actions.
_load_env_file()
_setup_environment()


def _detect_and_link_device(pattern, symlink_path, label):
    """Check for device matching glob pattern, create symlink if needed.

    Returns the resolved device path or None.
    """
    import glob as glob_mod
    devices = sorted(glob_mod.glob(pattern))
    if not devices:
        print(f'[SETUP]   {label:20s}: NOT FOUND ({pattern})')
        # Remove stale symlink
        if os.path.islink(symlink_path):
            try:
                subprocess.run(
                    ['sudo', 'rm', '-f', symlink_path],
                    capture_output=True, timeout=5)
            except Exception:
                pass
        return None

    dev = devices[0]
    current_target = None
    try:
        current_target = os.path.realpath(symlink_path)
    except OSError:
        pass

    if current_target == os.path.realpath(dev):
        print(f'[SETUP]   {label:20s}: {dev} → {symlink_path} (already linked)')
    else:
        print(f'[SETUP]   {label:20s}: {dev} → {symlink_path} (creating symlink)')
        try:
            result = subprocess.run(
                ['sudo', 'ln', '-sf', dev, symlink_path],
                capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print(f'[SETUP]   WARNING: symlink failed: {result.stderr.strip()}')
                print(f'[SETUP]   Run manually: sudo ln -sf {dev} {symlink_path}')
        except subprocess.TimeoutExpired:
            print(f'[SETUP]   WARNING: sudo timed out (password prompt?)')
            print(f'[SETUP]   Run manually: sudo ln -sf {dev} {symlink_path}')
        except FileNotFoundError:
            print(f'[SETUP]   WARNING: sudo not found')
    return dev


def _detect_ch340_lidar_vs_wonderecho():
    """Distinguish LiDAR from WonderEcho among CH340 (ttyUSB*) devices."""
    import glob as glob_mod
    ttyusb = sorted(glob_mod.glob('/dev/ttyUSB*'))
    lidar_dev = None
    wonderecho_dev = None

    for dev in ttyusb:
        try:
            result = subprocess.run(
                ['udevadm', 'info', '-a', '-n', dev],
                capture_output=True, text=True, timeout=5)
            info = result.stdout.lower()
        except Exception:
            info = ''

        if any(kw in info for kw in ['wonderecho', 'wonder_echo', 'ai voice', 'voice box']):
            wonderecho_dev = dev
        else:
            # Default non-WonderEcho CH340 to LiDAR
            if lidar_dev is None:
                lidar_dev = dev

    # If exactly 2 CH340 devices and WonderEcho wasn't identified by udevadm,
    # assume the second one is WonderEcho (both appear as identical CH340s).
    if wonderecho_dev is None and len(ttyusb) == 2 and lidar_dev is not None:
        other = [d for d in ttyusb if d != lidar_dev]
        if other:
            wonderecho_dev = other[0]
            print(f'[SETUP]   WonderEcho: guessed {wonderecho_dev} '
                  f'(2 CH340 devices, {lidar_dev} assigned to LiDAR)')

    return lidar_dev, wonderecho_dev


def _setup_hardware():
    """Auto-detect USB devices and create /dev symlinks.

    Returns list of fatal errors (empty = all critical hardware found).
    """
    OK = '\033[92m OK \033[0m'       # green
    WARN = '\033[93mMISS\033[0m'     # yellow
    FAIL = '\033[91mFAIL\033[0m'     # red
    LINK = '\033[96m -> \033[0m'     # cyan arrow
    DIM = '\033[2m'
    RESET = '\033[0m'

    print()
    print('\033[1m  Hardware Detection\033[0m')
    print('  ' + '-' * 50)

    fatal_errors = []
    devices = []  # collect (name, status, detail) for table

    # STM32 controller — only ACM device on this robot
    stm32 = _detect_and_link_device('/dev/ttyACM*', '/dev/rrc', 'STM32 controller')
    if stm32:
        devices.append(('STM32 Controller', OK, f'{stm32}{LINK}/dev/rrc'))
    else:
        devices.append(('STM32 Controller', FAIL, 'not found (/dev/ttyACM*)'))
        fatal_errors.append('STM32 controller not found. Check USB to controller board.')

    # CH340 devices (LiDAR and WonderEcho both use CH340)
    lidar_dev, wonderecho_dev = _detect_ch340_lidar_vs_wonderecho()

    if lidar_dev:
        current = os.path.realpath('/dev/ldlidar') if os.path.exists('/dev/ldlidar') else None
        if current != os.path.realpath(lidar_dev):
            try:
                subprocess.run(
                    ['sudo', 'ln', '-sf', lidar_dev, '/dev/ldlidar'],
                    capture_output=True, timeout=10)
            except Exception:
                pass
        devices.append(('LiDAR (LD19)', OK, f'{lidar_dev}{LINK}/dev/ldlidar'))
    else:
        devices.append(('LiDAR (LD19)', FAIL, 'not found (no /dev/ttyUSB*)'))
        fatal_errors.append('LiDAR not found. Check USB connection.')

    if wonderecho_dev:
        current = os.path.realpath('/dev/wonderecho') if os.path.exists('/dev/wonderecho') else None
        if current != os.path.realpath(wonderecho_dev):
            try:
                subprocess.run(
                    ['sudo', 'ln', '-sf', wonderecho_dev, '/dev/wonderecho'],
                    capture_output=True, timeout=10)
            except Exception:
                pass
        devices.append(('WonderEcho', OK, f'{wonderecho_dev}{LINK}/dev/wonderecho'))
    else:
        devices.append(('WonderEcho', WARN, f'{DIM}not found (voice disabled){RESET}'))

    # Camera — detected by USB vendor ID, no symlink needed
    try:
        lsusb = subprocess.run(
            ['lsusb'], capture_output=True, text=True, timeout=5)
        if '3251:1930' in lsusb.stdout:
            devices.append(('Depth Camera', OK, 'Aurora 930 (USB 3251:1930)'))
        else:
            devices.append(('Depth Camera', WARN, f'{DIM}not found{RESET}'))
    except Exception:
        devices.append(('Depth Camera', WARN, f'{DIM}lsusb failed{RESET}'))

    # Print table
    for name, status, detail in devices:
        print(f'  [{status}]  {name:20s} {detail}')
    print('  ' + '-' * 50)

    return fatal_errors


def _check_openai_connectivity():
    """Ping OpenAI API to verify key and network connectivity."""
    import urllib.request
    import urllib.error
    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        print('[SETUP] OpenAI API: SKIPPED (no key)')
        return
    try:
        req = urllib.request.Request(
            'https://api.openai.com/v1/models',
            headers={'Authorization': f'Bearer {api_key}'},
        )
        resp = urllib.request.urlopen(req, timeout=5)
        print(f'[SETUP] OpenAI API: OK (HTTP {resp.status})')
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print(f'[SETUP] OpenAI API: FAILED — invalid API key (HTTP 401)')
        else:
            print(f'[SETUP] OpenAI API: FAILED — HTTP {e.code}')
    except Exception as e:
        print(f'[SETUP] OpenAI API: FAILED — {e}')


def _pre_launch_setup(context, *args, **kwargs):
    """OpaqueFunction that runs all setup before nodes launch."""
    print()
    print('\033[1m' + '=' * 52 + '\033[0m')
    print('\033[1m  Jeeves Agent — Pre-launch Setup\033[0m')
    print('\033[1m' + '=' * 52 + '\033[0m')

    # env + .env already loaded at module level, just do hardware + validation
    hardware = LaunchConfiguration('hardware').perform(context)
    fatal_errors = _setup_hardware()
    if fatal_errors and hardware == 'true':
        print()
        print('\033[91;1m  LAUNCH ABORTED — Required hardware missing:\033[0m')
        for err in fatal_errors:
            print(f'\033[91m    * {err}\033[0m')
        print()
        print('  Fix connections and relaunch, or skip with:')
        print('    \033[2mros2 launch autonomous_explorer jeeves_agent.launch.py hardware:=false\033[0m')
        print()
        import sys
        sys.exit(1)

    # Validate API keys
    provider = LaunchConfiguration('llm_provider').perform(context)
    if provider == 'dryrun':
        print('[SETUP] DRY-RUN mode — no API keys needed')
    elif provider == 'openai' and not os.environ.get('OPENAI_API_KEY'):
        print('[SETUP] FATAL: OPENAI_API_KEY not set! LLM calls WILL fail.')
        print('[SETUP]   Add to .env: OPENAI_API_KEY=sk-...')
        print('[SETUP]   Or: export OPENAI_API_KEY="sk-..."')
    elif provider == 'claude' and not os.environ.get('ANTHROPIC_API_KEY'):
        print('[SETUP] FATAL: ANTHROPIC_API_KEY not set! LLM calls WILL fail.')
        print('[SETUP]   Add to .env: ANTHROPIC_API_KEY=sk-ant-...')

    if os.environ.get('VOICE_ENABLED', 'true') == 'true' and not os.environ.get('OPENAI_API_KEY'):
        print('[SETUP] WARNING: OPENAI_API_KEY not set — voice TTS/STT will use fallbacks (gTTS/Google)')

    # Ping OpenAI to verify connectivity + key
    if provider in ('openai',) or os.environ.get('OPENAI_API_KEY'):
        _check_openai_connectivity()

    oai_key = os.environ.get('OPENAI_API_KEY', '')
    ant_key = os.environ.get('ANTHROPIC_API_KEY', '')
    print('=' * 60)
    print(f'  Provider:   {provider}')
    print(f'  Machine:    {os.environ.get("MACHINE_TYPE")}')
    print(f'  RMW:        {os.environ.get("RMW_IMPLEMENTATION")}')
    print(f'  OpenAI key: {"SET (" + oai_key[:8] + "...)" if oai_key else "NOT SET"}')
    print(f'  Claude key: {"SET (" + ant_key[:8] + "...)" if ant_key else "NOT SET"}')
    print(f'  STM32:      {"OK" if os.path.exists("/dev/rrc") else "MISSING"}')
    print(f'  LiDAR:      {"OK" if os.path.exists("/dev/ldlidar") else "MISSING"}')
    print('=' * 60)

    return []  # OpaqueFunction must return a list of actions


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
            default_value=os.environ.get('LLM_PROVIDER', 'openai'),
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
            default_value='false',
            description='Launch curses dashboard in a detached screen session (attach: screen -r jeeves_dash)',
        ),
        DeclareLaunchArgument(
            'model_profile',
            default_value='',
            description='Model config profile: cloud, local, hybrid, budget, dryrun (empty=use defaults)',
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
        remappings=[('odometry/filtered', 'odom')],
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
    # Log directory: ~/mentorpi_explorer/logs/ (same as JSONL data logs)
    explorer_log_dir = os.path.expanduser(
        os.environ.get('EXPLORER_LOG_DIR', '~/mentorpi_explorer/logs'))
    os.makedirs(explorer_log_dir, exist_ok=True)
    explorer_log_file = os.path.join(
        explorer_log_dir,
        f'explorer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

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
                'model_profile': LaunchConfiguration('model_profile'),
                'use_nav2': True,
                'agent_mode': True,
            },
        ],
        output='both',
        additional_env={'EXPLORER_LOG_FILE': explorer_log_file},
    )
    # Log file path printed for easy access
    print(f'[SETUP] Explorer log: {explorer_log_file}')

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

    # ── Pre-launch setup (env vars, symlinks, hardware detection) ──
    pre_launch = OpaqueFunction(function=_pre_launch_setup)

    # ── Environment Variables ──
    # These MUST use SetEnvironmentVariable so they propagate to all child
    # processes (both C++ and Python nodes) spawned by ros2 launch.
    env_actions = [
        SetEnvironmentVariable('MACHINE_TYPE',
                               os.environ.get('MACHINE_TYPE', 'MentorPi_Tank')),
        SetEnvironmentVariable('DEPTH_CAMERA_TYPE',
                               os.environ.get('DEPTH_CAMERA_TYPE', 'aurora')),
        SetEnvironmentVariable('LIDAR_TYPE',
                               os.environ.get('LIDAR_TYPE', 'LD19')),
        SetEnvironmentVariable('need_compile',
                               os.environ.get('need_compile', 'True')),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp'),
    ]

    # CycloneDDS XML config (ManySocketsMode=false for 20+ nodes)
    cyclonedds_xml = os.path.expanduser('~/.cyclonedds.xml')
    if os.path.exists(cyclonedds_xml):
        env_actions.append(
            SetEnvironmentVariable(
                'CYCLONEDDS_URI', f'file://{cyclonedds_xml}'),
        )

    # Forward API keys to child processes if loaded from .env
    for key in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY'):
        val = os.environ.get(key, '')
        if val:
            env_actions.append(SetEnvironmentVariable(key, val))

    return LaunchDescription(
        args + [pre_launch] + env_actions + [
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
