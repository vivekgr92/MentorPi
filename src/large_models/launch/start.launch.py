import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # Check the environment variable 'need_compile' to determine the path of the peripherals package(检查环境变量 need_compile 决定 peripherals 包路径)
    compiled = os.environ.get('need_compile', 'False')
    if compiled == 'True':
        peripherals_package_path = get_package_share_directory('peripherals')
    else:
        peripherals_package_path = '/home/ubuntu/ros2_ws/src/peripherals'

    # Include the child launch file from the large_models package(引入 large_models 包中的子 Launch 文件)
    vocal_detect_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('large_models'), 'launch/vocal_detect.launch.py'))
    )

    agent_process_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('large_models'), 'launch/agent_process.launch.py'))
    )

    tts_node_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('large_models'), 'launch/tts_node.launch.py'))
    )

    

    # Enable the USB camera node(启动 USB 摄像头节点)
    camera_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        output='screen',
        parameters=[os.path.join(peripherals_package_path, 'config', 'usb_cam_param.yaml'),
                    ]
    )

    # Return all nodes and child Launch(返回所有节点和子 Launch)
    return LaunchDescription([
        vocal_detect_launch,
        agent_process_launch,
        tts_node_launch,
        #camera_node,
        ])
