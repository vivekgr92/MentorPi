import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction
from launch.actions import IncludeLaunchDescription, OpaqueFunction

def generate_launch_description():
    # Check the environment variable 'need_compile' to determine the peripherals package path(检查环境变量 need_compile 决定 peripherals 包路径)
    compiled = os.environ.get('need_compile', 'False')
    if compiled == 'True':

        controller_package_path = get_package_share_directory('controller')
        peripherals_package_path = get_package_share_directory('peripherals')
    else:
        controller_package_path = '/home/ubuntu/ros2_ws/src/driver/controller'
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
    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(controller_package_path, 'launch/controller.launch.py')),
    )
    
    sonar_controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_package_path, 'launch/sonar_controller_node.launch.py')),
    )

 
   # Launch camera-related nodes(启动摄像头相关节点)
    usb_cam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_package_path, 'launch/usb_cam.launch.py')),
    )

    # Return all nodes and child launches(返回所有节点和子 Launch)
    return LaunchDescription([
        vocal_detect_launch,
        agent_process_launch,
        tts_node_launch,
        controller_launch,
        sonar_controller_launch,
        usb_cam_launch,
        #camera_node,
        ])
