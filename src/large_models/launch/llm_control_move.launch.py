import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, EnvironmentVariable
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction, ExecuteProcess

def launch_setup(context):
    mode = LaunchConfiguration('mode', default=1)
    mode_arg = DeclareLaunchArgument('mode', default_value=mode)


    machine_type = EnvironmentVariable('MACHINE_TYPE', default_value='default_machine')

    log_machine_type = ExecuteProcess(
        cmd=['echo', 'MACHINE_TYPE is set to: ', machine_type],
        shell=True
    )

    controller_package_path = get_package_share_directory('controller')
    large_models_package_path = get_package_share_directory('large_models')

    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(controller_package_path, 'launch/controller.launch.py')),
    )

    large_models_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(large_models_package_path, 'launch/start.launch.py')),
        launch_arguments={'mode': mode}.items(),
    )


    llm_control_move_node = Node(
        package='large_models',
        executable='llm_control_move',
        output='screen',
        parameters=[{'machine_type': machine_type}]  # Pass machine_type as a parameter(将 machine_type 作为参数传递)
    )

    return [mode_arg,
            log_machine_type,  # Add ExecuteProcess action(添加 ExecuteProcess action)
            controller_launch,
            large_models_launch,
            llm_control_move_node,
            ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function = launch_setup)
    ])


# if __name__ == '__main__':
#     # Create a LaunchDescription object(创建一个LaunchDescription对象)
#     ld = generate_launch_description()
#
#     ls = LaunchService()
#     ls.include_launch_description(ld)
#     ls.run()
