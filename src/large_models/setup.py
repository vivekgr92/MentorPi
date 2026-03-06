import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'large_models'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='1270161395@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vocal_detect = large_models.vocal_detect:main',
            'llm_visual_patrol = large_models.llm_visual_patrol:main',
            'llm_color_track = large_models.llm_color_track:main',
            'llm_control_move = large_models.llm_control_move:main',
            'agent_process = large_models.agent_process:main',
            'tts_node = large_models.tts_node:main',
            'function_call = large_models.function_call:main',
            'color_sorting = large_models.color_sorting_node:main',
            'waste_classification = large_models.waste_classification:main',
            'llm_control_servo = large_models.llm_control_servo:main',
            'llm_color_sorting = large_models.llm_color_sorting:main',
            'llm_waste_classification = large_models.llm_waste_classification:main',
            'vllm_with_camera = large_models.vllm_with_camera:main',
            'vllm_track = large_models.vllm_track:main',
            'navigation_controller = large_models.navigation_controller:main',
            'vllm_navigation = large_models.vllm_navigation:main',
            'vllm_navigation_transport = large_models.navigation_transport.vllm_navigation_transport:main',
        ],
    },
)
