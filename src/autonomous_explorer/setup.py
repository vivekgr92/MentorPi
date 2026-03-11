import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'autonomous_explorer'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.*'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.*'))),
        (os.path.join('share', package_name, 'scripts'),
            glob(os.path.join('scripts', '*.py')) +
            glob(os.path.join('scripts', '*.sh'))),
        (os.path.join('share', package_name, 'models'),
            glob(os.path.join('models', '*.onnx'))),
    ],
    install_requires=[
        'setuptools',
        'anthropic',
        'openai',
    ],
    zip_safe=True,
    maintainer='vivek',
    maintainer_email='vivek@example.com',
    description='Autonomous exploration with LLM vision brain',
    license='MIT',
    tests_require=['pytest'],
    scripts=[
        'scripts/analytics_dashboard.py',
        'scripts/dashboard.py',
        'scripts/dataset_export.py',
        'scripts/jeeves_knowledge.py',
    ],
    entry_points={
        'console_scripts': [
            'explorer_node = autonomous_explorer.explorer_node:main',
            'semantic_map_publisher = autonomous_explorer.semantic_map_publisher:main',
            'dashboard = autonomous_explorer.dashboard:main',
        ],
    },
)
