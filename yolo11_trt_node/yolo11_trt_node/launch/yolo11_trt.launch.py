from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolo11_trt_node',
            executable='yolo11_trt_node.py',
            name='yolo11_trt',
            output='screen',
            parameters=[{'engine_path': '/path/to/yolo11.engine'}]
        )
    ])
