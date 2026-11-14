import os

import launch
import launch.actions
import launch.events

import launch_ros
import launch_ros.actions
import launch_ros.events
import launch_ros.parameter_descriptions

from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch_ros.actions import Node

import lifecycle_msgs.msg

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    ld = launch.LaunchDescription()

    localization_param_dir = launch.substitutions.LaunchConfiguration(
        "localization_param_dir",
        default=os.path.join(
            get_package_share_directory("lidar_localization_ros2"),
            "param",
            "localization.yaml",
        ),
    )

    map_data_path = launch.substitutions.LaunchConfiguration(
        "map_data_path", default="/root/datasets/my_map_data"
    )

    lidar_localization = launch_ros.actions.LifecycleNode(
        name="lidar_localization",
        namespace="",
        package="lidar_localization_ros2",
        executable="lidar_localization_node",
        parameters=[
            localization_param_dir,
            {
                "map_path": launch_ros.parameter_descriptions.ParameterValue(
                    launch.substitutions.PathJoinSubstitution(
                        [map_data_path, "PCD", "map_final_visualization.pcd"]
                    ),
                    value_type=str,
                )
            },
        ],
        remappings=[("/cloud", "/livox/point")],
        output="log",
    )

    lifecycle_manager = launch_ros.actions.Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[
            {
                "autostart": True,
                "node_names": ["lidar_localization"],
                "bond_timeout": 0.0,
            }
        ],
    )

    ld.add_action(lidar_localization)
    ld.add_action(lifecycle_manager)

    return ld
