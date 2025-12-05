from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory, get_package_prefix
import os
import sys

launch_args = [
    DeclareLaunchArgument(
        name="namespace", default_value="ov_srvins", description="namespace"
    ),
    DeclareLaunchArgument(
        name="ov_enable", default_value="true", description="enable SRVINS node"
    ),
    DeclareLaunchArgument(
        name="rviz_enable", default_value="false", description="enable rviz node"
    ),
    DeclareLaunchArgument(
        name="config",
        default_value="euroc_mav",
        description="euroc_mav, tum_vi, rpng_aruco...",
    ),
    DeclareLaunchArgument(
        name="config_path",
        default_value="",
        description="path to estimator_config.yaml. If not given, determined based on provided 'config' above",
    ),
    DeclareLaunchArgument(
        name="verbosity",
        default_value="INFO",
        description="ALL, DEBUG, INFO, WARNING, ERROR, SILENT",
    ),
    DeclareLaunchArgument(
        name="use_stereo",
        default_value="true",
        description="if we have more than 1 camera, if we should try to track stereo constraints between pairs",
    ),
    DeclareLaunchArgument(
        name="max_cameras",
        default_value="1",
        description="how many cameras we have 1 = mono, 2 = stereo, >2 = binocular (all mono tracking)",
    ),
    DeclareLaunchArgument(
        name="save_total_state",
        default_value="false",
        description="record the total state with calibration and features to a txt file",
    ),
    DeclareLaunchArgument(
        name="dobag",
        default_value="false",
        description="if we should play back the bag",
    ),
    DeclareLaunchArgument(
        name="bag",
        default_value="",
        description="path to the rosbag to play",
    ),
    DeclareLaunchArgument(
        name="bag_start",
        default_value="0.0",
        description="start time of the bag in seconds",
    ),
    DeclareLaunchArgument(
        name="bag_rate",
        default_value="1.0",
        description="playback rate of the bag",
    ),
    DeclareLaunchArgument(
        name="bag_delay",
        default_value="3.0",
        description="delay in seconds before starting bag playback (gives time for nodes to start)",
    ),
]


def launch_setup(context):
    config_path = LaunchConfiguration("config_path").perform(context)
    if not config_path:
        configs_dir = os.path.join(get_package_share_directory("ov_srvins"), "config")
        available_configs = os.listdir(configs_dir)
        config = LaunchConfiguration("config").perform(context)
        if config in available_configs:
            config_path = os.path.join(
                get_package_share_directory("ov_srvins"),
                "config",
                config,
                "estimator_config.yaml",
            )
        else:
            return [
                LogInfo(
                    msg="ERROR: unknown config: '{}' - Available configs are: {} - not starting OpenVINS".format(
                        config, ", ".join(available_configs)
                    )
                )
            ]
    else:
        if not os.path.isfile(config_path):
            return [
                LogInfo(
                    msg="ERROR: config_path file: '{}' - does not exist. - not starting OpenVINS".format(
                        config_path
                    )
                )
            ]
    node1 = Node(
        package="ov_srvins",
        executable="run_subscribe_msckf",
        condition=IfCondition(LaunchConfiguration("ov_enable")),
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        parameters=[
            {"verbosity": LaunchConfiguration("verbosity")},
            {"use_stereo": LaunchConfiguration("use_stereo")},
            {"max_cameras": LaunchConfiguration("max_cameras")},
            {"save_total_state": LaunchConfiguration("save_total_state")},
            {"filepath_est": "/workspace/results/ov_estimate.txt"},
            {"filepath_std": "/workspace/results/ov_estimate_std.txt"},
            {"filepath_gt": "/workspace/results/ov_groundtruth.txt"},
            {"config_path": config_path},
        ],
    )

    node2 = Node(
        package="rviz2",
        executable="rviz2",
        condition=IfCondition(LaunchConfiguration("rviz_enable")),
        arguments=[
            "-d"
            + os.path.join(
                get_package_share_directory("ov_srvins"), "launch", "display_ros2.rviz"
            ),
            "--ros-args",
            "--log-level",
            "warn",
        ],
    )

    # ROS2 bag player with delay
    bag_path = LaunchConfiguration("bag").perform(context)
    bag_rate = LaunchConfiguration("bag_rate").perform(context)
    bag_start = LaunchConfiguration("bag_start").perform(context)
    bag_delay = float(LaunchConfiguration("bag_delay").perform(context))
    dobag = LaunchConfiguration("dobag").perform(context).lower() == "true"

    nodes_to_launch = [node1, node2]

    if dobag and bag_path:
        bag_player = ExecuteProcess(
            cmd=[
                "ros2", "bag", "play",
                bag_path,
                "--rate", bag_rate,
                "--start-offset", bag_start,
            ],
            output="screen",
        )
        # Delay the bag playback to give nodes time to start
        delayed_bag_player = TimerAction(
            period=bag_delay,
            actions=[bag_player],
        )
        nodes_to_launch.append(delayed_bag_player)

    return nodes_to_launch


def generate_launch_description():
    opfunc = OpaqueFunction(function=launch_setup)
    ld = LaunchDescription(launch_args)
    ld.add_action(opfunc)
    return ld
