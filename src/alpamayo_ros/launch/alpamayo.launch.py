from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Launch Alpamayo node that listens to live camera + odometry topics."""
    # Camera topics and their corresponding camera indices:
    # 0=Front left, 1=Front, 2=Front right, 3=Rear left, 4=Rear, 5=Rear right, 6=Front telephoto
    default_camera_topics = [
        "/image_raw/compressed",   # front_wide  -> index 1
    ]
    default_camera_indices = [1]
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model_name_or_path",
                default_value="nvidia/Alpamayo-1.5-10B",
            ),
            DeclareLaunchArgument(
                "processor_name_or_path",
                default_value="Qwen/Qwen3-VL-2B-Instruct",
            ),
            DeclareLaunchArgument(
                "vlm_name_or_path",
                default_value="nvidia/Cosmos-Reason2-8B",
            ),
            DeclareLaunchArgument("offline_mode", default_value="false"),
            Node(
                package="alpamayo_ros",
                executable="alpamayo_node",
                prefix=["python3"],
                name="alpamayo_node",
                output="screen",
                parameters=[
                    {
                        "model_name_or_path": LaunchConfiguration("model_name_or_path"),
                        "vlm_name_or_path": LaunchConfiguration("vlm_name_or_path"),
                        "processor_name_or_path": LaunchConfiguration(
                            "processor_name_or_path"
                        ),
                        "offline_mode": LaunchConfiguration("offline_mode"),
                        "auto_run": True,
                        "camera_topics": default_camera_topics,
                        "camera_indices": default_camera_indices,
                        "odometry_topic": "/localization/kinematic_state",
                        "trajectory_topic": "/alpamayo/predicted_trajectory",
                        "cot_topic": "/alpamayo/reasoning",
                        "cot_with_stamped_topic": "/alpamayo/reasoning_stamped",
                        "inference_period_sec": 1.0,
                    }
                ],
            )
        ]
    )
