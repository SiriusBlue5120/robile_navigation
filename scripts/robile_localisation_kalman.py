#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from robile_interfaces.msg import PositionLabelled, PositionLabelledArray
from nav_msgs.msg import Odometry
import numpy as np
import tf2_ros
from tf_transformations import euler_from_quaternion, quaternion_from_euler


class LocalisationUsingKalmanFilter(Node):
    """
    Landmark based localisation using Kalman Filter
    This is a partially structured class for AMR assignment
    """

    def __init__(self):
        super().__init__('localisation_using_kalman_filter')

        # declaring and getting parameters from yaml file
        self.declare_parameters(
            namespace='',
            parameters=[
                ('map_frame', 'map'),
                ('odom_frame', 'odom'),                
                ('laser_link_frame', 'base_laser_front_link'),
                ('real_base_link_frame', 'real_base_link'),
                ('scan_topic', 'scan'),
                ('odom_topic', 'odom'),
                ('rfid_tag_poses_topic', 'rfid_tag_poses'),
                ('initial_pose_topic', 'initialpose'),
                ('real_base_link_pose_topic', 'real_base_link_pose'),
                ('estimated_base_link_pose_topic', 'estimated_base_link_pose'),
                ('minimum_travel_distance', 0.1),
                ('minimum_travel_heading', 0.1),
                ('rfid_tags.A', [0.,0.]),
                ('rfid_tags.B', [0.,0.]),
                ('rfid_tags.C', [0.,0.]),
                ('rfid_tags.D', [0.,0.]),
                ('rfid_tags.E', [0.,0.]),                        
            ])

        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.odom_frame = self.get_parameter('odom_frame').get_parameter_value().string_value
        self.laser_link_frame = self.get_parameter('laser_link_frame').get_parameter_value().string_value
        self.real_base_link_frame = self.get_parameter('real_base_link_frame').get_parameter_value().string_value
        self.scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.rfid_tag_poses_topic = self.get_parameter('rfid_tag_poses_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value
        self.real_base_link_pose_topic = self.get_parameter('real_base_link_pose_topic').get_parameter_value().string_value
        self.estimated_base_link_pose_topic = self.get_parameter('estimated_base_link_pose_topic').get_parameter_value().string_value
        self.minimum_travel_distance = self.get_parameter('minimum_travel_distance').get_parameter_value().double_value
        self.minimum_travel_heading = self.get_parameter('minimum_travel_heading').get_parameter_value().double_value
        self.rfid_tags_A = self.get_parameter('rfid_tags.A').get_parameter_value().double_array_value
        self.rfid_tags_B = self.get_parameter('rfid_tags.B').get_parameter_value().double_array_value
        self.rfid_tags_C = self.get_parameter('rfid_tags.C').get_parameter_value().double_array_value
        self.rfid_tags_D = self.get_parameter('rfid_tags.D').get_parameter_value().double_array_value
        self.rfid_tags_E = self.get_parameter('rfid_tags.E').get_parameter_value().double_array_value

        # setting up laser scan and rfid tag subscribers
        self.rfid_tag_subscriber = self.create_subscription(PositionLabelledArray, self.rfid_tag_poses_topic, self.rfid_callback, 10)
        self.real_laser_link_subscriber = self.create_subscription(PoseStamped, self.real_base_link_pose_topic, self.real_base_link_pose_callback, 10)        
        # self.estimated_robot_pose_publisher = self.create_publisher(PoseStamped, self.estimated_base_link_pose_topic, 10)
        self.estimated_robot_pose_publisher = self.create_publisher(PoseWithCovarianceStamped, self.estimated_base_link_pose_topic, 10)
        
        # setting up tf2 listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Dict mapping tag names to their respective variable names
        self.tag_map = {
            "A": self.rfid_tags_A,
            "B": self.rfid_tags_B,
            "C": self.rfid_tags_C,
            "D": self.rfid_tags_D,
            "E": self.rfid_tags_E
        }

        # State matrix:
        # position x, position y, heading position (yaw) theta,
        # velocity x, velocity y, heading velocity (yaw) omega
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.timestamp = {"sec": 0, "nanosec": 0}

        # Debug
        self.debug = True


    def get_detected_tags(self, msg: PositionLabelledArray) -> dict[str, np.ndarray]:
        """
        Parses PositionLabelledArray into a dictionary of detected tags by name: position
        """
        detected_tags = {}
        for tag in msg.positions:
            tag: PositionLabelled
                
            tag_name = tag.name
            tag_position = np.array([tag.position.x, tag.position.y, tag.position.z])
            
            detected_tags.update({tag_name: tag_position})

        return detected_tags
    

    def create_pose_from_state(self, state, covariance, frame_id, timestamp) -> PoseWithCovarianceStamped:
        """
        Create a PoseWithCovarianceStamped object from given state and covariance matrices
        """
        pose_cov = PoseWithCovarianceStamped()

        # Header
        pose_cov.header.frame_id = frame_id

        pose_cov.header.stamp.sec = timestamp["sec"]
        pose_cov.header.stamp.nanosec = timestamp["nanosec"]

        # Position
        pose_cov.pose.pose.position.x = state[0]
        pose_cov.pose.pose.position.y = state[1]
        pose_cov.pose.pose.position.z = 0.0

        # Orientation
        state_quat = quaternion_from_euler(0.0, 0.0, state[2])

        pose_cov.pose.pose.orientation.x = state_quat[0]
        pose_cov.pose.pose.orientation.y = state_quat[1]
        pose_cov.pose.pose.orientation.z = state_quat[2]
        pose_cov.pose.pose.orientation.w = state_quat[3]

        # Covariance
        # TODO: Modify this to use the correct covariance
        # Zero for now
        state_covariance = np.zeros_like(pose_cov.pose.covariance)
        pose_cov.pose.covariance = state_covariance

        return pose_cov

 
    def rfid_callback(self, msg: PositionLabelledArray):
        """
        Based on the detected RFID tags, performing measurement update
        """
        ### YOUR CODE HERE ###

        detected_tags = self.get_detected_tags(msg)

        if self.debug:
            self.get_logger().info(f"Detected tags: {detected_tags}")
        
        return


    def real_base_link_pose_callback(self, msg: PoseStamped):
        """
        Updating the base_link pose based on the update in robile_rfid_tag_finder.py
        """

        self.get_logger().info(f"real_base_link_pose msg: {msg}")

        yaw = euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])[2]
        self.real_laser_link_pose = [msg.pose.position.x, msg.pose.position.y, yaw]

    
    def motion_update(self, state: np.ndarray, control_input: np.ndarray, time_step: float):
        """
        Update estimate of state with control input
        """

        state_motion_prediction = np.array(state)

        # TODO: control update

        return state_motion_prediction
    
    def kalman_filter_gain (
                            cov_matrix : np.array,
                            measurement_matrix: np.array,
                            measurement_noise : np.array
                            ):

        kalman_gain = cov_matrix @ measurement_matrix.T @ np.linalg.inv(
        measurement_matrix @ cov_matrix @ measurement_matrix.T + measurement_noise)

        return kalman_gain
    

def main(args=None):
    rclpy.init(args=args)

    try:
        localisation_using_kalman_filter = LocalisationUsingKalmanFilter()
        rclpy.spin(localisation_using_kalman_filter)

    finally:
        localisation_using_kalman_filter.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
