#!/usr/bin/env python3

import time

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import (PoseStamped, PoseWithCovarianceStamped,
                               TransformStamped, Twist)
# from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf_transformations import euler_from_quaternion, quaternion_from_euler

from robile_interfaces.msg import PositionLabelled, PositionLabelledArray


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
        # Modifying name to be consistent
        # self.real_laser_link_subscriber = self.create_subscription(PoseStamped, self.real_base_link_pose_topic, self.real_base_link_pose_callback, 10)
        self.real_base_link_subscriber = self.create_subscription(PoseStamped, self.real_base_link_pose_topic, self.real_base_link_pose_callback, 10)
        # Modifying as per assignment requirement
        # self.estimated_robot_pose_publisher = self.create_publisher(PoseStamped, self.estimated_base_link_pose_topic, 10)
        self.estimated_robot_pose_publisher = self.create_publisher(PoseWithCovarianceStamped, self.estimated_base_link_pose_topic, 10)
        self.cmd_vel_subscriber = self.create_subscription(Twist, '\cmd_vel', self.cmd_vel_callback, 10)

        # setting up tf2 listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Dict mapping tag names to their respective variable names
        self.known_tags = {
            "A": np.array(self.rfid_tags_A),
            "B": np.array(self.rfid_tags_B),
            "C": np.array(self.rfid_tags_C),
            "D": np.array(self.rfid_tags_D),
            "E": np.array(self.rfid_tags_E)
        }
        self.tag_dim = 2

        # State matrix:
        # position x, position y, heading position (yaw) theta,
        self.state = np.array([0.0, 0.0, 0.0])
        self.cov_matrix = np.eye(3) * 0.001
        self.timestamp = {"sec": 0, "nanosec": 0}

        # Timing
        self.previous_pred_time = time.time()
        self.time_to_reset = 0.2

        # Control parameters
        self.pred_noise = np.eye(3)*0.01
        self.control_input = np.array([0.0, 0.0, 0.0])

        # Measurement parameters
        self.measurement_noise = np.eye(6) * 0.01

        # Debug
        self.verbose = True


    def cmd_vel_callback(self, msg:Twist):
        self.control_input = np.array([msg.linear.x,msg.linear.y,msg.angular.z])


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


    def convert_to_polar(self, cartesian_position: np.ndarray) -> np.ndarray:
        """
        Convert given 2D position array from cartesian to polar coordinates
        """
        assert cartesian_position.shape[-1] == 2, "Expected 2D cartesian coordinates"
        assert len(cartesian_position.shape) <= 2,  "Expected array of 2D cartesian coordinates"

        if len(cartesian_position.shape) == 1:
            cartesian_position = cartesian_position[np.newaxis]

        polar_position = np.zeros_like(cartesian_position)

        polar_position[:, 0] = np.linalg.norm(cartesian_position[:, :], axis=1)
        polar_position[:, 1] = np.arctan2(cartesian_position[:, 1], cartesian_position[:, 0])

        return polar_position


    def create_pose_from_state(self, position, orientation,
                               covariance, frame_id, timestamp) -> PoseWithCovarianceStamped:
        """
        Create a PoseWithCovarianceStamped object from given state and covariance matrices
        """
        pose_cov = PoseWithCovarianceStamped()

        # Header
        pose_cov.header.frame_id = frame_id

        pose_cov.header.stamp.sec = timestamp["sec"]
        pose_cov.header.stamp.nanosec = timestamp["nanosec"]

        # Position
        pose_cov.pose.pose.position.x = position[0]
        pose_cov.pose.pose.position.y = position[1]
        pose_cov.pose.pose.position.z = position[2]

        # Orientation
        state_quat = \
                quaternion_from_euler(orientation[0], orientation[1], orientation[2])

        pose_cov.pose.pose.orientation.x = state_quat[0]
        pose_cov.pose.pose.orientation.y = state_quat[1]
        pose_cov.pose.pose.orientation.z = state_quat[2]
        pose_cov.pose.pose.orientation.w = state_quat[3]

        # Covariance
        pose_cov.pose.covariance = covariance.flatten()

        return pose_cov


    def calculate_transform(self, target_frame, source_frame) -> TransformStamped:
        transform: TransformStamped = \
            self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())

        if self.verbose:
            self.get_logger().info(f"transform btw target {target_frame} and " + \
                                   f"source {source_frame}: {transform}")

        return transform


    def rfid_callback(self, msg: PositionLabelledArray):
        """
        Based on the detected RFID tags, performing measurement update
        """
        ### YOUR CODE HERE ###

        detected_tags = self.get_detected_tags(msg)
        num_tags = len(detected_tags.keys())

        if not num_tags:
            self.get_logger().info("No tags were detected. Skipping measurement update")

            return

        if self.verbose:
            self.get_logger().info(f"Detected tags: {detected_tags}")

        # Crafting matrix  of tags
        known_tag_positions = np.zeros((num_tags * self.tag_dim, 1))
        measured_tag_positions = np.zeros((num_tags * self.tag_dim, 1))
        for index, tag_name in enumerate(detected_tags.keys()):
            known_tag_positions[index * self.tag_dim : index * self.tag_dim+2, :] = \
                np.array(self.known_tags[tag_name])[:2][np.newaxis].T
            measured_tag_positions[index * self.tag_dim : index * self.tag_dim+2, :] = \
                detected_tags[tag_name][:2][np.newaxis].T    

        # Crafting matrix of tags in polar coordinates
        known_tags_polar = np.zeros(known_tag_positions.shape)
        measured_tags_polar = np.zeros(measured_tag_positions.shape)
        for index in range(num_tags):
            known_tags_polar[self.tag_dim * index : self.tag_dim * index + 2] = \
                self.convert_to_polar(known_tag_positions[self.tag_dim * index : self.tag_dim * index + 2].flatten()).T
            measured_tags_polar[self.tag_dim * index : self.tag_dim * index + 2] = \
                self.convert_to_polar(measured_tag_positions[self.tag_dim * index : self.tag_dim * index + 2].flatten()).T

        # Measurement update
        x, y, theta = self.state.flatten()

        # Transform known tags into robot frame (using h)
        predicted_tags_polar = np.zeros((self.tag_dim * num_tags, 1))
        for index in range(num_tags):
            alpha_j = known_tags_polar[self.tag_dim * index + 1][0]
            r_j = known_tags_polar[self.tag_dim * index][0]

            predicted_tags_polar[self.tag_dim * index : self.tag_dim * index+2] = \
                np.array([[r_j - (x * np.cos(alpha_j) + y * np.sin(alpha_j))],
                          [alpha_j - theta]])

        # Innovation
        v_t = measured_tags_polar - predicted_tags_polar

        # Jacobian of transformation matrix (H)
        H = np.zeros((self.tag_dim * num_tags, 3))
        for index in range(num_tags):
            alpha_j = known_tags_polar[self.tag_dim * index+1][0]
            r_j = known_tags_polar[self.tag_dim * index][0]
            
            H[self.tag_dim * index:(self.tag_dim * index)+2, :] = np.array([
                            [0, 0, -1],
                            [-np.cos(alpha_j), -np.sin(alpha_j), 0]
                        ])

        # Varying measurement noise with number of tags - more tags better
        measurement_noise = self.measurement_noise / num_tags

        # Innovation covariance
        sigma: np.ndarray = H @ self.cov_matrix @ H.T + measurement_noise

        # Kalman gain
        kalman_gain: np.ndarray = self.kalman_filter_gain(self.cov_matrix, H, sigma)

        # Updating state and covariance with measurement
        self.state: np.ndarray = (self.state[np.newaxis].T + kalman_gain @ v_t).flatten()
        self.cov_matrix: np.ndarray = self.cov_matrix - kalman_gain @ sigma @ kalman_gain.T


    def real_base_link_pose_callback(self, msg: PoseStamped):
        """
        Updating the base_link pose based on the update in robile_rfid_tag_finder.py
        """
        self.get_logger().info(f"real_base_link_pose msg: {msg}")

        yaw = euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y,
                                            msg.pose.orientation.z, msg.pose.orientation.w])[2]
        self.real_laser_link_pose = [msg.pose.position.x, msg.pose.position.y, yaw]

        time_step = time.time() - self.previous_pred_time
        if time_step > self.time_to_reset:
            time_step = self.time_to_reset

        self.state, self.cov_matrix = \
            self.motion_update(self.state, self.cov_matrix, 
                               self.control_input, time_step,
                               self.pred_noise)

        position = [self.state[0], self.state[1], 0.0]
        orientation = [0.0, 0.0, self.state[2]]

        covariance = np.zeros((6, 6))
        covariance[:2, :2] = self.cov_matrix[:2, :2]
        covariance[-1, -1] = self.cov_matrix[-1, -1]
        covariance[-1, :2] = self.cov_matrix[-1, :2]
        covariance[:2, -1] = self.cov_matrix[:2, -1]

        self.timestamp = {
                    "sec": msg.header.stamp.sec,
                    "nanosec": msg.header.stamp.nanosec
                    }

        # Craft correct covariance from computed covariance
        msg = self.create_pose_from_state(position, orientation, covariance, 
                                          self.odom_frame, self.timestamp)

        self.estimated_robot_pose_publisher.publish(msg)


    def motion_update(self, state: np.ndarray, cov_matrix: np.ndarray, 
                      control_input: np.ndarray, time_step: float, 
                      pred_noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Update estimate of state with control input
        Assuming state to be 3x1 and control input as velocity 3x1
        """
        # Control update
        F_k_1 = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
        G_k_1 = np.zeros((3, 3))
        np.fill_diagonal(G_k_1, time_step)

        # State and covariance prediction
        x_k = F_k_1 @ state[np.newaxis].T + G_k_1 @ control_input[np.newaxis].T
        P_k = F_k_1 @ cov_matrix @ np.transpose(F_k_1) + pred_noise

        return x_k.flatten(), P_k


    def kalman_filter_gain (self, cov_matrix : np.array, measurement_matrix: np.array,
                            sigma: np.array) -> np.ndarray:

        kalman_gain: np.ndarray = cov_matrix @ measurement_matrix.T @ np.linalg.pinv(sigma)

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
