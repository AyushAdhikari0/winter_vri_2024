import argparse
import os
import rosbag
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
from cv_bridge import CvBridge


def parse_rosbag_file(file_path):
    """
    Parse the path to a rosbag file.

    Parameters:
    file_path (str): Path to the rosbag file.

    Returns:
    str: The file path if it is valid, otherwise raises an error.
    """
    if not os.path.isfile(file_path):
        raise argparse.ArgumentTypeError(f"File {file_path} does not exist")
    return file_path

def load_tf_messages(bag_file, topic, parent_frame, child_frame):
    """
    Load tf2_msgs/TFMessage from a ROS bag given a topic, parent frame, and child frame.

    Parameters:
    - bag_file (str): The path to the ROS bag file.
    - topic (str): The topic to read the TF messages from.
    - parent_frame (str): The parent frame ID.
    - child_frame (str): The child frame ID.

    Returns:
    - list: A list of TFMessage containing the transformations between the specified frames.
    """
    transformations = []

    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic]):
            for transform in msg.transforms:
                if transform.header.frame_id == parent_frame and transform.child_frame_id == child_frame:
                    transformations.append(transform)

    return transformations

def load_images_from_rosbag(bag_file, topic_name):
    """
    Load images from a ROS bag file.

    Parameters:
    - bag_file (str): path to the ROS bag file.
    - topic_name (str): name of the image topic in the ROS bag.

    Returns:
    - images (list): images from back
    - timestamps (list): timestamp of each image
    """
    # Initialize the CvBridge
    bridge = CvBridge()

    images = []
    timestamps = []
    
    with rosbag.Bag(bag_file, 'r') as bag:
        # Iterate through the messages in the specified topic
        for _, msg, timestamp in bag.read_messages(topics=[topic_name]):
            # Convert the ROS image message to an OpenCV image
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            images.append(cv_image)
            timestamps.append(timestamp)
    
    return images, timestamps

def filter_tf_messages(tf_messages, timestamp, max_time_diff=0.01):
    """
    Filter a list of TF messages to find all messages that are at the given timestamp
    and within the maximum time difference.

    Parameters:
    - tf_messages (list): A list of TF messages to filter.
    - timestamp (rospy.Time): The target timestamp to filter the messages.
    - max_time_diff (rospy.Duration): The maximum allowable time difference.

    Returns:
    - list: A list of TF messages that match the criteria.
    """
    filtered_messages = []
    for tf_message in tf_messages:
        # Assume tf_message is of type geometry_msgs/TransformStamped
        msg_time = tf_message.header.stamp
        time_diff = abs((msg_time - timestamp).to_sec())
        
        if time_diff <= max_time_diff:
            filtered_messages.append(tf_message)
    
    return filtered_messages


def tf_messages_to_transformation(tf_messages):
    """
    Convert a list of TF messages to positions, Euler rotation angles,
    find the median across all poses, and convert to a 4x4 homogeneous transformation matrix.

    Parameters:
    - tf_messages (list): A list of TF messages.

    Returns:
    - np.ndarray: A 4x4 homogeneous transformation matrix representing the median pose.
    """
    positions = []
    euler_angles = []

    for tf_message in tf_messages:
        # Extract translation
        translation = tf_message.transform.translation
        position = np.array([translation.x, translation.y, translation.z])
        positions.append(position)

        # Extract rotation (quaternion)
        rotation = tf_message.transform.rotation
        quaternion = [rotation.x, rotation.y, rotation.z, rotation.w]

        # Convert quaternion to Euler angles (roll, pitch, yaw)
        r = R.from_quat(quaternion)
        euler = r.as_euler('xyz', degrees=False)  # 'xyz' indicates the rotation order
        euler_angles.append(euler)

    # Calculate median position and Euler angles
    median_position = np.median(positions, axis=0)
    median_euler = np.median(euler_angles, axis=0)

    # Convert median position and Euler angles to a 4x4 homogeneous transformation matrix
    r = R.from_euler('xyz', median_euler, degrees=False)
    rotation_matrix = r.as_matrix()
    
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = median_position

    return transformation_matrix

def calculate_transformation_board_to_camera(tfmsgs_endeffector_to_robot_base, image_timestamp, tf_robot_base_to_camera, tf_checkerboard_to_endeffector, max_time_diff=0.01):
    """
    Calculate the 4 by 4 homogeneous transformation matrix that will transform a point on the checkerboard/board to the camera by finding nearest end effector pose in time.
    This funcion takes in a list of end effector poses and finds the median pose across poses that are within the image timestamp.
    Resulting pose is calculated through the following chain of transformations: camera <- robot base <- end effector <- board

    Parameters:
    - tfmsgs_endeffector_to_robot_base (list): tf messages of transformation of end effector to robot base
    - image_timestamp (ros.timestamp): timestamp of image 
    - tf_robot_base_to_camera (np.ndarray): A 4x4 homogeneous transformation matrix of robot base to camera
    - tf_checkerboard_to_endeffector (np.ndarray): A 4x4 homogeneous transformation matrix of checkerboard/board to end effector
    - max_time_diff (float): threshold to find the nearest tf poses to the image_timestamp

    Returns:
    - np.ndarray: A 4x4 homogeneous transformation matrix
    """

    tfmsgs_endeffector_to_robot_base_filter = filter_tf_messages(tfmsgs_endeffector_to_robot_base, image_timestamp, max_time_diff)
    tf_endeffector_to_robot_base = tf_messages_to_transformation(tfmsgs_endeffector_to_robot_base_filter)
    tf_board_to_camera = tf_robot_base_to_camera @ tf_endeffector_to_robot_base @ np.linalg.inv(tf_checkerboard_to_endeffector)

    return tf_board_to_camera


def transform_matrix_to_rodrigues_and_position(matrix):
    """
    Convert a 4x4 transformation matrix to Rodrigues rotation vector (axis-angle) and position vector.

    Parameters:
    - matrix (np.ndarray): The 4x4 transformation matrix.

    Returns:
    - tuple: A tuple containing the Rodrigues rotation vector (axis-angle) and position vector.
    """
    # Extract rotation and translation components
    rotation_matrix = matrix[:3, :3]
    translation_vector = matrix[:3, 3]

    # Convert rotation matrix to Rodrigues rotation vector (axis-angle)
    rvec, _ = cv2.Rodrigues(rotation_matrix)

    return rvec.flatten(), translation_vector

def project_points_board_to_camera(pts, tf_board_to_camera, camera_matrix, distortion_coefficients=[0,0,0,0,0]):

    rvec, tvec = transform_matrix_to_rodrigues_and_position(np.linalg.inv(tf_board_to_camera))
    projected_pts = cv2.projectPoints(pts, rvec, tvec, camera_matrix, distortion_coefficients)[0]

    projected_pts = np.squeeze(projected_pts)

    if len(projected_pts.shape) == 1:
        projected_pts = projected_pts.reshape((1,2))
        # temp = projected_pts
        # projected_pts = np.zeros((1,1,2))
        # projected_pts[0] = temp
    
    return projected_pts
    