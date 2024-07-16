import yaml
import os
import numpy as np

def check_yaml_file_exists(file_path):
    """
    Check if a YAML file exists at the given path.

    Parameters:
    file_path (str): Path to the YAML file.

    Returns:
    bool: True if the file exists and is a file, False otherwise.
    """
    return os.path.isfile(file_path)

def load_configs(file_path):
    """Load config file from a given path

    Args:
        file_path (str): Path of the file

    Returns:
        config (python obj): Python object containing configs
    """
    with open(file_path, 'r') as stream:
        config = yaml.safe_load(stream)
        return config
    
def load_camera_intrinsics(file_path):
    """Load TF matrix, camera matrix and distortion matrix from a yaml file.

    Args:
        file_path (str): Path of the file

    Returns:
        camera_matrix (numpy array): intrinsic camera matrix [3 x 3]
        distortion_coefficients (numpy array): distortion coefficients in OpenCV format [1 x 5]
        image_size (integer list): image height, image width [1 x 2]
    """

    configs = load_configs(file_path)

    fx = configs["camera"]["fx"]
    fy = configs["camera"]["fy"]
    cx = configs["camera"]["cx"]
    cy = configs["camera"]["cy"]
    k1 = configs["camera"]["k1"]
    k2 = configs["camera"]["k2"]
    k3 = configs["camera"]["k3"]
    p1 = configs["camera"]["p1"]
    p2 = configs["camera"]["p2"]

    image_width = configs["camera"]["image_width"]
    image_height = configs["camera"]["image_height"]

    camera_matrix = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1]])
    
    distortion_coefficients = np.array([k1, k2, p1, p2, k3])

    image_size = (image_height,image_width)

    return camera_matrix, distortion_coefficients, image_size


def load_ros_topic_names(file_path):
    """
    Load ROS topic names from a YAML file.

    Parameters:
    file_path (str): Path to the YAML file containing the ROS topics.

    Returns:
    dict: A dictionary containing the topic names for "image", "tf", and "event".
    """
    configs = load_configs(file_path)
    
    topics = configs.get('topics', {})
    image_topic = topics.get('image')
    tf_topic = topics.get('tf')
    event_topic = topics.get('event')
    
    return {
        'image': image_topic,
        'tf': tf_topic,
        'event': event_topic
    }


def load_parameters_from_yaml(file_path, keys):
    """
    Load parameters from a YAML file based on a list of keys.

    Parameters:
    file_path (str): Path to the YAML file.
    keys (list of str): List of keys to extract from the YAML file.

    Returns:
    dict: Dictionary containing the keys and their corresponding values from the YAML file.
    """
    # Load the YAML file
    configs = load_configs(file_path)
    
    # Extract the requested parameters
    parameters = {}
    for key in keys:
        if key in configs:
            parameters[key] = configs[key]
        else:
            parameters[key] = None  # Or handle missing keys as needed
    if len(parameters.values()) == 1:
        return list(parameters.values())[0]
    else:
        return parameters


