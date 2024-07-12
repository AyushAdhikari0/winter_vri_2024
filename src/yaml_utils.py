import yaml
import numpy as np

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
    
def load_camera_intrinsics_from_config(file_path):
    """Load TF matrix, camera matrix and distortion matrix from a yaml file.

    Args:
        file_path (str): Path of the file

    Returns:
        camera_matrix (numpy array): intrinsic camera matrix [3 x 3]
        distortion_coefficients (numpy array): distortion coefficients in OpenCV format [1 x 5]
        image_size (integer list): image height, image width [1 x 2]
    """

    configs = load_configs(file_path)

    fx = configs["fx"]
    fy = configs["fy"]
    cx = configs["cx"]
    cy = configs["cy"]
    k1 = configs["k1"]
    k2 = configs["k2"]
    k3 = configs["k3"]
    p1 = configs["p1"]
    p2 = configs["p2"]

    image_width = configs["image_width"]
    image_height = configs["image_height"]

    camera_matrix = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1]])
    
    distortion_coefficients = np.array([k1, k2, p1, p2, k3])

    image_size = (image_height,image_width)

    return camera_matrix, distortion_coefficients, image_size