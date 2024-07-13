import argparse
import os

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