# test
# /c/Users/yush7/Desktop/satellite project files
import argparse
import sys
import rosbag
import cv2
from cv_bridge import CvBridge
import numpy as np
import os

from src.evaluation_metrics import evaluate_number_accuracy_for_frame
from src.utils import parse_rosbag_file
from src.yaml_utils import load_camera_intrinsics, check_yaml_file_exists, load_ros_topic_names, load_parameters_from_yaml
from src.functions import record_y_averages_in_results_file,get_lighting_and_exposure, show_all_graphs_per_lighting_condition_from_results_file, getCanny, create_results_dict, makeGraphFromTextFile, write_results_dict_to_text_file, get_corners, get_processed_images, use_feature_detector, filter_keypoints_with_mask, order_points_clockwise, get_circle_grid_mask, drawPoints, filter_border_lines, makeGraph, drawLinesPolar, filter_lines_intersect, img_from_events, showImage, undistort_image, getIntersect, isClose2D, isClose
from src.feature_detectors import sift_detector, surf_detector, orb_detector, brief_detector

def read_and_record_rosbag_data(bag_file, delay, experiment_dict, camera_matrix, distortion_coefficients):

    pattern = experiment_dict["pattern"]
    num_features = pattern[0] * pattern[1]

    feature_detect_toggle = experiment_dict["feature_toggle"]

    # event camera characteristics
    event_buffer_time_ns = experiment_dict["event_buffer_time_ns"]

    # storing the corners as detected from Hough Line transforms
    previous_corners = [[14, 18], [334, 12], [326, 248], [21, 227]] # initial guess if no corners
  
    # Open the ROS bag file
    bag = rosbag.Bag(bag_file, 'r')

    # event camera bins (separated by the buffer period)
    binned_events = []
    current_bin = []

    # bool to check for first message
    first_message = True

    # frame counter for RGB cam
    frame_counter = 0

    # per frame dictionary for rgb camera
    per_frame_dict= {}

    # per time dictionary for event camera
    per_time_dict = {}
    
    # Iterate through the messages in the specified topic
    for topic, msg, t in bag.read_messages(topics=topics_dict.values()):

        if (topic == topics_dict.get('image')):
            # Convert the ROS Image message to an OpenCV image

            # get processed images
            colour_img, gray_img, canny = get_processed_images(msg, camera_matrix, distortion_coefficients)

            # get corners using houghLines
            corners = get_corners(canny, colour_img, previous_corners)
            drawPoints(corners, colour_img)     

            # apply feature detection methods  

            per_frame_dict[frame_counter] = create_results_dict(feature_detect_toggle, frame_counter, gray_img, corners, colour_image=colour_img, event_flag=False)

            # results stored as:
                # {
                #     frame_1 :
                #         {
                #             'sift' : sift_accuracy,
                #             'orb'  : orb_accuracy,
                #             'brief' : brief_accuracy,
                #             'hough' : hough_accuracy
                #         }
                    
                #     frame_2 :
                #         {
                #             'sift' : sift_accuracy,
                #             'orb'  : orb_accuracy,
                #             'brief' : brief_accuracy,
                #             'hough' : hough_accuracy
                #         }
                # }

            showImage("Colour", colour_img, delay)
            
            frame_counter +=1

        elif (topic == topics_dict.get('tf')):
            # print((topic, t, '\b  n'))
            pass

        elif (topic == topics_dict.get('event')):

            if 'corners' not in locals():
                corners = previous_corners

            # sort all events into bins of the event buffer length 
            for event in msg.events:

                x,y,ts, polarity = event.x, event.y, event.ts, event.polarity

                if first_message:
                    start_time = ts.to_nsec()
                    previous_time_ns = 0
                    first_message = False   

                current_time_ns = ts.to_nsec()
                current_bin.append(event)
                # print(current_time_ns)

                if (current_time_ns - previous_time_ns > event_buffer_time_ns):
                    binned_events.append(current_bin)

                    # create image from current event bin once the buffer is exceeded
                    event_img = img_from_events(current_bin, camera_matrix, distortion_coefficients)

                    # event_img = cv2.Canny(event_img, 100, 200)
                    
                    # erode and dilate image to remove noise
                    kernel = np.ones((3,3),np.uint8)
                    event_img = cv2.morphologyEx(event_img, cv2.MORPH_OPEN, kernel)

                    per_time_dict[current_time_ns-start_time] = create_results_dict(feature_detect_toggle, current_time_ns, event_img, corners, event_img, event_flag=True)

                    current_bin = []
                    previous_time_ns = current_time_ns
 
    # Close the bag file
    bag.close()
    
    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

    write_results_dict_to_text_file(per_frame_dict, bag_file, event_flag=False)

    write_results_dict_to_text_file(per_time_dict, bag_file, event_flag=True)

    bag_text_file_name = bag_file[:-4]+'_rgb.txt'
    y_averages_rgb = makeGraphFromTextFile(file_location=bag_text_file_name, make_graph_boolean=False)   
    bag_text_file_name = bag_file[:-4]+'_evt.txt'
    y_averages_evt = makeGraphFromTextFile(file_location=bag_text_file_name, make_graph_boolean=False)   
    
    return y_averages_rgb, y_averages_evt
    
def iterate_through_every_bag_file(directory):

    progress_counter = 1

    for bagName in os.listdir(directory):

        print(bagName)
        # print(os.listdir(directory))

        if bagName.endswith(".bag"):
            bag_file = directory + bagName
            
            y_averages_rgb, y_averages_evt = read_and_record_rosbag_data(bag_file, delay, experiment_dict, camera_matrix, distortion_coefficients)
            record_y_averages_in_results_file(directory, bagName, y_averages_rgb,event_flag=False)
            record_y_averages_in_results_file(directory, bagName, y_averages_evt,event_flag=True)

            print(progress_counter,'bags completed')
            progress_counter+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a path to a rosbag file.")
    parser.add_argument('rosbag_path', type=parse_rosbag_file, help='Path to a rosbag file')
    args = parser.parse_args()
    bag_file = args.rosbag_path
    print(f"Path to rosbag file: {bag_file}")

    # load config file
    config_file = "config.yaml"

    if not check_yaml_file_exists(config_file):
        print(f"Error: The file '{config_file}' does not exist.")
        sys.exit(1)

    camera_matrix, distortion_coefficients, image_size = load_camera_intrinsics(config_file)
    topics_dict = load_ros_topic_names(config_file)
    experiment_dict = load_parameters_from_yaml(config_file, ["pattern", "event_buffer_time_ns", "feature_toggle"])

    delay = int(load_parameters_from_yaml(config_file, ["delay"]))

    # bag_file = 'C:/Users/yush7/Desktop/vri_files_2024/data/calib/hi2.bag'
    # lz4_file = 'C:/Users/yush7/Desktop/vri_files_2024/data/hi2.bag'
    # bag_file = 'C:/Users/yush7/Desktop/satellite project files/data/test1.bag'
    # bag_file = 'C:/Users/yush7/Desktop/satellite project files/data/calib/calibration_data.bag'
    # bag_file = '/home/ayush/Data/tst2.bag' 
    # bag_file = '/home/ayush/Data/dataset/ra_50_exp_20_000.bag' 

    directory = '/home/ayush/Data/dataset/'

    iterate_through_every_bag_file(directory)
    show_all_graphs_per_lighting_condition_from_results_file(directory,'results_evt.txt')
    show_all_graphs_per_lighting_condition_from_results_file(directory,'results_rgb.txt')

    # read_and_record_rosbag_data(bag_file, delay, experiment_dict, camera_matrix, distortion_coefficients)
