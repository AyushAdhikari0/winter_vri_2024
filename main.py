# test
# /c/Users/yush7/Desktop/satellite project files
import argparse
import sys
import rosbag
import cv2
from cv_bridge import CvBridge
import numpy as np
import os

from src.evaluation_metrics import evaluate_number_accuracy_for_frame, write_histogram_to_text_file
from src.utils import parse_rosbag_file
from src.yaml_utils import load_camera_intrinsics, check_yaml_file_exists, load_ros_topic_names, load_parameters_from_yaml
from src.functions import get_corners, get_processed_images, use_feature_detector, filter_sift_keypoints, order_points_clockwise, get_circle_grid_mask, drawPoints, filter_border_lines, makeGraph, drawLinesPolar, filter_lines_intersect, img_from_events, showImage, undistort_image, getIntersect, isClose2D, isClose
from src.feature_detectors import sift_detector, surf_detector, orb_detector, brief_detector


def read_and_record_rosbag_data(bag_file, delay, experiment_dict, camera_matrix, distortion_coefficients):

    pattern = experiment_dict["pattern"]
    num_features = pattern[0] * pattern[1]

    feature_detect_toggle = experiment_dict["feature_toggle"]

    # event camera characteristics
    event_buffer_time_ns = experiment_dict["event_buffer_time_ns"]

    # storing the corners as detected from Hough Line transforms
    previous_corners = [[14, 18], [334, 12], [326, 248], [21, 227]] # initial guess if no corners

    # list of [(frame, accuracy)]
    rgb_accuracy_histogram = []

    # list of [(time since start, accuracy)]
    event_accuracy_histogram = []
    
    # Open the ROS bag file
    bag = rosbag.Bag(bag_file, 'r')

    # event camera bins (separated by the buffer period)
    binned_events = []
    current_bin = []

    # bool to check for first message
    first_message = True

    # frame counter for RGB cam
    frame_counter = 0

    # frame histogram
    per_frame_dict= {}


    
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

            # results dict   
            per_frame_dict[frame_counter] = {}

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

            if feature_detect_toggle.get('sift'):
                keypoints, mask = use_feature_detector('sift', gray_img, corners, filter_toggle=True)
                per_frame_dict[frame_counter]['sift'] = evaluate_number_accuracy_for_frame(keypoints) 

            if feature_detect_toggle.get('orb'):
                keypoints, mask = use_feature_detector('orb', gray_img, corners, filter_toggle=True)
                per_frame_dict[frame_counter]['orb'] = evaluate_number_accuracy_for_frame(keypoints) 

            if feature_detect_toggle.get('brief'):
                keypoints, mask = use_feature_detector('brief', gray_img, corners, filter_toggle=True)
                per_frame_dict[frame_counter]['brief'] = evaluate_number_accuracy_for_frame(keypoints) 
             
            if feature_detect_toggle.get('hough'):
                centres = use_feature_detector('hough', gray_img, corners, filter_toggle=True, canny=canny, colour_img=colour_img)
                per_frame_dict[frame_counter]['hough'] = evaluate_number_accuracy_for_frame(centres) 

            showImage("Colour", colour_img, delay)
            
            rgb_perc = 100* len(centres)/num_features
            rgb_accuracy_histogram.append((frame_counter, rgb_perc))

            print("RGB Camera:", rgb_perc, "percent of features detected")
            frame_counter +=1


        elif (topic == topics_dict.get('tf')):
            # print((topic, t, '\b  n'))
            pass

        elif (topic == topics_dict.get('event')):

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

                    # apply circle detector
                    circles = cv2.HoughCircles(event_img, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50,param2=8,minRadius=5,maxRadius=8)

                    event_img = cv2.cvtColor(event_img, cv2.COLOR_GRAY2BGR)

                    # stores centres of circles detected by event camera
                    event_centres = []
                    if circles is not None:

                        j = 0
                        colour_increment = 255/len(circles[0,:])

                        for i in circles[0,:]:

                        # filter points that are close to the corners 
                            for (x,y) in corners:
                                if isClose2D((i[0],i[1]),(x,y), 15):
                                    break
                            else:
                                i = [int(a) for a in i]   

                                # draw centre
                                cv2.circle(event_img,(i[0],i[1]),2,(0,255,0),3)
                                
                                event_centres.append((i[0], i[1])) 

                                # draw the outer circle
                                if j==0:
                                    # draw first circle in green
                                    cv2.circle(event_img,(i[0],i[1]),i[2],(0,255,0),2)
                                else:  
                                    # draw next circles in red to blue
                                    cv2.circle(event_img,(i[0],i[1]),i[2],(255-j,0,j),2)
                                j += colour_increment

                    showImage("Event", event_img, delay)
                    current_bin = []
                    previous_time_ns = current_time_ns

                    event_perc = 100* len(event_centres)/num_features
                    event_accuracy_histogram.append((ts.to_nsec()-start_time, event_perc))

                    print("Event Camera:", event_perc, "percent of features detected")
 
    # Close the bag file
    bag.close()
    
    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

    # make Graphs
    # trim last datapoint
    event_accuracy_histogram = event_accuracy_histogram[1:]

    print(per_frame_dict)

    # write_histogram_to_text_file(event_accuracy_histogram, bag_file)
    makeGraph([t for (t,y) in event_accuracy_histogram], [[y for (t,y) in event_accuracy_histogram]], ["EVENT"], "Nano seconds from start", "Percentage of features detected (%)", "Event Camera accuracy")
    makeGraph([t for (t,y) in rgb_accuracy_histogram], [[y for (t,y) in rgb_accuracy_histogram]], ["RGB"], "Frame number", "Percentage of features detected (%)","RGB accuracy per frame" )
    
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
    read_and_record_rosbag_data(bag_file, delay, experiment_dict, camera_matrix, distortion_coefficients)