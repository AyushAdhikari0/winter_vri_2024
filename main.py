# test
# /c/Users/yush7/Desktop/satellite project files
import argparse
import sys
import rosbag
import cv2
from cv_bridge import CvBridge
import numpy as np
from src.utils import parse_rosbag_file
from src.yaml_utils import load_camera_intrinsics, check_yaml_file_exists, load_ros_topic_names, load_parameters_from_yaml
from src.functions import makeGraph, drawLinesPolar, filter_lines_intersect, img_from_events, showImage, undistort_image, getIntersect, isClose2D, isClose

def read_images_from_rosbag(bag_file, delay, experiment_dict, camera_matrix, distortion_coefficients):

    pattern = experiment_dict["pattern"]
    num_features = pattern[0] * pattern[1]


    feature_detect_toggle = experiment_dict["feature_toggle"]

    # event camera characteristics
    event_buffer_time_ns = experiment_dict["event_buffer_time_ns"]

    # Initialize the CvBridge class
    bridge = CvBridge()

    # storing the corners as detected from Hough Line transforms
    previous_corners = []

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
    
    # Iterate through the messages in the specified topic
    for topic, msg, t in bag.read_messages(topics=topics_dict.values()):

        if (topic == topics_dict.get('image')):
            # Convert the ROS Image message to an OpenCV image
            colour_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            colour_img = undistort_image(colour_img, camera_matrix, distortion_coefficients)

            
            # grayscale 
            gray_img = cv2.cvtColor(colour_img, cv2.COLOR_RGB2GRAY)

            # #binarise image

            # _, binarised_img = cv2.threshold(gray_img, thresh=50, maxval=255, type=cv2.THRESH_BINARY)
            # binarised_img = cv2.adaptiveThreshold(gray_img,maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            #                                          thresholdType = cv2.THRESH_BINARY, blockSize=11, C=2)

            # # blur image
            # blurred_img = cv2.medianBlur(binarised_img,5)

            # canny edge detection
            canny = cv2.Canny(gray_img, 100, 180)

            # get lines from image
            border = cv2.HoughLines(canny, 1,np.pi/180, 100)

            if border is not None:

                vert_lines = []
                hori_lines = []

                # filter horizontal and vertical lines
                for line in border:
                    rho,theta = line[0]

                    if theta < np.pi/4 or theta > 3*np.pi/4:
                        vert_lines.append((rho,theta))
                    else:
                        hori_lines.append((rho,theta))

                # filter duplicate intersecting lines in horizontal and vertical sets
                if len(border) != 4:
                    vert_lines = filter_lines_intersect(vert_lines)
                    hori_lines = filter_lines_intersect(hori_lines)

       
                border = [[[rho,theta]] for [rho,theta] in vert_lines + hori_lines]

                drawLinesPolar(border, colour_img)   

                corners = []    

                # get corners from horizontal and vertical line intersects
                for h_line in hori_lines:
                    for v_line in vert_lines:
                        corners.append(getIntersect(h_line,v_line))

                # if you get 4 corners, use those, otherwise use previous corners        
                if (len(corners) != 4):
                    corners = previous_corners
                else:
                    previous_corners = corners

            else:
                print("board not found, use the old dimensions")
                corners = previous_corners

            # draw corners
            for (x,y) in corners:
                x = int(x)
                y = int(y)
                cv2.circle(colour_img,(x,y),2,(0,255,0),3)
     
            # circleGrid method

            if feature_detect_toggle.get('circlesGrid'):

                _, centres = cv2.findCirclesGrid(gray_img, pattern)

                # print(centres)

                if centres is not None:
                    for point in centres:
                        cv2.circle(colour_img, (int(point[0][0]), int(point[0][1])), radius=5, color=(0, 0, 255), thickness=-1)  # Red points with radius 5
                else:
                    print("No circles found : circlesGrid method")

            input_img = canny
            
            if feature_detect_toggle.get('hough'):
                               
                # get circles from canny image, filter by radius
                circles = cv2.HoughCircles(input_img, cv2.HOUGH_GRADIENT, dp=1, minDist=12, param1=50,param2=8,minRadius=3,maxRadius=8)
                
                showImage("RGB Canny", canny, delay)
                
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    colour_increment = 255/len(circles[0,:])

                    j = 0
                    centres = []

                    for i in circles[0,:]:
                        # filter points that are close to the corners 
                        for (x,y) in corners:
                            if isClose2D((i[0],i[1]),(x,y), 15):
                                break
                        else:
                            # draw circle points    
                            cv2.circle(colour_img,(i[0],i[1]),2,(0,255,0),3)
                            centres.append((i[0], i[1]))  
                            # draw the outer circle
                            if j == 0:
                                # draw first circle in green
                                cv2.circle(colour_img,(i[0],i[1]),i[2],(0,255,0),2)
                            else:  
                                # draw next circles in red to blue
                                cv2.circle(colour_img,(i[0],i[1]),i[2],(255-j,0,j),2)
                            j += colour_increment

                    centres = np.array(centres, dtype=np.int32).reshape((-1, 1, 2))

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
                    previous_time_ns = start_time
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
                    event_accuracy_histogram.append((start_time - ts.to_nsec(), event_perc))
            
                    print("Event Camera:", event_perc, "percent of features detected")
                    # time.sleep(1)
 
            
    # Close the bag file
    bag.close()
    
    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

    # make Graphs
    # trim last datapoint
    event_accuracy_histogram = event_accuracy_histogram[1:]
    makeGraph(event_accuracy_histogram, ["EVENT"], "Nano seconds from start", "Percentage of features detected (%)", "Event Camera accuracy")
    makeGraph(rgb_accuracy_histogram, ["RGB"], "Frame number", "Percentage of features detected (%)","RGB accuracy per frame" )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a path to a rosbag file.")
    parser.add_argument('rosbag_path', type=parse_rosbag_file, help='Path to a rosbag file')
    args = parser.parse_args()
    bag_file = args.rosbag_path
    print(f"Path to rosbag file: {bag_file}")

    # load config file
    config_file = "cfg.yaml"

    if not check_yaml_file_exists(config_file):
        print(f"Error: The file '{config_file}' does not exist.")
        sys.exit(1)

    camera_matrix, distortion_coefficients, image_size = load_camera_intrinsics(config_file)
    topics_dict = load_ros_topic_names(config_file)
    experiment_dict = load_parameters_from_yaml(config_file, ["pattern", "event_buffer_time_ns", "feature_toggle"])

    # delay for how long the images are displayed for 
    delay = 100


    
    


    
    # read in images and undistort

    # read in events, convert to images, and undistort

    # apply feature detector to images and capture results

    # apply feature detector to event images and capture results


    # bag_file = 'C:/Users/yush7/Desktop/vri_files_2024/data/calib/hi2.bag'
    # lz4_file = 'C:/Users/yush7/Desktop/vri_files_2024/data/hi2.bag'
    # bag_file = 'C:/Users/yush7/Desktop/satellite project files/data/test1.bag'
    # bag_file = 'C:/Users/yush7/Desktop/satellite project files/data/calib/calibration_data.bag'
    # bag_file = '/home/ayush/Data/tst2.bag'

  
    read_images_from_rosbag(bag_file, delay, experiment_dict, camera_matrix, distortion_coefficients)