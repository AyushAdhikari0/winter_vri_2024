# test
# /c/Users/yush7/Desktop/satellite project files

import rosbag
from sensor_msgs.msg import Image
import cv2

from cv_bridge import CvBridge
import time
import lz4.frame
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

image_width =346
image_height =260

# Camera intrinsics
fx = 298.67179975
fy =298.59720232
cx =176.38660484
cy =119.96968706

# Camera distortion
k1 =-0.37074772
k2 =0.14798075
k3 =0.0
p1 =0.00251884
p2 =-0.00170605

pattern = (10,7)
num_features = pattern[0] * pattern[1]

topics_dict = {'image' : '/dvs/image_raw', 
                'tf'  : '/tf',
                'event' : '/dvs/events'}

feature_detect_toggle = {'hough' : 1,
                        'circlesGrid' : 0}

# delay for how long the images are displayed for 
delay = 100

# event camera characteristics
event_buffer_time_ns = 5e8
previous_time_ns = 0
current_time_ns = 1

# storing the corners as detected from Hough Line transforms
previous_corners = []

# list of [(frame, accuracy)]
rgb_accuracy_histogram = []

# list of [(time since start, accuracy)]
event_accuracy_histogram = []

def read_images_from_rosbag(bag_file):
    # Initialize the CvBridge class
    bridge = CvBridge()
    
    # Open the ROS bag file
    bag = rosbag.Bag(bag_file, 'r')

    # event camera bins (separated by the buffer period)
    binned_events = []
    current_bin = []

    # bool to check for first message
    first_message = True

    # frame counter for RGB cam
    frame_counter = 0

    global event_accuracy_histogram
    global rgb_accuracy_histogram
    
    # Iterate through the messages in the specified topic
    for topic, msg, t in bag.read_messages(topics=topics_dict.values()):

        if (topic == topics_dict.get('image')):
            # Convert the ROS Image message to an OpenCV image
            colour_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            colour_img = undistort_image(colour_img)

            
            # grayscale 
            gray_img = cv2.cvtColor(colour_img, cv2.COLOR_RGB2GRAY)

            # #binarise image

            # _, binarised_img = cv2.threshold(gray_img, thresh=50, maxval=255, type=cv2.THRESH_BINARY)
            # binarised_img = cv2.adaptiveThreshold(gray_img,maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            #                                          thresholdType = cv2.THRESH_BINARY, blockSize=11, C=2)

            # # blur image
            # blurred_img = cv2.medianBlur(binarised_img,5)

            # canny edge detection
            canny = cv2.Canny(gray_img, 50, 200)

            # get lines from image
            border = cv2.HoughLines(canny, 1,np.pi/180, 100)

            global previous_corners

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
                    first_message = False

                global current_time_ns
                global previous_time_ns

                current_time_ns = ts.to_nsec()
                current_bin.append(event)
                # print(current_time_ns)

                if (current_time_ns - previous_time_ns > event_buffer_time_ns):
                    binned_events.append(current_bin)

                    # create image from current event bin once the buffer is exceeded
                    event_img = img_from_events(current_bin)

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

"""
Function: makes a graph using the matplotlib library's pyplot class.

Parameters:
time_list   : a list of timestamps
y_lists     : a list of y-variable lists to plot, e.g. [acc_x, acc_y, acc_z] or [int_temp]    
series_names: a list of series name strings in the same order as 'y_lists', e.g. ["x-direction", "y-direction", "z-direction"]
x_label     : a string of the x axis label.
y_label     : a string of the y axis label
title       : a string of the title
"""
def makeGraph(data_list : list, series_names : list, x_label : str, y_label : str, title : str):

    plt.figure()

    x_list = []
    y_list = []

    for (x,y) in data_list:
        x_list.append(x)
        y_list.append(y)

    # for y_list in y_lists:
    #     plt.plot(time_list, y_list)

    plt.plot(x_list, y_list)
    
    plt.legend(series_names)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # plt.grid(b=True, which='major', color='grey', linestyle='-')
    # plt.grid(b=True, which='minor', color='lightgrey', linestyle='--')
    plt.minorticks_on() 
    plt.show()

def showImage(title, image, duration):
    if duration==0:
        return
    else:
        cv2.imshow(title, image)
        cv2.waitKey(duration)
        return

def undistort_image (image):
    camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]])
    
    # Create the distortion coefficients array
    dist_coeffs = np.array([k1, k2, p1, p2, k3])

    return cv2.undistort(image, camera_matrix, dist_coeffs)

def dbscan_filter(circles):

    points = np.array([[a,b] for (a,b,c) in circles[0,:]])

    eps = 20    # The maximum distance between two samples for one to be considered as in the neighborhood of the other
    min_samples = 8  # The number of samples (or total weight) in a neighborhood for a point to be considered as a core point

    # Create and fit the DBSCAN model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)

    labels = set(labels)

    # Step 5: Plot the results
# Get unique labels
    unique_labels = set(labels)

    print(unique_labels)
# Create a color map for different clusters
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    # Plot each cluster
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = points[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=6)


    # get indexes to delete from original list

    # delete outliers from original list

    return 

def drawLinesPolar(lines, image):

    point_lines = []

    if lines is not None:
        for line in lines:
            rho, theta = line[0][0], line[0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            if len(image.shape) == 2:
                cv2.line(image, (x1, y1), (x2, y2), 255, 2)
            elif len(image.shape) == 3:
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            point_lines.append([(x1,y1),(x2,y2)])

    return point_lines

# filters a list of lines based on if they intersect inside the frame
def filter_lines_intersect(vert_lines):

    remove_set = set()

    for line1 in vert_lines:
        for line2 in vert_lines:
            # not the same line
            if line1!= line2:

                #not parallel
                if getIntersect(line1,line2) != None:
                    x,y = getIntersect(line1,line2)

                    # intersect in the image dimensions
                    if abs(x) < image_width *1.5 and abs(y) < image_height*1.5:

                        #not already in remove list
                        if (line1 not in remove_set) and (line2 not in remove_set):
                            remove_set.add(line2)

    output = list(set(vert_lines).difference(remove_set)) 

    return output

def isClose(v1,v2,thresh):
    return (abs(v1 - v2) <= thresh)

def isClose2D(coord1,coord2,thresh):
    x1,y1 = coord1
    x2,y2 = coord2
    return ((x1-x2)**2 + (y1-y2)**2 <= thresh**2)
        
# gets the interection point of two lines in the form rho, theta
def getIntersect (line1, line2):
    
    a1,b1,c1 = polar_to_cartesian(line1[0],line1[1])
    a2,b2,c2 = polar_to_cartesian(line2[0],line2[1])

    # Create coefficient matrix and constant vector
    A = np.array([[a1, b1], [a2, b2]])
    B = np.array([c1, c2])
    
    # Check if lines are parallel
    if np.linalg.det(A) == 0:
        return None  # Lines are parallel, no intersection or infinite intersections
    
    # Solve the system of linear equations
    x, y = np.linalg.solve(A, B)
    return [x, y]
     

def polar_to_cartesian(rho, theta):
    """Convert polar coordinates (rho, theta) to Cartesian form coefficients (a, b, c) for the line equation ax + by = c."""
    a = np.cos(theta)
    b = np.sin(theta)
    c = rho
    return a, b, c

def img_from_events(event_bin):
    blank_image = np.zeros((image_height,image_width), np.uint8)

    for event in event_bin:
        x,y = event.x, event.y
        blank_image[y,x] = 255

    blank_image = undistort_image(blank_image)

    return blank_image


if __name__ == "__main__":
    
    # bag_file = 'C:/Users/yush7/Desktop/vri_files_2024/data/calib/hi2.bag'
    # lz4_file = 'C:/Users/yush7/Desktop/vri_files_2024/data/hi2.bag'
    # bag_file = 'C:/Users/yush7/Desktop/satellite project files/data/test1.bag'
    # bag_file = 'C:/Users/yush7/Desktop/satellite project files/data/calib/calibration_data.bag'
    bag_file = '/home/ayush/Data/tst2.bag'

    # decompress_lz4_file(lz4_file, bag_file)
  
    read_images_from_rosbag(bag_file)