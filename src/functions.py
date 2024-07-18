import rosbag
from sensor_msgs.msg import Image
import cv2
import yaml

from cv_bridge import CvBridge
import time
import lz4.frame
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os
from cv_bridge import CvBridge

from src.feature_detectors import sift_detector, orb_detector, brief_detector
from src.evaluation_metrics import evaluate_number_accuracy_for_frame

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

image_width = config.get("camera").get("image_width")
image_height = config.get("camera").get("image_height")

# Camera intrinsics
fx = config.get("camera").get("fx")
fy = config.get("camera").get("fy")
cx = config.get("camera").get("cx")
cy = config.get("camera").get("cy")

# Camera distortion
k1 =config.get("camera").get("k1")
k2 =config.get("camera").get("k2")
k3 =config.get("camera").get("k3")
p1 =config.get("camera").get("p1")
p2 =config.get("camera").get("p2")

delay = config.get("delay")

show_image_boolean = config.get("show_images")

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
def makeGraph(time_list : list, y_lists : list, series_names : list, x_label : str, y_label : str, title : str):

    plt.figure()

    for y_list in y_lists:
        plt.plot(time_list, y_list)
    
    plt.legend(series_names)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(which='major', color = 'grey', linestyle='-')
    # plt.grid(b=True, which='minor', color='lightgrey', linestyle='--')
    plt.minorticks_on()

    plt.show() 


def showImage(title, image, duration):
    if show_image_boolean:
        if duration==0:
            return
        else:
            cv2.imshow(title, image)
            cv2.waitKey(duration)
            return
    return

def undistort_image (image, camera_matrix, distortion_coefficients):
    return cv2.undistort(image, camera_matrix, distortion_coefficients)

def dbscan_filter(points_list, eps=2, min_samples=1):
    '''
    
    eps : The maximum distance between two samples for one to be considered as in the neighborhood of the other
    min_samples : the number of samples (or total weight) in a neighborhood for a point to be considered as a core point

    
    '''

    points = np.array(points_list)

    # Create and fit the DBSCAN model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)

    # labels = set(labels)

    print(labels)

    # Step 5: Plot the results
    # Get unique labels
    unique_labels = set(labels) - {-1}

    print(unique_labels)
    
    # image stuff here

    return labels, unique_labels

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

def img_from_events(event_bin, camera_matrix, distortion_coefficients):
    blank_image = np.zeros((image_height,image_width), np.uint8)

    for event in event_bin:
        x,y = event.x, event.y
        blank_image[y,x] = 255

    blank_image = undistort_image(blank_image, camera_matrix, distortion_coefficients)

    return blank_image

def filter_border_lines(border):

    '''Function: Filters the lines found by the Hough line detector by sorting into vertical and horizontal,
    then filters almost (co-linear) lines 
    
    Args:
        border: lines detected by cv2.HoughLines()
    
    Returns: 
        border: filtered list
        vert_lines: a subset list of vertical lines
        hori_lines: a subset list of horizontal lines 
    '''
    vert_lines = []
    hori_lines = []

    # filter horizontal and vertical lines
    for line in border:
        rho,theta = line[0]

        if theta < np.pi/4 or theta > 3*np.pi/4:
            vert_lines.append((rho,theta))
        else:
            hori_lines.append((rho,theta))

    vert_lines = filter_lines_intersect(vert_lines)
    hori_lines = filter_lines_intersect(hori_lines)

    border = [[[rho,theta]] for [rho,theta] in vert_lines + hori_lines]
    return border, vert_lines, hori_lines

def drawPoints(points, image, colour=(0,255,0)):

    '''
    Function: draws a list of points on a provided image

    Args: 
        points: a list of points in the form [(x1,y1), (x2,y2) ... ]
        image : cv2 image
    
    '''

    channels = len(image.shape)

    for (x,y) in points:
        x = int(x)
        y = int(y)
        if channels == 2:
            if isinstance(colour, int):
                cv2.circle(image,(x,y),2,colour,3) 
            else:
                cv2.circle(image,(x,y),2,120,3)
        if channels == 3:
            cv2.circle(image,(x,y),2,colour,3)
    return image

def get_circle_grid_mask(corners,r_circs=1):

    a3_width = 42
    a3_height = 29.7

    y_circs = [3.36, 7.36, 11.36, 15.36, 19.36, 23.36, 27.36]
    x_circs = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38]

    y_circs = [int(y/a3_height * 297) for y in y_circs]
    x_circs = [int(x/a3_width * 420) for x in x_circs]
    r_circs = round(r_circs/a3_height *  297)

    # using same aspect ratio of a3 page to make a high def mask
    mask = np.zeros((297, 420, 1), dtype=np.uint8)

    for y in y_circs:
        for x in x_circs:
            cv2.circle(mask, (x,y), r_circs, (255,255,255), -1)
    
    pts1 = np.float32([[0,0],[420,0],[420,297],[0,297]])
    pts2 = np.float32(corners)

    m = cv2.getPerspectiveTransform(pts1,pts2)

    mask = cv2.warpPerspective(mask,m,(image_width,image_height))

    # binarise the mask

    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    drawPoints(corners,mask, colour=254)
    
    return mask

def order_points_clockwise(points):

    # list of points in the form [(x1,y1), (x2,y2) ... ]

    center = (round(image_width/2), round(image_height/2))

    angles = [np.arctan2(point[1] - center[1], point[0] - center[0]) for point in points]
    
    # Order points based on angles (clockwise)
    ordered_points = [point for _, point in sorted(zip(angles, points))]

    return ordered_points

def filter_keypoints_with_mask(keypoints, mask):

    '''
    Functions:
        Filters the keypoints found by sift
    
    Args:
        keypoints : a list of keypoints.
        mask : an image mask of the circle board.
    
    '''
    
    # check if we get keypoints from sift/orb/brief or centres from hough
      
    filtered_keypoints = []
    bad_points = []

    kp_coords = []

    showImage("mask", mask, delay)

    flood_colour = 1

    kp_colour_index_dict = {}

    for keypoint in keypoints:

        # non Hough
        if not type(keypoint) == np.ndarray:
            x,y = keypoint.pt

            x = round(x)
            y = round(y)
            temp = keypoint
        
        # Hough
        else:
            x,y = keypoint[0]
            temp = (x,y)

        if int(mask[y,x]) == 255:
            filtered_keypoints.append(temp) 
            kp_coords.append((x,y))
            cv2.floodFill(image=mask, mask=None, seedPoint=(x,y), newVal=flood_colour)
            kp_colour_index_dict[str(flood_colour)] = keypoint
            flood_colour+=1

        else:
            # filtered_keypoints.append((x,y))
            # kp_colour_index_dict[str(flood_colour)] = keypoint
            bad_points.append((x,y))
            pass

    showImage("mask", mask, 10)
    
    return filtered_keypoints, bad_points


def is_point_in_quadrilateral(point,corners):

    px,py = point

    a,b,c,d = corners

    qx1, qy1 = a
    qx2, qy2 = b
    qx3, qy3 = c
    qx4, qy4 = d

    # Check the signs of cross products
    d1 = cross_product_sign(px, py, qx1, qy1, qx2, qy2)
    d2 = cross_product_sign(px, py, qx2, qy2, qx3, qy3)
    d3 = cross_product_sign(px, py, qx3, qy3, qx4, qy4)
    d4 = cross_product_sign(px, py, qx4, qy4, qx1, qy1)

    # Check if the point is inside the quadrilateral
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0) or (d4 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0) or (d4 > 0)

    return (has_neg and has_pos)
    
def use_feature_detector(feature_type, gray_img, corners, canny= None, colour_img= None, filter_toggle=True):
    '''
    Function:
        Uses the feature detector defined by the type input.
        Also filters the features and returns the number of circles detected 
    
    Args:
        feature_type : a string for selecting the type of feature detection (can be 'sift', 'orb', 'brief')
        gray_img : greyscale image to get features from
        corners : a list of corners [(x1,y1), (x2,y2) ... ] of the board, this is for the mask filtering
    
    Returns:
        num_keypoints : number of features detected
        
    '''
    if feature_type == 'sift':
        keypoints, descriptors, output_img = sift_detector(gray_img)
        mask = get_circle_grid_mask(corners, r_circs=0.8)
    elif feature_type == 'brief':
        keypoints, descriptors, output_img = brief_detector(gray_img)
        mask = get_circle_grid_mask(corners, r_circs=0.8)
    elif feature_type == 'orb':
        keypoints, descriptors, output_img = orb_detector(gray_img)
        mask = get_circle_grid_mask(corners)   
    elif feature_type == 'hough':
        keypoints = hough_feature_detection(gray_img, corners, colour_img)
        mask = get_circle_grid_mask(corners)   
    else:
        print("Incorrect feature detection type given")
        return

    if (filter_toggle == True):
        keypoints, bad_keypoints = filter_keypoints_with_mask(keypoints, mask)
        if feature_type != 'hough':
            # print(bad_keypoints)
            # print(len(bad_keypoints))
            if len(bad_keypoints) > 0:
                drawPoints(bad_keypoints,output_img)               

            output_img = cv2.drawKeypoints(gray_img, keypoints, None)

            showImage(feature_type, output_img, delay)
        else:
            drawHoughCircles(keypoints, colour_img)

        return keypoints, mask
    return keypoints, mask

# defined new function for canny for default values
def getCanny(gray_img, min_thresh = 100, max_thresh=180):
    return cv2.Canny(gray_img, min_thresh, max_thresh)



def hough_feature_detection(canny, corners, colour_img):

    # get circles from canny image, filter by radius

    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, dp=1, minDist=12, param1=50,param2=8,minRadius=3,maxRadius=8)

    if circles is None:
        return []

    centres = [(i[0],i[1]) for i in circles[0,:]]

    filter_centres(centres, corners)

    centres = np.array(centres, dtype=np.int32).reshape((-1, 1, 2))

    return centres
    
    # # showImage("Hough Canny", canny, delay)
    
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     colour_increment = 255/len(circles[0,:])

    #     j = 0
    #     centres = []

    #     for i in circles[0,:]:
    #         # filter points that are close to the corners and outside the board


    #         else:
    #             # draw circle points    
    #             cv2.circle(colour_img,(i[0],i[1]),2,(0,255,0),3)
    #             centres.append((i[0], i[1]))  
    #             # draw the outer circle
    #             if j == 0:
    #                 # draw first circle in green
    #                 cv2.circle(colour_img,(i[0],i[1]),i[2],(0,255,0),2)
    #             else:  
    #                 # draw next circles in red to blue
    #                 cv2.circle(colour_img,(i[0],i[1]),i[2],(255-j,0,j),2)
    #             j += colour_increment

    #     centres = np.array(centres, dtype=np.int32).reshape((-1, 1, 2))
    #     return centres
    # return []

def get_processed_images(msg, camera_matrix, distortion_coefficients, desired_encoding='bgr8'):

        # Initialize the CvBridge class
    bridge = CvBridge()

    colour_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    colour_img = undistort_image(colour_img, camera_matrix, distortion_coefficients)

    gray_img = cv2.cvtColor(colour_img, cv2.COLOR_RGB2GRAY)

    _, binarised_img = cv2.threshold(gray_img, thresh=50, maxval=255, type=cv2.THRESH_BINARY)
    # binarised_img = cv2.adaptiveThreshold(gray_img,maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
    #                                          thresholdType = cv2.THRESH_BINARY, blockSize=11, C=2)

    # # blur image
    # blurred_img = cv2.medianBlur(binarised_img,5)

    # canny edge detection

    canny = getCanny(gray_img)

    return colour_img, gray_img, canny

def get_corners(canny, colour_img, previous_corners):
            
                        # get lines from image
    border = cv2.HoughLines(canny, 1,np.pi/180, 100)

    if border is not None:

        border, vert_lines, hori_lines = filter_border_lines(border)
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

            corners = order_points_clockwise(corners)
            previous_corners = corners

    else:
        # print("board not found, use the old dimensions")
        corners = previous_corners
    
    return corners

def write_results_dict_to_text_file(per_frame_dict, bag_file,event_flag=False):

    y_averages_list=[]

    # print(per_frame_dict)

    if event_flag:
        results_suffix = "_evt.txt"
    else:
        results_suffix = "_rgb.txt"

    # file path is same as bag files
    txt_file_name = bag_file[:-4]+ results_suffix

    headings = []

    for _, result in per_frame_dict.items():
        separator = ','
        headings = list(result.keys())
        headings_string = separator.join(headings)
        headings_string = "time," + headings_string + "\n" if event_flag else "frame," + headings_string + "\n"
        # print(headings)
        break

    # open the file
    with open(txt_file_name, "w") as myFile:
        #initialise headings

        myFile.write(headings_string)

        for frame, result in per_frame_dict.items():
           
            currentLine = str(frame) 

            for heading in headings:
                currentLine = currentLine + ',' + str(result[heading])
            
            currentLine += '\n'
            myFile.write(currentLine)

    myFile.close()

def makeGraphFromTextFile(file_location, make_graph_boolean=False, x_axis_index=0,y_axes_indices=None, lighting_condition=None, event_flag=None):

    txt_file_name = file_location[:-4]+".txt"

    headings_string = ""

    with open(txt_file_name, "r") as myFile:
        headings_string = myFile.readline()
        headings_list = convert_csv_line_to_list(headings_string)

        if y_axes_indices == None:
            y_axes_indices = [x_axis_index+1, len(headings_list)-1]

        y_headings_list = headings_list[y_axes_indices[0]:y_axes_indices[1]+1]
        x_heading = headings_list[x_axis_index]

        # print(headings_list)
        # print(currentLine)

        x_list = []
        y_lists = []

        for i in range(len(y_headings_list)):
            y_lists.append([])

        for currentLine in myFile:

            # get rid of newline operator in headings list
            line_list = convert_csv_line_to_list(currentLine)

            if (line_list[1] == lighting_condition or lighting_condition==None):
            
                for i in range(len(line_list)):
                    # print(line_list, i, line_list[i],list(range(y_axes_indices[0], y_axes_indices[1]+1)))
                    if i == x_axis_index:
                        x_list.append(float(line_list[i]))
                    elif i in range(y_axes_indices[0], y_axes_indices[1]+1):
                        print(i-y_axes_indices[0], float(line_list[i]))
                        y_lists[i-y_axes_indices[0]].append(float(line_list[i]))
                    # time.sleep(1)

    myFile.close()

    # print(x_list)
    # print(y_lists)
    # print(len(x_list))
    # print(len(y_lists))
    # print(range(y_axes_indices[0], y_axes_indices[1]))

    # sort x and y_lists in ascending x list

    zipped_lists = [list(zip(x_list, y_list)) for y_list in y_lists]

    # Sort the zipped lists based on the x values
    sorted_zipped_lists = [sorted(zipped_list, key=lambda x: x[0]) for zipped_list in zipped_lists]

    # Unzip the sorted lists
    x_list = [x for x, _ in sorted_zipped_lists[0]]
    y_lists = [[y for _, y in sorted_zipped_list] for sorted_zipped_list in sorted_zipped_lists]

    if make_graph_boolean:

        event_tag = ' for the DVS Event Camera' if event_flag else ' for the DVS RGB Camera'

        lighting_condition_dict = {'00' : 'No lights' + event_tag,
                                   '0' : 'No lights, colocated light at 1%'+event_tag,
                                   '01' : 'Ra at 1% power'+event_tag,
                                   '50' : 'Ra at 50 % power'+event_tag,
                                   '100' : 'Ra at full 100% power'+event_tag}

        makeGraph(x_list, y_lists, y_headings_list, x_heading+'time (nanoseconds)', 'percentage accuracy (%)', lighting_condition_dict[lighting_condition])
    
    y_averages = [headings_list[1:], [sum(y_list) / len(y_list) for y_list in y_lists]]

    print(y_averages)
    return y_averages

        
def cross_product_sign(x1, y1, x2, y2, x3, y3):
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

def convert_csv_line_to_list(line_string):
    line_list = line_string.split(',')
    line_list[-1] = line_list[-1][:-1]
    return line_list

def create_results_dict(feature_detect_toggle, frame_counter, gray_img, corners, colour_image, event_flag=False):
    
    results_dict = {}

    if feature_detect_toggle.get('sift'):
        keypoints, mask = use_feature_detector('sift', gray_img, corners, filter_toggle=True)
        results_dict['sift_accuracy'] = evaluate_number_accuracy_for_frame(keypoints) 

    if feature_detect_toggle.get('orb'):
        keypoints, mask = use_feature_detector('orb', gray_img, corners, filter_toggle=True)
        results_dict['orb_accuracy'] = evaluate_number_accuracy_for_frame(keypoints) 

    if feature_detect_toggle.get('brief'):
        keypoints, mask = use_feature_detector('brief', gray_img, corners, filter_toggle=True)
        results_dict['brief_accuracy'] = evaluate_number_accuracy_for_frame(keypoints) 
        
    if feature_detect_toggle.get('hough'):
        if event_flag:
            centres, _ = use_feature_detector('hough', gray_img, corners, filter_toggle=True, colour_img=colour_image)
        else:
            centres, _ = use_feature_detector('hough', getCanny(gray_img), corners, filter_toggle=True, colour_img=colour_image)
        results_dict['hough_accuracy'] = evaluate_number_accuracy_for_frame(centres) 

    return results_dict


def drawHoughCircles(centres, colour_img):

    if (len(centres) == 0):
        return 

    display_image = colour_img

    if centres is not None:
        centres = np.uint16(np.around(centres))
        colour_increment = 255/len(centres[0,:])

        j = 0

        for i in centres:

            # draw circle points    
            cv2.circle(display_image,(i[0],i[1]),2,(0,255,0),3)
            # draw the outer circle
            if j == 0:
                # draw first circle in green
                cv2.circle(display_image,(i[0],i[1]),6,(0,255,0),2)
            else:  
                # draw next circles in red to blue
                cv2.circle(display_image,(i[0],i[1]),6,(255-j,0,j),2)
            j += colour_increment
    
    showImage("Event Hough", display_image, delay)


def filter_centres(centres, corners):

    for (x,y) in centres.copy():

        if is_point_in_quadrilateral((x,y), corners):
            centres.remove((x,y))
            continue

        for (cx,cy) in corners:
            if isClose2D((x,y),(cx,cy), 15):
                centres.remove((x,y))

def show_all_graphs_per_lighting_condition_from_results_file(directory, results_file_name, lighting_conditions=['00','0','01','50','100']):

    if results_file_name not in {'results_rgb.txt', 'results_evt.txt'}:
        print("Incorrect results file name input")
        return
    
    event_flag = (results_file_name[-7:] == "evt.txt")

    for lighting_condition in lighting_conditions:
        makeGraphFromTextFile(directory+results_file_name,x_axis_index=2,y_axes_indices=[3,6],make_graph_boolean=True,lighting_condition=lighting_condition, event_flag=event_flag)

    # rgb results

def get_lighting_and_exposure(rosbag_name):
    split_list = rosbag_name.split('_')

    for i in range(len(split_list)):
        if i == 1:
            lighting = split_list[i]
        elif split_list[i-1] == 'exp':
            exposure = split_list[i:]
        
    exposure = ''.join(exposure)
    # cut off .txt suffix and '_evt' or '_rgb' flag
    exposure = exposure[:-8]

    # print(lighting, exposure)
    # quit()
        
    return lighting, exposure

def record_y_averages_in_results_file(directory, bag_name, y_averages, event_flag=False):

    result_txt_file_name = 'results_evt.txt' if event_flag else 'results_rgb.txt'
    record_path = directory+result_txt_file_name

    if not os.path.exists(record_path):
        print("The results file does not exist. Initialising results file")

        with open(record_path, "w") as myFile:
            header_line = "rosbag_name,lighting,exposure," + ','.join(y_averages[0])+'\n'
            myFile.write(header_line)
        myFile.close()

    with open(record_path, "r+") as myFile:

        for line in myFile:
            pass
        lighting, exposure = get_lighting_and_exposure(bag_name)
        result_line = bag_name + ',' + lighting +',' + exposure +',' + ','.join([str(num) for num in y_averages[1]])+'\n'
        myFile.write(result_line)
    myFile.close()





