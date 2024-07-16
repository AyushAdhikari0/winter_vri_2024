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
    if duration==0:
        return
    else:
        cv2.imshow(title, image)
        cv2.waitKey(duration)
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

def filter_sift_keypoints(keypoints, mask):

    '''
    Functions:
        Filters the keypoints found by sift
    
    Args:
        keypoints : a list of keypoints.
        mask : an image mask of the circle board.
    
    '''
    
    filtered_keypoints = []
    bad_points = []

    kp_coords = []

    showImage("mask", mask, delay)

    flood_colour = 1

    kp_colour_index_dict = {}

    for keypoint in keypoints:

        x,y = keypoint.pt

        x = round(x)
        y = round(y)

        # print(mask[y,x])

        if int(mask[y,x]) == 255:
            filtered_keypoints.append(keypoint)
            kp_coords.append((x,y))
            cv2.floodFill(image=mask, mask=None, seedPoint=(x,y), newVal=flood_colour)
            kp_colour_index_dict[str(flood_colour)] = keypoint
            flood_colour+=1

        else:
            # filtered_keypoints.append((x,y))
            # kp_colour_index_dict[str(flood_colour)] = keypoint
            bad_points.append((x,y))
            pass

    # labels, unique_labels = dbscan_filter(kp_coords, eps=25)

    # current_cluster = -1

    # for keypoint, cluster in zip(filtered_keypoints.copy(), labels):
    #     x,y = keypoint.pt

    #     if cluster == current_cluster:
    #         filtered_keypoints.remove(keypoint)
    #         # bad_points.append((x,y))
    #     else:
    #         current_cluster = cluster
    #         print(x,y,cluster)

    showImage("mask", mask, 10)
    
    return filtered_keypoints, bad_points

    
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
        centres = hough_feature_detection(canny, colour_img, corners)
    else:
        print("Incorrect feature detection type given")
        return

    if feature_type != 'hough':
        if (filter_toggle == True):
            keypoints, bad_keypoints = filter_sift_keypoints(keypoints, mask)

            print(bad_keypoints)
            print(len(bad_keypoints))
            if len(bad_keypoints) > 0:
                drawPoints(bad_keypoints,output_img)               

        # keypoints, _ = list(kp_colour_index_dict.values())

        output_img = cv2.drawKeypoints(gray_img, keypoints, None)

        showImage(feature_type, output_img, delay)

        return keypoints, mask
    else:
        return centres

def hough_feature_detection(canny, colour_img, corners):
    
    # get circles from canny image, filter by radius
    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, dp=1, minDist=12, param1=50,param2=8,minRadius=3,maxRadius=8)
    
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
        return centres
    return []

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
    canny = cv2.Canny(gray_img, 100, 180)

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
        print("board not found, use the old dimensions")
        corners = previous_corners
    
    return corners

# def dictToList(per_frame_dict):

#     x_values = []
#     y_values = []

#     for frame in per_frame_dict.keys():


#     for frame, acc_dict in per_frame_dict.items():
        
        


