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
