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
import string

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

topics_dict = {'image' : '/dvs/image_raw', 
                'tf'  : '/tf',
                'event' : '/dvs/events'}

feature_detect_toggle = {'hough' : 1,
                        'circlesGrid' : 0}

delay = 10

event_buffer_time_ns = 1e9
previous_time_ns = 0
current_time_ns = 1

previous_corners = []

def read_images_from_rosbag(bag_file):
    # Initialize the CvBridge class
    bridge = CvBridge()
    
    # Open the ROS bag file
    bag = rosbag.Bag(bag_file, 'r')

    # print(type(bag))
    
    # Iterate through the messages in the specified topic
    for topic, msg, t in bag.read_messages(topics=topics_dict.values()):

        if (topic == topics_dict.get('image')):
            # Convert the ROS Image message to an OpenCV image
            colour_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            colour_img = undistort_image(colour_img)

            
            # grayscale 
            gray_img = cv2.cvtColor(colour_img, cv2.COLOR_RGB2GRAY)

            # binarise image

            # _, binarised_img = cv2.threshold(gray_img, thresh=50, maxval=255, type=cv2.THRESH_BINARY)
            binarised_img = cv2.adaptiveThreshold(gray_img,maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                                     thresholdType = cv2.THRESH_BINARY, blockSize=11, C=2)

            # cv2.imshow("Image", binarised_img)

            # quit()
            # blur image

            # blurred_img = cv2.medianBlur(binarised_img,5)

            canny = cv2.Canny(gray_img, 100, 200)

            border = cv2.HoughLines(canny, 1,np.pi/180, 100)

            # border = filter_lines(border)

            print(border)

            # time.sleep(4)

            # border now has lines in the form of start and end points.
            drawLines(border, colour_img)   

            if border is not None and len(border) == 4: 

                # border = [(line[0][0], line[0][1]) if line[0][1] < 2.35 else (line[0][0], line[0][1]-np.pi) for line in border ]
                vert_lines = []
                hori_lines = []

                for line in border:
                    rho,theta = line[0]

                    # filter horizontal and vertical lines
                    if theta < np.pi/4 or theta > 3*np.pi/4:
                        vert_lines.append((rho,theta))
                    else:
                        hori_lines.append((rho,theta))


                    # print(x,y)
                    # if theta > 270 degrees:
                    #     invert theta
                
                # border = sorted(border, key=lambda x:x[0][1])
                print(border)
                print("H",hori_lines)
                print("V", vert_lines)

                corners = []    
                
                # get corners from horizontal and vertical line intersects
                for h_line in hori_lines:
                    for v_line in vert_lines:
                        corners.append(getIntersect(h_line,v_line))

            else:
                print("board not found, use the old dimensions")

                if 'corners' not in locals():
                    corners = previous_corners
                    print("yeeet", corners)

            print("corners", corners)
            for (x,y) in corners:
                x = int(x)
                y = int(y)
                # print(y)
                cv2.circle(colour_img,(x,y),2,(0,255,0),3)



            # corners = np.array(corners,dtype=np.int32).reshape((-1,1,2))
            # print(corners)

            # # Create a mask with the same dimensions as the image
            # mask = np.zeros(colour_img.shape[:2], dtype=np.uint8)

            # # Fill the quadrilateral on the mask with white color
            # cv2.fillPoly(mask, [corners], 255)

            # # Extract the desired region using the mask
            # cropped_image = cv2.bitwise_and(colour_img, colour_img, mask=mask)

            # # Create a white background to place the cropped region
            # background = np.zeros_like(colour_img)
            # background.fill(255)

            # # Extract only the region defined by the quadrilateral
            # result = cv2.bitwise_and(background, background, mask=mask)
            # result[mask == 255] = cropped_image[mask == 255]

            # # Display the original and the cropped image
            # plt.figure(figsize=(12, 6))
            # plt.subplot(121), plt.imshow(cv2.cvtColor(colour_img, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
            # plt.subplot(122), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title('Cropped Image')
            # plt.show()
                        
            input_img = canny

            # circleGrid method

            if feature_detect_toggle.get('circlesGrid'):

                _, centres = cv2.findCirclesGrid(gray_img, pattern)

                # print(centres)

                if centres is not None:
                    for point in centres:
                        cv2.circle(colour_img, (int(point[0][0]), int(point[0][1])), radius=5, color=(0, 0, 255), thickness=-1)  # Red points with radius 5
                else:
                    print("No circles found : circlesGrid method")
            # cv2.imshow("image", colour_img)
            # cv2.waitKey(delay)

            
            if feature_detect_toggle.get('hough'):

                               
                # canny param
                circles = cv2.HoughCircles(input_img, cv2.HOUGH_GRADIENT, dp=1, minDist=12, param1=50,param2=8,minRadius=3,maxRadius=8)
                
                cv2.imshow("Canny", canny)
                cv2.waitKey(delay)

                # quit()

                # quit()
                
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    colour_increment = 255/len(circles[0,:])

                    # filtered_points = dbscan_filter(circles)

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
                    
                            # centres = np.append(centres,(i[0], i[1]))
                            # print(i)

                    # print('centres', centres)

                    centres = np.array(centres, dtype=np.int32).reshape((-1, 1, 2))




                    # lines = cv2.HoughLinesPointSet(centres,
                    #                    lines_max=10,\
                    #                    threshold=1,\
                    #                     min_rho=0, \
                    #                     max_rho=300, \
                    #                     rho_step=1, \
                    #                     min_theta = 0, \
                    #                     max_theta = np.pi/8,\
                    #                     theta_step = np.pi/360)
                    
                    # print(lines)

                    # Draw lines
                    # if lines is not None:
                    #     for line in lines:
                    #         rho, theta = line[0][0], line[0][1]
                    #         a = np.cos(theta)
                    #         b = np.sin(theta)
                    #         x0 = a * rho
                    #         y0 = b * rho
                    #         x1 = int(x0 + 1000 * (-b))
                    #         y1 = int(y0 + 1000 * (a))
                    #         x2 = int(x0 - 1000 * (-b))
                    #         y2 = int(y0 - 1000 * (a))
                    #         cv2.line(colour_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # print("num_circles", circles[0,:])    
                    # cv2.imshow("Image", input_img)
                    # cv2.waitKey(delay)

            cv2.imshow("Colour", colour_img)
            cv2.waitKey(delay)

        elif (topic == topics_dict.get('tf')):
            # print((topic, t, '\b  n'))
            pass

        elif (topic == topics_dict.get('event')):
            
            binned_events = []
            current_bin = []

            # sort all events into bins of length 
            for event in msg.events:
                x,y,ts, polarity = event.x, event.y, event.ts, event.polarity

                global current_time_ns
                global previous_time_ns

                current_time_ns = ts.to_nsec()
                current_bin.append(event)
                # print(current_time_ns)


                if (current_time_ns - previous_time_ns > event_buffer_time_ns):
                    print(current_bin)
                    binned_events.append(current_bin)
                    event_img = img_from_events(current_bin)
                    # event_img = cv2.Canny(event_img, 100, 200)
                    circles = cv2.HoughCircles(event_img, cv2.HOUGH_GRADIENT, dp=1, minDist=12, param1=50,param2=8,minRadius=3,maxRadius=8)

                    event_img = cv2.cvtColor(event_img, cv2.COLOR_GRAY2RGB)
                    if circles is not None:

                        j = 0
                        colour_increment = 255/len(circles[0,:])

                        for i in circles[0,:]:
                        # filter points that are close to the corners 
                        # for (x,y) in corners:
                        #     print("hi")
                        #     if isClose2D((i[0],i[1]),(x,y), 15):
                        #         break
                        # else:
                        # draw circle points 
                            print(i)
                            i = [int(a) for a in i]   
                            cv2.circle(event_img,(i[0],i[1]),2,(0,255,0),3)
                            # centres.append((i[0], i[1]))  
                            # draw the outer circle
                            if 1 == 2:
                                # draw first circle in green
                                cv2.circle(event_img,(i[0],i[1]),i[2],(0,255,0),2)
                            else:  
                                # draw next circles in red to blue
                                cv2.circle(event_img,(i[0],i[1]),i[2],(255-j,0,j),2)
                            j += colour_increment
                  

                    cv2.imshow("Event", event_img)
                    # cv2.waitKey(2000) 
                    current_bin = []
                    previous_time_ns = current_time_ns
                    # time.sleep(1)


                # print(x,y,ts,polarity)
                print(current_time_ns)
                # print("{{{^}}}")
                pass


            print((topic, t, '\n'))


        # match (topic):
        #     case image_topic:
                
                
        #     case tf_topic:
        #         print(('topic:', topic,',',t, '\n'))
        
            
    # Close the bag file
    bag.close()
    
    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

# def decompress_lz4_file(input_file_path, output_file_path):
#     # Open the compressed file in binary read mode
#     with open(input_file_path, 'rb') as compressed_file:
#         # Open the output file in binary write mode
#         with open(output_file_path, 'wb') as decompressed_file:
#             # Create a decompressor object
#             decompressor = lz4.frame.LZ4FrameDecompressor()
#             # Read and decompress the file in chunks
#             for chunk in iter(lambda: compressed_file.read(4096), b''):
#                 decompressed_file.write(decompressor.decompress(chunk))

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

def drawLines(lines, image):

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

def filter_lines (border_lines):

    lines = border_lines[0]
    print("BORDR", lines)

    for i in border_lines:
        print("YUUH", i)
    # lines = border_lines[0]
    # print("INDXED", lines)

    filtered_lines = []


    rho_thresh = 1
    theta_thresh = 3

    # rho_thresh = 300
    # theta_thresh = 20 * np.pi / 180

    remove_set = set()
    if border_lines[0] is not None:
        for ((rho1, theta1)) in border_lines:

            m1 = -np.cos(theta1)/np.sin(theta1)
            c1 = rho1 / np.sin(theta1)

            for [(rho2, theta2)] in border_lines[0]:

                m2 = -np.cos(theta2)/np.sin(theta2)
                c2 = rho2 / np.sin(theta2)

                if isClose(m1,m2, theta_thresh) and isClose(c1,c2, rho_thresh) and (rho1, theta1) != (rho2, theta2):
                    if (rho1,theta1) in remove_set or (rho2,theta2) in remove_set:
                        pass
                    else:
                        print("added to remove set:", (rho1,theta1))
                        remove_set.add((rho1,theta1))
    
    # remove the lines

    print("TO REMOVE", remove_set)
    print("LINES", lines)
    print("FULL LIST", lines)

    for [rho, theta] in border_lines[0]:
        if (rho,theta) not in remove_set:
            filtered_lines.append([rho,theta])
            print("added to filtered lines")

    print([filtered_lines])

    return [filtered_lines]


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
    bag_file = '/home/ayush/Data/tst1.bag'

    # decompress_lz4_file(lz4_file, bag_file)
  
    read_images_from_rosbag(bag_file)