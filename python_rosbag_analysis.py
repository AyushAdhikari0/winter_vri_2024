import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import lz4.frame
import numpy as np

def read_images_from_rosbag(bag_file, topic_dict):
    # Initialize the CvBridge class
    bridge = CvBridge()
    
    # Open the ROS bag file
    bag = rosbag.Bag(bag_file, 'r')

    print(type(bag))
    
    # Iterate through the messages in the specified topic
    for topic, msg, t in bag.read_messages(topics=topic_dict.values()):


        if (topic == topic_dict.get('image')):
            # Convert the ROS Image message to an OpenCV image
            colour_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # grayscale 
            gray_img = cv2.cvtColor(colour_img, cv2.COLOR_RGB2GRAY)

            # binarise image

            # _, binarised_img = cv2.threshold(gray_img, thresh=50, maxval=255, type=cv2.THRESH_BINARY)
            # binarised_img = cv2.adaptiveThreshold(gray_img,maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                                    #  thresholdType = cv2.THRESH_BINARY, blockSize=11, C=2)

            # blur image

            # blurred_img = cv2.medianBlur(binarised_img,5)

            input_img = gray_img

            # circleGrid method

            # param 2 decrease,
            circles = cv2.HoughCircles(input_img, cv2.HOUGH_GRADIENT, dp=1, minDist=12, param1=50,param2=11,minRadius=3,maxRadius=8)

            if circles is not None:
                circles = np.uint16(np.around(circles))

                for i in circles[0,:]:
                    # draw the outer circle
                    cv2.circle(colour_img,(i[0],i[1]),i[2],(0,255,0),2)
                    # draw the center of the circle
                    cv2.circle(colour_img,(i[0],i[1]),2,(0,0,255),3)

                print("num_circles", circles[0,:])

            cv2.imshow("Image", colour_img)
            cv2.waitKey(1000)
            cv2.imshow("Image", input_img)
            cv2.waitKey(1000)

        elif (topic == topic_dict.get('tf')):
            # print((topic, t, '\b  n'))
            pass

        elif (topic == topic_dict.get('event')):
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

def hough_circled_img(gray_img):

    return circled_img

if __name__ == "__main__":
    
    # bag_file = 'C:/Users/yush7/Desktop/satellite project files/data/calib/hi2.bag'
    # lz4_file = 'C:/Users/yush7/Desktop/satellite project files/data/hi2.bag'
    bag_file = 'C:/Users/yush7/Desktop/satellite project files/data/hi2.bag'

    # decompress_lz4_file(lz4_file, bag_file)


    topics_dict = {'image' : '/dvs/image_raw', 
                   'tf'  : '/tf',
                   'event' : '/dvs/events'}

    
    read_images_from_rosbag(bag_file, topics_dict)