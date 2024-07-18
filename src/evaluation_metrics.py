import time
import yaml
# from src.functions import isClose2D
import cv2

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

pattern = config["pattern"]

def evaluate_number_accuracy_for_frame(keypoints):

    num_points = pattern[0] * pattern[1]

    return 100*len(keypoints) /num_points


# def get_adjacent_keypoint_colour_indexes(mask, init_coord, init_colour, search_radius=30):

#     x,y = init_coord
#     keepSearching = True
#     r = 0
#     adjacent_colours = set()

#     image_height, image_width = mask.shape

#     while keepSearching:

#         left = (max(x-r,0),y)
#         right = (min(x+r, image_width-1), y)
#         up = (x, max(y-r,0))
#         down = (x, min(y+r,image_height-1))

#         direction_list= [left, right, up, down]

#         for direction in direction_list:
#             test_pixel = str(mask[direction[1], direction[0]])
#             if (test_pixel not in {0,init_colour, 254, 255}): 
#                 adjacent_colours.add(test_pixel)
#         r +=1
#         if r > search_radius:
#             # print(adjacent_colours)
#             # time.sleep(1)
#             break


    
