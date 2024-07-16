import cv2

def sift_detector(gray_image):
    """
    Extracts SIFT features from an input gray image.

    Parameters:
    image (numpy.ndarray): Input gray image.

    Returns:
    keypoints (list of cv2.KeyPoint): Detected keypoints.
    descriptors (numpy.ndarray): Descriptors of the detected keypoints.
    output_image (numpy.ndarray): Image with drawn keypoints for visualization.
    """

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect SIFT features and compute the descriptors
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    # Draw keypoints on the image for visualization (optional)
    output_image = cv2.drawKeypoints(gray_image, keypoints, None)
    
    return keypoints, descriptors, output_image

def surf_detector(gray_image, hessian_threshold=400):
    """
    Extracts SURF features from an input gray image.

    Parameters:
    gray_image (numpy.ndarray): Input gray image
    hessian_threshold (float): Threshold for the Hessian keypoint detector used in SURF.
                               Higher values result in fewer keypoints. Default is 400.

    Returns:
    keypoints (list of cv2.KeyPoint): Detected keypoints.
    descriptors (numpy.ndarray): Descriptors of the detected keypoints.
    output_image (numpy.ndarray): Image with drawn keypoints for visualization.
    """
    
    # Initialize the SURF detector
    surf = cv2.xfeatures2d.SURF_create(hessian_threshold)
    
    # Detect SURF features and compute the descriptors
    keypoints, descriptors = surf.detectAndCompute(gray_image, None)
    
    # Draw keypoints on the image for visualization (optional)
    output_image = cv2.drawKeypoints(gray_image, keypoints, None)
    
    return keypoints, descriptors, output_image

def brief_detector(gray_image):
    """
    Extracts BRIEF features from an input grayscale image.

    Parameters:
    gray_image (numpy.ndarray): Input grayscale image.

    Returns:
    keypoints (list of cv2.KeyPoint): Detected keypoints.
    descriptors (numpy.ndarray): Descriptors of the detected keypoints.
    output_image (numpy.ndarray): Image with drawn keypoints for visualization.
    """
    # Initialize the STAR detector (keypoint detector) and the BRIEF extractor
    star = cv2.xfeatures2d.StarDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    
    # Detect keypoints using the STAR detector
    keypoints = star.detect(gray_image, None)
    
    # Compute the BRIEF descriptors
    keypoints, descriptors = brief.compute(gray_image, keypoints)
    
    # Draw keypoints on the image for visualization (optional)
    output_image = cv2.drawKeypoints(gray_image, keypoints, None)
    
    return keypoints, descriptors, output_image

def orb_detector(gray_image):
    """
    Extracts ORB features from an input grayscale image.

    Parameters:
    gray_image (numpy.ndarray): Input grayscale image.

    Returns:
    keypoints (list of cv2.KeyPoint): Detected keypoints.
    descriptors (numpy.ndarray): Descriptors of the detected keypoints.
    output_image (numpy.ndarray): Image with drawn keypoints for visualization.
    """
    # Initialize the ORB detector
    orb = cv2.ORB_create()
    
    # Detect ORB features and compute the descriptors
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    
    # Draw keypoints on the image for visualization (optional)
    output_image = cv2.drawKeypoints(gray_image, keypoints, None)
    
    return keypoints, descriptors, output_image