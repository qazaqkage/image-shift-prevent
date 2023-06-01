import cv2
from skimage import io


def extract_sift_features(image_path):
    # Load the image
    image = io.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a SIFT object
    sift = cv2.SIFT_create()

    # Detect and compute SIFT features
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return keypoints, descriptors