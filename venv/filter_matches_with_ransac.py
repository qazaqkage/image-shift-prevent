import cv2
import numpy as np


def filter_matches_with_ransac(keypoints1, keypoints2, matches, reprojection_threshold=3.0, num_iterations=1000):
    # Convert keypoints to numpy arrays
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # RANSAC parameters
    ransac_model = cv2.RANSAC
    ransac_params = dict(
        maxIters=num_iterations,
        confidence=0.99,
        ransacReprojThreshold=reprojection_threshold
    )

    # Find the homography matrix using RANSAC
    _, mask = cv2.findHomography(points1, points2, ransac_model, **ransac_params)

    # Filter matches using the RANSAC mask
    filtered_matches = [m for m, is_valid in zip(matches, mask.flatten().tolist()) if is_valid]

    return filtered_matches