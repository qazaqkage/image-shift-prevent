import cv2
import numpy as np


def prevent_image_shift(reference_img, target_img, filtered_matches, keypoints1, keypoints2):
    # Convert keypoints to numpy array
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)

    # Calculate shift using Optical Flow
    flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY),
                                        cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY),
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Apply shift compensation
    shifted_points2 = points2 + flow[points2[:, 0, 1].astype(np.int32), points2[:, 0, 0].astype(np.int32)]

    # Calculate the average shift vector
    shift_vector = np.mean(shifted_points2 - points1, axis=0)

    # Apply inverse shift to target image
    M = np.float32([[1, 0, -shift_vector[0, 0]], [0, 1, -shift_vector[0, 1]]])
    corrected_img = cv2.warpAffine(target_img, M, (target_img.shape[1], target_img.shape[0]))

    return corrected_img