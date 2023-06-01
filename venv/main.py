import cv2
from extract_sift_features import extract_sift_features
from match_sift_features import match_sift_features
from filter_matches_with_ransac import filter_matches_with_ransac
from prevent_image_shift import prevent_image_shift


def main():
    # Example usage
    image1 = cv2.imread("image1.jpg")
    image2 = cv2.imread("image2.jpg")

    # Extract SIFT features
    keypoints1, descriptors1 = extract_sift_features("image1.jpg")
    keypoints2, descriptors2 = extract_sift_features("image2.jpg")

    # Match SIFT features
    matched_image, good_matches, keypoints1, keypoints2 = match_sift_features(image1, image2)

    # Filter matches with RANSAC
    filtered_matches = filter_matches_with_ransac(keypoints1, keypoints2, good_matches)

    # Prevent image shift
    corrected_image = prevent_image_shift(image1, image2, filtered_matches, keypoints1, keypoints2)

    # Display the results or save the images
    cv2.imshow("Matched Image", matched_image)
    cv2.imshow("Corrected Image", corrected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()