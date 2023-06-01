import cv2


def match_sift_features(image1, image2):
    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Create a SIFT object
    sift = cv2.SIFT_create()

    # Detect and compute SIFT features for both images
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Create a BFMatcher object
    bf = cv2.BFMatcher()

    # Perform the feature matching
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.80 * n.distance:
            good_matches.append(m)

    # Draw the matches

    good_matches = filter_matches_with_ransac(keypoints1, keypoints2, good_matches)

    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return matched_image, good_matches, keypoints1, keypoints2