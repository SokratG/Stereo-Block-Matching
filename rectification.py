import cv2
import numpy as np

FLANN_INDEX_KDTREE = 1
TREES = 5
CHECKS = 50
GOOD_MATCH_PERCENT = 0.15

def FlannMatcherBuilder(index_params=dict(algorithm=FLANN_INDEX_KDTREE, trees=TREES), search_params=dict(checks=CHECKS)):
    return cv2.FlannBasedMatcher(index_params, search_params)


class StereoRectify:

    def __init__(self, feature_detector_builder=cv2.BRISK_create, matcher_builder=FlannMatcherBuilder, count_best_matches=2):
        self.feature_detector = feature_detector_builder()
        self.matcher = matcher_builder()
        self.k = count_best_matches
    
    def detect_and_match(self, im_left, im_right, mask1=None, mask2=None):
        l_pts, l_desc = self.feature_detector.detectAndCompute(im_left, mask1)
        r_pts, r_desc = self.feature_detector.detectAndCompute(im_right, mask2)
        
        if (len(l_desc) == 0 or len(r_desc) == 0):
            raise Exception('Error: No feature points in images...')

        f_l_desc, f_r_desc = np.float32(l_desc), np.float32(r_desc)

        matches = self.matcher.knnMatch(f_l_desc, f_r_desc, self.k)

        if (len(matches) == 0):
            raise Exception('Error: No matching points in images...')

        return l_pts, r_pts, matches

    def homography_Uncalibrate(self, pts1, pts2, img_size):
        pts1, pts2 = np.int32(pts1), np.int32(pts2)
        
        F, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        pts1, pts2 = pts1[inliers.ravel() == 1], pts2[inliers.ravel() == 1]

        f_pts1, f_pts2  = np.float32(pts1), np.float32(pts2)
        _, H1, H2 = cv2.stereoRectifyUncalibrated(f_pts1, f_pts2, F, img_size)
        return H1, H2

    def rectify(self, im_left, im_right, kpts_threshold=0.75, mask1=None, mask2=None):
        l_pts, r_pts, matches = self.detect_and_match(im_left, im_right)

        pts1, pts2 = [], []

        for i, (m, n) in enumerate(matches):
            if m.distance < kpts_threshold*n.distance:       
                pts1.append(l_pts[m.queryIdx].pt)
                pts2.append(r_pts[m.trainIdx].pt)
        
        h1, w1, _ = im_left.shape
        h2, w2, _ = im_right.shape

        H1, H2 = self.homography_Uncalibrate(pts1, pts2, (w1, h1))

        left_rect = cv2.warpPerspective(im_left, H1, (w1, h1))
        right_rect = cv2.warpPerspective(im_right, H2, (w2, h2))

        return left_rect, right_rect
        