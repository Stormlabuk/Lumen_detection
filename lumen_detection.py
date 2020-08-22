#!/usr/bin/env python
from __future__ import division

import numpy as np
import cv2
import warnings


class LumenDetection(object):
    def __init__(self):
        using_video_file = False

        if using_video_file:
            self.video_capture = cv2.VideoCapture('/home/path/to/video.mp4')
        else:
            self.video_capture = cv2.VideoCapture(0)

        # feature detector to determine if the current video frame contains a lumen
        self.feature_detector = cv2.FastFeatureDetector_create()
        # the presence of a lumen will return a high number of detected features
        # adjust this threshold to
        self.features_threshold = 1
        # set this flag to true in order to see the detected features in the current frame
        self.draw_features = True

    def control_loop(self):
        while True:
            # get next video frame
            ret, frame = self.video_capture.read()

            # check if high number of features indicate a lumen to segment
            features = self.feature_detector.detect(frame, None)
            if len(features) < self.features_threshold:
                warnings.warn('No lumen present in current frame')
                self.display_video(frame, features)
            else:
                segmented_frame = self.segment(frame)
                blurred = cv2.GaussianBlur(segmented_frame, (5, 5), 0)

                grey = self.BGRtoGRAY(blurred)
                # Leave darkest pixels
                grey[grey > (grey.min() + 15)] = 0

                x, y = self.getCenter(grey)
                result = cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                self.display_video(frame, None)

    def display_video(self, frame, features):
        if self.draw_features and features is not None:
            frame = cv2.drawKeypoints(frame, features, None, color=(0, 255, 0), flags=0)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.video_capture.release()
            cv2.destroyAllWindows()

    def getCenter(self, grey):

        M = cv2.moments(grey)

        cnts = cv2.findContours(grey.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY

    def BGRtoGRAY(self, image):
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return grey

    def segment(self, frame):

        blur = cv2.medianBlur(frame, 51)

        # Get red channel
        r = blur.copy()
        r[:, :, 0] = 0
        r[:, :, 1] = 0

        # Convert to greyscale
        grey = self.BGRtoGRAY(r)

        # Downsample 1/2 x 1/2
        scale = 2
        w = int(grey.shape[1] / scale)
        h = int(grey.shape[0] / scale)

        half = cv2.pyrDown(grey, dstsize=(w, h))
        img = half.copy()

        # Largest grey value
        L = img.max()

        # mean grey value
        muT = np.mean(img)

        # Number of pixels
        N = img.size

        Gv = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
        Gh = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)
        g = np.sqrt((Gh ** 2) + (Gv ** 2))
        gmean = np.mean(g)
        if gmean == 0:
            return frame

        h = np.zeros(L)
        h_prime = np.zeros(L)
        w = np.zeros(L)
        sigma_T = 0
        maxKt = 0
        max_eta = 0
        optimal_thresh = 0
        final = img.copy()
        for i in xrange(0, L):
            # Number of pixels at threshold i
            h[i] = len(np.extract(img == i, img))
            if h[i] == 0:
                h[i] == 1

            w[i] = (L / 1) * (1 / gmean)
            h_prime[i] = w[i] * h[i]
            sigma_T = sigma_T + ((i - muT) ** 2) * (h_prime[i] / N)

        for t in xrange(0, L):
            lambda_t = (-t + L) * (1 / gmean)

            # Classes
            C0 = np.extract(img <= t, img)
            C1 = np.extract(img > t, img)
            if C0.any() and C1.any():

                # mean of classes
                mu_0 = np.mean(C0)
                mu_1 = np.mean(C1)

                omega_0 = float(len(C0)) / float(N)
                omega_1 = float(len(C1)) / float(N)

                sigma_B = (omega_0 * omega_1) * ((mu_1 - mu_0) ** 2)

                eta_t = sigma_B / sigma_T
                kt = lambda_t * eta_t
                if kt > maxKt:
                    maxKt = kt
                    optimal_thresh = t

                # Threshold using calculated threshold value (otimal threshold)
        temp_img = img.copy()

        w = int(temp_img.shape[1] * scale)
        h = int(temp_img.shape[0] * scale)

        # Scale up to original size
        temp_img = cv2.pyrUp(temp_img, dstsize=(w, h))

        ret, mask = cv2.threshold(temp_img, optimal_thresh, 255, cv2.THRESH_BINARY)

        # Invert mask to the region we want (the lumen)
        mask_inv = cv2.bitwise_not(mask)

        # Get masked images
        # colour_masked = cv2.bitwise_and(crop,crop,mask = mask_inv)
        grey_masked = cv2.bitwise_and(temp_img, temp_img, mask=mask_inv)

        # Find contours in masked image and get largest region
        cnts = cv2.findContours(grey_masked.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        mask = np.ones(frame.shape[:2], dtype="uint8") * 255
        c = max(cnts, key=cv2.contourArea)

        # Mask to leave only the largest thresholded region
        mask_cnt = cv2.drawContours(mask, [c], -1, 0, -1)
        mask_cnt_inv = cv2.bitwise_not(mask_cnt)
        LRC = cv2.bitwise_and(frame, frame, mask=mask_cnt_inv)

        ## Zabulis et al.
        smax = 0
        for i in cnts:
            A = cv2.contourArea(i)

            if A != 0 and cv2.arcLength(i, True) != 0:
                C = A / cv2.arcLength(i, True)

                mask = np.ones(grey.shape[:2], dtype="uint8") * 255
                mask_cnt = cv2.drawContours(mask, [i], -1, 0, -1)
                mask_inv = cv2.bitwise_not(mask_cnt)
                colour_masked = cv2.bitwise_and(frame, frame, mask=mask_inv)

                mI = cv2.mean(grey, mask=mask_inv)
                I = 1 + (1 - mI[0])

                S = (I ** 2) * C * A
                if S > smax:
                    smax = S
                    LRC = colour_masked.copy()
                    LRC[mask == 255] = (255, 255, 255)

        return LRC


if __name__ == "__main__":
    obj = LumenDetection()
    obj.control_loop()
