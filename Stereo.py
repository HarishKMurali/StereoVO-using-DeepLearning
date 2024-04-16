import os
import numpy as np
import cv2
from scipy.optimize import least_squares
from tqdm import tqdm
from hitnet import *


class Stereo:
    
    def __init__(self, frameNo):
        self.frameNo = frameNo
        self.img_L = []
        self.img_R = []
        self.kp_l = []
        self.kp_r = []
        self.des_l = []
        self.des_r = []
        self.points3D = []
        self.lk_params = dict(winSize=(15, 15),
                              flags=cv2.MOTION_AFFINE,
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

        block = 11
        P1 = block * block * 8
        P2 = block * block * 32
        self.disparityCompute = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=block, P1=P1, P2=P2)
        self.disparity = []
        self.HITNET_disparity = []
        self.fastFeatures = cv2.FastFeatureDetector_create()

    def formTransformationMatrix(self, R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t.reshape(3,)
        return T
    
    def Matching(self, img_L, img_R):
        self.img_L = img_L
        self.img_R = img_R
        self.kp_l, self.des_l = self.KpDes(self.img_L)
        self.kp_r, self.des_r = self.KpDes(self.img_R)
    
        
    def KpDes(self, IMG):
    
        orb = cv2.ORB_create()
    
        fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
    
        kp = fast.detect(IMG, None)
        
    
        kp, des = orb.compute(IMG, kp)
        
        return kp, des
        
    def DesMatch(self, img_L, kp_l, des_l, img_R, kp_r, des_r, mono_points3D = None):
    
        bfObj = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
        matches = bfObj.match(des_l,des_r)
        
        pts_L = []
        pts_R = []
        desL = []
        desR = []
        kpL = []
        kpR = []
        
        minDist = 10000
        maxDist = 0
        
    
        for i in range(len(matches)):
            dist = matches[i].distance
            if dist < minDist:
                minDist = dist
            if dist > maxDist:
                maxDist = dist
        
        good = []
        mono = []
        
        for i in range(len(matches)):
            if matches[i].distance <= max(2 * minDist, 30):
                pts_L.append(kp_l[matches[i].queryIdx].pt)
                pts_R.append(kp_r[matches[i].trainIdx].pt)
                kpL.append(kp_l[matches[i].queryIdx])
                kpR.append(kp_r[matches[i].trainIdx])
                desL.append(des_l[matches[i].queryIdx])
                desR.append(des_r[matches[i].trainIdx])
                
                if mono_points3D is not None:
                    mono.append(mono_points3D[matches[i].queryIdx])
                good.append(matches[i])
                
        pts_L = np.array(pts_L)
        pts_R = np.array(pts_R)
        kpL = np.array(kpL)
        kpR = np.array(kpR)
        desL = np.array(desL)
        desR = np.array(desR)
        mono = np.array(mono)
        
        return pts_L, pts_R, kpL, kpR, desL, desR, good, mono

    def findWorldPts(self, pts_L, pts_R, focalLength, baseLength, cx, cy):

        points3D = []
        matchesL = []
        matchesR = []
        for i in range(len(pts_R)):
        
            d = pts_L[i][0] - pts_R[i][0]
        
        
    
            Z = calcZ
            matchesL.append(pts_L[i])
            matchesR.append(pts_R[i])
        
    
    
            points3D.append([X, Y, Z])
        
        return np.array(points3D), matchesL, matchesR


    def computeScale(self, monoPoints, stereoPoints):
    
        scales = []

        for i in range(len(monoPoints)):
        
            monoMag = ((monoPoints[i][0])**2 + (monoPoints[i][1])**2 + (monoPoints[i][2])**2)**0.5
            stereoMag = ((stereoPoints[i][0])**2 + (stereoPoints[i][1])**2 + (stereoPoints[i][2])**2)**0.5
        
            scales.append(monoMag/stereoMag)
    
        return np.mean(scales)
    
    def poseEstimation(self, previous, current, focalLength, baseLength , cx , cy, prevT_L, P1, P2):
    
        pts_prev, pts_current, kpL, kpR, desL, desR, good, _ = self.DesMatch(previous.img_L, previous.kp_l, previous.des_l, current.img_L, current.kp_l, current.des_l)
    
        E, mask = cv2.findEssentialMat(pts_current, pts_prev, focalLength, (cx, cy), method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
        _, R, t, _ = cv2.recoverPose(E, pts_current, pts_prev, focal=focalLength, pp = (cx,cy))
    

        pts_prev = pts_prev[mask.ravel() == 1]
        pts_current = pts_current[mask.ravel() == 1]
        kpL = kpL[mask.ravel() == 1]
        kpR = kpR[mask.ravel() == 1]
        desL = desL[mask.ravel() == 1]
        desR = desR[mask.ravel() == 1]
    
        P3 = P1.copy()
        P3[:3, :3] = P3[:3, :3] @ R
        P3[0:3,:] = t
    
        mono_points3D = cv2.triangulatePoints(P1, P3, pts_prev.T, pts_current.T)
        mono_points3D = mono_points3D.T
    
    
        pts_L, pts_R, kpL, kpR, desL, desR, good, mono_points3D = self.DesMatch(previous.img_L, kpL, desL, previous.img_R, previous.kp_r, previous.des_r, mono_points3D)
    
        E, mask = cv2.findEssentialMat(pts_R, pts_L, focalLength, (cx, cy), method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
        pts_L = pts_L[mask.ravel() == 1]
        pts_R = pts_R[mask.ravel() == 1]
        kpL = kpL[mask.ravel() == 1]
        kpR = kpR[mask.ravel() == 1]
        desL = desL[mask.ravel() == 1]
        desR = desR[mask.ravel() == 1]
        mono_points3D = mono_points3D[mask.ravel() == 1]
    
        previous.points3D = cv2.triangulatePoints(P1, P2, pts_L.T, pts_R.T)
        previous.points3D = previous.points3D.T
    
    
        scale = self.computeScale(mono_points3D, previous.points3D)
    
        # print("SCALE: ", scale)
    
        R_prev = prevT_L[0:3, 0:3]
        t_prev = np.reshape(prevT_L[0:3, 3], (3,1))
    
        R_curr = R_prev @ R
        t_curr = t_prev + 1*(R_prev @ t)
    
    
        currentT = self.formTransformationMatrix(R_curr, t_curr)
    
        return currentT
        
    def reprojection_residuals(self, dof, q1, q2, Q1, Q2, P1, P2):
        r = dof[:3]
        R, _ = cv2.Rodrigues(r)
        t = dof[3:]
        transf = self.formTransformationMatrix(R, t)

        f_projection = np.matmul(P1, transf)
        b_projection = np.matmul(P1, np.linalg.inv(transf))

        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        q1_pred = Q2.dot(f_projection.T)
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        q2_pred = Q1.dot(b_projection.T)
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        return residuals

    def get_tiled_keypoints(self, img, tile_h, tile_w):
        def get_kps(x, y):
            impatch = img[y:y + tile_h, x:x + tile_w]

            keypoints = self.fastFeatures.detect(impatch)

            for pt in keypoints:
                pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

            if len(keypoints) > 10:
                keypoints = sorted(keypoints, key=lambda x: -x.response)
                return keypoints[:10]
            return keypoints
        h, w, *_ = img.shape

        kp_list = [get_kps(x, y) for y in range(0, h, tile_h) for x in range(0, w, tile_w)]

        kp_list_flatten = np.concatenate(kp_list)
        return kp_list_flatten

    def track_keypoints(self, img1, img2, kp1, max_error=5):
        trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)

        trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **self.lk_params)

        trackable = st.astype(bool)

        under_thresh = np.where(err[trackable] < max_error, True, False)

        trackpoints1 = trackpoints1[trackable][under_thresh]
        trackpoints2 = np.around(trackpoints2[trackable][under_thresh])

        h, w = img1.shape
        in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
        trackpoints1 = trackpoints1[in_bounds]
        trackpoints2 = trackpoints2[in_bounds]

        return trackpoints1, trackpoints2

    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
        def get_idxs(q, disp):
            q_idx = q.astype(int)
            disp = np.array(disp).T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)
        
        disp1, mask1 = get_idxs(q1, disp1)
        disp2, mask2 = get_idxs(q2, disp2)
        
        in_bounds = np.logical_and(mask1, mask2)
        
        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]
        
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2
        
        return q1_l, q1_r, q2_l, q2_r

    def calc_3d(self, q1_l, q1_r, q2_l, q2_r, P1, P2):
        # print(P1.shape, P2.shape, q1_l.T.shape, q1_r.T.shape)
        Q1 = cv2.triangulatePoints(P1, P2, q1_l.T, q1_r.T)
        Q1 = np.transpose(Q1[:3] / Q1[3])

        Q2 = cv2.triangulatePoints(P1, P2, q2_l.T, q2_r.T)
        Q2 = np.transpose(Q2[:3] / Q2[3])
        return Q1, Q2

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def estimate_pose(self, q1, q2, Q1, Q2, P1, P2, max_iter=100):
        early_termination_threshold = 5

        min_error = float('inf')
        early_termination = 0

        for _ in range(max_iter):
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            in_guess = np.zeros(6)
            opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                    args=(sample_q1, sample_q2, sample_Q1, sample_Q2, P1, P2))

            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2, P1, P2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))

            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_threshold:
                break

        r = out_pose[:3]
        R, _ = cv2.Rodrigues(r)
        t = out_pose[3:]
        transformation_matrix = self._form_transf(R, t)
        return transformation_matrix

    def poseEstimationUsing3DPoints(self, previous, current, prev_T, P1, P2):
        img1_l, img2_l = previous.img_L, current.img_L

        kp1_l = self.get_tiled_keypoints(img1_l, 10, 20)

        tp1_l, tp2_l = self.track_keypoints(img1_l, img2_l, kp1_l)

        current.disparity = np.divide(current.disparityCompute.compute(img2_l, current.img_R).astype(np.float32), 16)

        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1_l, tp2_l, previous.disparity, current.disparity)


        Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r, P1, P2)

        transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2, P1, P2)
        # print(transformation_matrix)
        return np.matmul(prev_T, transformation_matrix)
    
    def poseEstimationUsingHITNET(self, previous, current, prev_T, P1, P2, hitnet_depth):

        img1_l, img2_l = previous.img_L, current.img_L

        kp1_l = self.get_tiled_keypoints(img1_l, 10, 20)

        tp1_l, tp2_l = self.track_keypoints(img1_l, img2_l, kp1_l)

        # previous.HITNET_disparity = hitnet_depth(previous.img_L, previous.img_R)
        # current.HITNET_disparity = hitnet_depth(current.img_L, current.img_R)
        # print("HERE", current.HITNET_disparity)
        # print(type(current.HITNET_disparity))
        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1_l, tp2_l, np.array(previous.HITNET_disparity), np.array(current.HITNET_disparity))


        Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r, P1, P2)

        transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2, P1, P2)
        # print(transformation_matrix)
        return np.matmul(prev_T, transformation_matrix)

