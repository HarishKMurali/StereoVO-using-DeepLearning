import cv2
import numpy as np
import os
import sys
import pickle
from Stereo import *
from hitnet import *

from numpy import arccos, array
from numpy.linalg import norm
import math

SEQUENCE = "10"

def angles(u, v): 
  
  if norm(u)==0 or norm(v)==0:
      return 180
  return math.degrees(arccos(u.dot(v)/(norm(u)*norm(v))))

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def getGroundTruth(frameNo, t):
    ss = np.fromstring(t[frameNo], dtype=np.float64, sep=' ')
    ss = np.array(ss).reshape(3, 4)
    ss = np.vstack((ss, [0, 0, 0, 1]))
    return ss

def loadCalib(filepath):
    with open(filepath, 'r') as f:
        params = np.float64(np.array(f.readline().split()).reshape(-1,1)[1:])
        P_l = np.reshape(params, (3, 4))
        K_l = P_l[0:3, 0:3]
        params = np.float64(np.array(f.readline().split()).reshape(-1,1)[1:])
        P_r = np.reshape(params, (3, 4))
        K_r = P_r[0:3, 0:3]
    return K_l, P_l, K_r, P_r

def find_distance(pointa, pointb):
    return np.sqrt(np.sum((pointa-pointb)**2, axis=0))

def main():
    
    # fname = "DepthImage.avi"
    odom = np.zeros((1980,1980,3), dtype=np.uint8)
    h1 = odom.shape[0]
    w1 = odom.shape[1]

    from tensorflow.python.client import device_lib 
    print(device_lib.list_local_devices())
    
    sequence = SEQUENCE

    
    try:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        fname = "DepthImage_" + sequence + ".avi"
        fps = 30.0
        videoWriter = cv2.VideoWriter(fname, fourcc, fps, (1980, 1980))
    except:
        print("Error: can't create output video: %s" % fname)
        sys.exit()
    
    frameNo = 0
    
    currentT_3D_hitnet = np.eye(4)
    currentT_3D = np.eye(4)
    currentT_2D = np.eye(4)
    
    kpR = []
    datasetDir = "../dataset/sequences/"

    K_l, P1, K_r, P2 = loadCalib(datasetDir + sequence + '/calib.txt')
    timeFile = open("../dataset/poses/" + sequence + ".txt","r")
    
    focalLength = float(P1[0][0])
    baseLength =  -1*float(P2[0][3])/float(P2[0][0]) # base = -P1(1,4)/P1(1,1) (in meters)
    cx = float(P1[0][2])
    cy = float(P1[1][2])
    t = timeFile.readlines()

    model_type = ModelType.eth3d

    if model_type == ModelType.middlebury:
        model_path = "models/middlebury_d400.pb"
    elif model_type == ModelType.flyingthings:
        model_path = "models/flyingthings_finalpass_xl.pb"
    elif model_type == ModelType.eth3d:
        model_path = "models/eth3d.pb"


    hitnet_depth = HitNet(model_path, model_type)

    frameNo = 0
    classes = []
    HITNET_disparities = []
    imageNames = os.listdir(datasetDir + sequence + "/image_0")
    imageNames.sort()
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 1
    thickness = 2
    odom = cv2.putText(odom, 'Ground Truth', (100, 100), font,  fontScale, (0, 255, 0), thickness, cv2.LINE_AA) 
    odom = cv2.putText(odom, 'Monocular Visual Odometry', (100, 150), font,  fontScale, (0, 0, 255), thickness, cv2.LINE_AA) 
    odom = cv2.putText(odom, 'Traditional Stereo Visual Odometry', (100, 200), font,  fontScale, (255, 0, 0), thickness, cv2.LINE_AA) 
    odom = cv2.putText(odom, 'Deep Learning Stereo Visual Odometry', (100, 250), font,  fontScale, (0, 255, 255), thickness, cv2.LINE_AA)

    myEntry = None
    loaded = False

    try:
        with open('HITNET_disparities_'+ sequence +'.pkl', 'rb') as file:
            myEntry = pickle.load(file)
            loaded = True
    except:
        print("No Disparities found, finding them from beginning")

    if not myEntry:
        for image in imageNames:
            print(frameNo)
            
            current = Stereo(frameNo)
            
            imgL = cv2.imread(datasetDir + sequence + "/image_0/" + image)
            imgR = cv2.imread(datasetDir + sequence + "/image_1/" + image)
            
            current.imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            current.imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            current.Matching(current.imgL, current.imgR)

            current.disparity = np.divide(current.disparityCompute.compute(imgL, imgR).astype(np.float32), 16)
            current.HITNET_disparity = hitnet_depth(current.img_L, current.img_R)

            HITNET_disparities.append(current.HITNET_disparity)

            frameNo += 1

            classes.append(current)
    
            db = {}
            db['HITNET_disparities'] = HITNET_disparities

        with open('HITNET_disparities_' + sequence + '.pkl', 'wb') as file:
            pickle.dump(db, file)  
    
    frameNo = 0
    rmse_hitnet = 0
    rmse_2d = 0
    rmse_3d = 0
    rmse_hitnets = []
    rmse_2ds = []
    rmse_3ds = []
    angle_error_2d = 0
    angle_error_3d = 0
    angle_error_hitnet = 0
    angle_error_2ds = []
    angle_error_3ds = []
    angle_error_hitnets = []
    initial_x = 512
    initial_y = 1000

    for image in imageNames:
        
        if loaded:
            
            current = Stereo(frameNo)

            imgL = cv2.imread(datasetDir + sequence + "/image_0/" + image)
            imgR = cv2.imread(datasetDir + sequence + "/image_1/" + image)
            
            imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            current.Matching(imgL, imgR)

            # current.disparity = np.divide(current.disparityCompute.compute(imgL, imgR).astype(np.float32), 16)
            current.HITNET_disparity = myEntry['HITNET_disparities'][frameNo]
            # print(current.HITNET_disparity)
        else:
            current = classes[frameNo]

        current.disparity = np.divide(current.disparityCompute.compute(imgL, imgR).astype(np.float32), 16)
        
        current_GT = getGroundTruth(frameNo, t)
        
        if frameNo>0:
            currentT_3D_hitnet = current.poseEstimationUsingHITNET(previous, current, currentT_3D_hitnet, P1, P2, hitnet_depth)
            currentT_3D = current.poseEstimationUsing3DPoints(previous, current, currentT_3D, P1, P2)
            currentT_2D = current.poseEstimation(previous, current, focalLength, baseLength , cx , cy, currentT_2D, P1, P2)
        else:
            currentT_3D_hitnet = current_GT
            currentT_3D = current_GT
            currentT_2D = current_GT
            
        print(frameNo, currentT_3D_hitnet[:3,3], current_GT[:3,3])
        rmse_hitnet += find_distance(currentT_3D_hitnet[:3,3], current_GT[:3,3])
        rmse_2d += find_distance(currentT_2D[:3,3],current_GT[:3,3])
        rmse_3d += find_distance(currentT_3D[:3,3],current_GT[:3,3])
        
        rmse_hitnets.append(rmse_hitnet)
        rmse_2ds.append(rmse_2d)
        rmse_3ds.append(rmse_3d)

        if frameNo:
            GT_vector = current_GT[:3,3] - previous_GT[:3,3]

            hitnet_vector = currentT_3D_hitnet[:3,3] - previous_hitnet[:3,3]
            _2D_vector = currentT_2D[:3,3] - previous_2D[:3,3]
            _3D_vector = currentT_3D[:3,3] - previous_3D[:3,3]
            
            if norm(_2D_vector)==0:
                angle_error_2d += angles(GT_vector,prev_2D_vector)
            else:
                angle_error_2d += angles(GT_vector,_2D_vector)
            
            if norm(_3D_vector)==0:
                angle_error_3d += angles(GT_vector,prev_3D_vector)
            else:
                angle_error_3d += angles(GT_vector,_3D_vector)
            
            if norm(hitnet_vector)==0:
                angle_error_hitnet += angles(GT_vector,prev_hitnet_vector)
            else:
                angle_error_hitnet += angles(GT_vector,hitnet_vector)
            
            if np.isnan(angle_error_3d):
                print(GT_vector,_3D_vector)
                cv2.waitKey(0)
                
        
            angle_error_2ds.append(angle_error_2d)
            angle_error_3ds.append(angle_error_3d)
            angle_error_hitnets.append(angle_error_hitnet)

            prev_2D_vector = _2D_vector
            prev_3D_vector = _3D_vector
            prev_hitnet_vector = hitnet_vector
            
            print(f"angle error: 2D: {angle_error_2d}, 3D: {angle_error_3d}, Hitnet:{angle_error_hitnet}")
        print(F"RMSE: 2D: {rmse_2d}, 3D:{rmse_3d}, hitnet:{rmse_hitnet}")
        
        
        odom[:45,:] = np.zeros((45, odom.shape[1],3))
        
        odom = cv2.putText(odom,str(frameNo), (100, 25), font,  fontScale, (0, 255, 0), thickness, cv2.LINE_AA) 
        
        odom = cv2.circle(odom, (int(current_GT[0,3]) + initial_x, int(-current_GT[2,3]) + initial_y), 2, (0,255,0), 2)
        odom = cv2.circle(odom, (int(currentT_2D[0,3]) + initial_x, int(-1*currentT_2D[2,3]) + initial_y), 2, (0,0,255), 2)
        odom = cv2.circle(odom, (int(currentT_3D[0,3]) + initial_x, int(-1*currentT_3D[2,3]) + initial_y), 2, (255,0,0), 2)
        odom = cv2.circle(odom, (int(currentT_3D_hitnet[0,3]) + initial_x, int(-1*currentT_3D_hitnet[2,3]) + initial_y), 2, (0,255,255), 2)
        
        previous = current
        previous_GT = current_GT
        previous_2D = currentT_2D
        previous_3D = currentT_3D
        previous_hitnet = currentT_3D_hitnet
        
        cv2.imshow("Left Image", current.img_L)
        videoWriter.write(odom)
        
        frameNo = frameNo + 1
        cv2.imshow("Map", odom)
        key_pressed = cv2.waitKey(1) & 0xFF
        
        if key_pressed == ord('q') or key_pressed == 27:
            break
    db2 = {}
    db2['rmse_hitnet'] = rmse_hitnets
    db2['rmse_2d'] = rmse_2ds
    db2['rmse_3d'] = rmse_3ds
    db2['angle_error_hitnet'] = angle_error_hitnets
    db2['angle_error_2d'] = angle_error_2ds
    db2['angle_error_3d'] = angle_error_3ds

    cv2.imwrite("Odom_" + sequence + ".png", odom)

    with open('RMSE_' + sequence + '.pkl', 'wb') as file:
        pickle.dump(db2, file) 

    videoWriter.release()


    
if __name__ == "__main__":
    main()