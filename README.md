# Stereo Visual Odometry using Deep Learning

This implementation is done to support a study that presents a deep learning-based approach for stereo-visual odometry estimation. While conventional methods dominate the field, research on deep learning applications remains scarce. Our proposed algorithm is evaluated using the KITTI stereo odometry dataset and compared against traditional monocular and traditional stereo-visual odometry techniques to assess its efficacy. Results demonstrate the superiority of our approach over traditional methods, highlighting its potential for advancing stereo-visual odometry methodologies.

Three different methods are implemented in this project. Namely,
1. Traditional Monocular Visual Odometry
2. Traditional Stereo Visual Odometry
3. Deep learning Stereo Visual Odometry

# Flow for the SVO implementation

![Flow for the SVO implementation](doc\img\flow_chart.jpg)
*Flow for the SVO implementation*
# Prerequisites

OpenCV, Numpy, Tensorflow

# Dataset

[KITTI 2012 Dataset](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)

# Model

HITNet model used for disparity map estimation uses the pretrained weights which can be found at [ETH3D](https://www.eth3d.net/)

# How to run?

1. Download the model and place it in the models folder
2. Download the KITTI sequences that you want to run the odometry tests for and place it in a folder named "dataset" just oitside the location of this git repo
3. Set the sequence number you want to execute as the SEQUENCE variable in main file and run the file. 

# Outputs

Running the main file will generate a pkl file which stores in the disparity map calculated by the HITNet model. It also shows how different model estimates the odometry of the sequence and saves the final odometry as an image with the name "Odom_{sequence_number}.png" and the whole run as a video named "DepthImage12_{sequence_number}.avi"


# Results for some sequences



![Sequence 3, 7, 10](doc\img\Odom_3_7_10.png)
*Odometry plots for 3 different sequences: Green line represents Ground truth, Red represents Traditional Monocular Odometry, Blue represents Traditional SVO and Yellow represents HITNet (Deep learning based) SVO*