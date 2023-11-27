##########################################################
"""
This is a tutorial demo of how to use MMGaze.

Format to use this file:
    python MMGaze_demo.py --videoPath './data/Raw_videos/011.mp4'  --head_num 2
    
Input: 
    1. Path of raw video (with audio, conversation video)
    2. Head num: the subject you want to detect

Output:
    1. Gaze target visualization frames
    2. Speaker & non-speaker visualisation frames
    3. Data of  Gaze target, frame_name and head postion

"""
##########################################################


import pandas as pd
import os
# import cv2
# import matplotlib.patches as patches
# from PIL import Image
# import matplotlib.pyplot as plt
# from matplotlib import image
# import warnings
# from PIL import ImageChops
# import datetime
# from matplotlib.animation import FuncAnimation, PillowWriter

import Data_Preprocessing as DP
import subprocess
import Com_fusion_speaker as CF
import utils.Mask_RCNN.tools.test_demo as Gaze_detect

# from matplotlib.font_manager import FontProperties
# import operator
# import joblib
# import re
import argparse
# import pickle
# import ast

import MLP_Best_Gaze

# Input resource path, output path, and head_num (if there are 3 heads in the video, then use 3 times of this command with head_num 1,2,3) 
# videoPath = './data/Raw_videos/011.mp4'  
# head_num= 1

parser = argparse.ArgumentParser()
parser.add_argument('--videoPath', type=str, help='video with audio', default='./data/Raw_videos/011.mp4')
parser.add_argument('--head_num', type=str, help='head number in the video', default=1)

args = parser.parse_args()

videoPath = args.videoPath
head_num= args.head_num

imgPath = './output/image_frame/011/'+ str(head_num)+'/'
os.makedirs(imgPath, exist_ok=True)

data_path = os.path.abspath('./')

##################################################################################
# Split videos into sequences(frames)
##################################################################################
DP.Video2Pic(videoPath,imgPath,head_num)



##################################################################################
# Detect speakers by audio&video
##################################################################################
new_directory =  './utils/Speaker_detector/'
os.chdir(new_directory)

videoPath = '../..'+videoPath[1:]
video_name = videoPath[videoPath.rfind('/') + 1:-4]
speaker_detect_out =  data_path +'/output/speaker_detect/'+ video_name+'/'

subprocess.run(["bash", "build_face_map_Generate_FaceMap.sh", 
                videoPath, video_name, speaker_detect_out], check=True)

##################################################################################
# Fusion speakers
##################################################################################
new_directory =  '../../'
os.chdir(new_directory)

raw_path = imgPath
speaker_path = './output/speaker_detect/'+video_name+'/face_maps/'+video_name +'/'
nonspeaker_path = speaker_path
output_path = './data/output/fusion_speaker/'+video_name+'/com_speaker_feature/'
os.makedirs(output_path, exist_ok=True)

CF.fusion(raw_path, speaker_path, nonspeaker_path, output_path)

##################################################################################
# Gaze Detection
##################################################################################
imagepath = output_path  
savepath= './output/gaze_detection_all/'  +video_name+'/'
config_file ='./utils/Mask_RCNN/yuqi_x101_32x4d_fpn_1x_HS/mask_rcnn_x101_32x4d_fpn_1x_coco.py'    
checkpoint_file= './utils/Mask_RCNN/yuqi_x101_32x4d_fpn_1x_HS/epoch_12.pth' 
all_points_output_file ='./data/output/fusion_speaker/'+video_name+'/x101_1x_test.txt'
best_gaze_output_file='./output/gaze_detection/'+video_name+'/best_gaze.txt'
out_path_cat = './output/gaze_detection/'+video_name +'/'
os.makedirs(out_path_cat, exist_ok=True)
mlp_checkpoint = "./utils/Mask_RCNN/yuqi_x101_32x4d_fpn_1x_HS/model_mlp.m"
head_bbox_file = './output/speaker_detect/011/head_boxes/'+video_name+'/'+video_name+'_frame_faces_boxes.npy'
Gaze_detect.gaze_detect(config_file,checkpoint_file ,imagepath,savepath ,all_points_output_file)
df_result = pd.read_csv(all_points_output_file)


MLP_Best_Gaze.mlp_best_gaze(imgPath, df_result,video_name,head_bbox_file, mlp_checkpoint,imagepath,savepath,config_file,checkpoint_file,all_points_output_file,best_gaze_output_file,out_path_cat)
