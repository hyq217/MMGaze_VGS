# Get VGS video dataset
# Annotation format: [image_name,head_bbox_x_min,head_bbox_y_min,head_bbox_x_max,head_bbox_y_max,gaze_x,gaze_y]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import warnings
import operator
import joblib
import os
import re
import argparse

model_mlp1 = joblib.load("/content/drive/MyDrive/summer_project/MLP_HEAD_GAZE_PAIR/model_mlp.m")
f = open('/content/drive/MyDrive/summer_project/VGS/VGS_Dataset/x101_1x_test.txt','rb+')
f_lines = f.read()



parser = argparse.ArgumentParser()
parser.add_argument('--GT_label_all_videoFormat', type=str, help='head information for each frame', default='/content/drive/MyDrive/summer_project/VGS/VGS_Dataset/VideoGazeSpeech/GT_VideoFormat/GT_label_all_videoFormat.txt')
parser.add_argument('--mlp_checkpoint', type=str, help='mlp_checkpoint', default='/content/drive/MyDrive/summer_project/MLP_HEAD_GAZE_PAIR/model_mlp.m')
parser.add_argument('--gaze_maskrcnn', type=str, help='gaze prediected by maskrcnn', default='/content/drive/MyDrive/summer_project/VGS/VGS_Dataset/x101_1x_test.txt')
parser.add_argument('--gaze_maskrcnn_mod', type=str, help='gaze prediected by maskrcnn,modify format', default='./output3/')

args = parser.parse_args()

# Get full name list concluding path of the images
def get_all_file(dir_name, file_extension):
    """
    Iterate Over Data
    """
    fullname_list, filename_list = [], []
    for root, dirs, files in os.walk(dir_name):
        for filename in files:
            if ("Detectors" not in os.path.join(root, filename)) and filename[-len(file_extension):] == file_extension:
                # fullname, concluding path
                fullname_list.append(os.path.join(root, filename))
                filename_list.append(filename)
    return fullname_list, filename_list


f_lines1 = f_lines.decode('ISO-8859-1')  # encoding may vary!
f_lines2 = re.split(r",|\\n", f_lines1) 
frame_list = [[]]

i=0
a = {'image_name':'gaze_box'}
gaze_boxes = 0
frame_name = ''
def move_dupli(mm):
  ls=[0]
  ls[0]=mm[0]
  for i in range(1,len(mm)):
      if mm[i]!=ls[-1]:
          ls.append(mm[i])
  return (''.join(ls))

for item in f_lines2:
  # if i<5:
    # print('-----')
    # print(item)
    i+=1
    
    item1 = re.split(r"]]|\n", item) 
    j=0
    gaze_box1 = [[]]
    for item in item1:      
      # print('***********')
      if item.__contains__("jpg"):
        frame_name = item
        # print('frame_name:',frame_name)
        frame_list.append(frame_name)

      elif item is not None:
        
        gaze_boxes = item
        # print('gaze_boxes:',gaze_boxes)
        j+=1
        gaze_box1.append(gaze_boxes)

      with open(args.gaze_maskrcnn_mod+frame_name[:-4]+".csv","w") as f:
        mm = move_dupli(','.join(str(gaze_box1).split()).replace("'","").replace("'","").split("[],,")[1][:-3]+']')
        f.write(str(mm))

# get head position

gt =pd.read_table(args.GT_label_all_videoFormat, sep=',',header=None)     

gt['image_name'] = gt[0]
gt['head_position_x'] = ((gt[1]+gt[3])/2).astype(int)
gt['head_position_y'] = ((gt[2]+gt[4])/2).astype(int)
gt['gaze_x'] = gt[5]
gt['gaze_y'] = gt[6]

gt['k'] = (gt['gaze_y']-gt['head_position_y'])/(gt['gaze_x']-gt['head_position_x'])
# gt.fillna(0,inplace=True)
gt.fillna(0, inplace=True)
gt['k'][np.isinf(gt['k'])] = 0


# get best gaze point from mask_rcnn prediction
import numpy as np
df = pd.DataFrame(columns=['video_name', 'head_num', 'frame_seq','frame_name', 'head_position','gaze_point'])
model_mlp1 = joblib.load(args.mlp_checkpoint)

fullname_list, filename_list = get_all_file(args.gaze_maskrcnn_mod, 'csv')
i=0

gaze_tmp = [[]]
# for each frame (with multiple gaze_boxes predicted by maskrcnn)
for frame_gaze_path,frame_name in zip (fullname_list, filename_list):
  # if i<2:
    # convert format to boxes
    video_name = frame_name[:3] + '.mp4'
    head_num = frame_name[3:4]
    frame_seq = frame_name[4:-4]
    f_gaze = open(frame_gaze_path,'rb+')
    f_gaze_lines = f_gaze.read()
    f_gaze_lines = str(f_gaze_lines).split('],[')
    gaze_num = len(f_gaze_lines)

    frame_name_1 = (frame_name[:-4]+'.jpg')
    # print(frame_name_1)

    dev = 0.0

    # get related head point
    try:
      head_position_x = gt.loc[gt[0] == frame_name[:-4]+'.jpg','head_position_x'].iloc[0]
      head_position_y = gt.loc[gt[0] == frame_name[:-4]+'.jpg','head_position_x'].iloc[0]
      head = [[head_position_x,head_position_y]]
    #   model_mlp1 = joblib.load("/content/drive/MyDrive/summer_project/MLP_HEAD_GAZE_PAIR/model_mlp.m")
      gaze_k_predict = model_mlp1.predict(head)[0]
      head_box = [gt.loc[gt[0] == frame_name[:-4]+'.jpg',1].iloc[0],gt.loc[gt[0] == frame_name[:-4]+'.jpg',2].iloc[0],gt.loc[gt[0] == frame_name[:-4]+'.jpg',3].iloc[0],gt.loc[gt[0] == frame_name[:-4]+'.jpg',4].iloc[0]]
      
      j = 1
      # for each gaze_box predicted by maskrcnn
      for item in f_gaze_lines:
        item = item.replace('[','').replace(',]','').replace("b",'').replace("'",'').split(',')[:-1]
        item = [ float(x) for x in item ]
        # get gaze point
        gaze_x = ((item[0]+item[2])/2)
        gaze_y = ((item[1]+item[3])/2)
        gaze = [[gaze_x,gaze_y]]
        # calculate k
        k_maskrcnn = (gaze_y - head_position_y)/(gaze_x - head_position_x)



        if j == 1 :
          dev = abs(k_maskrcnn-gaze_k_predict)
          gaze_tmp = gaze
        elif j> 1:
          if dev > abs(k_maskrcnn-gaze_k_predict):
            dev = abs(k_maskrcnn-gaze_k_predict)
            gaze_tmp = gaze
          else:
            continue

          
        j+=1
      df = df.append({'video_name': video_name, 'head_num': head_num, 'frame_seq':frame_seq,'frame_name':frame_name_1, 'head_position':head_box, 'gaze_point':gaze_tmp}, ignore_index=True)
      i+=1
    except Exception:
        continue


df.sort_values(by=['frame_name'], ascending=True, inplace=True)
df1 = df.drop(columns=['Unnamed: 0'])

df1.to_csv('KETI_Sequence.csv')