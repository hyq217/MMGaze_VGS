# Get VGS video dataset
# Annotation format: [image_name,head_bbox_x_min,head_bbox_y_min,head_bbox_x_max,head_bbox_y_max,gaze_x,gaze_y]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# import warnings
# import operator
import joblib
import os
# import re
# import argparse
import ast
import matplotlib.patches as patches
from functools import partial
from PIL import Image


##################################################################################
# Gaze Detection
##################################################################################
#  gaze_k_predict
def calculate_gaze_k(head_point,model_mlp):
    gaze_k_predict = model_mlp.predict([head_point])[0]
    return gaze_k_predict

# find nearest gaze_point by gaze_k_predict
def calculate_closest_gaze(gaze_point, gaze_k_predict, head_point):
    print(gaze_point)
    print(gaze_k_predict)
    print(head_point)
    print('===========')
    closest_gaze = None
    min_deviation = float('inf')

    # for gaze, gaze_k_predict, head_point in zip(gaze_points, gaze_k_predicts, head_points):
    try: 
        k_maskrcnn = (gaze_point[1] - head_point[1]) / (gaze_point[0] - head_point[0])
        deviation = abs(k_maskrcnn - gaze_k_predict)

        if deviation < min_deviation:
            min_deviation = deviation
            closest_gaze = gaze_point
    except ZeroDivisionError:
        pass

    return closest_gaze

# pick nearest line of gaze_k_predict
def filter_closest_gaze(group):
    closest_gaze_idx = (group['gaze_k_predict'] - group['gaze_k_predict'].values[0]).abs().idxmin()
    closest_gaze_row = group.loc[closest_gaze_idx]
    return closest_gaze_row

def bboxes_all(im, image_name, x_scale, y_scale, item, out_path):
    # Create figure and axes
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(im)
    width, height = im.size
    # Create a Rectangle patch
    rect = patches.Rectangle((item[0], item[1]), item[2]-item[0], item[3]-item[1], linewidth=1, edgecolor='g', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.plot(x_scale, y_scale, 'ro', color='g', markersize=8)
    plt.axis('off')
    #draw arrow
    norm_p = [x_scale, y_scale]
    plt.plot((norm_p[0],(item[0]+item[2])/2), (norm_p[1],(item[1]+item[3])/2), '-', color=(0,1,0,1))

    plt.savefig(out_path, bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
    plt.clf()
    plt.cla()
    plt.close()
    # plt.imshow()
    return
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

def move_dupli(mm):
  ls=[0]
  ls[0]=mm[0]
  for i in range(1,len(mm)):
      if mm[i]!=ls[-1]:
          ls.append(mm[i])
  return (''.join(ls))


# Define a function to parse the bounding box string into a list of integers
def parse_bbox(bbox_str):
    bbox_list = ast.literal_eval(bbox_str)
    return [int((bbox_list[0] + bbox_list[2]) / 2), int((bbox_list[1] + bbox_list[3]) / 2)]


def mlp_best_gaze(imgPath,df_result,video_name,head_bbox_file, mlp_checkpoint,imagepath,savepath,config_file,checkpoint_file,all_points_output_file,best_gaze_output_file,out_path_cat):

  model_mlp = joblib.load(mlp_checkpoint)

  arr = np.load(head_bbox_file,allow_pickle=True).item()

  df = pd.DataFrame(arr)

  df['frame_num'] = range(len(df))
  df['frame_num'] = df['frame_num'].apply(str).apply(lambda x:str(x).zfill(3))
  df=pd.melt(df,id_vars='frame_num',var_name='head_num',value_name='head_bbox')
  df['head_num'] = df['head_num'].apply(pd.to_numeric)
  df['head_num'] = df['head_num'].map(lambda x: x+1).apply(str)
  df['video_id'] = video_name

  df['frame_name'] = df['video_id'].str.cat(df['head_num'])
  df['frame_name'] = df['frame_name'].str.cat(df['frame_num'])
  df["frame_name"] =df["frame_name"]+'.jpg'

  df_head_bbox = df[['frame_name','head_bbox']]

  merged_df = pd.merge(df_result, df_head_bbox, on='frame_name', how='inner')

  gaze_maskrcnn= all_points_output_file
  gaze_maskrcnn_mod='./data/output/'+video_name+ '/'

  if not os.path.exists(gaze_maskrcnn_mod):
      os.makedirs(gaze_maskrcnn_mod)


  merged_df['head_num'] = merged_df['frame_name'].str[3:4]
  merged_df['frame_seq'] = merged_df['frame_name'].str[4:-4]
  merged_df['head_point'] = merged_df['head_bbox'].apply(lambda bbox: [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)])
  merged_df['gaze_point'] = merged_df['gaze_bbox'].apply(parse_bbox)

  # calculate_gaze_k_with_model = partial(calculate_gaze_k, model=model_mlp)
  calculate_gaze_k_with_model = lambda x: calculate_gaze_k(x, model_mlp)


  merged_df['gaze_k_predict'] = merged_df['head_point'].apply(calculate_gaze_k_with_model)
  # closest_gaze to 'gaze_best'
  merged_df['gaze_best'] = merged_df.apply(lambda row: calculate_closest_gaze(row['gaze_point'], row['gaze_k_predict'],row['head_point']), axis=1)
  filtered_df = pd.DataFrame(columns=merged_df.columns)

  grouped = merged_df.groupby('frame_name')

  # calculate nearest points on each group, output filtered_df
  for name, group in grouped:
      closest_gaze_row = filter_closest_gaze(group)
      filtered_df = pd.concat([filtered_df, closest_gaze_row.to_frame().T])

  #final gaze prediction
  filtered_df = filtered_df.reset_index(drop=True)
  filtered_df.to_csv(best_gaze_output_file, index=False)

  # draw gaze target in frame
  fullname_list, filename_list = get_all_file(imgPath, 'jpg')
  for frame_path,frame_name in zip(fullname_list, filename_list):
      frame_raw = Image.open(frame_path)
      frame_raw = frame_raw.convert('RGB')
      out_path = out_path_cat+ frame_name
      width, height = frame_raw.size
      print('Generate frame with prediction',out_path)

      # get gaze point
      try:
          gaze_best_point = filtered_df.loc[filtered_df['frame_name'].str.contains(frame_name[:-4])]['gaze_best'].tolist()[0]
          x_scale = gaze_best_point[0]
          y_scale = gaze_best_point[1]

          #get head_bbox
          head_box = filtered_df.loc[filtered_df['frame_name'].str.contains(frame_name[:-4])]['head_bbox'].tolist()
          head_xmin = int((head_box[0][0]))
          head_ymin = int((head_box[0][1]))
          head_xmax = int((head_box[0][2]))
          head_ymax = int((head_box[0][3]))
          head_bbox = [head_xmin,head_ymin,head_xmax,head_ymax]
      except :
          print('-------')
          continue

      bboxes_all(frame_raw, frame_name, x_scale, y_scale, head_bbox, out_path)

