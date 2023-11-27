from mmdet.apis import init_detector, inference_detector
from mmdet.apis import show_result_pyplot
import os
import argparse
import pandas as pd
import ast

# parser = argparse.ArgumentParser()
# parser.add_argument('--imagepath', type=str, help='imagepath', default='../../../output/fusion_speaker/012/')
# parser.add_argument('--savepath', type=str, help='output save path', default='../../../output/gaze_detection_all/')
# parser.add_argument('--config_file', type=str, help='config file for model', default='../yuqi_x101_32x4d_fpn_1x_HS/mask_rcnn_x101_32x4d_fpn_1x_coco.py')
# parser.add_argument('--checkpoint_file', type=str, help='checkpoint for model', default='../yuqi_x101_32x4d_fpn_1x_HS/epoch_12.pth')
# parser.add_argument('--all_points_output_file', type=str, help='all estimated gaze boxes for each subject', default='../../../output/fusion_speaker/012/x101_1x_test.txt')

# args = parser.parse_args()

device = 'cuda:0'
# init a detector
# config_file = args.config_file
# checkpoint_file = args.checkpoint_file
# imagepath = args.imagepath
# savepath = args.savepath
# all_points_output_file = args.all_points_output_file

# def gaze_all_result(config_file, checkpoint_file, imagepath,savepath, all_points_output_file):
# imagepath = r'../../../output/fusion_speaker/012/' #Path of Test dataset to load images
# savepath = r'../../../output/gaze_detection_all/' #Path to save images
# config_file = r'../yuqi_x101_32x4d_fpn_1x_HS/mask_rcnn_x101_32x4d_fpn_1x_coco.py' #path to load network model
# checkpoint_file = r'../yuqi_x101_32x4d_fpn_1x_HS/epoch_12.pth'  #path for pre-trained parameters

def gaze_detect(config_file,checkpoint_file ,imagepath,savepath,all_points_output_file): 
    model = init_detector(config_file, checkpoint_file, device=device)


# inference the demo image
# f= open(all_points_output_file,'w')
# for filename in os.listdir(imagepath):
#     img = os.path.join(imagepath, filename)
#     result = inference_detector(model, img)
#     out_file = os.path.join(savepath, filename)
#     print('---img_name is:', filename)
#     # if result is not null:
#     # print('-----type',type(result[0][0]))
#     # print('-------------result',result[0][0])
    
#     line = str(filename)+','+str(result[0][0])
#     f.write('\n'+line)
#     # print('txt line is',line)

#     show_result_pyplot(model, img, result, out_file = out_file,  score_thr=0, title='result', wait_time=0, palette=None)
    
# f.close()

# print('All gaze result saved in :',all_points_output_file)


    df_result = pd.DataFrame(columns=['frame_name', 'gaze_bbox', 'gaze_score'])

    # loop every frame
    for filename in os.listdir(imagepath):
        img = os.path.join(imagepath, filename)
        result = inference_detector(model, img)
        out_file = os.path.join(savepath, filename)
        print('---img_name is:', filename)

        for gaze_result in result[0]:
            # if len(gaze_result) == 5:  # 确保长度为 5
                for box in gaze_result.tolist():
                    print('------------box-------',box)
                    gaze_bbox = box[:4]
                    score = box[4]
                    # df_a = df_a.append({'gaze_bbox': gaze_bbox, 'score': score}, ignore_index=True)
                    
                    # line = str(filename) + ',' + str(gaze_result.tolist())
                    
                    # print('line',line)
                    df_result = df_result.append({'frame_name': filename, 'gaze_bbox': gaze_bbox, 'gaze_score': score}, ignore_index=True)
            # else:
            #     print('Result is not predicted',result[0])

        show_result_pyplot(model, img, result, out_file=out_file, score_thr=0, title='result', wait_time=0, palette=None)

    df_result.to_csv(all_points_output_file, index=False)

    print('All gaze results saved in:', all_points_output_file)