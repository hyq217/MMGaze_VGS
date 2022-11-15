from mmdet.apis import init_detector, inference_detector
from mmdet.apis import show_result_pyplot
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--imagepath', type=str, help='imagepath', default='/home/zzq/yuqi_summer_project/com_speaker/temp_data_processing/com_speaker_feature/')
parser.add_argument('--savepath', type=str, help='output save path', default='../output/gaze_detection_all/')
parser.add_argument('--config_file', type=str, help='config file for model', default='../yuqi_x101_32x4d_fpn_1x_HS/mask_rcnn_x101_32x4d_fpn_1x_coco.py')
parser.add_argument('--checkpoint_file', type=str, help='checkpoint for model', default='../yuqi_x101_32x4d_fpn_1x_HS/epoch_12.pth')
args = parser.parse_args()

# imagepath = r'/home/zzq/yuqi_summer_project/mmdetection-master/data/coco/val2017/' #Path of Test dataset to load images
# savepath = r'/home/zzq/yuqi_summer_project/mmdetection-master/video_coco_test/output/' #Path to save images
# config_file = r'/home/zzq/yuqi_summer_project/mmdetection-master/yuqi_x101_32x4d_fpn_1x_HS/mask_rcnn_x101_32x4d_fpn_1x_coco.py' #path to load network model
# checkpoint_file = r'/home/zzq/yuqi_summer_project/mmdetection-master/yuqi_x101_32x4d_fpn_1x_HS/epoch_12.pth'  #path for pre-trained parameters
device = 'cuda:0'
# init a detector
model = init_detector(args.config_file, args.checkpoint_file, device=device)

# video_name = (args.imagepath)[-7:-4]
# print('Now is working on mp4', video_name)

# save_path = args.savepath + '/' + video_name + '/'

# if(os.path.exists(save_path) == False):
#     os.makedirs(save_path)

# inference the demo image
for filename in os.listdir(args.imagepath):
    f= open('./x101_1x_test_mod.txt','a')
    img = os.path.join(args.imagepath, filename)
    result = inference_detector(model, img)
    out_file = os.path.join(args.imagepath, filename)
    print('---img_name is:', filename)
    # if result is not null:
    # print('-----type',type(result[0][0]))
    print('-------------result',result[0][0])
    
    line = str(filename)+','+str(result[0][0])
    f.write('\n'+line)
    print('txt line is',line)


    show_result_pyplot(model, img, result, out_file = out_file,  score_thr=0, title='result', wait_time=0, palette=None)
    
    f.close()
