from mmdet.apis import init_detector, inference_detector
from mmdet.apis import show_result_pyplot
import os
 

# parser = argparse.ArgumentParser()
# parser.add_argument('--imagepath', type=str, help='imagepath', default='/home/zzq/yuqi_summer_project/com_speaker/temp_data_processing/com_speaker_feature/')
# # parser.add_argument('--savepath', type=str, help='output save path', default='../output/gaze_detection_all/')
# parser.add_argument('--config_file', type=str, help='config file for model', default='../yuqi_x101_32x4d_fpn_1x_HS/mask_rcnn_x101_32x4d_fpn_1x_coco.py')
# parser.add_argument('--checkpoint_file', type=str, help='checkpoint for model', default='../yuqi_x101_32x4d_fpn_1x_HS/epoch_12.pth')
# parser.add_argument('--his_path', type=str, help='his_path', default='../../../output/history/')
# args = parser.parse_args()


imagepath = r'/home/zzq/yuqi_summer_project/com_speaker/temp_data_processing/com_speaker_feature/' #Path of Test dataset to load images
savepath = r'../../../output/gaze_detection_all/' #Path to save images
config_file = r'../yuqi_x101_32x4d_fpn_1x_HS/mask_rcnn_x101_32x4d_fpn_1x_coco.py' #path to load network model
checkpoint_file = r'../yuqi_x101_32x4d_fpn_1x_HS/epoch_12.pth'  #path for pre-trained parameters
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)


# inference the demo image
for filename in os.listdir(imagepath):
    f= open('./x101_1x_test.txt','a')
    img = os.path.join(imagepath, filename)
    result = inference_detector(model, img)
    out_file = os.path.join(savepath, filename)
    print('---img_name is:', filename)
    # if result is not null:
    # print('-----type',type(result[0][0]))
    print('-------------result',result[0][0])
    
    line = str(filename)+','+str(result[0][0])
    f.write('\n'+line)
    print('txt line is',line)


    show_result_pyplot(model, img, result, out_file = out_file,  score_thr=0, title='result', wait_time=0, palette=None)
    
    f.close()
