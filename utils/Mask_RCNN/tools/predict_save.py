import mmcv
import os
import numpy as np
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector, show_result


config_file = 'yuqi_test_HS/mask_rcnn_r101_fpn_1x_coco.py'
checkpoint_file = 'yuqi_test_HS/epoch_12.pth'

model = init_detector(config_file,checkpoint_file)

img_dir = 'data/VOCdevkit/VOC2007/JPEGImages/'
out_dir = 'yuqi_test_HS/results/'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

img = 'test.jpg'
result = inference_detector(model,img)
show_result(img, result, model.CLASSES, out_file='testOut.jpg')