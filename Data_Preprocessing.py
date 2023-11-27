'''
Input: Video Path
Output: Frames of the video
Function: Split video into frames
'''

import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os
from PIL import Image
# import argparse, os

# parser = argparse.ArgumentParser()
# parser.add_argument('--videoPath', type=str, help='model weights', default='./data/Raw_Video/videos/012.mp4')
# parser.add_argument('--imgPath', type=str, help='video or image', default='./output/image_frame/012/')
# parser.add_argument('--head_num', type=str, help='head number in the video', default=1)

# args = parser.parse_args()


def Video2Pic(videoPath,imgPath,head_num):
    # videoPath = "D:\BirminghamDataScience\Semester2\Summer Project\GT_label_split/raw_videos/001.mp4"  # Video path
    # imgPath = "D:\PROJECT\PYCHARM\PythonProject_HYQ\summer_project\JPEGImages/"  # Path of saving frames
    cap = cv2.VideoCapture(videoPath)
    print('Now deal with video:',(videoPath)[-7:-4])

    fps = cap.get(cv2.CAP_PROP_FPS)  # get FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # get height
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # get weight
    suc = cap.isOpened()  # whether open successfully
    print("suc", suc)
    frame_count = 0
    while suc:

        suc, frame = cap.read()

        print('------------',imgPath ,videoPath,head_num,frame_count)
        
        # frame_name = args.imgPath + (args.videoPath)[-7:-4] + args.head_num + str(frame_count).zfill(3) + ".jpg"
        frame_name = imgPath + (videoPath)[-7:-4] + str(head_num) + str(frame_count).zfill(3) + ".jpg"

        print('Save image into path:',frame_name)
        # cv2.imwrite(imgPath + "%d.jpg" %frame_count, frame)
        if(os.path.exists(imgPath) == False):
            os.makedirs(imgPath)
        try:
            cv2.imwrite(frame_name, frame)
            frame_count += 1
            cv2.waitKey(1)
        except:
            continue

    cap.release()
    print("Converting videos to images finished！")


# if __name__ == '__main__':
#     Video2Pic()
