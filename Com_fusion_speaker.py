# Function: Combine speaker features to raw frame
import cv2

import os


# Get full name list concluding path of the images
def get_all_file(dir_name, file_extension):
    """
    Iterate Over Data
    """
    fullname_list, filename_list = [], []
    for root, dirs, files in os.walk(dir_name):
        for filename in files:
            if ("Detectors" not in os.path.join(root, filename)) and filename[-3:] == file_extension:
                # fullname, concluding path
                fullname_list.append(os.path.join(root, filename))
                filename_list.append(filename)
    return fullname_list, filename_list


def com_speaker_feature(speaker_frame, no_speaker_frame, raw_frame, file_name,output_path):
    # This is to combine speakers，head position together with raw file
    frame = cv2.imread(raw_frame)
    # print('==========no_speaker_frame type', no_speaker_frame)

    no_speak_h = cv2.imread(no_speaker_frame[0])
    speak_h = cv2.imread(speaker_frame[0])
    masked_frame = cv2.addWeighted(no_speak_h, 0.5, speak_h, 1, 0)
    com_frame = cv2.addWeighted(masked_frame, 0.7, frame, 1, 0)
    # print('com_frame',com_frame)
    # print('no_speak_h',no_speak_h)
    # filename format = videoNum_frameNum
    cv2.imwrite(output_path + file_name , com_frame)
    # cv2.imshow(output_path +'/com_speaker_feature/'+ file_name , com_frame)
    print('Fusion image saved in',output_path + file_name)



def com_head_feature(speaker_frame, no_speaker_frame, raw_frame, file_name, output_path):
    # This is to combine head position with raw file
    frame = cv2.imread(raw_frame)
    # cv2.imshow('frame',frame)
    no_speak_h = cv2.imread(no_speaker_frame[0])
    speak_h = cv2.imread(speaker_frame[0])
    masked_frame = cv2.addWeighted(no_speak_h, 1, speak_h, 1, 0)
    print('speaker',speak_h)
    print('no_speak_h',no_speak_h)
    com_frame = cv2.addWeighted(masked_frame, 0.5, frame, 1, 0)

    # filename format = videoNum_frameNum
    cv2.imwrite(output_path +'/com_head_feature/'+ file_name, com_frame)
    print('Fusion image saved in',output_path +'/com_head_feature/'+ file_name)


def fusion(raw_path, speaker_path, nonspeaker_path, output_path):
    print('========================Start to combine the speaker and raw frames.========================')
    print('now is at path:',os.getcwd())

    raw_fullname, raw_filename = get_all_file(raw_path, 'jpg')
    speaker_fullname, speaker_filename = get_all_file(speaker_path, 'png')
    sp_fullname, sp_filename = get_all_file(raw_path, 'jpg')
    for f_path, f_name in zip(raw_fullname, raw_filename):
        video_num = f_name[:3]
        frame_num = f_name[4:7]
        print('------video_num,frame_num',video_num,frame_num)
        speaker_path = [s for s in speaker_fullname if s.split('/')[-1][:3]==video_num and s.split('/')[-1].split('_')[1][1:].zfill(3)==frame_num and s.split('/')[-1].split('_')[2]=='speaker.png']
        nonspeaker_path = [s for s in speaker_fullname if s.split('/')[-1][:3]==video_num and s.split('/')[-1].split('_')[1][1:].zfill(3)==frame_num and s.split('/')[-1].split('_')[2]=='nonspeaker.png']

        if len(speaker_path) != 0:
            com_speaker_feature(speaker_path, nonspeaker_path, f_path, f_name, output_path)
            # com_head_feature(speaker_path, nonspeaker_path, f_path, f_name, output_path)
        else:
            continue
