#This is a whole flow demo shell file

# 1. Split raw video into frames

### Step1: Split videos into sequences(frames)

echo '=======================================Now is splitting video into frames===================================='

python Data_Preprocessing.py --videoPath ./data/Raw_Video/videos/012.mp4  --imgPath ./output/image_frame/012


# ### Step2: Go to folder './utils/Speaker_detector', By fusion audio&video to detect speakers, here we use pre-trained model for speaker dectection
#(1) Install environment yaml file from google drive: https://drive.google.com/file/d/1bVlD1bTi1-auGVF1qTJB3iXrSOxudyN3/view?usp=sharing
#(2) create conda environment, you should also install tensorflow mannually:

conda env create -f ./configs/Speaker_Detector/Speaker_Detection.yaml
conda activate yuqi_summer3

# You will still need to mannually install some libraries and create some folders, no worries, just follow the warning is ok

echo '=======================================Now is detecting speakers by audio&video===================================='
cd ./utils/Speaker_detector/

sh build_face_map_Generate_FaceMap.sh ../../data/Raw_Video/videos/012.mp4  012  ../../data/output/speaker_detect/

# ### Step3: Go to folder './utils/Mask_RCNN/'. Train VGS model to detect gaze points. Here we use pre_trained checkpoint directly, skip this step.
echo '=======================================Train VGS model to detect gaze points. Here we use pre_trained checkpoint directly, skip this step===================================='

# ### Step4: Go to folder './utils/Mask_RCNN/'. Detect gaze points from Mask_RCNN

cd ../Mask_RCNN/
#(1) Install environment yaml file from google drive: https://drive.google.com/drive/folders/1gcACzyd_8tXTU4HsVNFLyC0_ZBCd7iwm?usp=sharing
#(2) create conda environment from environment.yaml: https://drive.google.com/drive/folders/1XN_GjB-eygYPSX5YPXE_FpQyaCY0SbSs?usp=share_link:
conda activate yuqi_maskrcnn
cd ./tools/

python test_yuqi.py


### Step5: Go to home folder. Use MLP [checkpoint]() to detect best gaze point.

```
python MLP_Best_Gaze.py
```

# In this step you will generate the final csv file [here](https://drive.google.com/file/d/1JBDwW9fbwGz-gl2hzAI9voNz3PR9Rf0T/view?usp=sharing).


# ## Contact
# If you have any questions, please email Yuqi Hou at houyuqi21701@gmail.com

