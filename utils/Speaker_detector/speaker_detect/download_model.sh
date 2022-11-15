# SyncNet model

wget http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model -O speaker_detect/data/syncnet_v2.model

# For the pre-processing pipeline

wget http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/face_detection_tf.zip -O facedet.zip

mkdir speaker_detect/protos
unzip facedet.zip -d speaker_detect/protos/
rm -f facedet.zip

cat /dev/null > speaker_detect/protos/__init__.py
cat /dev/null > speaker_detect/utils/__init__.py