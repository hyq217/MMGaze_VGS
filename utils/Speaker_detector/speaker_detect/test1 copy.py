#!/usr/bin/python
"""
This file is the run_pipeline.py form syncnet_python
It has been modified to:
  1) work with scenedetect v0.5 instead of only 0.3.5
  2) not work only on videos with fps 25
"""
import sys, time, os, pdb, argparse, pickle, subprocess
import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
import cv2
import yaml

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.detectors import ContentDetector

from scipy.interpolate import interp1d
from utils import label_map_util
from scipy.io import wavfile
from scipy import signal