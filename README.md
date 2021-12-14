# FallNet

This repository contains PyTorch models of I3D  based on the following repositories: 

    • https://github.com/piergiaj/pytorch-i3d/blob/master/pytorch_i3d.py 
Models: I3D 

Datasets: UCF-101, HMDB51, OOPS 

#FallAction Dataset

Download FallAction Dataset from site: https://figshare.com/articles/dataset/FallAction_dataset_zip/17157845


Download the HMDB51 dataset from:

Download videos and train/test splits: https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/.

Follows steps for UCF-101

Convert from avi to jpg files using utils/video_jpg_ucf101_hmdb51.py

• python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory

Generate n_frames files using utils/n_frames_ucf101_hmdb51.py

• python utils/n_frames_ucf101_hmdb51.py jpg_video_directory

Generate annotation file in json format similar to ActivityNet using utils/ucf101_json.py
annotation_dir_path includes classInd.txt, trainlist0{1, 2, 3}.txt, testlist0{1, 2, 3}.txt

• python utils/ucf101_json.py annotation_dir_path




# Installation
Clone and install:

git clone https://github.com/Nigam-Niti/FallNet.git 

cd FallNet 

pip install -r requirements.txt 

OR

directly go to colab link https://colab.research.google.com/drive/1Wpn07PMyokfecEDfeXkCHnxPYTtz0Gh_?usp=sharing


# Requirements 
    • Python 2.7 or 3.5+ (developed with 3.7)
    • Numpy (developed with 1.15.0)
    • PyTorch >= 0.4.0
    • TensorboardX (optional)
    • PIL (optional)
    • six
      
    • numpy
      
    • scipy
      
    • matplotlib
      
    • libavformat-dev libavdevice-dev
      
    • av == 6.2.0
