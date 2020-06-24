# Understanding Complex Group Interactions From Videos - Source code


## Prerequisites

The project uses OpenPose for pose estimation from video recording.
Please follow installation guide at https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md. Scripts assume that open pose binary is located at `./openpose/build/examples/openpose/openpose.bin` 

The project use Python3 and following python libraries: cv2 (opencv-python), pytorch, numpy, scipy, face_recognition (all installable through pip).

All scripts assumes that there are videos available in folder `./videosLabelled`
These videos are not distributed with the source code to maintain privacy of the participants.
All scripts except video files to be located at `./videosLabelled/{video_id}/webcam.mp4`.
The video activity label data are expected to be in `./videosLabelled/{video_id}/data.csv`.

To reproduce the result following scripts needs to be run in order:

## Student recognition

`run-openpose-hands.sh`: This script will run openPose pose estimation on all videos saving result into `./run-result-hands`

`run-images.sh`: Produces cut out face images of persons from each video every second under `./run-result-face-images/{video_id}/`. Couple of these images needs to be placed into `./student-faces/{video_id}/` corresponding to individual students (named `A_0.jpg`, `A_1.jpg` ... for student A).

`run-recognition.sh`: This script will use OpenPose data, videos, and student faces and output open pose data together with identification in 2 formats under `./data/{video_id}.csv` and `./data/{video_id}.json`

## Data preprocessing

Folder `classifier` contains implementation of second stage of machine learning pipeline

`classifier/prepare_data.py`: Run this script to transform data from csv file to numpy format. This format is much faster to load.

`classifier/normalize_features .py`: Script to generate normalised data and save in files.

`classifier/prepare_data_10fps.py`: Script to sample original data to 10 frames per second.

`classifier/normalize_features_10fps.py`: Script to normalise data sampled to 10 frames per second.


## Classifier

Run script `run-learn.sh` to train the model. This script trains the model on 7-folds, excluding 2 videos in each. Results are written into the log files, including performance on the testing set.
Few things can be changed in this script.

## Tests
To run unit test execute: `python3 -m unittest` in individual folders.
