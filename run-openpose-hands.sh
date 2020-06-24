# limit number of used GPU devices
export CUDA_VISIBLE_DEVICES=1,2

for VIDEO in  1542 1543 1544 1550 1552 1565 1568 1576 1584 1585 1593 1600 1602 1606
do
    RESULTS=./run-result-hands
    VIDEO_RESULT=$RESULTS/$VIDEO

    echo ------------------
    echo running: $VIDEO
    echo

    mkdir -p $RESULTS/$VIDEO
    ./openpose/build/examples/openpose/openpose.bin \
        --model_folder ./openpose/models/ \
        --video ./videosLabelled/$VIDEO/webcam.mp4 \
        --write_json=$VIDEO_RESULT \
        --display=0 \
        --render_pose 0 \
        --model_pose BODY_25 \
        --cli_verbose=1 \
        --hand
done