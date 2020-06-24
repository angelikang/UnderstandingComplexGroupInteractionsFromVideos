import sys
import cv2
import face_recognition
from pathlib import Path

video_id = sys.argv[1]
 
video = f'./videosLabelled/{video_id}/webcam.mp4'
resultFolder = f'./run-result-face-images/{video_id}/'

Path(resultFolder).mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)

frames = 0
snapshots = 0
snapshotEachSeconds = 1.0

while(cap.isOpened()):
    
    ret, frame = cap.read()
    frames += 1
    videoTime = frames / fps
    if not ret:
        break

    if snapshots % 10 == 0:
        print("time ", videoTime)
        cv2.imwrite(resultFolder + str(snapshots) + "_img.jpg", frame)

        face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="cnn")    

        faceno = 0
        for face_location in face_locations:

            # Print the location of each face in this image
            top, right, bottom, left = face_location
            print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

            top = max(0, top - 20)
            left = max(0, left - 20)
            bottom = bottom + 20
            right = right + 20

            image = frame[top:bottom, left:right]

            if len(face_recognition.face_encodings(image)) > 0:
                cv2.imwrite(resultFolder + str(snapshots) + "_face_" + str(faceno) + ".jpg", image)
                faceno += 1


    snapshots += 1

    # skip frames until 
    while (cap.isOpened() and videoTime < snapshots * snapshotEachSeconds):
        cap.read()
        frames += 1
        videoTime = frames / fps

cap.release()
cv2.destroyAllWindows()
