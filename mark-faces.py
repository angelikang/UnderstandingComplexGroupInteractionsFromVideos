import sys
import cv2
import face_recognition
import scipy.optimize as opt
import json
import csv
import math
import dlib
from pathlib import Path
from os import listdir
from os.path import isfile, join
from numpy import genfromtxt

from person import Person

video_id = sys.argv[1]

video = f'./videosLabelled/{video_id}/webcam.mp4'
resultFolder = f'./run-result-4/{video_id}/'
sourceFolder = f'./run-result-hands/{video_id}/'
studentFolder = f'./student-faces/{video_id}/'

resultJson = f'./data/{video_id}.json'
resultCsv = f'./data/{video_id}.csv'

data = genfromtxt(f'./videosLabelled/{video_id}/data.csv', delimiter=',',  names=True)
jsonFile = open(resultJson, 'w')
csvFile = open(resultCsv, 'w')
csvWritter  = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

print(studentFolder)

data_result_folder = f'./run-result-id-data/{video_id}/'

def getEncoding(img):
    loaded_image = face_recognition.load_image_file(studentFolder + img)
    return face_recognition.face_encodings(loaded_image)[0]

names = [
    "A",
    "B",
    "C",
]

counts = { "A": 0, "B": 0, "C": 0 }

def loadEncodings():
    known_faces = []
    ids = []
    images = [f for f in listdir(studentFolder) if isfile(join(studentFolder, f))]

    for image in images:
        for id, name in enumerate(names):
            if image.startswith(name):
                #print(image)
                known_faces.append(getEncoding(image))
                ids.append(id)

    return known_faces, ids
        
known_faces, ids = loadEncodings()
persons = []

UNKNOWN = 3

last_known_positions = [
    None,
    None,
    None
]

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)

frames = 0
snapshots = 0
snapshotEachSeconds = 1.0
currentDataRow = 0

while(cap.isOpened()):
    
    while frames < 0:
        cap.read()
        frames += 1
        snapshots += 1

    ret, frame = cap.read()
    videoTime = frames / fps
    if not ret:
        break

    # load persons
    with open(sourceFolder + "webcam_" + "000{:09d}".format(frames) + "_keypoints.json") as file:
        openPose = json.load(file)

    perviousPersons = persons
    persons = []

    for personData in openPose["people"]:
        persons.append(Person(personData))

    # update probability based on previous frame
    for person in persons:
        
        person.draw(frame)

        if len(perviousPersons) > 0: 

            distances = [person.distance(previous) for previous in perviousPersons]
            minDistance = min(distances)
            id = distances.index(minDistance)
            previous = perviousPersons[id]


            person.previousFrameDistance = minDistance

            if minDistance < 100:
                person.mixInProbabilities(previous, 1)
            if minDistance < 500:
                person.mixInProbabilities(previous, 0.999)
            elif minDistance < 1000:
                person.mixInProbabilities(previous, 0.95)
            elif minDistance < 1200:
                person.mixInProbabilities(previous, 0.7)

            (left, top, right, bottom) = person.extendedFaceBoundingBox()
            cv2.putText(frame, str(minDistance),(int(left),int(bottom)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
    
    # try to recognize using faces every 10th frame
    if frames % 10 == 0:

        faces = []
        persons_with_faces = []
        for person in persons:

            bb = person.faceBoundingBoxFromPose() # .extendedFaceBoundingBox()
            if bb != None:

                left, top, right, bottom = bb
                faces.append((
                    int(top), 
                    int(right), 
                    int(bottom), 
                    int(left)))
                persons_with_faces.append(person)
        
        face_encodings = face_recognition.face_encodings(frame, faces)
        
        for i in range(len(persons_with_faces)):
            person = persons_with_faces[i]
            encoding = face_encodings[i]

            distanceToKnown = face_recognition.face_distance(known_faces, encoding)

            # lets take best match higher then 0.5
            minDistance = min(distanceToKnown)
            id = distanceToKnown.tolist().index(minDistance)

            if minDistance < 0.5:
                
                # detect wrong identification if there was different pose identified strongly with this id:
                id = ids[id]
                false_positiove = False
                for other in persons:
                    if person != other and other.idProbs[id] > 0.7:
                        false_positiove = True
                        break

                if not false_positiove:
                    person.identify(id, 0.5)
                    left, top, right, bottom = person.faceBoundingBoxFromPose() # .extendedFaceBoundingBox()

                    cv2.putText(frame, "!id: " + str(names[ids[id]]),(int(right),int(bottom)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)


    # pick unique best matches
    probs = [p.idProbs[0:3] for p in persons]
    negativeProbs = [[1-p for p in row] for row in probs]

    identified_persons = []

    if len(probs) > 0:
    
        [found_ids, known_ids] = opt.linear_sum_assignment(negativeProbs)

        for i in range(len(known_ids)):
            name = names[known_ids[i]]
            prob = probs[found_ids[i]][known_ids[i]]
            person = persons[found_ids[i]]

            # it is very likely the the whole team is captured by the camera
            # to get good result we can use quite low threshold and rely on hungarian algorithm 
            # to uniquely identify students
            if prob > 0.30:

                person.opData["person_id"] = name
                person.opData["meta"] = {
                    "classification": data[currentDataRow][name],
                    "id_prob" : prob,
                    "id_probs" : person.idProbs,
                    "previous_frame_distance" : person.previousFrameDistance
                }
                identified_persons.append(person)

                face = person.faceBoundingBoxFromPose()
                if face != None:
                    p_left, p_top, p_right, p_bottom = face
                    cv2.rectangle(frame, (int(p_left),int(p_top)), (int(p_right), int(p_bottom)), (0,255,255), 1)
            
                    cv2.putText(frame, name + " " + str(prob),(int(p_left),int(p_top)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
    
    d = resultFolder + "/" + str(math.floor(frames / 1000))
    Path(d).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(d + "/000{:09d}".format(frames) + "_probabilistic_id.jpg", frame)

    result_data = {}
    echo = "frame: " + str(frames) + " "

    for p in identified_persons:
        counts[p.opData["person_id"]] += 1
        result_data[p.opData["person_id"]] = p.opData
        echo += p.opData["person_id"] + " " + str(p.opData["meta"]["id_prob"]) + " " + str(p.opData["meta"]["previous_frame_distance"]) + " "


    out_data = []
    for name in names:
        if name in result_data: 
            p = result_data[name]
            out_data.append(p["meta"]["classification"])
        else:
            out_data.append(0)

    for name in names:
        if name in result_data: 
            p = result_data[name]
            out_data.extend(p["pose_keypoints_2d"])
            out_data.extend(p["hand_left_keypoints_2d"])
            out_data.extend(p["hand_right_keypoints_2d"])
        else:
            out_data.extend([0] * (75 + 63 + 63))


    csvWritter.writerow(out_data)

    print(echo)

    result = {
        "frame": frames,
        "people": result_data
    }
    Path(data_result_folder).mkdir(parents=True, exist_ok=True)

    json.dump(result, jsonFile)
    jsonFile.write('\n')

    frames += 1
    snapshots += 1

    if data[currentDataRow]["last_frame"] < frames:
        currentDataRow += 1

print("frames: " + str(frames) + " counts: " + str(counts))

cap.release()
jsonFile.close()
csvFile.close()