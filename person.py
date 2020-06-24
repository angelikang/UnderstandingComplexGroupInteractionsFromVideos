import sys
import cv2
import math

BODY_NOSE = 0
BODY_NECT = 1
BODY_RIGHT_EYE = 15
BODY_LEFT_EYE = 16
BODY_RIGHT_EAR = 17
BODY_LEFT_EAR = 18

# Taken from OpenPose source code for reference (https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#keypoint-ordering-in-cpython)
# Result for BODY_25 (25 body parts consisting of COCO + foot)
# const std::map<unsigned int, std::string> POSE_BODY_25_BODY_PARTS {
#     {0,  "Nose"},
#     {1,  "Neck"},
#     {2,  "RShoulder"},
#     {3,  "RElbow"},
#     {4,  "RWrist"},
#     {5,  "LShoulder"},
#     {6,  "LElbow"},
#     {7,  "LWrist"},
#     {8,  "MidHip"},
#     {9,  "RHip"},
#     {10, "RKnee"},
#     {11, "RAnkle"},
#     {12, "LHip"},
#     {13, "LKnee"},
#     {14, "LAnkle"},
#     {15, "REye"},
#     {16, "LEye"},
#     {17, "REar"},
#     {18, "LEar"},
#     {19, "LBigToe"},
#     {20, "LSmallToe"},
#     {21, "LHeel"},
#     {22, "RBigToe"},
#     {23, "RSmallToe"},
#     {24, "RHeel"},
#     {25, "Background"}


BODY_PAIRS = [
    (17,15),
    (15,0), 
    (0,16), 
    (16,18),

    (0,1),

    (4,3),
    (3,2),
    (2,1),

    (1,5),
    (5,6),
    (6,7),

    (1,8),
    (8,9),
    (9,10),  
    (10,11),
    (11,24),
    (11,22),
    (22,23),

    (1,2),
    (1,5), 
    (2,3),
    (3,4), 
    (5,6),
    (6,7),  
    
    (8,12),
    (12,13),
    (13,14), 
    (14,21),
    (14,19),
    (19,20),
]


class Person:
    def __init__(self, opData):
        self.opData = opData
        self.idProbs = [1,1,1,1]
        self.previousFrameDistance = math.inf
        self.normaliseProbs()

    def normaliseProbs(self):
        s = sum(self.idProbs)
        for i in range(len(self.idProbs)):
            self.idProbs[i] /= s

    def getPosePoint(self, data, index):
        return (data[index * 3], data[index * 3 + 1], data[index * 3 + 2])

    def includePoints(self, bb, points):
        left, top, right, bottom = bb

        for x, y, c in points:
            if c > 0: 
                top = min(top, y)
                bottom = max(bottom, y)
                left = min(left, x)
                right = max(right, x)

        return (left, top, right, bottom)

    def emptyBB(self):
        top = math.inf
        bottom = 0
        left = math.inf
        right = 0
        return (left, top, right, bottom)

    def isEmpty(self, bb):
        left, top, right, bottom = bb
        return top == math.inf and bottom == 0 and left == math.inf and right == 0

    def extendBox(self, bb, t, r, b, l):
        left, top, right, bottom = bb

        height = bottom - top
        width = right - left

        top = top - t * height
        right = right + r * width
        bottom = bottom + b * height
        left = left - l * width

        return (left, top, right, bottom)


    def faceBoundingBoxFromPose(self):
        body = self.opData["pose_keypoints_2d"]
        nose = self.getPosePoint(body, BODY_NOSE)
        leye = self.getPosePoint(body, BODY_LEFT_EYE)
        reye = self.getPosePoint(body, BODY_RIGHT_EYE)
        lear = self.getPosePoint(body, BODY_LEFT_EAR)
        rear = self.getPosePoint(body, BODY_RIGHT_EAR)

        top = math.inf
        bottom = 0
        left = math.inf
        right = 0

        center = self.includePoints(self.emptyBB(), [nose, leye, reye])
        if self.isEmpty(center):
            return None

        bb = self.extendBox(center, 1, 0.5, 2.5, 0.5)
        return  self.includePoints(bb, [lear, rear])

    def faceBoundingBox(self):
        print( self.opData["face_keypoints_2d"])
        face = self.opData["face_keypoints_2d"]
        top = math.inf
        bottom = 0
        left = math.inf
        right = 0

        it = iter(face)
        for x in it:
            y = next(it)
            next(it)
            top = min(top, y)
            bottom = max(bottom, y)
            left = min(left, x)
            right = max(right, x)
        return (left, top, right, bottom)
    
    def extendedFaceBoundingBox(self):
        (left, top, right, bottom) = self.faceBoundingBox()
        height = bottom - top
        width = right - left

        if width <= 2:
            return (0,0,0,0)
        top = top - 0.3 * height
        right = right + 0.1 * width
        bottom = bottom + 0.1 * height
        left = left - 0.1 * width
        return (left, top, right, bottom)

    def distance(self, other):
        """Calculate Distance between 2 persons as sum of open pose keypoint distances"""

        MISSING_KEYPOINT_DISTANCE = 50
        keypoints = self.opData["pose_keypoints_2d"]
        otherKeypoints = other.opData["pose_keypoints_2d"]
        
        distance = 0
        it = iter(zip(keypoints, otherKeypoints))
        for (x1,x2) in it:
            (y1, y2) = next(it)
            if ((x1 == 0 and y1 == 0 and 
                x2 != 0 and y2 != 0) or
                (x1 != 0 and y1 != 0 and 
                x2 == 0 and y2 == 0)):
                distance += MISSING_KEYPOINT_DISTANCE
            else:
                distance += math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            next(it)
        return distance
            
    def identify(self, id, probability):
        # in case of fail give equal chance of other ids
        probabilityOfWrong = (1 - probability) / (len(self.idProbs) - 1)
        for i in range(len(self.idProbs)):
            if i == id:
                self.idProbs[i] *= probability
            else:
                self.idProbs[i] *= probabilityOfWrong
        self.normaliseProbs()


    def mixInProbabilities(self, p, probabilitySame):
        self.idProbs = [me * (1 - probabilitySame) + other * probabilitySame for (me, other) 
                            in zip(self.idProbs, p.idProbs)]

    def draw(self, frame):
        pose = self.opData["pose_keypoints_2d"]
        for i1,i2 in BODY_PAIRS:
            x1 = pose[i1 * 3]
            y1 = pose[i1 * 3 + 1]
            x2 = pose[i2 * 3]
            y2 = pose[i2 * 3 + 1]
            cv2.circle(frame, (int(x1), int(y1)), 2, (0,0,255), 1)

            if (x1 != 0 or y1 != 0) and (x2 != 0 or y2 != 0):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 1)
