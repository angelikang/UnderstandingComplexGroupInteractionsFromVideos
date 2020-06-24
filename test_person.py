import unittest
import person
import numpy as np

class PersonTests(unittest.TestCase):
    def test_create(self):
        p = person.Person(TEST_PERSON)
        self.assertEqual(sum(p.idProbs), 1)

    def test_faceBoundingBox_correct(self):
        p = person.Person(TEST_PERSON)
        face = p.faceBoundingBoxFromPose()
        self.assertFalse(p.isEmpty(face))
        self.assertEqual(face, (-2.0, 0, 2.0, 1))

    def test_mixInProbabilities(self):
        p1 = person.Person(TEST_PERSON)
        p2 = person.Person(TEST_PERSON)

        p2.identify(0, 1) # 100% identified as id = 0
        p1.mixInProbabilities(p2, 0.6)
        # probability for id = 0 is combined as: 0.6 * 1 + 0.4 * 0.25 = 0.7
        self.assertListEqual(p1.idProbs, [0.7, 0.1, 0.1, 0.1 ])

    def test_identify(self):
        p = person.Person(TEST_PERSON)
        p.identify(1, 0.7)  # 70% identified as id = 1
        # comparison needs to ignore floating point arithmetic error
        self.assertAlmostEqual(p.idProbs[1], 0.7)
        self.assertAlmostEqual(p.idProbs[0], 0.1)
        self.assertAlmostEqual(p.idProbs[2], 0.1)
        self.assertAlmostEqual(p.idProbs[3], 0.1)

    def test_distance(self):
        p1 = person.Person(TEST_PERSON)
        p2 = person.Person(EMPTY_PERSON)

        self.assertEquals(p1.distance(p1), 0)

TEST_PERSON = { "pose_keypoints_2d" : [
    0,0,1, # BODY_NOSE
    0,0,0, # BODY_NECT
    0,0,0,
    0,0,0,
    0,0,0,
    0,0,0,
    0,0,0,
    0,0,0,
    0,0,0,
    0,0,0,
    0,0,0,
    0,0,0,
    0,0,0,
    0,0,0,
    0,0,0,
    1,0,1, # BODY_RIGHT_EYE
    -1,0,1, # BODY_LEFT_EYE
    1,1,1, # BODY_RIGHT_EAR
    -1,1,1, # BODY_LEFT_EAR
    0,0,0,
    0,0,0,
    0,0,0,
    0,0,0,
    0,0,0,
    0,0,0]}