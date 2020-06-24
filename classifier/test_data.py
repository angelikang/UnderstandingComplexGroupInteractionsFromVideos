import unittest
import data
import numpy as np

class DataTests(unittest.TestCase):
    def test_k_fold(self):
        for i  in range(6):
            test = data.k_fold_testing(i, 2)
            train = data.k_fold_training(i, 2)

            self.assertEqual(len(test), 2)
            self.assertEqual(len(train), len(data.videos) - 2)
            self.assertSetEqual(set(data.videos), set(test + train))
    
    def test_center_position(self):
        features = np.array([[[
            0,0,0,
            1,1,0,
            2,2,0 ]]])

        data.center_position(features)
        self.assertListEqual(list(features[0,0]), [
            -1,-1,0,
            0,0,0,
            1,1,0])

    def test_transform_target(self):
        result = data.transform_target(np.array([0,1,2,3]))
        self.assertListEqual(list(result), 
            [1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0, 
             0, 0, 0, 1,])
