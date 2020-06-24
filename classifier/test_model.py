import unittest
import torch
import model

class ModelTests(unittest.TestCase):
    def test_TeamModel(self):
        m = model.TeamModelFaster(3, 201, 4)
        self.model_forward(m)

    def test_SimpleModel(self):
        m = model.SimpleModel(3, 201, 4)
        self.model_forward(m)

    def test_TeamModel2(self):
        m = model.TeamModel2(3, 201, 4, dropout_p=0.2)
        self.model_forward(m)

    def test_SimpleModel2(self):
        m = model.SimpleModel2(3, 201, 4, dropout_p=0.2)
        self.model_forward(m)
    
    def test_SimpleModelDouble(self):
        m = model.SimpleModelDouble(3, 201, 4, dropout_p=0.2)
        self.model_forward(m)
        

    def model_forward(self, m): 
        torch.autograd.set_detect_anomaly(True)
        m.cuda()
        data = torch.rand(300, 3, 201).cuda()
        result = m(data)
        self.assertEqual(result.shape, torch.Size([300, 3, 4]))
