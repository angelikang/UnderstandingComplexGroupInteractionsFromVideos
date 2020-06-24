import torch
import torch.optim as optim
import model
import torch.nn as nn
import data
import numpy as np
import random
import sys
import math
from pathlib import Path

# Helper function to transform features for model
def most_common(arr):
        counts = np.bincount(arr)
        return np.argmax(counts)

def most_common_row(data):
    return np.apply_along_axis(most_common,0, data)

# Helper function to count results
def add_results(results, out, target):
    out_labels = np.argmax(out.cpu().detach(), axis=1)
    for i in range(len(out_labels)):
        results[i, target[i], out_labels[i]] += 1

# read in arguments
model_name = sys.argv[1]
model_type = sys.argv[2]
data_fold = int(sys.argv[3])

Path(f'./models/{model_name}/').mkdir(parents=True, exist_ok=True)

# print all output to log file
f = open(f'./models/{model_name}/_run.log','w')
sys.stdout = f
sys.stderr = f

print ("Model: ", model_type)
print ("Fold: ", data_fold)

torch.autograd.set_detect_anomaly(False)

batch_size = 300
training_set = data.k_fold_training(data_fold, 2)
testing_set = data.k_fold_testing(data_fold, 2)

print("Training set: ", training_set)

# Construct model class based on command line argument
Model = getattr(model, model_type)

net = Model(3, 75 + 63 + 63, 4, dropout_p=0.2)
net.cuda()

# optimizer can be changed here
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# potential other optimizers
#optimizer = optim.SGD(net.parameters(), lr=0.01)
#optimizer = optim.Adam(net.parameters())

w = torch.tensor([11.687, 4.157, 1.955, 6.161]).cuda()
criterion = nn.CrossEntropyLoss(weight=w)

# Uncomment for dynamic reduction of learning rate
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, 
#     mode='min', 
#     factor=0.1, 
#     patience=10, 
#     verbose=True, 
#     threshold=0.00001, 
#     threshold_mode='rel', 
#     cooldown=0, 
#     min_lr=0, 
#     eps=1e-08)

# Training and testing loop, implemented according to PyTorch manual (https://pytorch.org/docs/stable/optim.html)

for epoch in range(500):
    epoch_loss = 0.0
    step_count = 0
    
    print(f"New Epoch: {epoch}")
    video_run = 0
    random.shuffle(training_set)

    results_epoch = np.zeros((3, 4, 4),dtype=np.int32)

    # Training: 
    net.train()
    for video_id in training_set:
        print(f"New Video: {video_id}")

        results = np.zeros((3, 4, 4),dtype=np.int32)
        features, target_labels = data.load_10fps(video_id)

        # Cross entropy loss expect target to be index of class rather than the 1d array of class indicators
        target = np.argmax(target_labels, axis=2)

        running_loss = 0.0
        i = 0
        step = 0
        while i < features.shape[0]:

            next = min(i+batch_size, features.shape[0])

            batch_input = torch.from_numpy(features[i:next]).cuda()
            batch_target = torch.from_numpy(most_common_row(target[i:next])).cuda()

            optimizer.zero_grad()   # zero the gradient buffers
            output = net(batch_input)

            # take output of the last frame to represent behaviour in the 30s window
            class_count = 4
            window_output = output[-1].view(-1, class_count)

            # calculate loss and update weights
            loss = criterion(window_output, batch_target)
            loss.backward(retain_graph=True)
            optimizer.step()

            add_results(results, window_output, batch_target)

            step += 1
            running_loss += loss.item() * batch_size
            i = next

        
        print('[%d, %5d] loss: %.6f' %
                (epoch, i, running_loss / i))

        print ("video results: \n", np.sum(results, axis=0))
        epoch_loss += running_loss
        step_count += i

        results_epoch += results
        
        running_loss = 0.0
        f.flush()
        video_run +=1
    
    # Uncomment for dynamic reduction of learning rate
    # scheduler.step(epoch_loss / step_count)

    print ("Epoch loss: ", epoch_loss / step_count)
    print ("Step count: ", step_count)

    print ("epoch results: \n", np.sum(results_epoch, axis=0))
    print ("individual: \n", results_epoch)


    torch.save(net.state_dict(), f"models/{model_name}/model-{epoch}.torch")
    f.flush()

    print("Testing: \n")

    net.eval()
    test_results = np.zeros((3, 4, 4),dtype=np.int32)

    for video_id in testing_set:
        print(f"Eval Video: {video_id}")
        results = np.zeros((3, 4, 4),dtype=np.int32)
        features, target_labels = data.load_10fps(video_id)
        # Cross entropy loss expect target to be index of class rather than the 1d array of class indicators
        target = np.argmax(target_labels, axis=2)

        i = 0
        while i < features.shape[0]:

            next = min(i+batch_size, features.shape[0])

            batch_input = torch.from_numpy(features[i:next]).cuda()
            batch_target = torch.from_numpy(most_common_row(target[i:next])).cuda()
            
            output = net(batch_input)
            class_count = 4
            window_output = output[-1].view(-1, class_count)
            add_results(results, window_output, batch_target)
            i = next

        print ("video results: \n", np.sum(results, axis=0))
        test_results += results 

    print ("overall: \n", np.sum(test_results, axis=0))
    print("individual: \n", test_results)
    print("---")
   
    f.flush()
