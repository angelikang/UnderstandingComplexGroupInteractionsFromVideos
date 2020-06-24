
import data
import numpy as np

for video in data.videos:
    print(f"normalizing video {video}")
    
    # features dimensions: frame, person, features
    features, _ = data.load_10fps(video)
    data.center_position(features)
    np.save(f"data/{video}-features-10fps-normalized.npy", features)
