
import data
import numpy as np

for video in data.videos:
    print(f"normalizing video {video}")
    
    # features dimensions: frame, person, features
    features, _ = data.load_prepared(video)
    data.center_position(features)
    np.save(f"data/{video}-features-normalized.npy", features)
