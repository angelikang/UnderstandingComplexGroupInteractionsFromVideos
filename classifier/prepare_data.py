import data
import numpy as np

for video in data.videos:
    features, target = data.prepare(video)
    np.save(f"data/{video}-features.npy", features)
    np.save(f"data/{video}-target.npy", target)

