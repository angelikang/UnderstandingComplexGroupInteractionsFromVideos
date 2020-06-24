import data
import numpy as np

# prepare video data to be 10 fps
for video in data.videos:
    features, target = data.load_prepared(video)
    original_fps = data.video_fps(video)
    target_fps = 10
    ratio =  target_fps / original_fps

    rows = []
    next_frame = 0
    for i in range(target.shape[0]):
        if next_frame <= i * ratio:
            next_frame += 1
            rows.append(i)
            
    trim = len(rows) - (len(rows) % (30 * target_fps))
    rows = rows[:trim]

    target = np.take(target, rows, 0)
    features = np.take(features, rows, 0)

    print(next_frame)
    print()

    np.save(f"data/{video}-features-10fps.npy", features)
    np.save(f"data/{video}-target-10fps.npy", target)
