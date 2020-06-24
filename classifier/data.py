import numpy as np

videos = ["1542", "1543", "1544", "1550", "1552", "1565", "1568", "1576", "1584", "1585", "1593", "1600", "1602", "1606"]

def k_fold_training(i, size):
    return videos[0:i * size] + videos[(i + 1) * size:]

def k_fold_testing(i, size):
    return videos[i * size:(i + 1) * size]

def transform_target(row):
    result = np.zeros(4 * len(row), dtype='float32')
    for x, value in enumerate(row):
        result[int(value + (x * 4))] = 1
    return result

def load_video_data(video_id):
    return np.genfromtxt(f'./data/{video_id}_missing_label.csv', delimiter=',', dtype='float32')
    
def prepare(video_id):
    video_data = load_video_data(video_id)
    target = video_data[:, 0:3]
    features = video_data[:, 3:]

    target = np.apply_along_axis(transform_target, 1, video_data[:, 0:3])
    target = target.reshape(target.shape[0], 3, 4)
    features = features.reshape(features.shape[0], 3, 75 + 63 + 63)

    return features, target   

def load_prepared(video_id):
    return np.load(f"data/{video_id}-features.npy"), np.load(f"data/{video_id}-target.npy")

def load_prepared_normalized(video_id):
    return np.load(f"data/{video_id}-features-normalized.npy"), np.load(f"data/{video_id}-target.npy")

def load_original_data(video_id):
    return  np.genfromtxt(f'./videosLabelled/{video_id}/data.csv', delimiter=',',  names=True)

def load_10fps(video_id):
    return np.load(f"data/{video_id}-features-10fps.npy"), np.load(f"data/{video_id}-target-10fps.npy")

def load_10fps_normalized(video_id):
    return np.load(f"data/{video_id}-features-10fps-normalized.npy"), np.load(f"data/{video_id}-target-10fps.npy")


def video_fps(video_id):
    data = load_original_data(video_id)
    return data["fps"][0]

def center_position(features):
    BODY_NOSE = 0
    BODY_NECT = 1

    for frame in range(features.shape[0]):
        for person in range(features.shape[1]):
            neck_x = features[frame, person, BODY_NECT * 3]
            neck_y = features[frame, person, BODY_NECT * 3 + 1]

            for p in range(int(features.shape[2] / 3.0)):
                features[frame, person, p * 3] -= neck_x
                features[frame, person, p * 3 + 1] -= neck_y

