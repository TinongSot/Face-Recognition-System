from .preprocess import fetch_lfw_people
import tqdm
import numpy as np

if __name__ == "__main__":
    X, Y = fetch_lfw_people(shuffle=False)

    __x = []
    for x in tqdm.tqdm(X):
        __x.append(x())

    x_data = np.array(__x)
    x_data.dump("Dataset/x.npy")
    Y.dump("Dataset/y.npy")
