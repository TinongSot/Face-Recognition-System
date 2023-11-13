import dill
import tqdm
from preprocess.preprocess import dataset, fetch_lfw_people
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np

if __name__ == "__main__":
    # with open("Dataset/catalog.dill", "rb") as f:
    #     data = dill.load(f)
    #     print(data)
    # buf = data[0][0]()
    # data = LazyData("Dataset/Raw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg", load_strategy="image")

    # dst = dataset("Dataset/Raw", shuffle=False)
    # # data = dst[0][0]()
    # # img = Image.fromarray(data, "RGB").convert("L")
    # # data = np.array(img) / 255
    # # plt.imshow(data, cmap="gray", vmin=0, vmax=1)
    # # plt.colorbar()
    # # plt.show()

    # X = dst[:, 0]
    # Y = dst[:, 1]

    # print(X.shape)
    # print(X[0:5])
    # print(Y.shape)
    # print(Y[0:5])

    X, Y = fetch_lfw_people()
    __x = []
    for x in tqdm.tqdm(X):
        __x.append(x())

    x_data = np.array(__x)
    x_data.dump("Dataset/x.npy")
    Y.dump("Dataset/y.npy")
