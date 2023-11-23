import os
from re import X
import tqdm
from Code.preprocess.preprocess import dataset, fetch_lfw_people
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from PIL import Image
import numpy as np
from rich import print
from Code.preprocess.visualization import visualize_image
from Code.preprocess.export import dump_test_files

if __name__ == "__main__":
    plt.style.use("ggplot")
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

    # X, Y = fetch_lfw_people(shuffle=False, min_faces=60, max_faces=100, hard_limit=True)
    # X, Y = fetch_lfw_people(shuffle=True)

    # __x = []
    # for x in range(10):
    #     img = Image.fromarray(X[x](), "RGB").convert("L")
    #     data = np.array(img) / 255
    #     __x.append(data)

    # print(X.shape)
    # print(X[0:5])
    # print(Y.shape)
    # print(Y[0:5])
    # __x = []
    # for x in tqdm.tqdm(X):
    #     __x.append(x())

    # x_data = np.array(__x)
    # x_data.dump("Dataset/x.npy")
    # Y.dump("Dataset/y.npy")
    # print(X.shape)
    # print(len(set(Y)))

    # freq = {}
    # for y in Y:
    #     if y not in freq:
    #         freq[y] = 0
    #     freq[y] += 1

    # sort = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    # print(sort)
    # max_value = sort[0][1]
    # counts = []
    # for i in range(0, max_value):
    #     # count how many people have at least i images
    #     count = 0
    #     for _, v in sort:
    #         if v >= i:
    #             count += 1

    #     counts.append(count)
    # print(counts)
    # print(counts[20])

    # plt.plot(counts)
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.show()
    # X = np.load("Dataset/x_train.npy", allow_pickle=True)
    # Y = np.load("Dataset/y_train.npy", allow_pickle=True)
    # x_test = np.load("Dataset/x_test.npy", allow_pickle=True)
    # y_test = np.load("Dataset/y_test.npy", allow_pickle=True)
    # print(f"{X.shape = }")
    # print(f"{len(set(Y)) = }")
    # print(f"{len(set(y_test)) = }")
    # fig = plt.figure()
    # ax = fig.add_subplot(xticks=[], yticks=[])
    # ax.imshow(X[0], cmap="gray")
    # ax.set_title(Y[0])
    # plt.show()
    # print(X[0])
    # print(Y[0])

    data = np.load("Dataset/data.npz", allow_pickle=True)
    X = data["x_test"]
    Y = data["y_test"]

    dump_test_files(X, Y, path_prefix=os.path.join("Dataset", "test"))

    fig, axes = plt.subplots(3, 3)
    axes = axes.flatten()

    visualize_image(axes, X, Y, fig)
    plt.tight_layout()
    plt.show()
    # data = np.load("Dataset/data.zip", allow_pickle=True)
    # X = data["x_train"]
    # print(X.shape)
