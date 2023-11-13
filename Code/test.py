import dill
from preprocess.preprocess import dataset
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

    dst = dataset("Dataset/Raw", shuffle=False)
    data = dst[0][0]()
    img = Image.fromarray(data, "RGB").convert("L")
    data = np.array(img) / 255
    plt.imshow(data, cmap="gray", vmin=0, vmax=1)
    plt.colorbar()
    plt.show()
