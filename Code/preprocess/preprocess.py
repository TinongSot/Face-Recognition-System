import os
import tqdm
import re
import numpy as np
import matplotlib.image as mpimg
from typing import Any, Callable

import dill


def identity(x):
    return x


class LazyData:
    def __init__(
        self,
        filename: str,
        load_strategy: str | Callable[[str | bytes], Any] | None = None,
    ):
        self._filename = filename
        self._data = None
        self._load_strategy = load_strategy

    def __call__(self):
        if self._data is None:
            self._build_data()
        return self._data

    def __load_strategy_image(buf) -> np.ndarray[np.uint8]:
        img = mpimg.imread(buf, format="jpg")
        return img

    def _build_data(self):
        print("Loading data from", self._filename)
        with open(self._filename, "rb") as f:
            if self._load_strategy is None:
                self._data = f.read()
            elif self._load_strategy == "image":
                self._data = LazyData.__load_strategy_image(self._filename)
            else:
                buf = f.read()
                self._data = self._load_strategy(buf)


def dataset(path, shuffle=True, random_state=None, verbose=True):
    ds = []
    tracker = tqdm.tqdm if verbose else identity
    pattern = re.compile(r"(.*)_(?:\d{4}).jpg")

    for root, dirs, files in tracker(os.walk(path)):
        for file in files:
            if file.endswith(".jpg"):
                match = pattern.match(file)
                if match:
                    target = match.group(1)
                ds.append([LazyData(os.path.join(root, file)), target])

    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(ds)

    return np.array(ds)


if __name__ == "__main__":
    # data = LazyData(
    #     "Dataset/Raw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg", load_strategy="image"
    # )
    # buf = data()
    # print(buf.shape)
    # nbuf = data()
    # print(nbuf.shape)
    data = dataset("Dataset/Raw")
    print(data[0:5])
    print(data.shape)

    with open("Dataset/catalog.dill", "wb") as f:
        dill.dump(data, f)
